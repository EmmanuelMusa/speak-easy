"""LiveCleanup: stable-sentence finalization, correction merging, tail stitch."""

from app.cleanup import looks_like_enumeration
from app.focus import Surrounding
from app.live_cleanup import LiveCleanup, _split_sentences


class FakeSession:
    def __init__(self):
        self.parts = []

    def committed_parts(self):
        return list(self.parts)


class FakeCleaner:
    """Marks each cleaned chunk so the stitch is visible; records calls."""

    def __init__(self):
        self.calls = []  # (raw, context, surrounding)

    def clean(self, text, context=None, surrounding=None, reformat=True):
        self.calls.append((text, context, surrounding))
        return f"<{text}>"


def _live(session, cleaner, surrounding=None, context=None):
    return LiveCleanup(
        session, cleaner,
        context_provider=lambda: context,
        surrounding_provider=lambda: surrounding,
    )


def test_split_sentences_respects_abbreviations():
    assert _split_sentences("I arrive at 3 p.m. sharp. Then we start.") == \
        ["I arrive at 3 p.m. sharp.", "Then we start."]


def test_only_stable_complete_sentences_are_cleaned():
    session, cleaner = FakeSession(), FakeCleaner()
    live = _live(session, cleaner)
    session.parts = ["we shipped it."]          # last part: still mutable
    live._poll_once()
    assert cleaner.calls == []
    session.parts = ["we shipped it.", "the docs are"]  # first part now stable
    live._poll_once()
    assert [c[0] for c in cleaner.calls] == ["we shipped it."]
    live._poll_once()                            # no new stable sentence
    assert len(cleaner.calls) == 1


def test_correction_cue_re_cleans_previous_sentence():
    session, cleaner = FakeSession(), FakeCleaner()
    live = _live(session, cleaner)
    session.parts = ["the meeting is at 9am.", "no sorry at 3pm.", "and"]
    live._poll_once()
    # Second sentence corrects the first: merged and cleaned together.
    assert cleaner.calls[-1][0] == "the meeting is at 9am. no sorry at 3pm."
    assert live._cleaned == ["<the meeting is at 9am. no sorry at 3pm.>"]


def test_finalize_cleans_tail_and_stitches():
    session, cleaner = FakeSession(), FakeCleaner()
    live = _live(session, cleaner)
    session.parts = ["we shipped it.", "docs are next"]
    live._poll_once()
    live._thread.start()
    out = live.finalize("we shipped it. docs are next week")
    assert out == "<we shipped it.> <docs are next week>"
    # Later chunks get the already-cleaned text as context, not history.
    assert cleaner.calls[-1][1] == "<we shipped it.>"


def test_finalize_with_no_precleaned_sentences_cleans_everything():
    session, cleaner = FakeSession(), FakeCleaner()
    live = _live(session, cleaner, context="history text")
    live._thread.start()
    out = live.finalize("just one short thing")
    assert out == "<just one short thing>"
    assert cleaner.calls[0][1] == "history text"  # first chunk: history


def test_looks_like_enumeration_needs_two_strong_cues():
    assert looks_like_enumeration("first the budget second the timeline")
    assert looks_like_enumeration("number one open it number two save it")
    assert not looks_like_enumeration("first I went to the store")  # one cue
    assert not looks_like_enumeration("then I left and finally came home")  # weak


class _PassthroughCleaner:
    """Returns each chunk unchanged; records the reformat flag it was called
    with (streaming must pass reformat=False for individual sentences)."""

    def __init__(self):
        self.calls = []  # (text, reformat)

    def clean(self, text, context=None, surrounding=None, reformat=True):
        self.calls.append((text.strip(), reformat))
        return text.strip()


def test_finalize_reformats_split_enumeration_into_a_list():
    # Streaming split the list across sentences; finalize restructures the
    # assembled ordinal-led sentences into a numbered list deterministically —
    # no extra model call, and the per-sentence cleans must NOT each reformat.
    session, cleaner = FakeSession(), _PassthroughCleaner()
    live = _live(session, cleaner)
    session.parts = ["We need three things.", "First, the budget.",
                     "Second, the timeline.", "third the plan"]
    live._poll_once()
    live._thread.start()
    raw = "We need three things. First, the budget. Second, the timeline. third the plan"
    out = live.finalize(raw)
    assert out == ("We need three things:\n1. The budget.\n2. The timeline.\n"
                   "3. The plan.")
    assert all(reformat is False for _, reformat in cleaner.calls)


def test_finalize_leaves_non_enumeration_prose_alone():
    session, cleaner = FakeSession(), _PassthroughCleaner()
    live = _live(session, cleaner)
    session.parts = ["We shipped the release.", "docs are next"]
    live._poll_once()
    live._thread.start()
    out = live.finalize("We shipped the release. docs are next week")
    assert out == "We shipped the release. docs are next week"


def test_surrounding_split_first_gets_before_last_gets_after():
    sur = Surrounding(before="Existing text", after=" continues here")
    session, cleaner = FakeSession(), FakeCleaner()
    live = _live(session, cleaner, surrounding=sur)
    session.parts = ["first sentence.", "tail"]
    live._poll_once()
    live._thread.start()
    live.finalize("first sentence. tail words")
    first_sur = cleaner.calls[0][2]
    tail_sur = cleaner.calls[-1][2]
    assert first_sur.before == "Existing text" and first_sur.after == ""
    assert tail_sur.before == "" and tail_sur.after == " continues here"
    # Surrounding text present -> first chunk must NOT also get history.
    assert cleaner.calls[0][1] is None
