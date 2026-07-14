"""LiveCleanup: stable-sentence finalization, correction merging, tail stitch."""

from app.cleanup import looks_like_enumeration
from app.focus import Surrounding
from app.live_cleanup import LiveCleanup


class FakeSession:
    """Model-view sentences with a `stable` cut (how many are committed &
    stable) and an internal emit cursor mirroring the real session."""

    def __init__(self):
        self.sentences = []
        self.stable = 0
        self._emitted = 0

    def stable_sentences(self, source):
        out = self.sentences[self._emitted:self.stable]
        self._emitted = max(self._emitted, self.stable)
        return out

    def remaining_sentences(self, source):
        out = self.sentences[self._emitted:]
        self._emitted = len(self.sentences)
        return out


class FakeCleaner:
    """Marks each cleaned chunk so the stitch is visible; records calls."""

    def __init__(self):
        self.calls = []  # (raw, context, surrounding)
        from app.config import CleanupConfig
        self.cfg = CleanupConfig()

    def clean(self, model_text, fallback_text=None, context=None,
              surrounding=None, reformat=True):
        self.calls.append((model_text, context, surrounding))
        return f"<{model_text}>"


def _live(session, cleaner, surrounding=None, context=None):
    return LiveCleanup(
        session, cleaner,
        context_provider=lambda: context,
        surrounding_provider=lambda: surrounding,
    )


def test_only_stable_complete_sentences_are_cleaned():
    session, cleaner = FakeSession(), FakeCleaner()
    live = _live(session, cleaner)
    session.sentences = ["we shipped it."]      # nothing stable yet
    session.stable = 0
    live._poll_once()
    assert cleaner.calls == []
    session.sentences = ["we shipped it.", "the docs are"]
    session.stable = 1                          # first sentence now stable
    live._poll_once()
    assert [c[0] for c in cleaner.calls] == ["we shipped it."]
    live._poll_once()                           # no new stable sentence
    assert len(cleaner.calls) == 1


def test_correction_cue_re_cleans_previous_sentence():
    session, cleaner = FakeSession(), FakeCleaner()
    live = _live(session, cleaner)
    session.sentences = ["the meeting is at 9am.", "no sorry at 3pm."]
    session.stable = 2
    live._poll_once()
    assert cleaner.calls[-1][0] == "the meeting is at 9am. no sorry at 3pm."
    assert live._cleaned == ["<the meeting is at 9am. no sorry at 3pm.>"]


def test_finalize_cleans_tail_and_stitches():
    session, cleaner = FakeSession(), FakeCleaner()
    live = _live(session, cleaner)
    session.sentences = ["we shipped it.", "docs are next week"]
    session.stable = 1
    live._poll_once()
    live._thread.start()
    out = live.finalize("we shipped it. docs are next week")
    assert out == "<we shipped it.> <docs are next week>"
    assert cleaner.calls[-1][1] == "<we shipped it.>"


def test_finalize_with_no_precleaned_sentences_cleans_everything():
    session, cleaner = FakeSession(), FakeCleaner()
    live = _live(session, cleaner, context="history text")
    session.sentences = ["just one short thing"]
    session.stable = 0
    live._thread.start()
    out = live.finalize("just one short thing")
    assert out == "<just one short thing>"
    assert cleaner.calls[0][1] == "history text"


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
        from app.config import CleanupConfig
        self.cfg = CleanupConfig()

    def clean(self, model_text, fallback_text=None, context=None,
              surrounding=None, reformat=True):
        self.calls.append((model_text.strip(), reformat))
        return model_text.strip()


def test_finalize_reformats_split_enumeration_into_a_list():
    # Streaming split the list across sentences; finalize restructures the
    # assembled ordinal-led sentences into a numbered list deterministically —
    # no extra model call, and the per-sentence cleans must NOT each reformat.
    session, cleaner = FakeSession(), _PassthroughCleaner()
    live = _live(session, cleaner)
    session.sentences = ["We need three things.", "First, the budget.",
                         "Second, the timeline.", "third the plan"]
    session.stable = 3
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
    session.sentences = ["We shipped the release.", "docs are next week"]
    session.stable = 1
    live._poll_once()
    live._thread.start()
    out = live.finalize("We shipped the release. docs are next week")
    assert out == "We shipped the release. docs are next week"


def test_surrounding_split_first_gets_before_last_gets_after():
    sur = Surrounding(before="Existing text", after=" continues here")
    session, cleaner = FakeSession(), FakeCleaner()
    live = _live(session, cleaner, surrounding=sur)
    session.sentences = ["first sentence.", "tail words"]
    session.stable = 1
    live._poll_once()
    live._thread.start()
    live.finalize("first sentence. tail words")
    first_sur = cleaner.calls[0][2]
    tail_sur = cleaner.calls[-1][2]
    assert first_sur.before == "Existing text" and first_sur.after == ""
    assert tail_sur.before == "" and tail_sur.after == " continues here"
    assert cleaner.calls[0][1] is None
