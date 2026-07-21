"""Spoken emoji: "fire emoji" -> 🔥, and nothing else touched."""

from __future__ import annotations

import pytest

from app.emoji import ALIASES, EMOJI, apply_spoken_emoji as sub


def test_alias_resolves_to_the_emoji():
    assert sub("Great work on this fire emoji") == "Great work on this 🔥"
    assert sub("ship it rocket emoji") == "ship it 🚀"
    assert sub("that deserves a thumbs up emoji") == "that deserves a 👍"


def test_both_word_orders():
    assert sub("send a fire emoji") == "send a 🔥"
    assert sub("send an emoji fire") == "send an 🔥"


def test_plural_marker():
    assert sub("add some fire emojis") == "add some 🔥"


def test_longest_alias_wins():
    """'crying laughing' must not stop at the shorter 'crying' (😢)."""
    assert sub("crying laughing emoji") == "😂"
    assert sub("crying emoji") == "😢"
    assert sub("musical note emoji") == "🎵"
    assert sub("note emoji") == "📝"


def test_case_and_separator_insensitive():
    assert sub("FIRE EMOJI") == "🔥"
    assert sub("Thumbs-Up emoji") == "👍"
    assert sub("thumbs   up emoji") == "👍"


def test_tolerates_the_comma_the_stt_drops_in():
    assert sub("rocket, emoji") == "🚀"


def test_surrounding_punctuation_is_preserved():
    assert sub("ship it rocket emoji.") == "ship it 🚀."
    assert sub("really? fire emoji!") == "really? 🔥!"


def test_unknown_name_is_left_alone():
    """A missing entry should cost you an awkward sentence, never a mangled
    one."""
    assert sub("send a banana emoji") == "send a banana emoji"
    assert sub("emoji zzzznope") == "emoji zzzznope"


@pytest.mark.parametrize("text", [
    # The marker word is what makes this safe: no "emoji", no substitution.
    "Call the fire department",
    "That deserves a thumbs up",
    "I gave the rocket launch a star rating",
    "she has a heart of gold",
    # The reversed order is the risky one — these read as ordinary sentences
    # that happen to put a name right after the word "emoji".
    "use an emoji like a thumbs up",
    "does the emoji look right",
    "no emoji please",
])
def test_ordinary_speech_is_untouched(text):
    assert sub(text) == text


def test_no_emoji_word_short_circuits():
    long_prose = "the quick brown fox jumps over the lazy dog. " * 20
    assert sub(long_prose) == long_prose


def test_multiple_in_one_utterance():
    assert sub("the tea emoji and the coffee emoji") == "the 🍵 and the ☕"


def test_aliases_are_unique_across_the_table():
    """Two emoji answering to one name would make the winner depend on dict
    order. _build_aliases raises, so simply importing proves it — this states
    the rule and checks the table is non-trivial."""
    assert len(ALIASES) == sum(len(v) for v in EMOJI.values())
    assert len(EMOJI) > 100, "the curated set has shrunk unexpectedly"


def test_every_alias_actually_round_trips():
    """Guards against an alias that the regex can't match — e.g. one with
    punctuation that escaping or the \\b anchors would break."""
    for alias, char in ALIASES.items():
        assert sub(f"{alias} emoji") == char, f"{alias!r} did not resolve"


# --- wiring into the cleanup pipeline ----------------------------------------

def _cleaner(**kw):
    from app.cleanup import Cleaner
    from app.config import CleanupConfig
    cfg = CleanupConfig(enabled=False, **kw)   # enabled=False -> local path only
    return Cleaner(cfg)


def test_pass_is_wired_into_the_cleanup_pipeline():
    out = _cleaner().clean("ship it rocket emoji")
    assert "🚀" in out, out


def test_setting_disables_the_pass():
    out = _cleaner(spoken_emoji=False).clean("ship it rocket emoji")
    assert "🚀" not in out and "emoji" in out, out


def test_emoji_at_a_sentence_start_still_capitalizes_the_next_word():
    """The pass runs before capitalize_sentences precisely so this holds."""
    out = _cleaner().clean("rocket emoji ship it now")
    assert out.startswith("🚀"), out
    assert "Ship it now" in out, out


def test_leading_quote_or_bracket_also_capitalizes():
    """Not emoji-specific: the same gap swallowed capitalization after any
    non-letter opener."""
    from app.cleanup import capitalize_sentences as cap
    assert cap('"hello there"') == '"Hello there"'
    assert cap("(this is a note)") == "(This is a note)"
    assert cap("🔥 that was good") == "🔥 That was good"
    assert cap("Done. 🚀 ship it") == "Done. 🚀 Ship it"
    # A leading number must NOT drag capitalization onto a later word.
    assert cap("42 apples are nice") == "42 apples are nice"


# --- surviving the cleanup model ---------------------------------------------

def _llm_cleaner(reply: str):
    """A Cleaner whose Ollama call returns `reply`, to test what the guards do
    with it."""
    from unittest.mock import MagicMock, patch
    from app.cleanup import Cleaner
    from app.config import CleanupConfig
    fake = MagicMock()
    fake.json.return_value = {"response": reply}   # /api/generate's shape
    fake.raise_for_status.return_value = None
    cleaner = Cleaner(CleanupConfig(enabled=True))
    return cleaner, patch("app.cleanup.requests.post", return_value=fake)


def test_model_doing_the_substitution_itself_is_not_divergence():
    """The model usually converts "rocket emoji" to 🚀 on its own. The guard
    counts words, so it saw two words vanish for a character it cannot see —
    and threw away a perfectly good result."""
    cleaner, mocked = _llm_cleaner("So I think we should ship it today. 🚀")
    with mocked:
        out = cleaner.clean("so i think we should ship it today rocket emoji")
    assert out == "So I think we should ship it today. 🚀", out
    assert "I think" in out, "fell back to local cleanup, losing LLM casing"


def test_model_leaving_the_words_alone_also_works():
    cleaner, mocked = _llm_cleaner("So I think we should ship it today, rocket emoji.")
    with mocked:
        out = cleaner.clean("so i think we should ship it today rocket emoji")
    assert "🚀" in out and "emoji" not in out, out


def test_model_dropping_the_emoji_falls_back():
    """An emoji is not a word character, so too_divergent cannot see one go
    missing. Observed for real: asked for ☕, the model returned 🍵."""
    cleaner, mocked = _llm_cleaner("Lets grab 🍵 tomorrow morning.")
    with mocked:
        out = cleaner.clean("lets grab coffee emoji tomorrow morning")
    assert "☕" in out, out
    assert "🍵" not in out, out


def test_model_deleting_the_emoji_entirely_falls_back():
    cleaner, mocked = _llm_cleaner("Can you send me the file?")
    with mocked:
        out = cleaner.clean("can you send me the file thumbs up emoji")
    assert "👍" in out, out
