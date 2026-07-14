"""Pause-based punctuation: silence between segments becomes , or . and the
SendInput INPUT struct stays the size Win32 demands."""

import ctypes

from app.injection import _INPUT
from app.stt import (
    append_gap_punctuation, stitch_segments, classify_gap, collapse_ellipses,
    resolve_model, resolve_fallback, split_into_sentences, Transcript,
)


def _parts_boundaries(segs):
    from app.stt import stitch
    return stitch(segs)


def test_long_pause_becomes_full_stop_no_forced_capital():
    segs = [(0.0, 2.0, "we should ship it"), (3.1, 4.5, "the docs need a pass")]
    # Period from the pause, but the next word is NOT force-capitalized.
    assert stitch_segments(segs) == "we should ship it. the docs need a pass"


def test_short_pause_becomes_comma():
    segs = [(0.0, 2.0, "if the tests pass"), (2.5, 4.0, "we merge tonight")]
    assert stitch_segments(segs) == "if the tests pass, we merge tonight"


def test_tiny_gap_adds_nothing():
    segs = [(0.0, 2.0, "the quick brown"), (2.1, 3.0, "fox jumps")]
    assert stitch_segments(segs) == "the quick brown fox jumps"


def test_whisper_terminal_punctuation_wins():
    segs = [(0.0, 2.0, "Is it ready?"), (3.5, 4.5, "I think so.")]
    assert stitch_segments(segs) == "Is it ready? I think so."


def test_long_pause_upgrades_trailing_comma_to_full_stop():
    assert append_gap_punctuation("done,", 2.0) == "done."
    assert append_gap_punctuation("done,", 0.25) == "done,"
    segs = [(0.0, 2.0, "we shipped it,"), (4.0, 5.0, "the rest lands Friday")]
    assert stitch_segments(segs) == "we shipped it. the rest lands Friday"


def test_empty_segments_are_skipped_but_keep_the_timeline():
    segs = [(0.0, 2.0, "hello"), (2.2, 3.0, ""), (3.1, 4.0, "world")]
    assert stitch_segments(segs) == "hello world"


def test_thinking_pause_gets_no_punctuation():
    # A pause after a function word is the speaker thinking, not punctuating.
    assert append_gap_punctuation("we should", 2.0) == "we should"
    assert append_gap_punctuation("move it to the", 0.5) == "move it to the"
    assert append_gap_punctuation("because", 1.0) == "because"
    # Content words still get punctuated normally.
    assert append_gap_punctuation("we shipped the fix", 2.0) == "we shipped the fix."
    # Object-capable pronouns can end sentences and are not suppressed.
    assert append_gap_punctuation("I fixed it", 2.0) == "I fixed it."


def test_classify_gap_kinds():
    assert classify_gap("we shipped the fix", 2.0) == "period"
    assert classify_gap("if the tests pass", 0.3) == "comma"
    assert classify_gap("the quick brown", 0.1) == "none"
    # Function word -> speaker thinking, never punctuation.
    assert classify_gap("we should", 2.0) == "none"
    # Already ends with a Whisper terminal -> nothing to add.
    assert classify_gap("Is it ready?", 2.0) == "none"
    # A long stop after a trailing comma is a full stop.
    assert classify_gap("done,", 2.0) == "period"


def test_input_struct_matches_win32_size():
    """Regression: sizeof(INPUT) must be 40 on x64 (union sized for
    MOUSEINPUT). At 32, SendInput rejects every call and types nothing."""
    expected = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
    assert ctypes.sizeof(_INPUT) == expected


def test_collapse_ellipses():
    # Trailing "..." (trailing-off speech) -> a single period.
    assert collapse_ellipses("I was just thinking...") == "I was just thinking."
    assert collapse_ellipses("done…") == "done."
    # Internal ellipsis -> a single space.
    assert collapse_ellipses("wait... what") == "wait what"
    # A normal single period is untouched.
    assert collapse_ellipses("e.g. this") == "e.g. this"
    assert collapse_ellipses("all good.") == "all good."


def test_model_view_drops_pause_punctuation():
    segs = [(0.0, 2.0, "we should ship it"), (3.1, 4.5, "the docs need a pass")]
    parts, boundaries = _parts_boundaries(segs)
    # model view: no pause marks at all (the LLM will punctuate)
    assert resolve_model(parts, boundaries, "model") == \
        "we should ship it the docs need a pass"
    # "pauses" source delegates to the fallback view
    assert resolve_model(parts, boundaries, "pauses") == \
        "we should ship it. the docs need a pass"
    assert resolve_fallback(parts, boundaries) == \
        "we should ship it. the docs need a pass"


def test_split_into_sentences_uses_period_pauses_and_terminals():
    # period pause after part 0; Whisper terminal ends part 2.
    parts = ["we should ship it", "the docs need a pass.", "and then deploy"]
    boundaries = ["period", "none"]
    assert split_into_sentences(parts, boundaries) == [(0, 0), (1, 1), (2, 2)]


def test_split_into_sentences_respects_abbreviations():
    # A part ending in an abbreviation ("p.m.") is NOT a sentence end.
    parts = ["I arrive at 3 p.m.", "sharp then we start."]
    boundaries = ["none"]
    assert split_into_sentences(parts, boundaries) == [(0, 1)]


def test_transcript_views():
    t = Transcript(
        parts=["hello there", "world"], boundaries=["comma"],
    )
    assert t.model_text("model") == "hello there world"
    assert t.fallback_text == "hello there, world"
