"""Pause-based punctuation: silence between segments becomes , or . and the
SendInput INPUT struct stays the size Win32 demands."""

import ctypes

from app.injection import _INPUT
from app.stt import append_gap_punctuation, stitch_segments, classify_gap, collapse_ellipses


def test_long_pause_becomes_full_stop_and_capitalizes():
    segs = [(0.0, 2.0, "we should ship it"), (3.1, 4.5, "the docs need a pass")]
    assert stitch_segments(segs) == "we should ship it. The docs need a pass"


def test_short_pause_becomes_comma():
    segs = [(0.0, 2.0, "if the tests pass"), (2.5, 4.0, "we merge tonight")]
    assert stitch_segments(segs) == "if the tests pass, we merge tonight"


def test_tiny_gap_adds_nothing():
    segs = [(0.0, 2.0, "the quick brown"), (2.1, 3.0, "fox jumps")]
    assert stitch_segments(segs) == "the quick brown fox jumps"


def test_whisper_terminal_punctuation_wins():
    # Segment already ends with terminal punctuation: never doubled.
    segs = [(0.0, 2.0, "Is it ready?"), (3.5, 4.5, "I think so.")]
    assert stitch_segments(segs) == "Is it ready? I think so."


def test_long_pause_upgrades_trailing_comma_to_full_stop():
    # The voice outranks the language model: a 2s stop is not a comma.
    assert append_gap_punctuation("done,", 2.0) == "done."
    assert append_gap_punctuation("done,", 0.25) == "done,"  # short: keep
    segs = [(0.0, 2.0, "we shipped it,"), (4.0, 5.0, "the rest lands Friday")]
    assert stitch_segments(segs) == "we shipped it. The rest lands Friday"


def test_empty_segments_are_skipped_but_keep_the_timeline():
    segs = [(0.0, 2.0, "hello"), (2.2, 3.0, ""), (3.1, 4.0, "world")]
    # Gap measured from the empty segment's end (3.1 - 3.0): tiny, no comma.
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
