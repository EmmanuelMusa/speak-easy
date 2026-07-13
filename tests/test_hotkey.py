"""Tests for push-to-talk key validation — rejecting bad key names before
they can brick the live rebind (and get persisted to config)."""

from app.hotkey import _binding_is_valid


def test_valid_single_key():
    assert _binding_is_valid("f9")


def test_valid_chord_with_spaces():
    assert _binding_is_valid("control + shift + space")


def test_valid_chord_no_spaces():
    assert _binding_is_valid("control+alt+d")


def test_invalid_ctrl_alias():
    # The library wants "control", not "ctrl" — a classic silent-fail name.
    assert not _binding_is_valid("ctrl + space")


def test_invalid_esc_alias():
    assert not _binding_is_valid("esc")


def test_invalid_empty():
    assert not _binding_is_valid("")
    assert not _binding_is_valid("   ")


def test_invalid_dangling_plus():
    assert not _binding_is_valid("control +")
