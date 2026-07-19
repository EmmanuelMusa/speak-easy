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


def test_start_cue_is_debounced(monkeypatch):
    # A brief accidental trigger (released before the threshold) must NOT beep;
    # a genuine hold must. Regression: an accidental Ctrl+Shift+Space blip was
    # playing the start cue "on its own".
    import time
    from unittest.mock import MagicMock
    import app.hotkey as hk
    from app.config import Config

    played = []
    monkeypatch.setattr(hk.sound, "play_start_cue", lambda *a: played.append(a))
    monkeypatch.setattr(hk, "_CUE_DELAY_S", 0.05)

    cfg = Config()
    cfg.audio.start_sound = True
    cfg.context.surrounding = False   # don't spawn the surrounding-read thread
    cfg.stt.streaming = False         # don't build a StreamingSession

    fake = MagicMock()
    fake.cfg = cfg
    fake._busy.locked.return_value = False
    fake.recorder.recording = False
    fake._cue_timer = None

    # Quick tap: press then release before the delay -> cue cancelled.
    hk.PushToTalkApp._on_press(fake)
    hk.PushToTalkApp._on_release(fake)
    time.sleep(0.12)
    assert played == []

    # Genuine hold: press and wait past the delay -> cue plays once.
    fake._cue_timer = None
    hk.PushToTalkApp._on_press(fake)
    time.sleep(0.12)
    assert len(played) == 1
