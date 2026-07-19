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


def test_keep_warm_default_off():
    from app.config import Config
    assert Config().performance.keep_warm is False


def test_keep_warm_ping_nudges_stt_and_ollama():
    from unittest.mock import MagicMock
    import app.hotkey as hk
    from app.config import Config

    fake = MagicMock()
    fake.cfg = Config()
    fake.cfg.cleanup.enabled = True
    hk.PushToTalkApp._keep_warm_ping(fake)
    fake.transcriber.transcribe.assert_called_once()   # STT nudged
    fake.cleaner.warmup.assert_called_once()            # Ollama nudged


def test_keep_warm_ping_skips_ollama_when_cleanup_off():
    from unittest.mock import MagicMock
    import app.hotkey as hk
    from app.config import Config

    fake = MagicMock()
    fake.cfg = Config()
    fake.cfg.cleanup.enabled = False
    hk.PushToTalkApp._keep_warm_ping(fake)
    fake.transcriber.transcribe.assert_called_once()
    fake.cleaner.warmup.assert_not_called()


def test_start_cue_plays_immediately_on_press(monkeypatch):
    # The cue must sound the instant the key goes down — before the recorder is
    # even started — so it's immediate feedback, not delayed until speech.
    from unittest.mock import MagicMock
    import app.hotkey as hk
    from app.config import Config

    order = []
    monkeypatch.setattr(hk.sound, "play_start_cue",
                        lambda *a: order.append("cue"))

    cfg = Config()
    cfg.audio.start_sound = True
    cfg.context.surrounding = False
    cfg.stt.streaming = False

    fake = MagicMock()
    fake.cfg = cfg
    fake._busy.locked.return_value = False
    fake.recorder.recording = False
    fake.recorder.start.side_effect = lambda: order.append("record")

    hk.PushToTalkApp._on_press(fake)
    assert order == ["cue", "record"]   # cue fired first, synchronously


def test_start_cue_silent_when_disabled(monkeypatch):
    from unittest.mock import MagicMock
    import app.hotkey as hk
    from app.config import Config

    played = []
    monkeypatch.setattr(hk.sound, "play_start_cue", lambda *a: played.append(a))
    cfg = Config()
    cfg.audio.start_sound = False
    cfg.context.surrounding = False
    cfg.stt.streaming = False
    fake = MagicMock()
    fake.cfg = cfg
    fake._busy.locked.return_value = False
    fake.recorder.recording = False
    hk.PushToTalkApp._on_press(fake)
    assert played == []
