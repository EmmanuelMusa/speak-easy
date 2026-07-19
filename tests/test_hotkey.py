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


def _cue_fake(monkeypatch, level, recording=True):
    from unittest.mock import MagicMock
    import app.hotkey as hk
    from app.config import Config
    played = []
    monkeypatch.setattr(hk.sound, "play_start_cue", lambda *a: played.append(a))
    fake = MagicMock()
    fake.cfg = Config()
    fake._cue_gen = 1
    fake.recorder.recording = recording
    fake.recorder.level = level
    return hk, fake, played


def test_cue_plays_when_speech_is_heard(monkeypatch):
    # The cue sounds the moment the mic level clears the speech floor.
    hk, fake, played = _cue_fake(monkeypatch, level=0.05)   # clearly speaking
    hk.PushToTalkApp._cue_on_speech(fake, 1)
    assert len(played) == 1


def test_cue_stays_silent_on_a_silent_hold(monkeypatch):
    # A held key with no speech (silence / a spurious or stuck trigger) never
    # reaches the floor, so it never beeps. Regression: the "random" cue.
    monkeypatch.setattr(__import__("app.hotkey", fromlist=["_CUE_MAX_WAIT_S"]),
                        "_CUE_MAX_WAIT_S", 0.05)
    hk, fake, played = _cue_fake(monkeypatch, level=0.0002)  # noise floor
    hk.PushToTalkApp._cue_on_speech(fake, 1)
    assert played == []


def test_cue_bails_when_press_superseded_or_released(monkeypatch):
    # A stale generation (released / newer press) stops the cue even mid-speech.
    hk, fake, played = _cue_fake(monkeypatch, level=0.05)
    hk.PushToTalkApp._cue_on_speech(fake, 999)   # gen != fake._cue_gen
    assert played == []
    hk, fake, played = _cue_fake(monkeypatch, level=0.05, recording=False)
    hk.PushToTalkApp._cue_on_speech(fake, 1)
    assert played == []
