"""Recording-start cue: valid WAV synthesis, caching, and safe playback."""

import io
import wave

from app import sound


def test_render_cue_is_valid_mono_pcm_wav():
    data = sound._render_cue(0.18)
    w = wave.open(io.BytesIO(data))
    assert w.getnchannels() == 1
    assert w.getsampwidth() == 2
    assert w.getframerate() == sound._SAMPLE_RATE
    assert w.getnframes() > 0


def test_render_cue_starts_and_ends_near_silence():
    # The Hann envelope must fade in/out so there's no click at the edges.
    import struct
    data = sound._render_cue(0.5)
    w = wave.open(io.BytesIO(data))
    frames = w.readframes(w.getnframes())
    samples = struct.unpack("<%dh" % (len(frames) // 2), frames)
    assert abs(samples[0]) < 200 and abs(samples[-1]) < 200
    assert max(abs(s) for s in samples) > 1000  # but it's audible in the middle


def test_play_start_cue_never_raises_and_caches(monkeypatch):
    played = []

    class FakeWinsound:
        SND_MEMORY = 4
        SND_ASYNC = 1
        SND_NODEFAULT = 2

        def PlaySound(self, data, flags):
            played.append((data, flags))

    monkeypatch.setattr(sound.sys, "platform", "win32")
    monkeypatch.setitem(sound._cache, 0.2, b"")  # pre-seed nothing else
    sound._cache.clear()
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "winsound", FakeWinsound())

    sound.play_start_cue(0.2)
    sound.play_start_cue(0.2)
    assert len(played) == 2
    assert 0.2 in sound._cache  # rendered once, reused on the second call
