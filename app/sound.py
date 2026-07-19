"""Subtle audio cue for recording start.

A short, soft, low tone gives non-visual confirmation that the push-to-talk
hotkey registered — useful because your eyes are usually on what you're
dictating into, not the pill. The cue is rendered once to an in-memory WAV
and played asynchronously via winsound (stdlib, Windows) so the hotkey path
never blocks on audio. Silently no-ops off Windows or if audio is
unavailable.
"""

from __future__ import annotations

import io
import logging
import math
import struct
import sys
import threading
import wave

log = logging.getLogger(__name__)

_SAMPLE_RATE = 44100
#: rendered WAV bytes keyed by volume, so repeated presses don't re-synthesize.
_cache: dict[float, bytes] = {}


def _render_cue(volume: float) -> bytes:
    """A short, punchy notification 'thump' — a soft percussive knock, not a
    fading tone. The trick to reading as a *thump* rather than "something dying
    out" is a fast, percussive envelope: near-instant attack, a SHORT decay, and
    a clean cut — so it lands as one discrete hit and is gone.

    A quick pitch drop (≈200 → 120 Hz) gives the body; a tiny high click at the
    onset gives the attack "presence" that small speakers reproduce; tanh drive
    adds harmonics so the low body carries too. 16-bit mono WAV bytes.
    """
    dur = 0.14
    n = int(_SAMPLE_RATE * dur)
    f_start, f_end = 170.0, 92.0                    # deeper than before
    frames = bytearray()
    phase = 0.0
    for i in range(n):
        t = i / _SAMPLE_RATE
        f = f_end + (f_start - f_end) * math.exp(-t / 0.020)
        phase += 2 * math.pi * f / _SAMPLE_RATE
        attack = min(1.0, t / 0.0015)               # ~1.5 ms attack
        body = math.sin(phase) * math.exp(-t / 0.040)  # punchy, a touch more body
        click = math.sin(2 * math.pi * 520 * t) * math.exp(-t / 0.005) * 0.35
        release = min(1.0, (dur - t) / 0.012)       # clean cut, no lingering tail
        val = (body + click) * attack * release
        # Heavier drive: more harmonics of the deep fundamental -> the ear reads
        # it as deep AND it's plainly louder on small speakers.
        val = math.tanh(val * (4.0 + 2.0 * volume))
        val *= 0.92 * release
        frames += struct.pack("<h", int(max(-1.0, min(1.0, val)) * 32767))
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SAMPLE_RATE)
        w.writeframes(bytes(frames))
    return out.getvalue()


def play_start_cue(volume: float = 0.18) -> threading.Thread | None:
    """Play the recording-start cue without blocking the hotkey path. Never
    raises. Returns the player thread (or None if unavailable) for tests.

    winsound CANNOT play asynchronously from an in-memory WAV — SND_MEMORY |
    SND_ASYNC raises "Cannot play asynchronously from memory", which the old
    code swallowed at debug level, so the cue silently never played. Instead we
    play SYNCHRONOUSLY (the only mode that works from memory) on a throwaway
    daemon thread, so the ~150 ms playback still doesn't block the caller.
    """
    if sys.platform != "win32":
        return None
    try:
        import winsound
    except Exception:
        return None

    def _play() -> None:
        try:
            wav = _cache.get(volume)
            if wav is None:
                wav = _cache[volume] = _render_cue(volume)
            winsound.PlaySound(
                wav, winsound.SND_MEMORY | winsound.SND_NODEFAULT
            )
        except Exception as exc:  # audio device busy/missing — cue is optional
            log.debug("start cue failed: %s", exc)

    t = threading.Thread(target=_play, daemon=True, name="start-cue")
    t.start()
    return t
