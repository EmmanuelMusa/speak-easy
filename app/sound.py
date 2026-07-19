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
    """A deep, punchy 'thump' — a kick-drum-like cue rather than a chime, so it
    reads as bass and doesn't blend in with Windows' high notification sounds.

    A pure low tone is inaudible on laptop speakers, so we build the thump from
    a fast downward pitch sweep (≈150 Hz → 55 Hz) with a quick attack and
    exponential decay, then soft-saturate it (tanh). The saturation adds
    harmonics of the low fundamental that small speakers CAN reproduce, so the
    ear still perceives the deep pitch (missing-fundamental effect). 16-bit mono.
    """
    dur = 0.22
    n = int(_SAMPLE_RATE * dur)
    # Sweep from an audible mid pitch DOWN to a deep one: the mid start carries
    # on small speakers (which roll off below ~200 Hz), the low end gives the
    # "thump". Slow enough to spend real time in the audible band.
    f_start, f_end = 330.0, 72.0
    frames = bytearray()
    phase = 0.0
    for i in range(n):
        t = i / _SAMPLE_RATE
        f = f_end + (f_start - f_end) * math.exp(-t / 0.045)
        phase += 2 * math.pi * f / _SAMPLE_RATE
        attack = min(1.0, t / 0.003)         # ~3 ms soft attack (no hard click)
        decay = math.exp(-t / 0.085)         # body of the thump
        release = min(1.0, (dur - t) / 0.012)  # fade the tail to zero (no click)
        val = math.sin(phase) * attack * decay
        val = math.tanh(val * (2.2 + 2.6 * volume))  # drive -> audible harmonics
        val *= (0.6 + 0.4 * volume) * release  # level tracks setting; clean tail
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
