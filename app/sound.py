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
import wave

log = logging.getLogger(__name__)

_SAMPLE_RATE = 44100
#: rendered WAV bytes keyed by volume, so repeated presses don't re-synthesize.
_cache: dict[float, bytes] = {}


def _render_cue(volume: float) -> bytes:
    """A soft low 'start' blip: a G3 root with a gently-delayed fifth above
    it, under a raised-cosine (Hann) envelope so there are no edge clicks.
    ~150 ms, 16-bit mono WAV bytes."""
    dur = 0.15
    n = int(_SAMPLE_RATE * dur)
    f_root, f_fifth = 196.00, 293.66  # G3 + D4: a soft fifth reads as "on"
    frames = bytearray()
    for i in range(n):
        t = i / _SAMPLE_RATE
        env = 0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1))  # smooth in/out
        lift = min(1.0, t / 0.05)  # fifth eases in over the first 50 ms
        s = math.sin(2 * math.pi * f_root * t) + 0.5 * lift * math.sin(
            2 * math.pi * f_fifth * t
        )
        val = max(-1.0, min(1.0, s / 1.5)) * env * volume
        frames += struct.pack("<h", int(val * 32767))
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SAMPLE_RATE)
        w.writeframes(bytes(frames))
    return out.getvalue()


def play_start_cue(volume: float = 0.18) -> None:
    """Play the recording-start cue asynchronously. Never raises."""
    if sys.platform != "win32":
        return
    try:
        import winsound
    except Exception:
        return
    try:
        wav = _cache.get(volume)
        if wav is None:
            wav = _cache[volume] = _render_cue(volume)
        winsound.PlaySound(
            wav,
            winsound.SND_MEMORY | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
        )
    except Exception as exc:  # audio device busy/missing — cue is optional
        log.debug("start cue failed: %s", exc)
