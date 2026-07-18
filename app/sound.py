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
    """A soft, clearly-audible 'ready' blip: two short rising notes (E5 → B5),
    each under a raised-cosine (Hann) envelope so there are no edge clicks.
    Pitched high enough to carry over small laptop speakers (the old ~196 Hz
    tone was near-inaudible on them). 16-bit mono WAV bytes."""
    note, gap = 0.07, 0.012      # seconds per note / silent gap between
    notes = (659.25, 987.77)     # E5 -> B5: bright and pleasant
    frames = bytearray()
    for idx, f in enumerate(notes):
        n = int(_SAMPLE_RATE * note)
        for i in range(n):
            t = i / _SAMPLE_RATE
            env = 0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1))
            val = math.sin(2 * math.pi * f * t) * env * volume
            frames += struct.pack("<h", int(max(-1.0, min(1.0, val)) * 32767))
        if idx == 0:
            frames += b"\x00\x00" * int(_SAMPLE_RATE * gap)
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
