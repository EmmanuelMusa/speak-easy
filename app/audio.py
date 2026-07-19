"""Microphone capture for push-to-talk recording.

Records raw float32 mono audio into memory while the hotkey is held.
sounddevice is imported lazily so the module can be imported (e.g. by tests)
on machines without a working audio stack.
"""

from __future__ import annotations

import threading

import numpy as np

from .config import AudioConfig


def has_speech(audio, sample_rate: int, floor: float = 0.006) -> bool:
    """True if `audio` holds something loud enough to be speech, versus silence
    or steady background noise. A guard for engines without their own VAD
    (Parakeet transcribes the whole clip, so on a silent recording it can
    hallucinate a 'ghost' word).

    Speech is short loud bursts well above a quiet noise floor, so we take the
    loudest 30 ms window and require it to clear BOTH a small absolute floor
    (rejects true silence) AND a multiple of the noise floor (rejects flat hum/
    fan noise). Relative to the noise floor, so it works across mic gains."""
    if audio is None:
        return False
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size < int(0.12 * sample_rate):   # too brief to be a real dictation
        return False
    win = max(1, int(0.03 * sample_rate))
    n = audio.size // win
    if n == 0:
        return float(np.sqrt(np.mean(audio ** 2))) >= floor
    wr = np.sqrt(np.mean(audio[: n * win].reshape(n, win) ** 2, axis=1))
    peak = float(wr.max())
    noise = float(np.percentile(wr, 20))
    return peak >= floor and peak >= noise * 3.0


class Recorder:
    """Start/stop microphone capture; returns a float32 numpy array at stop."""

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self._chunks: list[np.ndarray] = []
        self._stream = None
        self._lock = threading.Lock()
        #: live RMS level of the mic (0.0 when not recording) — drives the
        #: overlay's waveform animation.
        self.level: float = 0.0

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd  # lazy: needs PortAudio at runtime only

        self._chunks = []

        def _callback(indata, frames, time_info, status):
            with self._lock:
                self._chunks.append(indata.copy())
            self.level = float(np.sqrt((indata ** 2).mean()))

        self._stream = sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            dtype="float32",
            callback=_callback,
        )
        self._stream.start()

    def snapshot(self) -> np.ndarray:
        """Return the audio captured so far without stopping the stream.

        Used by streaming transcription to run passes while recording
        continues. Cheap enough to call a few times per second.
        """
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            audio = np.concatenate(self._chunks, axis=0)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32)

    def stop(self) -> np.ndarray:
        """Stop capture and return the recorded audio as mono float32."""
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        self.level = 0.0
        self._stream.stop()
        self._stream.close()
        self._stream = None
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            audio = np.concatenate(self._chunks, axis=0)
            self._chunks = []
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32)
