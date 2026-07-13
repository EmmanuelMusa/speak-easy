"""Streaming transcription: transcribe while the hotkey is still held.

The batch pipeline pays the whole Whisper pass after key release. A
StreamingSession instead runs passes over the growing recording buffer in a
background thread and *commits* segments that ended comfortably before the
live edge (the speaker has moved past them, so their transcription is
stable). On release only the uncommitted tail — typically the last second
or two — remains to transcribe, so perceived STT latency stays near-constant
regardless of how long you spoke.

Committed text from THIS utterance is fed to later passes as Whisper's
initial_prompt, so each chunk is decoded knowing what came before (casing,
vocabulary, sentence flow) instead of starting cold at an arbitrary
boundary. Nothing from previous dictations or the target document is ever
given to Whisper — prompt text can leak into the transcript on ambiguous
audio, and the transcript is upstream of every guard.

If speech never pauses, nothing commits and finish() degrades gracefully to
the batch behavior (one pass over everything).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

from .stt import append_gap_punctuation

log = logging.getLogger(__name__)

# Committed text passed back into Whisper as initial_prompt is capped so a
# long dictation can't crowd out the audio (Whisper reserves 224 tokens for
# the prompt; ~600 chars stays safely under that).
_MAX_PROMPT_CHARS = 600


class StreamingSession:
    """One push-to-talk hold: incremental passes, then a final tail pass."""

    def __init__(
        self,
        transcriber,
        snapshot: Callable[[], np.ndarray],
        sample_rate: int,
        window_seconds: float = 3.0,
        margin_seconds: float = 1.2,
        poll_interval: float = 0.25,
    ):
        self._transcriber = transcriber
        self._snapshot = snapshot
        self._sr = sample_rate
        self._window = window_seconds
        self._margin = margin_seconds
        self._poll = poll_interval
        self._committed: list[str] = []
        self._committed_samples = 0
        #: absolute end time (s) of the last absorbed segment — pauses
        #: between segments become punctuation, across pass boundaries too.
        self._prev_end_abs: float | None = None
        self._passes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="stt-stream", daemon=True
        )

    def start(self) -> "StreamingSession":
        self._thread.start()
        return self

    def committed_parts(self) -> list[str]:
        """Snapshot of committed text parts. All parts except the LAST are
        immutable from here on (the last may still gain pause punctuation),
        which is what lets live cleanup work on a stable prefix."""
        return list(self._committed)

    def _loop(self) -> None:
        while not self._stop.wait(self._poll):
            try:
                self._pass_once()
            except Exception:
                # A failed pass just means less pre-committed text; the
                # final pass in finish() still covers the audio.
                log.exception("Streaming pass failed; will retry")

    def _pass_once(self) -> None:
        pending = self._snapshot()[self._committed_samples:]
        pending_dur = len(pending) / self._sr
        if pending_dur < self._window:
            return
        self._passes += 1
        segs = self._transcriber.transcribe_segments(
            pending, initial_prompt=self._context()
        )
        # Segments that ended well before the live edge are stable; anything
        # newer may still be mid-word and gets re-transcribed next pass.
        cutoff = pending_dur - self._margin
        base = self._committed_samples / self._sr
        stable = []
        last_end = 0.0
        next_start = None
        for start, end, text in segs:
            if end > cutoff:
                next_start = start
                break
            stable.append((start, end, text))
            last_end = end
        if last_end > 0.0:
            self._absorb(stable, base)
            # Advance through the trailing silence to the next speech onset
            # (capped at the stability cutoff): word-aligned segment ends
            # can be slightly early, and re-transcribing that residue makes
            # the boundary word come out twice.
            advance = min(next_start, cutoff) if next_start is not None else last_end
            self._committed_samples += int(max(advance, last_end) * self._sr)

    def _absorb(self, segs, base: float) -> None:
        """Append segment texts, turning inter-segment pauses (absolute
        timeline, so pass/tail boundaries count too) into punctuation."""
        for start, end, text in segs:
            if text:
                if self._committed and self._prev_end_abs is not None:
                    before = self._committed[-1]
                    self._committed[-1] = append_gap_punctuation(
                        before, (base + start) - self._prev_end_abs
                    )
                    if self._committed[-1] is not before and \
                            self._committed[-1].endswith("."):
                        text = text[0].upper() + text[1:]
                self._committed.append(text)
            self._prev_end_abs = base + end

    def finish(self, audio: np.ndarray) -> str:
        """Stop passes, transcribe the uncommitted tail, return full text."""
        self._stop.set()
        self._thread.join()  # let an in-flight pass land its commits
        tail = audio[self._committed_samples:]
        t0 = time.perf_counter()
        try:
            tail_segs = self._transcriber.transcribe_segments(
                tail, initial_prompt=self._context()
            )
        except Exception:
            # Whatever was committed already is still worth delivering.
            log.exception("Tail transcription failed")
            tail_segs = []
        self._absorb(tail_segs, self._committed_samples / self._sr)
        log.info(
            "Streaming: %d passes pre-committed %.1fs/%.1fs; tail %.1fs took %.2fs",
            self._passes,
            self._committed_samples / self._sr,
            len(audio) / self._sr,
            len(tail) / self._sr,
            time.perf_counter() - t0,
        )
        return " ".join(self._committed).strip()

    def _context(self) -> str | None:
        joined = " ".join(self._committed)
        return joined[-_MAX_PROMPT_CHARS:] if joined else None
