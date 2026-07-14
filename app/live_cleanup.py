"""Streaming cleanup: clean finished sentences while dictation continues.

The cleanup LLM's generation time scales with text length, so cleaning the
whole utterance at key release is the last latency that grows with how long
you spoke. This worker watches the streaming session's committed text and,
as soon as a sentence is complete AND stable (terminal punctuation, not in
the still-mutable last part), cleans it in the background. At release only
the unfinished tail sentence remains, so cleanup wait is near-constant.

Correctness carve-outs:
- A sentence containing a self-correction cue ("no sorry", "scratch that")
  may be correcting the PREVIOUS sentence, so the two are merged and
  re-cleaned together (same rule for the tail at release).
- The first chunk gets the surrounding-text continuation rules (mid-sentence
  casing); the final chunk gets the after-cursor rules (no stray trailing
  period). Middle chunks are ordinary sentences.
- Each chunk is guarded by the divergence check inside Cleaner.clean, so a
  bad LLM response degrades that one sentence to the local strip, not the
  whole dictation.

Enumerations ("first... second... third...", "number one... number two...")
dictated with pauses get split across sentences and would clean inline; a
holistic re-clean pass in finalize() reforms them into a numbered list when
the whole utterance looks like an enumeration (see _looks_like_enumeration).
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from .cleanup import _CORRECTION_CUE_RE, looks_like_enumeration, reformat_enumeration
from .focus import Surrounding

log = logging.getLogger(__name__)


class LiveCleanup:
    def __init__(
        self,
        session,
        cleaner,
        context_provider: Callable[[], str | None],
        surrounding_provider: Callable[[], "Surrounding | None"],
        poll_interval: float = 0.5,
    ):
        self._session = session
        self._cleaner = cleaner
        self._get_context = context_provider
        self._get_surrounding = surrounding_provider
        self._poll = poll_interval
        self._raw: list[str] = []      # raw sentences, aligned with _cleaned
        self._cleaned: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="live-cleanup", daemon=True
        )

    def start(self) -> "LiveCleanup":
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self._poll):
            try:
                self._poll_once()
            except Exception:
                log.exception("Live cleanup pass failed; final pass covers it")

    def _source(self) -> str:
        return getattr(self._cleaner.cfg, "punctuation_source", "model")

    def _poll_once(self) -> None:
        for sent in self._session.stable_sentences(self._source()):
            self._clean_one(sent, last=False)

    def _clean_one(self, raw_sent: str, last: bool) -> None:
        # A correction may refer to the previous sentence — re-clean both.
        if self._raw and _CORRECTION_CUE_RE.search(raw_sent):
            raw_sent = self._raw.pop() + " " + raw_sent
            self._cleaned.pop()
        first = not self._raw
        self._raw.append(raw_sent)
        self._cleaned.append(
            self._cleaner.clean(
                raw_sent,
                context=self._chunk_context(first),
                surrounding=self._chunk_surrounding(first=first, last=last),
                reformat=False,
            )
        )

    def _chunk_context(self, first: bool) -> str | None:
        if first:
            s = self._get_surrounding()
            if s is not None and s.before.strip():
                return None  # surrounding text subsumes dictation history
            return self._get_context()
        return " ".join(self._cleaned)[-600:] or None

    def _chunk_surrounding(self, first: bool, last: bool) -> "Surrounding | None":
        orig = self._get_surrounding()
        if orig is None:
            return None
        return Surrounding(
            before=orig.before if first else "",
            after=orig.after if last else "",
            app=orig.app,
        )

    def finalize(self, raw_full: str) -> str:
        """Stop the worker, clean the remaining sentences, return full text."""
        self._stop.set()
        self._thread.join()
        remaining = self._session.remaining_sentences(self._source())
        for i, sent in enumerate(remaining):
            self._clean_one(sent, last=(i == len(remaining) - 1))
        log.info("Live cleanup: %d sentence(s) assembled", len(self._cleaned))
        assembled = " ".join(p for p in self._cleaned if p).strip()
        # Streaming cleans one sentence at a time, so an enumeration dictated
        # with pauses ("...three ways. Firstly, I'll X. Secondly, I'll Y.")
        # arrives as separate cleaned sentences. Restructure the assembled text
        # into a numbered list deterministically — no extra model call, and it
        # works regardless of how the model handled each sentence. Skip when the
        # dictation continues text at the caret (a mid-sentence continuation
        # shouldn't become a standalone list).
        s = self._get_surrounding()
        mid = s is not None and (s.mid_sentence or s.continues_after)
        if not mid and looks_like_enumeration(raw_full):
            listed = reformat_enumeration(assembled)
            if listed != assembled:
                log.info("Live cleanup: enumeration reformatted into a list")
            return listed
        return assembled
