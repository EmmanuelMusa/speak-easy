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
import re
import threading
from typing import Callable

from .cleanup import _CORRECTION_CUE_RE, looks_like_enumeration, reformat_enumeration
from .focus import Surrounding

log = logging.getLogger(__name__)

# Sentence end: terminal punctuation not preceded by a single-letter
# abbreviation ("p.m.", "e.g.") and followed by a space or end-of-text.
_SENT_END_RE = re.compile(r"(?<![.\s][A-Za-z])[.?!]+(?=\s|$)")


def _split_sentences(chunk: str) -> list[str]:
    out, prev = [], 0
    for m in _SENT_END_RE.finditer(chunk):
        sent = chunk[prev:m.end()].strip()
        if sent:
            out.append(sent)
        prev = m.end()
    rest = chunk[prev:].strip()
    if rest:
        out.append(rest)
    return out


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
        self._consumed = 0             # chars consumed of " ".join(parts)
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

    def _poll_once(self) -> None:
        parts = self._session.committed_parts()
        if len(parts) < 2:
            return  # the last part may still mutate; nothing stable yet
        stable = " ".join(parts[:-1])
        region = stable[self._consumed:]
        last = None
        for m in _SENT_END_RE.finditer(region):
            last = m
        if last is None:
            return
        chunk, self._consumed = region[: last.end()], self._consumed + last.end()
        for sent in _split_sentences(chunk):
            self._clean_one(sent)

    def _clean_one(self, raw_sent: str) -> None:
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
                surrounding=self._chunk_surrounding(first=first, last=False),
                reformat=False,  # per-chunk; the list is built once at finalize
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
        """Stop the worker, clean the remaining tail, return the full text."""
        self._stop.set()
        self._thread.join()  # waits out an in-flight sentence cleanup
        tail = raw_full[self._consumed:].strip()
        if tail:
            if self._raw and _CORRECTION_CUE_RE.search(tail):
                tail = self._raw.pop() + " " + tail
                self._cleaned.pop()
            first = not self._raw
            self._cleaned.append(
                self._cleaner.clean(
                    tail,
                    context=self._chunk_context(first),
                    surrounding=self._chunk_surrounding(first=first, last=True),
                    reformat=False,  # per-chunk; list is built once below
                )
            )
        log.info(
            "Live cleanup: %d sentence(s) pre-cleaned, tail %d chars",
            len(self._cleaned) - (1 if tail else 0), len(tail),
        )
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
