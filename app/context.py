"""Cross-utterance context: recent dictations inform the next one.

What you dictated a moment ago is the best predictor of the vocabulary,
casing, and spelling of what you dictate next ("Kubernetes" stays
"Kubernetes", a follow-up sentence continues the same register). The store
keeps a few recent utterances in memory only — nothing is written to disk,
and context expires after a few minutes so stale text can't bleed into an
unrelated dictation.

Sole consumer is the cleanup LLM: `cleanup_context()` is included as
reference-only text so terms are spelled/cased consistently, and the
divergence guard rejects output that copies context words in. Whisper is
deliberately NOT a consumer — initial_prompt text can leak into the raw
transcript on ambiguous audio, upstream of every guard.
"""

from __future__ import annotations

import time
from typing import Callable

from .config import ContextConfig


class ContextStore:
    def __init__(self, cfg: ContextConfig, clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self._clock = clock
        self._entries: list[tuple[float, str]] = []  # (monotonic ts, text)

    def add(self, text: str) -> None:
        """Record a delivered dictation as context for the next one."""
        text = text.strip()
        if not text:
            return
        self._entries.append((self._clock(), text))
        del self._entries[: -self.cfg.max_utterances]

    def replace_last(self, text: str) -> None:
        """Swap the newest entry (a training correction supersedes it)."""
        text = text.strip()
        if self._entries and text:
            self._entries[-1] = (self._entries[-1][0], text)

    def _recent_text(self) -> str:
        cutoff = self._clock() - self.cfg.expiry_seconds
        return " ".join(t for ts, t in self._entries if ts >= cutoff)

    def cleanup_context(self) -> str | None:
        """Recent dictation text for the cleanup LLM, or None."""
        if not self.cfg.enabled:
            return None
        return self._recent_text()[-self.cfg.max_chars:] or None
