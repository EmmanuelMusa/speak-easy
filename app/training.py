"""Training mode: learn from user feedback on dictation output.

Negative feedback pairs (raw transcript -> user's ideal text) are stored in
training_data.jsonl and power two mechanisms:

1. Few-shot injection — the most recent corrections are appended to the
   cleanup LLM's system prompt as examples, changing behavior immediately.
2. Vocabulary mining — small word-level substitutions between what the
   system produced and what the user wanted (misheard names, jargon) are
   saved to learned_vocab.json and merged into the preserve-terms list.

No model weights are touched; everything is instant and local. The JSONL
doubles as a dataset if a real fine-tune is ever wanted.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = _ROOT / "training_data.jsonl"
VOCAB_PATH = _ROOT / "learned_vocab.json"

# Substituted spans longer than this are style changes, not vocabulary.
_MAX_VOCAB_SPAN = 3
_WORD_SPLIT_RE = re.compile(r"[^\w']+")

# A genuine misheard term looks/sounds like what the model produced
# ("web sockets" -> "WebSockets", "Ogi up" -> "Ogiop"). A grammatical rewrite
# does not ("But" -> "What", "handled" -> "handle"). Below this
# character-level similarity the substitution is a correction, not a mishear,
# and must NOT be learned as preserve-forever vocabulary.
_MISHEAR_SIMILARITY = 0.6

# Ordinary English. A term made only of these is language the model rewrote,
# not a name/acronym/jargon term to preserve — so it is never learned as
# vocabulary. (Capitalization alone can't tell "Ogiop" the name from
# "Whatever" the sentence-initial common word; a wordlist can.) Deliberately
# broad on function words and the highest-frequency content words; a real
# term ("Kubernetes", "WebSockets", "Ogiop") simply won't appear here.
_COMMON_WORDS = frozenset("""
a an the and or but nor so to of in on at by for with from as than that which
who whom whose this these those there here what whatever whichever whenever
however whoever wherever i we he she it they you me him her us them my our your
their his its mine ours yours theirs myself yourself itself am is are was were
be been being do does did doing done have has had having will would shall
should can could may might must ought need dare not no nor yes ok okay
if then else when while because although though unless until since whether
about above across after against along among around before behind below beneath
beside besides between beyond during except inside into near off onto out
outside over past through throughout toward under underneath up upon within
without very quite really just too also even still yet again ever never always
often sometimes usually rarely almost enough more most much many few less least
some any all both each every either neither none one two three four five six
seven eight nine ten first second third next last other another same such
thing things way ways stuff lot lots kind sort part place time times day days
love hate like want wants wanted need needs needed make makes made get gets got
go goes going gone went come comes came see sees saw seen know knows knew known
think thinks thought say says said tell tells told ask asks asked give gives
gave take takes took use uses used find finds found handle handles handled
manage managed update updates updated improve improves improved change changed
relevant important expertise punctuation comma commas period sentence word words
good bad big small new old high low long short right wrong true false
""".split())


class TrainingStore:
    def __init__(self, data_path: Path = DATA_PATH, vocab_path: Path = VOCAB_PATH):
        self.data_path = data_path
        self.vocab_path = vocab_path
        self._lock = threading.Lock()

    # -- recording -----------------------------------------------------------

    def record(self, raw: str, output: str, verdict: str, ideal: str | None = None) -> None:
        """Append one feedback entry; mine vocabulary on negative feedback."""
        entry = {
            "ts": time.time(),
            "raw": raw,
            "output": output,
            "verdict": verdict,          # "ok" | "bad"
            "ideal": ideal or None,
        }
        with self._lock:
            with open(self.data_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if verdict == "bad" and ideal and ideal.strip() and ideal.strip() != output.strip():
            learned = self._mine_vocab(output, ideal)
            if learned:
                self._add_vocab(learned)
                log.info("Learned vocabulary: %s", learned)

    # -- few-shot corrections ---------------------------------------------

    def _all_entries(self) -> list[dict]:
        if not self.data_path.exists():
            return []
        entries: list[dict] = []
        with open(self.data_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def corrections(self, n: int | None = 5) -> list[dict]:
        """Negative feedback entries that carry an ideal text (most recent
        `n`, or all if n is None)."""
        found = [
            e for e in self._all_entries()
            if e.get("verdict") == "bad" and e.get("ideal")
        ]
        return found if n is None else found[-n:]

    def delete_correction(self, ts) -> None:
        """Remove the correction with the given timestamp (undo a lesson)."""
        kept = [e for e in self._all_entries() if e.get("ts") != ts]
        with self._lock:
            with open(self.data_path, "w", encoding="utf-8") as fh:
                for e in kept:
                    fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    def few_shot_block(self, n: int = 5) -> str:
        """Correction examples formatted for the system prompt ('' if none)."""
        pairs = self.corrections(n)
        if not pairs:
            return ""
        blocks = [
            f"Input: {e['raw']}\nOutput: {e['ideal']}" for e in pairs
        ]
        return (
            "\n\nThis user has corrected past outputs. Match these exactly "
            "when similar input appears:\n" + "\n\n".join(blocks)
        )

    # -- vocabulary ------------------------------------------------------------

    def learned_vocab(self) -> list[str]:
        if not self.vocab_path.exists():
            return []
        try:
            return list(json.loads(self.vocab_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return []

    def _write_vocab(self, vocab: list[str]) -> None:
        with self._lock:
            self.vocab_path.write_text(
                json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8"
            )

    def _add_vocab(self, terms: list[str]) -> None:
        vocab = self.learned_vocab()
        existing = {t.lower() for t in vocab}
        vocab.extend(t for t in terms if t.lower() not in existing)
        self._write_vocab(vocab)

    def remove_vocab(self, term: str) -> None:
        """Forget a learned word (undo a bad vocabulary lesson)."""
        self._write_vocab(
            [t for t in self.learned_vocab() if t.lower() != term.lower()]
        )

    @staticmethod
    def _is_distinctive(term: str) -> bool:
        """True if `term` carries a name / acronym / jargon token worth
        preserving — a token of length >= 3 that is not ordinary English.
        "Ogiop"/"WebSockets"/"Boba tea" qualify; "Whatever" and
        "with the commas" are all common words, so they do not."""
        return any(
            len(t) >= 3 and t.lower() not in _COMMON_WORDS
            for t in term.split()
        )

    @staticmethod
    def _mine_vocab(output: str, ideal: str) -> list[str]:
        """Find short word substitutions where the model *misheard* a term
        (names, jargon) — worth preserving. Two gates keep ordinary grammar
        corrections out: the ideal term must look distinctive (not a plain
        lowercase word), and it must be orthographically close to what the
        model produced (a mishear, not a reword). Without these, every
        correction like "But" -> "What" got saved as preserve-forever vocab
        and polluted the cleanup prompt."""
        out_words = [w for w in _WORD_SPLIT_RE.split(output) if w]
        ideal_words = [w for w in _WORD_SPLIT_RE.split(ideal) if w]
        sm = difflib.SequenceMatcher(
            a=[w.lower() for w in out_words], b=[w.lower() for w in ideal_words]
        )
        terms: list[str] = []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op != "replace":
                continue
            if (i2 - i1) > _MAX_VOCAB_SPAN or (j2 - j1) > _MAX_VOCAB_SPAN:
                continue
            term = " ".join(ideal_words[j1:j2])
            if len(term) < 3 or not any(c.isalpha() for c in term):
                continue
            if not TrainingStore._is_distinctive(term):
                continue
            was = " ".join(out_words[i1:i2])
            similarity = difflib.SequenceMatcher(
                None, was.lower().replace(" ", ""), term.lower().replace(" ", "")
            ).ratio()
            if similarity < _MISHEAR_SIMILARITY:
                continue
            terms.append(term)
        return terms
