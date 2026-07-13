# Punctuation Authority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move punctuation and casing authority from deterministic pause heuristics onto the cleanup model, with an A/B toggle, while keeping the offline fallback and streaming pre-cleaning working.

**Architecture:** The transcript is carried as structured parts (raw Whisper segment texts + the pause kind between each pair: `none`/`comma`/`period`). Two resolvers derive a *model view* (Whisper punctuation only, for the LLM) and a *fallback view* (pause punctuation applied, for the offline strip). A `[cleanup] punctuation_source` toggle folds into the model resolver so every consumer — including streaming — honors it. Two always-on fixes (no pause-forced capital; collapse `…`/`...`) apply regardless.

**Tech Stack:** Python 3 (stdlib `tomllib`, `re`, `dataclasses`, `difflib`), faster-whisper, PySide6, requests → Ollama.

## Global Constraints

- No new third-party dependencies — stdlib + existing packages only.
- Tests run with `.venv/Scripts/python.exe -m pytest` (Windows, Git Bash).
- Use `127.0.0.1`, never `localhost` (Windows IPv6 penalty).
- Commit messages: NO `Co-Authored-By` trailer, NO AI-authorship attribution (user global rule).
- **This directory is not a git repository.** If `git` is unavailable, skip every "Commit" step (the work is still valid); do not run `git init` without the user's approval.
- Default cleanup model in `config.toml` is `llama3.2:3b`; `CleanupConfig` dataclass default is `llama3.1:8b`. Do not change these here.
- Existing behavior locked by tests must stay green except where a task explicitly rewrites a test.

**Reference spec:** `docs/superpowers/specs/2026-07-13-punctuation-authority-design.md`

---

## Execution note: two groups

- **Group A (Tasks 1–7)** — foundations, both always-on fixes, non-streaming model authority, and the toggle. After Task 7 the app is fully working: `…` collapse and "no pause forces a capital" apply everywhere; the non-streaming path and the toggle are complete. The streaming path still feeds pause-punctuated chunks to the LLM (interim) but already benefits from the always-on fixes. A natural review checkpoint.
- **Group B (Tasks 8–9)** — restructure streaming so the default (streaming) path also sends clean model-view text to the LLM and honors the toggle.

---

## Task 1: Config — `punctuation_source`

**Files:**
- Modify: `app/config.py` (CleanupConfig dataclass)
- Modify: `config.toml` (`[cleanup]` section)
- Test: `tests/test_config_punctuation.py` (create)

**Interfaces:**
- Produces: `CleanupConfig.punctuation_source: str` (default `"model"`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_punctuation.py`:

```python
"""punctuation_source config field: default and load from TOML."""
from app.config import CleanupConfig, load_config


def test_default_is_model():
    assert CleanupConfig().punctuation_source == "model"


def test_loads_from_toml(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[cleanup]\npunctuation_source = "pauses"\n', encoding="utf-8"
    )
    assert load_config(cfg_file).cleanup.punctuation_source == "pauses"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config_punctuation.py -v`
Expected: FAIL (`AttributeError: 'CleanupConfig' object has no attribute 'punctuation_source'`).

- [ ] **Step 3: Add the field**

In `app/config.py`, inside `@dataclass class CleanupConfig`, add after `battery_timeout_multiplier`:

```python
    # Where sentence punctuation/casing comes from. "model": the cleanup LLM
    # receives words + Whisper punctuation only and punctuates from context.
    # "pauses": the LLM also gets the deterministic pause-derived punctuation
    # (legacy behavior / A-B baseline). The offline fallback always keeps the
    # pause punctuation regardless of this setting.
    punctuation_source: str = "model"
```

- [ ] **Step 4: Document in config.toml**

In `config.toml`, inside `[cleanup]`, add after the `battery_timeout_multiplier` line:

```toml
# Where sentence punctuation & capitalization come from:
#   "model"  = the cleanup AI decides, from the words + context (recommended)
#   "pauses" = also feed it the pause-timing punctuation (older behavior)
# Flip this and restart to A/B the two. The offline fallback (AI off/unreachable)
# always keeps the pause punctuation either way.
punctuation_source = "model"
```

- [ ] **Step 5: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config_punctuation.py tests/test_training.py::test_save_config_updates_preserves_comments -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/config.py config.toml tests/test_config_punctuation.py
git commit -m "feat(config): add cleanup.punctuation_source toggle"
```

---

## Task 2: `classify_gap` — extract pause decision (behavior-preserving)

**Files:**
- Modify: `app/stt.py` (`append_gap_punctuation` and above)
- Test: `tests/test_stt.py` (add cases)

**Interfaces:**
- Produces: `classify_gap(prev_text: str, gap: float) -> str` returning `"none"`, `"comma"`, or `"period"`.
- `append_gap_punctuation(text, gap)` keeps its existing signature/behavior.

- [ ] **Step 1: Write the failing test**

In `tests/test_stt.py`, update the import line and append tests:

```python
from app.stt import append_gap_punctuation, stitch_segments, classify_gap
```

```python
def test_classify_gap_kinds():
    assert classify_gap("we shipped the fix", 2.0) == "period"
    assert classify_gap("if the tests pass", 0.3) == "comma"
    assert classify_gap("the quick brown", 0.1) == "none"
    # Function word -> speaker thinking, never punctuation.
    assert classify_gap("we should", 2.0) == "none"
    # Already ends with a Whisper terminal -> nothing to add.
    assert classify_gap("Is it ready?", 2.0) == "none"
    # A long stop after a trailing comma is a full stop.
    assert classify_gap("done,", 2.0) == "period"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_stt.py::test_classify_gap_kinds -v`
Expected: FAIL (`ImportError: cannot import name 'classify_gap'`).

- [ ] **Step 3: Implement `classify_gap`, rewrite `append_gap_punctuation` over it**

In `app/stt.py`, replace the whole `append_gap_punctuation` function with:

```python
def classify_gap(prev_text: str, gap: float) -> str:
    """Classify the pause that followed `prev_text`: 'none' | 'comma' |
    'period'. Mirrors append_gap_punctuation's decision so the two never
    diverge: a pause after a function word is the speaker thinking (none); a
    Whisper terminal already present needs nothing; a long stop is a period
    (upgrading a trailing comma); a short stop after a word is a comma."""
    if not prev_text or _pause_is_thinking(prev_text):
        return "none"
    last = prev_text[-1]
    if gap >= PERIOD_GAP_S:
        if last.isalnum() or last == ",":
            return "period"
        return "none"
    if gap >= COMMA_GAP_S and last.isalnum():
        return "comma"
    return "none"


def append_gap_punctuation(text: str, gap: float) -> str:
    """Punctuate `text` for a `gap`-second pause that followed it. Thin
    wrapper over classify_gap so a single place owns the decision."""
    kind = classify_gap(text, gap)
    if kind == "period":
        if text[-1].isalnum():
            return text + "."
        if text[-1] == ",":
            return text[:-1] + "."
        return text
    if kind == "comma":
        return text + ","
    return text
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_stt.py -v`
Expected: PASS (new `test_classify_gap_kinds` plus all existing `append_gap_punctuation`/`stitch_segments` tests still green — behavior unchanged).

- [ ] **Step 5: Commit**

```bash
git add app/stt.py tests/test_stt.py
git commit -m "refactor(stt): extract classify_gap from append_gap_punctuation"
```

---

## Task 3: `collapse_ellipses` helper

**Files:**
- Modify: `app/stt.py`
- Test: `tests/test_stt.py`

**Interfaces:**
- Produces: `collapse_ellipses(text: str) -> str`.

- [ ] **Step 1: Write the failing test**

Add to the `app.stt` import in `tests/test_stt.py`: `collapse_ellipses`. Append:

```python
def test_collapse_ellipses():
    # Trailing "..." (trailing-off speech) -> a single period.
    assert collapse_ellipses("I was just thinking...") == "I was just thinking."
    assert collapse_ellipses("done…") == "done."
    # Internal ellipsis -> a single space.
    assert collapse_ellipses("wait... what") == "wait what"
    # A normal single period is untouched.
    assert collapse_ellipses("e.g. this") == "e.g. this"
    assert collapse_ellipses("all good.") == "all good."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_stt.py::test_collapse_ellipses -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement**

In `app/stt.py`, add near the other helpers (after `append_gap_punctuation`):

```python
def collapse_ellipses(text: str) -> str:
    """Collapse ellipses Whisper emits for trailing-off speech: a trailing
    ellipsis becomes a single period; an internal one becomes a single space.
    A lone period (e.g. "e.g.", "3.14") is left alone — only runs of 2+ dots
    or the Unicode ellipsis are touched."""
    text = re.sub(r"\s*(?:…|\.{2,})\s*$", ".", text)   # trailing -> "."
    text = re.sub(r"\s*(?:…|\.{2,})\s*", " ", text)    # internal -> " "
    return text.strip()
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_stt.py::test_collapse_ellipses -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/stt.py tests/test_stt.py
git commit -m "feat(stt): add collapse_ellipses helper"
```

---

## Task 4: View resolvers, `Transcript`, `stitch`, `split_into_sentences` — drop pause-forced capitalization

**Files:**
- Modify: `app/stt.py` (`stitch_segments`, `transcribe`, add functions/dataclass)
- Test: `tests/test_stt.py` (update the 6 `stitch_segments` assertions; add resolver + split tests)

**Interfaces:**
- Produces:
  - `stitch(segs) -> tuple[list[str], list[str]]` (parts, boundaries).
  - `resolve_fallback(parts: list[str], boundaries: list[str]) -> str`.
  - `resolve_model(parts: list[str], boundaries: list[str], source: str) -> str`.
  - `ends_sentence(text: str) -> bool`.
  - `split_into_sentences(parts: list[str], boundaries: list[str]) -> list[tuple[int, int]]` (inclusive part-index ranges).
  - `@dataclass Transcript` with `.parts`, `.boundaries`, `.model_text(source: str) -> str`, `.fallback_text` (property).
  - `stitch_segments(segs) -> str` still returns the fallback view (now WITHOUT pause-forced capitalization).
  - `Transcriber.transcribe(audio, initial_prompt=None) -> Transcript`.

- [ ] **Step 1: Update the failing tests**

In `tests/test_stt.py`, extend the import:

```python
from app.stt import (
    append_gap_punctuation, stitch_segments, classify_gap, collapse_ellipses,
    resolve_model, resolve_fallback, split_into_sentences, Transcript,
)
```

Replace the six existing `stitch_segments` assertions (pause path no longer capitalizes — the downstream `strip_fillers`/LLM own casing):

```python
def test_long_pause_becomes_full_stop_no_forced_capital():
    segs = [(0.0, 2.0, "we should ship it"), (3.1, 4.5, "the docs need a pass")]
    # Period from the pause, but the next word is NOT force-capitalized.
    assert stitch_segments(segs) == "we should ship it. the docs need a pass"


def test_short_pause_becomes_comma():
    segs = [(0.0, 2.0, "if the tests pass"), (2.5, 4.0, "we merge tonight")]
    assert stitch_segments(segs) == "if the tests pass, we merge tonight"


def test_tiny_gap_adds_nothing():
    segs = [(0.0, 2.0, "the quick brown"), (2.1, 3.0, "fox jumps")]
    assert stitch_segments(segs) == "the quick brown fox jumps"


def test_whisper_terminal_punctuation_wins():
    segs = [(0.0, 2.0, "Is it ready?"), (3.5, 4.5, "I think so.")]
    assert stitch_segments(segs) == "Is it ready? I think so."


def test_long_pause_upgrades_trailing_comma_to_full_stop():
    assert append_gap_punctuation("done,", 2.0) == "done."
    assert append_gap_punctuation("done,", 0.25) == "done,"
    segs = [(0.0, 2.0, "we shipped it,"), (4.0, 5.0, "the rest lands Friday")]
    assert stitch_segments(segs) == "we shipped it. the rest lands Friday"


def test_empty_segments_are_skipped_but_keep_the_timeline():
    segs = [(0.0, 2.0, "hello"), (2.2, 3.0, ""), (3.1, 4.0, "world")]
    assert stitch_segments(segs) == "hello world"
```

Append resolver / view / split tests:

```python
def test_model_view_drops_pause_punctuation():
    segs = [(0.0, 2.0, "we should ship it"), (3.1, 4.5, "the docs need a pass")]
    parts, boundaries = _parts_boundaries(segs)
    # model view: no pause marks at all (the LLM will punctuate)
    assert resolve_model(parts, boundaries, "model") == \
        "we should ship it the docs need a pass"
    # "pauses" source delegates to the fallback view
    assert resolve_model(parts, boundaries, "pauses") == \
        "we should ship it. the docs need a pass"
    assert resolve_fallback(parts, boundaries) == \
        "we should ship it. the docs need a pass"


def test_split_into_sentences_uses_period_pauses_and_terminals():
    # period pause after part 0; Whisper terminal ends part 2.
    parts = ["we should ship it", "the docs need a pass.", "and then deploy"]
    boundaries = ["period", "none"]
    assert split_into_sentences(parts, boundaries) == [(0, 0), (1, 1), (2, 2)]


def test_split_into_sentences_respects_abbreviations():
    # A part ending in an abbreviation ("p.m.") is NOT a sentence end.
    parts = ["I arrive at 3 p.m.", "sharp then we start."]
    boundaries = ["none"]
    assert split_into_sentences(parts, boundaries) == [(0, 1)]


def test_transcript_views():
    t = Transcript(
        parts=["hello there", "world"], boundaries=["comma"],
    )
    assert t.model_text("model") == "hello there world"
    assert t.fallback_text == "hello there, world"
```

Add this helper near the top of `tests/test_stt.py` (after the imports):

```python
def _parts_boundaries(segs):
    from app.stt import stitch
    return stitch(segs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_stt.py -v`
Expected: FAIL (ImportError for `resolve_model`/`stitch`/`Transcript`, and the updated assertions).

- [ ] **Step 3: Implement in `app/stt.py`**

Add `from dataclasses import dataclass` to the imports at the top. Replace the existing `stitch_segments` function with the following block:

```python
_TERMINAL_RE = re.compile(r"(?<![.\s][A-Za-z])[.?!]+\s*$")


def ends_sentence(text: str) -> bool:
    """True if `text` ends with terminal punctuation that is not part of a
    single-letter abbreviation ("p.m.", "e.g.")."""
    return bool(_TERMINAL_RE.search(text))


def stitch(segs: "list[tuple[float, float, str]]") -> "tuple[list[str], list[str]]":
    """Split segments into raw parts + the pause kind between each pair.
    `boundaries[i]` is the pause between `parts[i]` and `parts[i+1]`."""
    parts: list[str] = []
    boundaries: list[str] = []
    prev_end: float | None = None
    for start, end, text in segs:
        if text:
            if parts and prev_end is not None:
                boundaries.append(classify_gap(parts[-1], start - prev_end))
            parts.append(text)
        prev_end = end
    return parts, boundaries


def resolve_fallback(parts: list[str], boundaries: list[str]) -> str:
    """The pause-punctuated view (for the offline strip). Applies the pause
    marks deterministically; does NOT force any capitalization."""
    if not parts:
        return ""
    out = [parts[0]]
    for i, kind in enumerate(boundaries):
        if kind == "period":
            if out[-1] and out[-1][-1].isalnum():
                out[-1] += "."
            elif out[-1].endswith(","):
                out[-1] = out[-1][:-1] + "."
        elif kind == "comma":
            if out[-1] and out[-1][-1].isalnum():
                out[-1] += ","
        out.append(parts[i + 1])
    return " ".join(out).strip()


def resolve_model(parts: list[str], boundaries: list[str], source: str) -> str:
    """The view the cleanup model receives. "model": words + Whisper
    punctuation only (pause marks dropped so the model punctuates from
    context). "pauses": the fallback view (A-B baseline / legacy)."""
    if source == "pauses":
        return resolve_fallback(parts, boundaries)
    return " ".join(p for p in parts if p).strip()


def split_into_sentences(
    parts: list[str], boundaries: list[str]
) -> "list[tuple[int, int]]":
    """Inclusive `(start, end)` part-index ranges, one per sentence. A
    sentence ends at part i when a `period` pause follows it or the part ends
    with a Whisper terminal; the final part always closes the last sentence."""
    sents: list[tuple[int, int]] = []
    start = 0
    n = len(parts)
    for i in range(n):
        end_here = ends_sentence(parts[i]) or (
            i < len(boundaries) and boundaries[i] == "period"
        )
        if i == n - 1:
            end_here = True
        if end_here:
            sents.append((start, i))
            start = i + 1
    return sents


@dataclass
class Transcript:
    """A transcribed utterance as structured parts + pause boundaries, with
    resolvers for the two views."""

    parts: list
    boundaries: list

    def model_text(self, source: str) -> str:
        return resolve_model(self.parts, self.boundaries, source)

    @property
    def fallback_text(self) -> str:
        return resolve_fallback(self.parts, self.boundaries)


def stitch_segments(segs: "list[tuple[float, float, str]]") -> str:
    """Backwards-compatible fallback-view join (now without pause-forced
    capitalization; casing is owned downstream)."""
    return resolve_fallback(*stitch(segs))
```

Then change `Transcriber.transcribe` to return a `Transcript`:

```python
    def transcribe(
        self,
        audio: "np.ndarray | str | Path",
        initial_prompt: str | None = None,
    ) -> "Transcript":
        """Transcribe a mono float32 array (16 kHz) or an audio file path,
        returning a Transcript (structured parts + pause boundaries)."""
        segs = self.transcribe_segments(audio, initial_prompt=initial_prompt)
        return Transcript(*stitch(segs))
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_stt.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/stt.py tests/test_stt.py
git commit -m "feat(stt): structured Transcript with model/fallback views; drop pause-forced capital"
```

---

## Task 5: Cleaner — two-view `clean`, ellipsis collapse, prompt rule

**Files:**
- Modify: `app/cleanup.py` (`SYSTEM_PROMPT`, `Cleaner.clean`, `Cleaner._finish`, imports)
- Test: `tests/test_smoke.py` (add two focused tests)

**Interfaces:**
- Consumes: `collapse_ellipses` from `app.stt`.
- Produces: `Cleaner.clean(model_text: str, fallback_text: str | None = None, context=None, surrounding=None, reformat: bool = True) -> str`. Existing single-positional callers keep working (`fallback_text` defaults to `model_text`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_smoke.py`:

```python
def test_clean_collapses_ellipses_without_llm():
    cfg = CleanupConfig(enabled=False)
    assert Cleaner(cfg).clean("so i was thinking...") == "So I was thinking."


def test_clean_uses_fallback_text_for_local_strip():
    # LLM off: the local strip runs on fallback_text (pause punctuation),
    # not the clean model_text.
    cfg = CleanupConfig(enabled=False)
    out = Cleaner(cfg).clean("we shipped it the docs are next",
                             fallback_text="we shipped it. the docs are next")
    assert out == "We shipped it. The docs are next."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smoke.py::test_clean_collapses_ellipses_without_llm tests/test_smoke.py::test_clean_uses_fallback_text_for_local_strip -v`
Expected: FAIL (ellipsis not collapsed / `fallback_text` unknown kwarg).

- [ ] **Step 3: Implement**

In `app/cleanup.py`:

(a) Add the import near the top (after `from .config import CleanupConfig`):

```python
from .stt import collapse_ellipses
```

(b) In `SYSTEM_PROMPT`, add one line to the numbered rules (after rule 6, before the `NEVER do anything else:` block):

```
7. Do not use "..." (ellipses) for trailing-off speech. End the sentence \
with a single period instead.
```

(c) Replace `Cleaner.clean` and `Cleaner._finish` with:

```python
    def clean(self, model_text: str, fallback_text: str | None = None,
              context: str | None = None, surrounding=None,
              reformat: bool = True) -> str:
        """Clean a transcript. `model_text` is the LLM's input (already
        resolved for the punctuation_source); `fallback_text` (defaults to
        `model_text`) is what the local strip runs on. Returns the best
        available cleaned form."""
        model_text = model_text.strip()
        if not model_text:
            return ""
        fb = (fallback_text if fallback_text is not None else model_text).strip()
        mid_sentence = surrounding is not None and surrounding.mid_sentence
        continues_after = surrounding is not None and surrounding.continues_after
        local = strip_fillers(
            fb, capitalize=not mid_sentence, ensure_period=not continues_after
        )
        reformat_ok = reformat and not (mid_sentence or continues_after)

        if not self.cfg.enabled:
            return self._finish(local, reformat_ok)
        try:
            polished = self._ollama_clean(model_text, context, surrounding)
        except Exception as exc:
            log.warning("Ollama cleanup failed (%s); using local cleanup", exc)
            return self._finish(local, reformat_ok)
        if not polished:
            return self._finish(local, reformat_ok)
        if too_divergent(model_text, polished):
            log.warning(
                "LLM output diverged from speech (%r); using local cleanup",
                polished,
            )
            return self._finish(local, reformat_ok)
        return self._finish(polished, reformat_ok)

    def _finish(self, text: str, reformat_ok: bool = True) -> str:
        """Deterministic post-step: collapse ellipses, then turn an
        ordinal-led enumeration into a numbered list (no-op for non-lists)."""
        text = collapse_ellipses(text)
        return reformat_enumeration(text) if reformat_ok else text
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smoke.py tests/test_training.py tests/test_context.py tests/test_power.py tests/test_focus.py -v`
Expected: PASS (existing single-positional `clean(...)` callers unaffected; two new tests pass).

- [ ] **Step 5: Commit**

```bash
git add app/cleanup.py tests/test_smoke.py
git commit -m "feat(cleanup): two-view clean, collapse ellipses, no-ellipsis prompt rule"
```

---

## Task 6: Hotkey non-streaming path + CLI dry-run wired to views

**Files:**
- Modify: `app/hotkey.py` (`_process`)
- Modify: `app/__main__.py` (`_dry_run`)
- Test: `tests/test_focus.py` already exercises the surrounding path via `clean`; add a hotkey-level unit is out of scope (heavy). Rely on the existing suite staying green.

**Interfaces:**
- Consumes: `Transcript.model_text(source)`, `Transcript.fallback_text`, `Cleaner.clean(model_text, fallback_text=...)`, `cfg.cleanup.punctuation_source`.

- [ ] **Step 1: Update `_process` non-streaming branch**

In `app/hotkey.py`, inside `_process`, locate:

```python
                if session is not None:
                    raw = session.finish(audio)
                else:
                    raw = self.transcriber.transcribe(audio)
                t_stt = time.perf_counter()
                if not raw:
                    log.info("No speech detected.")
                    return
```

Replace with:

```python
                source = self.cfg.cleanup.punctuation_source
                fallback_full = None
                if session is not None:
                    raw = session.finish(audio)
                else:
                    tr = self.transcriber.transcribe(audio)
                    raw = tr.model_text(source)
                    fallback_full = tr.fallback_text
                t_stt = time.perf_counter()
                if not raw:
                    log.info("No speech detected.")
                    return
```

Then locate:

```python
                if live is not None:
                    cleaned = live.finalize(raw)
                else:
                    cleaned = self.cleaner.clean(
                        raw,
                        context=None if has_before else self.context.cleanup_context(),
                        surrounding=surrounding,
                    )
```

Replace with:

```python
                if live is not None:
                    cleaned = live.finalize(raw)
                else:
                    cleaned = self.cleaner.clean(
                        raw,
                        fallback_text=fallback_full,
                        context=None if has_before else self.context.cleanup_context(),
                        surrounding=surrounding,
                    )
```

(Note: `session.finish(audio)` still returns a string in Group A — Task 8 adds the `source` argument and the fallback wiring for streaming.)

- [ ] **Step 2: Update the CLI dry-run**

In `app/__main__.py`, in `_dry_run`, replace:

```python
    raw = Transcriber(cfg.stt).transcribe(wav)
    print(f"[dry-run] raw transcript : {raw!r}")
    cleaned = Cleaner(cfg.cleanup).clean(raw)
```

with:

```python
    tr = Transcriber(cfg.stt).transcribe(wav)
    source = cfg.cleanup.punctuation_source
    print(f"[dry-run] raw transcript : {tr.model_text(source)!r}")
    cleaned = Cleaner(cfg.cleanup).clean(
        tr.model_text(source), fallback_text=tr.fallback_text
    )
```

- [ ] **Step 3: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS (streaming/live_cleanup tests still green — untouched in Group A).

- [ ] **Step 4: Commit**

```bash
git add app/hotkey.py app/__main__.py
git commit -m "feat(hotkey): wire non-streaming path + CLI to model/fallback views"
```

---

## Task 7: Streaming — remove pause-forced capitalization (Group A finish)

**Files:**
- Modify: `app/streaming.py` (`_absorb`)
- Test: `tests/test_streaming.py` (update one assertion)

**Interfaces:** none new — behavior-only change to the interim streaming path.

- [ ] **Step 1: Update the failing test**

In `tests/test_streaming.py`, replace `test_long_pause_becomes_full_stop_across_commits` body's assertion:

```python
    assert s._committed == ["we should ship it.", "also the docs need a pass"]
```

(The next segment is no longer force-capitalized.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_streaming.py::test_long_pause_becomes_full_stop_across_commits -v`
Expected: FAIL (still capitalized "Also").

- [ ] **Step 3: Remove the capitalization in `_absorb`**

In `app/streaming.py` `_absorb`, replace:

```python
                    self._committed[-1] = append_gap_punctuation(
                        before, (base + start) - self._prev_end_abs
                    )
                    if self._committed[-1] is not before and \
                            self._committed[-1].endswith("."):
                        text = text[0].upper() + text[1:]
```

with:

```python
                    self._committed[-1] = append_gap_punctuation(
                        before, (base + start) - self._prev_end_abs
                    )
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_streaming.py tests/test_context.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/streaming.py tests/test_streaming.py
git commit -m "fix(streaming): a pause no longer forces a capital letter"
```

**Group A complete — review checkpoint.** Always-on fixes live everywhere; non-streaming + toggle done.

---

## Task 8: Streaming — structured parts, sentence units, `finish(source)`

**Files:**
- Modify: `app/streaming.py` (`StreamingSession`)
- Test: `tests/test_streaming.py` (rewrite `_committed`-based assertions)

**Interfaces:**
- Consumes: `classify_gap`, `resolve_model`, `resolve_fallback`, `split_into_sentences` from `app.stt`.
- Produces on `StreamingSession`:
  - `self._parts: list[str]`, `self._boundaries: list[str]`, `self._emitted: int`.
  - `stable_sentences(source: str) -> list[str]`
  - `remaining_sentences(source: str) -> list[str]`
  - `fallback_text() -> str`
  - `finish(audio, source: str = "model") -> str`
  - (Removed: `committed_parts()`, `self._committed`.)

- [ ] **Step 1: Rewrite the streaming tests**

Replace the bodies that reference `s._committed` and `s.finish(...)`. Full replacements:

```python
def test_commits_only_settled_segments():
    fake = FakeTranscriber(
        segment_script=[[(0.0, 2.0, "hello there"), (2.5, 4.6, "world")]]
    )
    s = _session(fake, lambda: _audio(5.0))
    s._pass_once()
    assert s._parts == ["hello there"]
    assert s._committed_samples == int(2.5 * SR)


def test_finish_stitches_commits_and_tail():
    fake = FakeTranscriber(
        segment_script=[
            [(0.0, 2.0, "hello there"), (2.5, 4.6, "world")],
            [(0.0, 1.5, "world again")],
        ]
    )
    s = _session(fake, lambda: _audio(5.0))
    s._pass_once()
    s._thread.start()
    # Fallback view keeps the 0.5s pause as a comma; model view drops it.
    assert s.finish(_audio(6.0), "pauses") == "hello there, world again"
    n_samples, prompt = fake.segment_calls[1]
    assert n_samples == int(3.5 * SR)
    assert prompt == "hello there"


def test_long_pause_becomes_full_stop_across_commits():
    fake = FakeTranscriber(
        segment_script=[
            [(0.0, 2.0, "we should ship it")],
            [(1.5, 3.0, "also the docs need a pass")],
        ]
    )
    buf = {"dur": 5.0}
    s = _session(fake, lambda: _audio(buf["dur"]))
    s._pass_once()
    buf["dur"] = 8.0
    s._pass_once()
    assert s._parts == ["we should ship it", "also the docs need a pass"]
    assert s._boundaries == ["period"]


def test_finish_without_commits_degrades_to_batch():
    fake = FakeTranscriber(segment_script=[[(0.0, 1.0, "all of it")]])
    s = _session(fake, lambda: _audio(1.0))
    s._thread.start()
    out = s.finish(_audio(1.0))
    assert out == "all of it"
    assert fake.segment_calls[0] == (int(1.0 * SR), None)


def test_failed_pass_does_not_lose_audio():
    class Flaky(FakeTranscriber):
        def transcribe_segments(self, audio, initial_prompt=None):
            if not self.segment_calls:
                self.segment_calls.append((len(audio), initial_prompt))
                raise RuntimeError("transient")
            return super().transcribe_segments(audio, initial_prompt)

    import pytest

    fake = Flaky(segment_script=[[(0.0, 5.0, "recovered")]])
    s = _session(fake, lambda: _audio(5.0))
    with pytest.raises(RuntimeError):
        s._pass_once()
    s._thread.start()
    out = s.finish(_audio(5.0))
    assert out == "recovered"
    assert s._committed_samples == 0


def test_finish_survives_tail_transcription_failure():
    class DeadModel(FakeTranscriber):
        def transcribe_segments(self, audio, initial_prompt=None):
            raise RuntimeError("model gone")

    s = _session(DeadModel(), lambda: _audio(1.0))
    s._parts = ["what we already have"]
    s._thread.start()
    assert s.finish(_audio(2.0)) == "what we already have"


def test_stable_sentences_and_remaining():
    fake = FakeTranscriber()
    s = _session(fake, lambda: _audio(1.0))
    s._parts = ["we shipped it", "the docs are next", "and then"]
    s._boundaries = ["period", "none"]
    # "we shipped it" is a complete, stable sentence (period pause; not the
    # still-mutable last part). The rest is not yet stable.
    assert s.stable_sentences("model") == ["we shipped it"]
    assert s.stable_sentences("model") == []          # cursor advanced
    # finish-time: everything remaining, including the final partial part.
    assert s.remaining_sentences("model") == ["the docs are next and then"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_streaming.py -v`
Expected: FAIL (`_parts`/`stable_sentences` missing).

- [ ] **Step 3: Rewrite `StreamingSession`**

In `app/streaming.py`, update the import line:

```python
from .stt import classify_gap, resolve_fallback, resolve_model, split_into_sentences
```

In `__init__`, replace `self._committed: list[str] = []` with:

```python
        self._parts: list[str] = []
        self._boundaries: list[str] = []
        self._emitted = 0              # parts already yielded as sentences
```

Delete the `committed_parts` method entirely and replace `_absorb` with:

```python
    def _absorb(self, segs, base: float) -> None:
        """Append raw segment texts; record the inter-segment pause kind
        (absolute timeline, so pass/tail boundaries count) as a boundary."""
        for start, end, text in segs:
            if text:
                if self._parts and self._prev_end_abs is not None:
                    self._boundaries.append(
                        classify_gap(self._parts[-1], (base + start) - self._prev_end_abs)
                    )
                self._parts.append(text)
            self._prev_end_abs = base + end
```

Add the sentence-unit + resolver methods (place after `_absorb`):

```python
    def _resolve_range(self, a: int, b: int, source: str) -> str:
        return resolve_model(self._parts[a:b + 1], self._boundaries[a:b], source)

    def stable_sentences(self, source: str) -> list[str]:
        """Newly-complete sentences that are also stable (do not reach into
        the still-mutable last committed part). Advances an internal cursor."""
        if len(self._parts) < 2:
            return []
        stable_upto = len(self._parts) - 1  # last part may still mutate
        out: list[str] = []
        for a, b in split_into_sentences(self._parts, self._boundaries):
            if a < self._emitted:
                continue
            if b >= stable_upto:
                break
            out.append(self._resolve_range(a, b, source))
            self._emitted = b + 1
        return out

    def remaining_sentences(self, source: str) -> list[str]:
        """Every sentence not yet emitted, including the final partial one.
        Call after finish() has absorbed the tail."""
        out: list[str] = []
        for a, b in split_into_sentences(self._parts, self._boundaries):
            if a < self._emitted:
                continue
            out.append(self._resolve_range(a, b, source))
            self._emitted = b + 1
        return out

    def fallback_text(self) -> str:
        return resolve_fallback(self._parts, self._boundaries)
```

Replace `finish` and `_context`:

```python
    def finish(self, audio: np.ndarray, source: str = "model") -> str:
        """Stop passes, transcribe the uncommitted tail, return the full text
        in the requested view."""
        self._stop.set()
        self._thread.join()
        tail = audio[self._committed_samples:]
        t0 = time.perf_counter()
        try:
            tail_segs = self._transcriber.transcribe_segments(
                tail, initial_prompt=self._context()
            )
        except Exception:
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
        return resolve_model(self._parts, self._boundaries, source)

    def _context(self) -> str | None:
        joined = " ".join(self._parts)
        return joined[-_MAX_PROMPT_CHARS:] if joined else None
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_streaming.py tests/test_context.py -v`
Expected: PASS. (`test_context.py::test_streaming_prompts_contain_only_current_utterance` still passes: model view of `["and the deploy", "went fine"]` is `"and the deploy went fine"`, and `_context()` returns `"and the deploy"`.)

- [ ] **Step 5: Commit**

```bash
git add app/streaming.py tests/test_streaming.py
git commit -m "refactor(streaming): structured parts + pause boundaries, sentence units, finish(source)"
```

---

## Task 9: LiveCleanup — consume sentence units; wire hotkey streaming path

**Files:**
- Modify: `app/live_cleanup.py` (`LiveCleanup`)
- Modify: `app/hotkey.py` (`_process` streaming branch)
- Test: `tests/test_live_cleanup.py` (rewrite the session-driven tests)

**Interfaces:**
- Consumes: `session.stable_sentences(source)`, `session.remaining_sentences(source)`, `self._cleaner.cfg.punctuation_source`.
- `LiveCleanup.finalize(raw_full: str) -> str` unchanged signature.

- [ ] **Step 1: Rewrite the live-cleanup tests**

Replace `FakeSession` and the affected tests in `tests/test_live_cleanup.py`. New `FakeSession` and helper:

```python
class FakeSession:
    """Model-view sentences with a `stable` cut (how many are committed &
    stable) and an internal emit cursor mirroring the real session."""

    def __init__(self):
        self.sentences = []
        self.stable = 0
        self._emitted = 0

    def stable_sentences(self, source):
        out = self.sentences[self._emitted:self.stable]
        self._emitted = max(self._emitted, self.stable)
        return out

    def remaining_sentences(self, source):
        out = self.sentences[self._emitted:]
        self._emitted = len(self.sentences)
        return out
```

Update `FakeCleaner.clean` and `_PassthroughCleaner.clean` signatures to accept the two-view params (they ignore `fallback_text`):

```python
    def clean(self, model_text, fallback_text=None, context=None,
              surrounding=None, reformat=True):
        self.calls.append((model_text, context, surrounding))
        return f"<{model_text}>"
```

```python
    def clean(self, model_text, fallback_text=None, context=None,
              surrounding=None, reformat=True):
        self.calls.append((model_text.strip(), reformat))
        return model_text.strip()
```

Give `FakeCleaner`/`_PassthroughCleaner` a `cfg` so `_source()` works — add to each `__init__`:

```python
        from app.config import CleanupConfig
        self.cfg = CleanupConfig()
```

Replace the session-driven test bodies:

```python
def test_only_stable_complete_sentences_are_cleaned():
    session, cleaner = FakeSession(), FakeCleaner()
    live = _live(session, cleaner)
    session.sentences = ["we shipped it."]      # nothing stable yet
    session.stable = 0
    live._poll_once()
    assert cleaner.calls == []
    session.sentences = ["we shipped it.", "the docs are"]
    session.stable = 1                          # first sentence now stable
    live._poll_once()
    assert [c[0] for c in cleaner.calls] == ["we shipped it."]
    live._poll_once()                           # no new stable sentence
    assert len(cleaner.calls) == 1


def test_correction_cue_re_cleans_previous_sentence():
    session, cleaner = FakeSession(), FakeCleaner()
    live = _live(session, cleaner)
    session.sentences = ["the meeting is at 9am.", "no sorry at 3pm."]
    session.stable = 2
    live._poll_once()
    assert cleaner.calls[-1][0] == "the meeting is at 9am. no sorry at 3pm."
    assert live._cleaned == ["<the meeting is at 9am. no sorry at 3pm.>"]


def test_finalize_cleans_tail_and_stitches():
    session, cleaner = FakeSession(), FakeCleaner()
    live = _live(session, cleaner)
    session.sentences = ["we shipped it.", "docs are next week"]
    session.stable = 1
    live._poll_once()
    live._thread.start()
    out = live.finalize("we shipped it. docs are next week")
    assert out == "<we shipped it.> <docs are next week>"
    assert cleaner.calls[-1][1] == "<we shipped it.>"


def test_finalize_with_no_precleaned_sentences_cleans_everything():
    session, cleaner = FakeSession(), FakeCleaner()
    live = _live(session, cleaner, context="history text")
    session.sentences = ["just one short thing"]
    session.stable = 0
    live._thread.start()
    out = live.finalize("just one short thing")
    assert out == "<just one short thing>"
    assert cleaner.calls[0][1] == "history text"


def test_finalize_reformats_split_enumeration_into_a_list():
    session, cleaner = FakeSession(), _PassthroughCleaner()
    live = _live(session, cleaner)
    session.sentences = ["We need three things.", "First, the budget.",
                         "Second, the timeline.", "third the plan"]
    session.stable = 3
    live._poll_once()
    live._thread.start()
    raw = "We need three things. First, the budget. Second, the timeline. third the plan"
    out = live.finalize(raw)
    assert out == ("We need three things:\n1. The budget.\n2. The timeline.\n"
                   "3. The plan.")
    assert all(reformat is False for _, reformat in cleaner.calls)


def test_finalize_leaves_non_enumeration_prose_alone():
    session, cleaner = FakeSession(), _PassthroughCleaner()
    live = _live(session, cleaner)
    session.sentences = ["We shipped the release.", "docs are next"]
    session.stable = 1
    live._poll_once()
    live._thread.start()
    out = live.finalize("We shipped the release. docs are next week")
    assert out == "We shipped the release. docs are next week"


def test_surrounding_split_first_gets_before_last_gets_after():
    sur = Surrounding(before="Existing text", after=" continues here")
    session, cleaner = FakeSession(), FakeCleaner()
    live = _live(session, cleaner, surrounding=sur)
    session.sentences = ["first sentence."]
    session.stable = 1
    live._poll_once()
    live._thread.start()
    live.finalize("first sentence. tail words")
    first_sur = cleaner.calls[0][2]
    tail_sur = cleaner.calls[-1][2]
    assert first_sur.before == "Existing text" and first_sur.after == ""
    assert tail_sur.before == "" and tail_sur.after == " continues here"
    assert cleaner.calls[0][1] is None
```

Delete `test_split_sentences_respects_abbreviations` and the `_split_sentences` import (that behavior is now covered by `split_into_sentences` in `tests/test_stt.py`). Change the top import to:

```python
from app.cleanup import looks_like_enumeration
from app.focus import Surrounding
from app.live_cleanup import LiveCleanup
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_live_cleanup.py -v`
Expected: FAIL (LiveCleanup still calls `committed_parts`; `_split_sentences` import removed).

- [ ] **Step 3: Rewrite `LiveCleanup`**

In `app/live_cleanup.py`, remove the now-unused `_SENT_END_RE`, `_split_sentences`, and the char-offset fields. Replace the class internals as follows.

In `__init__`, delete `self._consumed = 0` and keep `self._raw`, `self._cleaned`. Replace `_poll_once`, `_clean_one`, and `finalize`:

```python
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

    def finalize(self, raw_full: str) -> str:
        """Stop the worker, clean the remaining sentences, return full text."""
        self._stop.set()
        self._thread.join()
        remaining = self._session.remaining_sentences(self._source())
        for i, sent in enumerate(remaining):
            self._clean_one(sent, last=(i == len(remaining) - 1))
        log.info("Live cleanup: %d sentence(s) assembled", len(self._cleaned))
        assembled = " ".join(p for p in self._cleaned if p).strip()
        s = self._get_surrounding()
        mid = s is not None and (s.mid_sentence or s.continues_after)
        if not mid and looks_like_enumeration(raw_full):
            listed = reformat_enumeration(assembled)
            if listed != assembled:
                log.info("Live cleanup: enumeration reformatted into a list")
            return listed
        return assembled
```

Keep `_chunk_context` and `_chunk_surrounding` unchanged. Update the module imports at the top to drop the sentence-split regex usage (leave `_CORRECTION_CUE_RE`, `looks_like_enumeration`, `reformat_enumeration`, `Surrounding`):

```python
from .cleanup import _CORRECTION_CUE_RE, looks_like_enumeration, reformat_enumeration
from .focus import Surrounding
```

- [ ] **Step 4: Wire the hotkey streaming path to `finish(source)`**

In `app/hotkey.py` `_process`, update the streaming branch (from Task 6) so `finish` receives the source and the fallback is captured for the streaming-STT-but-cleanup-off sub-case:

```python
                source = self.cfg.cleanup.punctuation_source
                fallback_full = None
                if session is not None:
                    raw = session.finish(audio, source)
                    fallback_full = session.fallback_text()
                else:
                    tr = self.transcriber.transcribe(audio)
                    raw = tr.model_text(source)
                    fallback_full = tr.fallback_text
```

(The `live.finalize(raw)` / `clean(raw, fallback_text=fallback_full, ...)` block from Task 6 is unchanged.)

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS (all files green).

- [ ] **Step 6: Commit**

```bash
git add app/live_cleanup.py app/hotkey.py tests/test_live_cleanup.py
git commit -m "refactor(live-cleanup): consume session sentence units; honor punctuation_source in streaming"
```

---

## Task 10: Manual verification (dry-run on both modes)

**Files:** none (verification only).

- [ ] **Step 1: Model mode dry-run**

Run: `.venv/Scripts/python.exe -m app --dry-run assets/sample.wav`
Expected: prints a raw transcript (model view — no pause-inserted `.`/`,`) and a cleaned output. No crash; no stray `...`.

- [ ] **Step 2: Pauses mode dry-run**

Temporarily set `punctuation_source = "pauses"` in `config.toml`, re-run the same command. Expected: the raw transcript now shows pause punctuation. Revert the setting.

- [ ] **Step 3: Full suite once more**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS.

---

## Self-Review (completed by plan author)

- **Spec coverage:** structured parts + resolvers (Tasks 4, 8); model/fallback views (Tasks 4–6, 8–9); no pause-forced capital (Tasks 4, 7); collapse ellipses + prompt rule (Tasks 3, 5); `punctuation_source` toggle folded into `resolve_model` (Tasks 1, 4) honored by streaming (Tasks 8–9); offline fallback keeps pause punctuation (Tasks 5–6, `fallback_text`); divergence guard unchanged, compares `model_text` (Task 5). All spec sections mapped.
- **Type consistency:** `classify_gap -> str`, `resolve_model(parts, boundaries, source)`, `resolve_fallback(parts, boundaries)`, `split_into_sentences -> list[tuple[int,int]]`, `Transcript.model_text(source)`/`.fallback_text`, `clean(model_text, fallback_text=None, ...)`, `finish(audio, source="model")`, `stable_sentences(source)`/`remaining_sentences(source)`/`fallback_text()` — names consistent across Tasks 4–9.
- **Placeholder scan:** none — every code and test step shows full content.
- **Known interim state:** after Group A (Task 7) the streaming path still feeds pause-punctuated chunks to the LLM; Group B (Tasks 8–9) closes that. Documented in the Execution note.
