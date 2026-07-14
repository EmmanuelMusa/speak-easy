# Cleanup Quality Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean each utterance holistically (one pass, whole text) instead of per pause-chunk, so punctuation/sentence-breaks/lists are decided globally; let the model format parallel-item lists; stop the divergence guard rejecting legitimate number→digit and list conversions; and make fix-in-place reliable.

**Architecture:** Keep streaming transcription; flip cleanup to a single holistic pass at release (per-chunk streaming stays opt-in). Broaden the cleanup prompt + relax the guard for lists/numbers. Harden `injector.replace_last`.

**Tech Stack:** Python 3 stdlib, faster-whisper, Ollama, PySide6.

## Global Constraints

- No new dependencies. Tests: `.venv/Scripts/python.exe -m pytest`.
- Git repo on branch `dev`; commit each task (targeted `git add`). No `Co-Authored-By`/AI-authorship line.
- Baseline before Task 1: full suite **150 passing**.
- Real cleanup **quality** (lists/punctuation) is verified by the user re-recording live; unit tests lock the mechanics (config, prompt content, guard behavior).

**Reference spec:** `docs/superpowers/specs/2026-07-14-cleanup-quality-overhaul-design.md`

---

## Task 1: Holistic cleanup by default

**Files:**
- Modify: `app/config.py` (`CleanupConfig.streaming`)
- Modify: `config.toml` (`[cleanup].streaming`)
- Test: `tests/test_smoke.py`

**Interfaces:** `CleanupConfig.streaming` default becomes `False`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smoke.py`:

```python
def test_cleanup_streaming_defaults_off_for_holistic_quality():
    # Holistic (whole-utterance) cleanup is the default; per-chunk streaming
    # is opt-in. Chunking at pauses was what broke punctuation/lists.
    from app.config import CleanupConfig
    assert CleanupConfig().streaming is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smoke.py::test_cleanup_streaming_defaults_off_for_holistic_quality -v`
Expected: FAIL (`streaming` currently defaults `True`).

- [ ] **Step 3: Flip the default**

In `app/config.py`, `@dataclass class CleanupConfig`, change the streaming field and its comment:

```python
    # Clean the WHOLE utterance in one pass at release (holistic): grammar
    # decides sentence breaks and lists, not your pauses. Set True for the older
    # per-chunk streaming (cleans finished sentences while you talk — faster on
    # long dictations, but chunks at pauses so punctuation/lists suffer).
    streaming: bool = False
```

In `config.toml`, under `[cleanup]`, change the streaming line + comment:

```toml
# Holistic cleanup (default): clean the whole utterance in one pass at release,
# so punctuation and lists are decided from the full text, not sliced at your
# pauses. Set true for per-chunk streaming (faster on long dictations, but lower
# punctuation/list quality). Streaming transcription ([stt].streaming) is separate.
streaming = false
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS. (The streaming-cleanup code path and its tests still exist and pass — this only changes the default; `test_streaming.py`/`test_live_cleanup.py` construct their objects directly and are unaffected.)

- [ ] **Step 5: Commit**

```bash
git add app/config.py config.toml tests/test_smoke.py
git commit -m "feat(cleanup): holistic cleanup by default (per-chunk streaming now opt-in)"
```

---

## Task 2: Divergence guard — allow number→digit and list conversions

**Files:**
- Modify: `app/cleanup.py` (`too_divergent` and its droppable sets)
- Test: `tests/test_smoke.py`

**Interfaces:** `too_divergent(raw, cleaned)` no longer flags spoken-number→digit conversions, and is lenient on dropped joining words when the output is a list.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_smoke.py`:

```python
def test_number_words_to_digits_not_rejected():
    from app.cleanup import too_divergent
    # "ten million" spoken -> "10 million" cleaned drops the WORD "ten"; the
    # guard must treat spoken numbers as legitimately convertible to digits.
    raw = "the budget is ten million to twenty million naira"
    clean = "The budget is 10 million to 20 million Naira."
    assert not too_divergent(raw, clean)


def test_parallel_item_list_not_rejected():
    from app.cleanup import too_divergent
    raw = ("the registration value is value ten million to twenty million value "
           "twenty one million to one hundred million value above five hundred million")
    clean = ("The registration value is:\n"
             "- Value: 10 million to 20 million\n"
             "- Value: 21 million to 100 million\n"
             "- Value: above 500 million")
    assert not too_divergent(raw, clean)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smoke.py -k "number_words or parallel_item" -v`
Expected: FAIL (spoken number words counted as dropped; over the `MAX_DROPPED_WORDS = 1` limit).

- [ ] **Step 3: Implement**

In `app/cleanup.py`, add a number-words set near the other droppable sets (after `_LIST_SCAFFOLD_WORDS`):

```python
# Spoken cardinal numbers legitimately become digits ("ten million" -> "10
# million"), which the word-level guard would otherwise see as dropped words.
_NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
    "thirty forty fifty sixty seventy eighty ninety hundred thousand million "
    "billion trillion and".split()
)
```

In `too_divergent`, extend the droppable set and raise the dropped-word ceiling
for list output. Replace the dropped-word block:

```python
    droppable = _DROPPABLE_WORDS | _NUMBER_WORDS
    limit = MAX_DROPPED_WORDS
    if _is_list(cleaned):
        # A dictated list legitimately sheds scaffolding and repeated joining
        # words as it becomes items, so be more forgiving there.
        droppable = droppable | _LIST_SCAFFOLD_WORDS
        limit = MAX_DROPPED_WORDS_LIST
    out_set = set(out_words)
    dropped = sum(
        1
        for w in raw_words
        if w not in out_set
        and w not in droppable
        and not _NOISE_RE.fullmatch(w)
    )
    return dropped > limit
```

Add the list ceiling constant next to `MAX_DROPPED_WORDS`:

```python
# When the output is a formatted list, allow more dropped joining words than the
# strict prose limit — items shed repeated lead-in words as they're bulleted.
MAX_DROPPED_WORDS_LIST = 4
```

- [ ] **Step 4: Run the suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smoke.py -q`
Expected: PASS (2 new tests; the existing `too_divergent` tests — `test_dropped_speaker_words_are_rejected`, `test_paraphrased_llm_output_is_rejected`, the enumeration-reformat guard tests — stay green: they use prose/ordinal cases with no number words).

- [ ] **Step 5: Commit**

```bash
git add app/cleanup.py tests/test_smoke.py
git commit -m "fix(cleanup): guard allows number->digit and list-formatting conversions"
```

---

## Task 3: Prompt — format parallel-item lists

**Files:**
- Modify: `app/cleanup.py` (`SYSTEM_PROMPT`)
- Test: `tests/test_smoke.py`

**Interfaces:** none (prompt content).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smoke.py`:

```python
def test_prompt_covers_parallel_item_lists():
    from app.cleanup import SYSTEM_PROMPT
    p = SYSTEM_PROMPT.lower()
    assert "parallel" in p or "same shape" in p  # parallel-item list rule present
    assert "- " in SYSTEM_PROMPT                  # a bulleted-list example present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smoke.py::test_prompt_covers_parallel_item_lists -v`
Expected: FAIL (no parallel-list rule/example yet).

- [ ] **Step 3: Extend the prompt**

In `app/cleanup.py`, in `SYSTEM_PROMPT`, immediately after list rule 4 (the
ordinal-enumeration rule ending "...never turn ordinary prose that happens to
say \"first\" into a list."), insert a parallel-list rule:

```
4b. If the speaker dictates a run of 2+ PARALLEL items with the same shape (each \
beginning the same way or repeating a label — e.g. "value X to Y, value A to B, \
value above Z", or "option one does…, option two does…"), format them as a \
BULLETED list (one item per line, "- "). Keep the introductory lead-in as a line \
ending in a colon. Convert spoken numbers to digits (ten million -> 10 million). \
Only for a genuine run of parallel items, never for ordinary prose.
```

And add a worked example in the Examples block (after the existing enumeration
examples), modeled on the user's case:

```
Input: for corporate bodies registering an engineering firm the initial registration costs scale based on your total business value value ten million to twenty million value twenty one million to one hundred million value above five hundred million
Output: For corporate bodies registering an engineering firm, the initial registration costs scale based on your total business value:
- Value: 10 million to 20 million
- Value: 21 million to 100 million
- Value: above 500 million
```

- [ ] **Step 4: Run the suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS (the new prompt test; existing cleaner tests still pass — they mock the Ollama response, so prompt text changes don't affect them).

- [ ] **Step 5: Commit**

```bash
git add app/cleanup.py tests/test_smoke.py
git commit -m "feat(cleanup): prompt formats parallel-item lists as bullets"
```

---

## Task 4: Fix-in-place reliability + diagnostics

**Files:**
- Modify: `app/injection.py` (`replace_last`)
- Test: `tests/test_injection.py`

**Interfaces:** `replace_last` behavior unchanged on success; clearer logging + a longer settle on the re-focus.

- [ ] **Step 1: Write the failing test**

In `tests/test_injection.py`, add a test that the abort path logs the mismatch
(so a live failure is diagnosable). Use the existing mocking style in that file;
if the module-level Win32 helpers are patched there, mirror that. Concretely:

```python
def test_replace_last_logs_reason_on_mismatch(caplog):
    import logging
    from unittest.mock import patch
    from app.injection import Injector
    from app.config import InjectionConfig
    inj = Injector(InjectionConfig())
    inj.last_text = "hello world"
    inj.last_hwnd = 1234
    with patch("app.injection._set_foreground_window"), \
         patch("app.injection._select_back"), \
         patch("app.injection._press_ctrl_c"), \
         patch("app.injection._get_clipboard_text", return_value="something else"), \
         patch("app.injection._tap"), \
         patch("app.injection._set_clipboard_text"):
        with caplog.at_level(logging.INFO):
            ok = inj.replace_last("goodbye world")
    assert ok is False
    assert any("changed since injection" in r.message.lower()
               or "expected" in r.message.lower() for r in caplog.records)
```

(If `tests/test_injection.py` already stubs these helpers differently, adapt the
patch targets to match — the assertion is: mismatch → `False` + an informative log.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_injection.py::test_replace_last_logs_reason_on_mismatch -v`
Expected: FAIL (the current abort log doesn't include the expected/got detail, or the settle is too short) — adjust to match the actual current message before implementing.

- [ ] **Step 3: Implement**

In `app/injection.py` `replace_last`:
- Increase the post-refocus settle from `time.sleep(0.12)` to `time.sleep(0.2)`
  (give the target app time to restore focus + caret after the feedback panel
  had focus).
- On the mismatch abort, log the expected vs got so failures are diagnosable:

```python
        if not selection_matches(old, selected):
            _tap(VK_RIGHT)  # deselect, leave the user's text untouched
            log.info("Replace aborted: text changed since injection "
                     "(expected %r, selection %r)", old, selected)
            if previous_clip is not None:
                try:
                    _set_clipboard_text(previous_clip)
                except Exception:
                    pass
            return False
```

- [ ] **Step 4: Run the suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_injection.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/injection.py tests/test_injection.py
git commit -m "fix(injection): steadier fix-in-place refocus + diagnostic abort log"
```

---

## Task 5: Manual verification (user + scripted)

**Files:** none.

- [ ] **Step 1: Scripted guard/prompt check**

Run: `.venv/Scripts/python.exe -m pytest -q` — full suite green.

- [ ] **Step 2: Live quality check (user, needs mic)**

With `[cleanup].streaming = false` (now default) and `ollama_model = llama3.1:8b`:
- Re-record the **engineering-fees block** → expect a bulleted list.
- Record an **ordinal list** ("first…, second…, third…") → expect a numbered list.
- Record a **pause-heavy sentence** → expect no stray full stops on the pauses.
- Insert text **mid-sentence** in your target app → expect it to flow (lowercase
  start, no stray trailing period) where the app exposes caret context.
- Submit a **correction** editing the *Ideal cleanup* field → expect the typed
  text to be replaced (check the log for the reason if it doesn't).

---

## Self-Review (completed by plan author)

- **Spec coverage:** holistic default (Task 1, also fixes cross-chunk mid-sentence casing); parallel-list prompt (Task 3); guard accepts number→digit + list conversions (Task 2); fix-in-place reliability + diagnostics (Task 4); manual quality verification (Task 5). Mid-sentence (#3) is substantially resolved by Task 1 (full-utterance surrounding context) and checked in Task 5.
- **Type consistency:** `CleanupConfig.streaming: bool`, `too_divergent(raw, cleaned) -> bool`, `_NUMBER_WORDS`/`MAX_DROPPED_WORDS_LIST`, `SYSTEM_PROMPT`, `replace_last -> bool` — consistent.
- **Placeholder scan:** none; Task 4's test note flags adapting patch targets to the existing `test_injection.py` style (the assertion is concrete).
- **Empirical caveat:** Tasks 2–3 lock the *mechanics* (guard accepts, prompt contains the rule); the actual list/punctuation *quality* from 8B is verified live in Task 5.
