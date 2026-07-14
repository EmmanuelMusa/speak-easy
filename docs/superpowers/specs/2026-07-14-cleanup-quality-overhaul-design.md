# Cleanup quality overhaul: holistic cleanup, list formatting, mid-sentence flow, fix-in-place

**Date:** 2026-07-14
**Status:** Approved direction (pending spec review), then implementation plan

## Problem (from live use, on llama3.1:8b)

- **Punctuation** — full stops land wherever the speaker pauses, even in
  `punctuation_source = "model"` mode.
- **Lists** — "first/second/third", "one/two/three" often stay one line;
  parallel-item lists (repeated "Value: X to Y …") are never listed. Wispr Flow
  turns the same audio into a proper list.
- **Mid-sentence insertion** — dictating between existing text inserts a
  standalone capitalized sentence with a trailing period instead of flowing in.
- **Corrections** — a submitted correction does not replace the already-typed
  text.

## Root cause (punctuation + lists)

Cleanup runs **per pause-chunk** (`[cleanup].streaming = true`, via
`live_cleanup.py`): the utterance is split at pauses and each chunk is cleaned by
the LLM in isolation (middle chunks get no preceding context). So every
pause-fragment is punctuated as a standalone sentence — full stops on pauses,
lists broken across sentences. Model size is not the bottleneck; the chunking is.

## Goals

Keep streaming **transcription** (fast release), but clean **holistically** so
the model punctuates and formats from the whole utterance; and fix the two
correction/flow bugs.

## Design

### 1. Holistic cleanup by default

- `CleanupConfig.streaming` default → `False`; `config.toml [cleanup].streaming =
  false`, with a comment: holistic = better punctuation/lists at a small extra
  release latency; `true` restores per-chunk streaming (faster, lower list/
  punctuation quality). The per-chunk path (`live_cleanup.py`) is **retained**
  as an opt-in; no code removed.
- Effect: with `[stt].streaming = true` (kept) and cleanup streaming off,
  `hotkey._process` transcribes incrementally, then at release runs
  `Cleaner.clean(full_transcript, fallback_text=…)` **once** over the whole text.
  The existing enumeration reformat already runs over the whole utterance here.

### 2. List formatting the model can do (ordinal + parallel)

- **Prompt** (`SYSTEM_PROMPT`): broaden the list rule. Today it lists only
  *ordinal-cued* enumerations (first/second, number one/two). Add: when the
  speaker dictates a run of **parallel items with the same shape** (e.g.
  "value 10 million to 20 million, value 21 million to 100 million, value above
  500 million"), format them as a **bulleted** list (`- item`), keeping any
  lead-in as a `…:` line. Add one worked example modeled on the
  engineering-fees block. Ordinal enumerations stay **numbered** (`1. `).
- **Divergence guard** (`too_divergent`): confirm a model-formatted bulleted/
  numbered list is accepted — list markers and the dropped joining words
  ("value", repeated units) must not trip the dropped-word check. `_LIST_ITEM_RE`
  already matches `- ` bullets; `_LIST_SCAFFOLD_WORDS` covers ordinal scaffolding.
  Extend the droppable set / leniency as needed so a correct parallel list is not
  rejected as divergent (verified against the engineering-fees example).
- **Deterministic `reformat_enumeration`** stays for ordinal lists (a reliable
  assist); parallel lists rely on the model. Runs holistically (one utterance),
  so it is no longer fighting per-chunk fragments.

### 3. Mid-sentence flow

- Investigate `focus.read_surrounding` / `Surrounding.mid_sentence` /
  `continues_after` and the `strip_fillers`/`_ollama_clean` casing path. When the
  caret's surrounding text **is** readable and shows we are mid-sentence
  (preceding text does not end a sentence, and/or text follows the caret),
  cleanup must lowercase the leading word (unless a proper noun / "I") and omit a
  trailing period. Make that path robust; where the app exposes **no** caret text
  (the read fails), document that it degrades to standalone-sentence casing (can't
  be helped without the context). The target app the user hit this in informs
  whether it's a readable-context bug or an unreadable-app limitation.

### 4. Fix-in-place reliability

- `_record_feedback` applies `injector.replace_last(ideal)` only when the
  **Ideal cleanup** field was edited (editing only "what you actually said" is a
  *speech* correction, not a text edit — unchanged). Two reliability issues to
  address: the expanded panel takes focus from the target, and `replace_last`
  then re-focuses `last_hwnd` and re-selects the injected span — fragile if the
  caret moved. Improve: re-focus + settle reliably before re-selecting, and make
  the abort path **log clearly why** (focus lost / selection changed) so failures
  are diagnosable. Confirm the flow end-to-end with a real target app.

## Testing

- **Config**: `CleanupConfig().streaming is False`; loads from TOML.
- **Prompt/guard (lists)**: `SYSTEM_PROMPT` contains the parallel-list rule +
  example; `too_divergent(raw, bulleted_list)` is `False` for the engineering-fees
  raw→list pair (and the existing ordinal-list guard tests stay green).
- **Holistic path**: with cleanup streaming off, `Cleaner.clean(full)` is the
  single cleanup call (no `live_cleanup`), verified via the existing mocked-Ollama
  cleaner tests (they already exercise the non-streaming path).
- **Mid-sentence / fix-in-place**: unit-test the pure decision points where
  possible (e.g., casing given a `Surrounding`); the end-to-end GUI behavior is
  the user's live check.
- Full suite stays green; real cleanup **quality** (lists/punctuation) is verified
  by the user re-recording the engineering-fees block and a pause-heavy utterance.

## Known tradeoffs

- Holistic cleanup adds release latency proportional to utterance length
  (~1–3 s on 8B/GPU for a paragraph); the per-chunk streaming opt-in remains for
  users who prioritize speed over list/punctuation quality.
- A local 8B model will not perfectly match a cloud model's formatting; the goal
  is to close most of the gap, with the deterministic reformat as a floor.
- Where an app exposes no caret context, mid-sentence continuation can't be
  inferred and falls back to standalone-sentence casing.

## Out of scope

- The fine-tune harness (B) / model integration (C) — separate roadmap items.
- Streaming the cleanup **output** tokens for perceived latency (a possible later
  optimization; not needed for quality).
