# Punctuation authority: model owns punctuation, pauses are advisory

**Date:** 2026-07-13
**Status:** Approved design, pending implementation plan

## Problem

Three punctuation/casing complaints, all traced to deterministic heuristics
rather than the cleanup model understanding context:

1. **Wrong full stops on a pause.** `append_gap_punctuation` (`app/stt.py`)
   inserts `.` at a >=0.65s inter-segment gap after a content word, before the
   LLM ever sees the text. The cleanup prompt asks the model to undo bad
   pause-punctuation, but a 3B model does so unreliably and the divergence
   guard (`too_divergent`) only compares words, so nothing enforces it.
2. **`...` overused for trailing thoughts.** Whisper emits `...` on
   trailing-off speech; the "keep every word verbatim" rule preserves it and no
   layer collapses it.
3. **Capital letter when dictating mid-text.** The pause path auto-capitalizes
   the next segment after an inserted period (`stitch_segments`,
   `StreamingSession._absorb`); `strip_fillers` also capitalizes when
   mid-sentence isn't detected.

## Goal

Move punctuation/casing authority from the deterministic pause heuristics onto
the cleanup model, while keeping the offline fallback and streaming
pre-cleaning working. Must work on both a small (3B) and a larger (8B) model,
with a switch to A/B the two behaviors.

## Core principle

Whisper's own (prosody) punctuation is trustworthy and stays in the text.
Pause-*gap*-derived punctuation stops being baked into a single string shared
by every consumer. The transcript is carried as **structured parts** — the raw
Whisper segment texts plus, between each pair, the *kind of pause* that
separated them (`none` / `comma` / `period`). Two resolver functions turn that
structure into whichever view a consumer needs:

| Consumer | View it gets | Behavior |
|---|---|---|
| Cleanup model (authority) | **model view** | Whisper punctuation only; pause gaps become plain spaces; no forced capitals. The model punctuates and cases from context. |
| Offline fallback (`strip_fillers`) | **fallback view** | Pause punctuation applied deterministically (today's behavior), so offline output keeps structure. |
| Streaming (`live_cleanup`) | model view + pause boundaries | Sentence splitting uses `period`-kind pauses and Whisper terminals, so pre-cleaning does not regress when pause periods are absent from the text. |

## Always-on fixes (independent of model and of the toggle)

1. **A pause never forces a capital.** Remove the `text[0].upper()` after a
   pause-inserted period in `stitch_segments` and `StreamingSession._absorb`.
   Casing is the model's job when cleanup is on, and `strip_fillers`'
   existing sentence-capitalization regex handles the fallback.
2. **Collapse `…` / `...`.** A new `collapse_ellipses(text)` helper: a trailing
   ellipsis is dropped (the existing `ensure_period` logic re-adds a single
   period where appropriate); an internal ellipsis becomes a single space.
   Applied to cleanup output (both LLM and fallback paths) so Whisper's
   ellipses don't survive. One prompt line tells the model not to emit `...`
   for trailing-off speech.

## Tunable A/B switch

New config `[cleanup] punctuation_source`:

- `"model"` (new default) — the model receives the **model view** (no pause
  punctuation). The model owns punctuation.
- `"pauses"` — the model receives the **fallback view** (pause punctuation
  baked in), reproducing today's input to the model. Escape hatch and A/B
  baseline.

The toggle is folded into the **model-view resolver** (`resolve_model(parts,
boundaries, source)`): in `"pauses"` mode it returns the fallback view. The
`source` is read from `[cleanup].punctuation_source` at the call sites that own
the config (hotkey and `live_cleanup`) and passed into the resolver — so the
toggle is applied entirely at resolution time. Every consumer of the "model
view" (including streaming) honors it with no extra plumbing, and `Cleaner.
clean` never needs to know about it (it receives two already-resolved strings).
The two always-on fixes still apply in `"pauses"` mode (no forced capital from a
pause; ellipses collapsed).

Flip the value, restart, compare — on 3B or 8B.

## Components and interfaces

### `app/stt.py`

- `classify_gap(prev_text: str, gap: float) -> "none" | "comma" | "period"` —
  extracts today's `append_gap_punctuation` decision logic (thinking-word
  suppression, comma vs period thresholds, "Whisper terminal already present ->
  none"). Pure and unit-tested.
- `append_gap_punctuation(text, gap)` — kept as a thin wrapper over
  `classify_gap` + application, so its existing unit tests and any external
  callers still work.
- `resolve_fallback(parts, boundaries) -> str` — rebuild today's
  pause-punctuated string (minus the removed forced capitalization).
- `resolve_model(parts, boundaries, source) -> str` — `source=="model"`: join
  with Whisper punctuation only, spaces at pauses, no forced capitals.
  `source=="pauses"`: delegate to `resolve_fallback`.
- `collapse_ellipses(text) -> str` — see fix 2.
- `stitch(segs) -> (parts, boundaries)` builds the structure. `Transcriber.
  transcribe(audio) -> Transcript` returns a `Transcript` carrying `parts` and
  `boundaries` with resolver accessors `model_text(source)` and `fallback_text`
  (thin wrappers over the two free functions). The caller supplies `source`
  from `[cleanup].punctuation_source`, so the STT layer never stores a cleanup
  setting. `stitch_segments(segs) -> str` retained as a thin `resolve_fallback`
  wrapper for compatibility.

### `app/streaming.py`

- `StreamingSession` stores `parts: list[str]` and `boundaries: list[str]`
  (kind between consecutive parts) instead of a pre-punctuated
  `_committed: list[str]`. `_absorb` appends raw Whisper text and computes the
  boundary via `classify_gap` on the absolute timeline (unchanged cross-pass
  behavior).
- Sentence assembly moves into the session (it owns the pause data). New:
  `stable_sentences(source) -> list[str]` yields committed sentences that are
  both complete (ended by a `period` pause or a Whisper terminal) and stable
  (not in the still-mutable last part), advancing an internal cursor.
  `finish(audio, source) -> str` transcribes the tail and returns the full
  model-view text; it also exposes any remaining un-popped sentences for
  `finalize`.

### `app/live_cleanup.py`

- Consumes **sentence units** from the session instead of tracking char
  offsets into a joined string. Per poll: clean each newly stable sentence
  (still merging a self-correction cue with the previous sentence). `finalize`:
  clean the remaining sentence(s), assemble, then run the existing
  enumeration-to-list reformat on the whole utterance.
- Streaming operates in model view only. A per-chunk LLM failure falls back to
  `strip_fillers` on the model-view chunk (no pause punctuation) — an
  acceptable, rare degradation on the fast path.

### `app/cleanup.py`

- `Cleaner.clean(model_text, fallback_text=None, context=None, surrounding=None,
  reformat=True)`. Both strings are already resolved by the caller (the toggle
  was applied in `resolve_model`), so `clean` is toggle-agnostic. `fallback_text`
  defaults to `model_text` when omitted (streaming chunks, tests). The LLM
  receives `model_text`; the local strip uses `fallback_text`; the divergence
  guard compares against `model_text`.
- `_finish` applies `collapse_ellipses` before the enumeration reformat, so both
  the LLM and fallback outputs are covered.
- One prompt line added for the `...` rule. The divergence guard is **not**
  otherwise changed: it already permits free repunctuation. Run-on risk on a
  weak model in `"model"` mode is a known tradeoff; the `"pauses"` toggle and a
  larger model are the escape hatches.

### `app/config.py` / `config.toml`

- `CleanupConfig.punctuation_source: str = "model"`. Read at the call sites
  (hotkey, `live_cleanup`) and passed to the view resolvers; not stored in the
  STT layer or read inside `Cleaner.clean`. Documented in `config.toml` with the
  A/B note.

## Data flow

- **Non-streaming:** `transcribe(audio) -> Transcript`; hotkey passes
  `model_text` + `fallback_text` into `clean`.
- **Streaming:** session yields model-view sentence units → `live_cleanup`
  cleans each → `finalize` assembles + reformats. `finish` returns the
  model-view full text used for the tail and the enumeration check.
- **Cleanup disabled:** `clean` returns `strip_fillers(fallback_text)` — offline
  keeps pause punctuation.

## Testing

- `test_stt.py`: assert the two views explicitly (2s pause → fallback
  `"...it. the docs..."` [capitalized later by `strip_fillers`], model
  `"...it the docs..."`); `classify_gap` unit cases; `collapse_ellipses`
  (trailing dropped, internal → space); no pause-forced capital.
- `test_streaming.py`: structured parts + boundaries; `stable_sentences` splits
  on `period` pauses and Whisper terminals; cross-pause A/B via
  `punctuation_source`.
- `test_live_cleanup.py`: sentence-unit consumption; per-chunk fallback on LLM
  failure; enumeration reformat preserved.
- `test_training.py` etc.: unaffected; keep green.

## Out of scope

- Acoustic/voice learning, feedback-panel redesign, correction retrieval — later
  items on the roadmap.
- Changing the divergence guard beyond the `model_text` basis.

## Known tradeoffs

- In `"model"` mode a weak (3B) model may under-punctuate (run-ons). Mitigated
  by the `"pauses"` toggle and larger models.
- Streaming per-chunk LLM failure falls back without pause punctuation (rare).
