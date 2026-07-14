# Feedback panel redesign — richer capture, progressive UI

**Date:** 2026-07-14
**Status:** Approved design (pending spec review), then implementation plan

## Problem

Training-mode feedback today ([`FeedbackPanel`](../../../app/overlay_ui.py)) is too thin to teach the system well:
- Times out after 6s (`FEEDBACK_TIMEOUT_MS`) even in training mode.
- Truncates the text to 52 chars.
- Uses 👍/👎 **emojis** (the rest of the app is vector).
- Captures only one binary verdict + one "ideal" field — no separation of *what was misheard* (STT) from *how it was cleaned* (LLM), and no graded/categorized signal.

## Goals (decided with the user)

- **Progressive**: after each dictation a small **collapsed rating strip** appears (no timeout); clicking **"Correct it"** expands into a full teaching form.
- **1–5 star rating**, drawn as **vector** stars (no emoji), filled = accent, empty = outline.
- **Four fields** in the expanded form: **Heard** (raw STT, read-only) · **Cleaned** (what it typed, read-only) · **What you actually said** (editable) · **Ideal cleanup** (editable).
- **Multi-select failure tags**: `misheard word`, `wrong punctuation`, `over-deleted`, `wrong casing`, `bad list` (several selectable at once).
- **Full text**, never truncated.
- **Learning scope**: capture the richer data and wire it into *today's* learning — few-shot from corrections as now, **plus** mine STT-misheard terms from the new true-transcript field. Store rating + tags for later. Similarity retrieval over all corrections (roadmap #2) and acoustic learning (roadmap #3) stay separate.

## Data schema

One JSONL line per feedback event ([training_data.jsonl](../../../training_data.jsonl)), a **backward-compatible superset** of today's `{ts, raw, output, verdict, ideal}` — old lines still load (readers use `.get()`):

```json
{
  "ts": 1784000000.0,
  "raw": "<STT text that was fed to cleanup>",
  "output": "<what cleanup produced / typed>",
  "rating": 3,                       // 1-5, or null (new)
  "transcript": "<what the user actually said>",  // or null (new)
  "ideal": "<what cleanup should have produced>", // or null
  "tags": ["wrong punctuation"],     // [] if none (new)
  "verdict": "bad"                   // "ok"|"bad", retained, derived (see below)
}
```

`verdict` is kept only for backward compatibility (existing data + the unchanged
few-shot mechanism); it is **derived**, not user-facing: `"ok"` when `rating == 5`
and no `ideal` correction, else `"bad"`.

## Interaction flows

**Collapsed strip** (shown after each dictation in training mode; never steals
keyboard focus — `WA_ShowWithoutActivating` + `WindowDoesNotAcceptFocus`):
full cleaned text (wrapped) · 5 stars · `Correct it ›`. **No timeout.**

- **Tap a star (N)** → fast path: submit `rating=N` with no correction, close.
  A low rating with no correction is still recorded as signal.
- **Click "Correct it ›"** → expand to the teaching form (the window may now take
  focus so fields are editable).
- **Superseded** (next dictation's feedback arrives) or **Cancel** → close,
  record nothing (same as today's ignored-timeout behavior).

**Teaching form** (expanded): title · settable star row · **Heard** (read-only
raw) · **Cleaned** (read-only output) · **What you actually said** (editable,
pre-filled with raw) · **Ideal cleanup** (editable, pre-filled with output) ·
multi-select tag chips · Cancel · Save.

- **Save** → submit `rating` (if set), `transcript` (only if edited, else null),
  `ideal` (only if edited, else null), `tags`. Close.
- Sending an edited field as null when unchanged avoids creating spurious
  "corrections" that equal the original.

## Components and interfaces

### `app/overlay_ui.py` — `FeedbackPanel` (the bulk of the work)

Rewrite the panel to the two-state progressive design above. Add a small vector
**star** widget (a `QPainterPath` star, filled/outline, click sets rating — same
approach as the existing vector gear `_gear_path`). Tag chips are toggle buttons
tracking a set. Remove `FEEDBACK_TIMEOUT_MS` and its `QTimer`.

- Child receives: `feedback {"id": N, "raw": "<raw>", "cleaned": "<output>"}`
  (today it only sends `preview`; it must now send **both** raw and cleaned).
- Child emits on submit:
  `{"type":"feedback","id":N,"rating":int|null,"transcript":str|null,"ideal":str|null,"tags":[str]}`.
- Extend the existing `selftest` stdin command to instantiate the panel in both
  collapsed and expanded states so the render paths are exercised headlessly.

### `app/overlay.py` — `Overlay`

- `request_feedback(raw, output)`: send `feedback {"id", "raw": raw, "cleaned": output}`.
  Track only the **most recent** request as a single slot,
  `self._pending = (id, raw, output)` (only one panel is ever shown at a time —
  a new request supersedes the old), replacing today's `_pending: dict`. This
  removes the growth/leak that a no-timeout, dismissable panel would otherwise
  cause (dismissed/superseded ids would never be popped from a dict).
- `_dispatch` for a `feedback` event: ignore it unless its `id` matches the
  current `self._pending`; if it is an actual answer (has a `rating` or an
  `ideal`), call `self.on_feedback(raw, output, rating, transcript, ideal, tags)`.
  A dismiss/no-answer event (or none at all) records nothing; the slot is
  replaced on the next request.
- New callback shape: `on_feedback(raw, output, rating, transcript, ideal, tags)`.

### `app/hotkey.py` — `PushToTalkApp._record_feedback`

New signature `_record_feedback(raw, output, rating, transcript, ideal, tags)`:
- Derive `verdict` (`"ok"` if `rating == 5 and not ideal` else `"bad"`).
- `self.training.record(raw, output, verdict, ideal, rating=rating, transcript=transcript, tags=tags)`.
- If `ideal`: `self.context.replace_last(ideal)`, and if `replace_on_correction`,
  `self.injector.replace_last(ideal)` (unchanged behavior, now gated on `ideal`
  presence rather than `verdict`).

### `app/training.py` — `TrainingStore`

- `record(raw, output, verdict, ideal=None, *, rating=None, transcript=None, tags=None)`:
  write all fields (`tags` defaults to `[]`). Vocabulary mining now runs on
  **two** diffs, both through the existing gated `_mine_vocab` (phonetic
  similarity + `_is_distinctive`):
  - **Cleanup terms** — if `ideal` and `ideal.strip() != output.strip()`:
    `_mine_vocab(output, ideal)`.
  - **STT-misheard terms** — if `transcript` and `transcript.strip() != raw.strip()`:
    `_mine_vocab(raw, transcript)` (the true-transcript side holds the word STT got wrong).
- `corrections(n=5)`: filter by **`ideal` present** (drop the `verdict=="bad"`
  dependency) — backward-compatible with existing data and tests. `few_shot_block`
  unchanged (raw→ideal, last 5).
- `rating`/`tags` are stored only (analytics + future use); no learning consumes
  them yet.

### Config

Remove `FEEDBACK_TIMEOUT_MS` usage (no timeout in training mode). No new config keys.

## Testing

- `test_training.py`: `record` persists `rating`/`transcript`/`tags`;
  STT-mishear vocab mined from `transcript` (e.g. raw `"meet with mr ogi up"`,
  transcript `"meet with Mr Ogiop"` → learns `"Ogiop"`); cleanup-term mining still
  works; `corrections()` returns `ideal`-bearing entries; few-shot unchanged.
- Overlay dispatch (`test_training.py`): a `feedback` event carrying
  `rating`+`transcript`+`ideal`+`tags` calls `on_feedback` with all six args; a
  rating-only event (no `ideal`) calls it with `ideal=None`; a dismissed/no-answer
  event does not call it and clears `_pending`. Update the existing
  `test_feedback_event_dispatch` / `test_feedback_timeout_not_recorded` to the new shape.
- `overlay_ui` `selftest`: instantiate `FeedbackPanel` collapsed + expanded, emit
  `selftest_ok` (render smoke, as done for the settings/review dialogs).

## Out of scope

- Similarity retrieval over all corrections (roadmap #2) — few-shot stays last-5.
- Acoustic/voice (LoRA) learning (roadmap #3) — but the new `transcript` field is
  the data foundation it will need.
- Deep `ReviewDialog` changes — it keeps working (reads `corrections()`); showing
  the star rating/tags there is optional polish, not required.

## Known tradeoffs

- The collapsed strip shows the **full** cleaned text (per the "show text fully"
  requirement), so a long dictation makes a taller strip. Acceptable — training
  mode is opt-in.
- `rating`/`tags` are captured but not yet consumed by the learning algorithm;
  they are forward-looking signal (analytics, future retrieval weighting).
