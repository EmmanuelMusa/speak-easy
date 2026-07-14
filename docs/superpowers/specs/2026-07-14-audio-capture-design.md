# Audio capture for acoustic-training pairs (Sub-project A)

**Date:** 2026-07-14
**Status:** Approved design (pending spec review), then implementation plan

**Parent project:** "Acoustic learning" decomposed into A (this — data capture),
B (offline fine-tune harness), C (adapted-model integration). This spec defines
the on-disk `(audio, transcript)` contract that B will consume.

## Problem

[`_process`](../../../app/hotkey.py) transcribes the recorded audio and then
discards it. The feedback panel's `transcript` field ("what you actually said",
Round 14) captures the *verbatim ground-truth* text, but with no audio paired to
it — so there is nothing to fine-tune Whisper's acoustics on.

## Goal

When the user submits a feedback correction that includes a `transcript`, persist
that dictation's audio as a WAV and link it to the training-log entry, producing a
verified `(audio, verbatim-transcript)` pair. Nothing else saves audio.

Why `transcript` (not `raw`, not `ideal`): supervised ASR fine-tuning minimizes the
gap between the model's prediction and a ground-truth target. `raw` is what Whisper
produced (the thing being corrected — zero learning signal); `ideal` trains the
separate text-cleanup LLM. Only the verbatim `transcript` is a valid acoustic target.

## What's captured

A pair is saved **only when a feedback submission carries a non-empty `transcript`**
(the user edited "what you actually said"). A submission with only a rating, ideal
cleanup, and/or tags saves no audio.

## Components and interfaces

### `app/training.py`

- New imports: `wave` (stdlib), `numpy as np`.
- Module constant `AUDIO_DIR = _ROOT / "training_audio"`.
- `TrainingStore.__init__(self, data_path=DATA_PATH, vocab_path=VOCAB_PATH,
  audio_dir=AUDIO_DIR)` — new `audio_dir` param, mirroring the existing path params
  so tests can point it at a `tmp_path`.
- `save_audio(self, audio, sample_rate: int) -> str | None` — writes `audio`
  (a float32 mono array in [-1, 1]) as a **16-bit PCM mono WAV** via the stdlib
  `wave` module into `audio_dir`, and returns a POSIX path of the form
  `"<audio_dir-name>/<file>.wav"` (e.g. `training_audio/1699999999123.wav`) — i.e.
  the path **relative to `audio_dir.parent`**, which is the project root in
  production (`AUDIO_DIR = _ROOT / "training_audio"`). This keeps the stored path
  portable and avoids assuming `audio_dir` lives under `_ROOT` (so tests can use a
  `tmp_path` dir). Filename is a millisecond timestamp (`int(time.time() * 1000)`),
  unique for manual corrections. Returns `None` for empty/degenerate audio or on any
  write error (logged, never raises — dictation must not break). Conversion: `clip`
  to [-1, 1], `* 32767` → int16 little-endian; `setnchannels(1)`, `setsampwidth(2)`,
  `setframerate(sample_rate)`.
- `record(..., audio_path: str | None = None)` — add `"audio": audio_path` to the
  JSONL entry (a backward-compatible new field; old lines lack it, readers `.get()`).

### `app/hotkey.py`

- `PushToTalkApp.__init__`: add `self._last_audio: tuple | None = None`.
- `_process`: when training is enabled, before requesting feedback, stash the
  dictation's audio: `self._last_audio = (audio, self.cfg.audio.sample_rate)` (then
  `self.overlay.request_feedback(raw, cleaned)` as today). One dictation runs at a
  time (busy-lock) and feedback is single-slot, so one stash slot is sufficient; a
  new dictation overwrites it.
- `_record_feedback(self, raw, output, rating, transcript, ideal, tags)`: pop the
  stash; if `transcript` is present and `self.cfg.training.save_correction_audio`
  and a stash exists, `audio_path = self.training.save_audio(audio, sample_rate)`;
  pass `audio_path=` into `record(...)`. Always clear the stash. (The existing
  verdict derivation, context/injector replace-on-ideal behavior is unchanged.)

### `app/config.py` / `config.toml`

- `TrainingConfig.save_correction_audio: bool = True` — capture is on by default
  (the point of training mode) but can be disabled for text-only training / privacy.
  Documented in `config.toml` under `[training]`.

### `.gitignore`

- Add `training_audio/` (voice recordings are local user data, like
  `training_data.jsonl` and `learned_vocab.json`, which are already ignored).

## Data model

Each corrected-with-transcript entry in `training_data.jsonl` gains:

```json
{ "...": "...", "transcript": "<verbatim ground truth>", "audio": "training_audio/<id>.wav" }
```

A training pair for Sub-project B = every entry where **both** `audio` and
`transcript` are non-null: resolve the waveform as `_ROOT / entry["audio"]` (the
stored path is relative to the project root) and use `entry["transcript"]` as the
target text. WAV format: mono, 16-bit PCM, at the capture sample rate
(`[audio].sample_rate`, 16 kHz) — already what Whisper expects.

## Testing

- `save_audio` writes a WAV that re-opens via `wave` with the expected
  `nchannels=1`, `sampwidth=2`, `framerate=sample_rate`, and frame count equal to
  the input length; returns a project-relative POSIX path under `training_audio/`.
- `record(audio_path="training_audio/x.wav")` stores it as `entry["audio"]`; a
  `record(...)` without `audio_path` stores `entry["audio"] is None`.
- `save_audio` on an empty array returns `None` and writes no file.
- Hotkey wiring: call `PushToTalkApp._record_feedback` unbound with a `MagicMock`
  self (as the existing `_record_feedback` test does) — with a `transcript` and
  `save_correction_audio=True` and a stashed `_last_audio`, it calls
  `training.save_audio` and passes the returned path into `record(...)`; with
  `transcript=None` it does not call `save_audio` and passes `audio_path=None`; the
  stash is cleared either way.
- Tests use a `tmp_path` `audio_dir` via the new `TrainingStore` param (update the
  shared `make_store` helper).

## Out of scope

- Sub-project B (offline LoRA fine-tune harness) and C (convert + load the adapted
  model). This spec only fixes the `(audio, transcript)` capture and its on-disk
  layout — the contract B reads.
- Retention limits / pruning of `training_audio/` — keep everything for now; a cap
  is a later addition (noted so it is a conscious omission).

## Known tradeoffs

- Audio is stored only for dictations the user explicitly corrected with a verbatim
  transcript — minimal footprint and privacy exposure, at the cost of a smaller
  dataset (which is the right trade: verified pairs only).
- No retention cap yet; corrected-only audio is modest, but a long-running user will
  want pruning eventually.
- The stash follows the single-slot feedback panel: if a correction is superseded by
  a new dictation before being submitted, its audio is dropped (never written) —
  acceptable, matches how superseded feedback is already discarded.
