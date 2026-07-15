# Pluggable STT engine: Whisper + Parakeet, user-selectable

**Date:** 2026-07-15
**Status:** Approved direction (feasibility proven), pending spec review → plan

## Problem / goal

Let the user choose the speech-to-text engine: faster-whisper (current, incl. the
newly-added `large-v3-turbo`) or **NVIDIA Parakeet TDT 0.6b** (via `onnx-asr`,
proven working on this machine: ~1 s CPU on `sample.wav`, well-punctuated, no
NeMo/PyTorch). Selectable in `config.toml` and the Settings dialog.

## Proven facts

- `onnx-asr` (pure-Python; deps NumPy + onnxruntime + huggingface-hub) loads
  `nemo-parakeet-tdt-0.6b-v2` and `model.recognize(np_float32_16k, sample_rate=16000)`
  returns punctuated text. `onnxruntime` 1.27.0 already installed (Python 3.14).
- Parakeet transcribes the whole clip fast enough at release that it does NOT
  need the while-you-talk streaming windowing (which is faster-whisper-specific).

## Design

### Config (`SttConfig`)

- `engine: str = "whisper"` — `"whisper" | "parakeet"`.
- `parakeet_model: str = "nemo-parakeet-tdt-0.6b-v2"` — the `onnx_asr` model id.
- Existing `model`, `streaming`, etc. remain (apply to the Whisper engine).

### `ParakeetTranscriber` (new `app/parakeet.py`)

Duck-types the parts of `Transcriber` the app uses:
- `__init__(cfg: SttConfig)`; `_load()` lazily `onnx_asr.load_model(cfg.parakeet_model,
  providers=<auto>)` (cached); `warmup()` = `_load()`.
- **Auto GPU** (mirrors Whisper's `device = "auto"`): at load, inspect
  `onnxruntime.get_available_providers()` and pass `providers` preferring
  `CUDAExecutionProvider` (then `CPUExecutionProvider`) when CUDA is available,
  else CPU only. So with `onnxruntime-gpu` installed the GPU is used with no
  config; with the CPU-only `onnxruntime` it runs on CPU. Log which provider was
  chosen. (CUDA libs come from the existing `requirements-gpu.txt` wheels.)
- `transcribe(audio, initial_prompt=None) -> Transcript`: `recognize` the audio
  (numpy float32 @ 16 kHz, or a wav path for `--dry-run`) and return
  `Transcript(parts=[text] if text else [], boundaries=[])`. No pause boundaries
  (Parakeet gives token timestamps, not the segment interface streaming needs) —
  the model view is the plain text, which is exactly what the holistic cleanup
  path consumes. Empty audio → empty Transcript.
- Import `onnx_asr` lazily inside `_load` so the base app never imports it; a
  missing package raises a clear "engine=parakeet needs `pip install -r
  requirements-parakeet.txt`" error.

### Wiring (`app/hotkey.py`)

- `__init__`: pick the transcriber by engine —
  `ParakeetTranscriber(cfg.stt)` if `cfg.stt.engine == "parakeet"` else
  `Transcriber(cfg.stt)`.
- `_on_press`: create a `StreamingSession` (+ `LiveCleanup`) **only** when
  `cfg.stt.engine == "whisper" and cfg.stt.streaming`. For Parakeet, `_session`
  stays `None`, so `_process` records the full clip and calls
  `self.transcriber.transcribe(audio)` once at release (the existing
  non-streaming path — already holistic-cleanup-friendly).
- `run()` warmup + `_reload_stt` (on a settings change) rebuild the correct
  transcriber by engine.
- `_settings_snapshot` / `_apply_settings`: carry `engine` (rebuild STT when it
  changes, like the model change).

### Settings dialog (`app/overlay_ui.py`)

- Add an **Engine** dropdown (`whisper` | `parakeet`) above the Speech-model
  field; include `engine` in the saved values and the snapshot. The Speech-model
  dropdown applies to the Whisper engine (a tooltip notes Parakeet uses its own
  model). `STT_MODELS` already includes `large-v3-turbo`.

### CLI (`app/__main__.py`)

- `_dry_run` picks the transcriber by `cfg.stt.engine` (same helper as hotkey),
  so `--dry-run` works with either engine.

### Packaging

- `requirements-parakeet.txt`: `onnx-asr` (CPU `onnxruntime` is already present).
  For GPU, a comment: `pip uninstall onnxruntime && pip install onnxruntime-gpu`
  (they conflict) — the CUDA/cuDNN libs come from `requirements-gpu.txt`. The base
  `requirements.txt` is unchanged — Parakeet stays opt-in.
- README: a short "Choosing an STT engine" note (Whisper vs Parakeet; how to
  install the Parakeet extra).

## Testing

- Config: `SttConfig().engine == "whisper"`, `.parakeet_model` default; load from TOML.
- `ParakeetTranscriber` (mock `onnx_asr.load_model` → a fake with `.recognize`
  returning a fixed string): `transcribe(np.zeros(...))` returns a `Transcript`
  whose `model_text("model")` is that string; empty audio → empty Transcript;
  `warmup()`/`_load()` caches (load called once).
- Engine selection: a helper `make_transcriber(cfg.stt)` (or the `__init__`
  branch) returns a `ParakeetTranscriber` for `engine="parakeet"` and a
  `Transcriber` otherwise — unit-test the selector without constructing the heavy
  app.
- Hotkey: `_on_press` does not create a `StreamingSession` when `engine ==
  "parakeet"` (test via a lightweight `MagicMock`-self, mirroring existing hotkey
  tests) — verify `self._session is None` / streaming skipped.
- Full suite stays green; live Parakeet quality is the user's check.

## Out of scope / tradeoffs

- Parakeet runs **non-streaming** (no while-you-talk transcription); acceptable
  because it's ~1 s. A future streaming adapter is possible but not needed.
- GPU for Parakeet needs `onnxruntime-gpu` (CPU works out of the box); documented.
- Pause-based punctuation (`punctuation_source = "pauses"`) has no effect under
  Parakeet (no pause boundaries) — the model already punctuates; `"model"` mode
  is the norm anyway.
- `parakeet-tdt-0.6b-v2` (English, top of the English leaderboard) is the default;
  `v3` (multilingual) is selectable via `parakeet_model`.
