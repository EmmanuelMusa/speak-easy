# Pluggable STT Engine (Whisper + Parakeet) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user pick the STT engine — faster-whisper (default) or NVIDIA Parakeet TDT via `onnx-asr` — in config and Settings; Parakeet runs non-streaming with auto-GPU.

**Architecture:** A `ParakeetTranscriber` duck-types the parts of `Transcriber` the app uses (`transcribe -> Transcript`, `_load`, `warmup`). A `make_transcriber(stt_cfg)` factory picks the backend by `[stt].engine`. Parakeet skips the Whisper-specific streaming pipeline.

**Tech Stack:** Python 3, faster-whisper, `onnx-asr` (+ onnxruntime, already installed), PySide6.

## Global Constraints

- No new **base** dependencies; `onnx-asr` is an opt-in extra (`requirements-parakeet.txt`). Base `requirements.txt` unchanged.
- Tests: `.venv/Scripts/python.exe -m pytest`. Commit each task (targeted `git add`). No `Co-Authored-By`/AI-authorship line.
- Default `[stt].engine = "whisper"` — nothing changes unless the user picks Parakeet.
- Parakeet assumes the app's fixed 16 kHz capture (same as Whisper). Auto-GPU: prefer `CUDAExecutionProvider` when available, else CPU.
- Baseline before Task 1: full suite **158 passing**.

**Reference spec:** `docs/superpowers/specs/2026-07-15-pluggable-stt-engine-design.md`

---

## Task 1: Config — `engine` + `parakeet_model`

**Files:** Modify `app/config.py` (`SttConfig`), `config.toml` (`[stt]`); Test `tests/test_smoke.py`.

**Interfaces:** `SttConfig.engine: str = "whisper"`, `SttConfig.parakeet_model: str = "nemo-parakeet-tdt-0.6b-v2"`.

- [ ] **Step 1: Failing test** — append to `tests/test_smoke.py`:

```python
def test_stt_engine_defaults():
    from app.config import SttConfig
    assert SttConfig().engine == "whisper"
    assert SttConfig().parakeet_model == "nemo-parakeet-tdt-0.6b-v2"
```

- [ ] **Step 2: Run → FAIL** (`.venv/Scripts/python.exe -m pytest tests/test_smoke.py::test_stt_engine_defaults -v`).

- [ ] **Step 3: Implement** — in `app/config.py` `@dataclass class SttConfig`, add after `streaming`:

```python
    # Speech-to-text engine: "whisper" (faster-whisper) or "parakeet"
    # (NVIDIA Parakeet TDT via onnx-asr — pip install -r requirements-parakeet.txt).
    engine: str = "whisper"
    # Parakeet model id for onnx-asr (English v2, or "nemo-parakeet-tdt-0.6b-v3"
    # for multilingual). Only used when engine = "parakeet".
    parakeet_model: str = "nemo-parakeet-tdt-0.6b-v2"
```

In `config.toml` `[stt]`, add after the `streaming = true` line:

```toml
# Speech-to-text engine: "whisper" (faster-whisper, streaming) or "parakeet"
# (NVIDIA Parakeet TDT via onnx-asr — run: pip install -r requirements-parakeet.txt;
# for GPU: pip uninstall onnxruntime && pip install onnxruntime-gpu). Non-streaming.
engine = "whisper"
parakeet_model = "nemo-parakeet-tdt-0.6b-v2"
```

- [ ] **Step 4: Run** full suite → PASS.
- [ ] **Step 5: Commit** `git add app/config.py config.toml tests/test_smoke.py` / `feat(stt): engine + parakeet_model config`.

---

## Task 2: `ParakeetTranscriber` + `make_transcriber` factory

**Files:** Create `app/parakeet.py`; Modify `app/stt.py` (add `make_transcriber`); Test `tests/test_parakeet.py` (create).

**Interfaces:**
- `ParakeetTranscriber(cfg: SttConfig)` with `_load()`, `warmup()`, `transcribe(audio, initial_prompt=None) -> Transcript`.
- `stt.make_transcriber(cfg: SttConfig)` → `ParakeetTranscriber` if `cfg.engine == "parakeet"` else `Transcriber`.
- `parakeet._providers() -> list[str]` (CUDA-then-CPU when available).

- [ ] **Step 1: Failing tests** — create `tests/test_parakeet.py`:

```python
"""Parakeet STT backend (onnx-asr), mocked so no model download/runtime."""
import sys
import types
from unittest.mock import MagicMock

import numpy as np

from app.config import SttConfig


def _install_fake_onnx_asr(recognize_return="hello world"):
    fake = types.ModuleType("onnx_asr")
    model = MagicMock()
    model.recognize.return_value = recognize_return
    fake.load_model = MagicMock(return_value=model)
    sys.modules["onnx_asr"] = fake
    return fake, model


def test_transcribe_returns_transcript_text(monkeypatch):
    fake, model = _install_fake_onnx_asr("Meet me at noon.")
    from app.parakeet import ParakeetTranscriber
    t = ParakeetTranscriber(SttConfig(engine="parakeet"))
    tr = t.transcribe(np.zeros(16000, dtype=np.float32))
    assert tr.model_text("model") == "Meet me at noon."
    model.recognize.assert_called_once()


def test_empty_audio_returns_empty(monkeypatch):
    _install_fake_onnx_asr("should not be called")
    from app.parakeet import ParakeetTranscriber
    t = ParakeetTranscriber(SttConfig(engine="parakeet"))
    tr = t.transcribe(np.array([], dtype=np.float32))
    assert tr.model_text("model") == ""


def test_load_is_cached(monkeypatch):
    fake, model = _install_fake_onnx_asr("x")
    from app.parakeet import ParakeetTranscriber
    t = ParakeetTranscriber(SttConfig(engine="parakeet"))
    t.transcribe(np.zeros(10, dtype=np.float32))
    t.transcribe(np.zeros(10, dtype=np.float32))
    fake.load_model.assert_called_once()  # model loaded once, reused


def test_make_transcriber_selects_engine():
    from app.stt import make_transcriber, Transcriber
    _install_fake_onnx_asr()
    from app.parakeet import ParakeetTranscriber
    assert isinstance(make_transcriber(SttConfig(engine="parakeet")), ParakeetTranscriber)
    assert isinstance(make_transcriber(SttConfig(engine="whisper")), Transcriber)
```

- [ ] **Step 2: Run → FAIL** (`app.parakeet` / `make_transcriber` don't exist).

- [ ] **Step 3: Implement** — create `app/parakeet.py`:

```python
"""Parakeet TDT speech-to-text via onnx-asr (optional engine).

Runs NVIDIA Parakeet TDT 0.6b through ONNX Runtime — no NeMo/PyTorch. Auto-uses
CUDA when onnxruntime-gpu is installed, else CPU. Non-streaming: the whole clip is
transcribed at release (Parakeet is fast enough not to need windowing).
"""

from __future__ import annotations

import logging

import numpy as np

from .config import SttConfig
from .stt import Transcript

log = logging.getLogger(__name__)


def _providers() -> list[str]:
    """Prefer CUDA when onnxruntime exposes it, else CPU."""
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]
    picked = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
              if p in avail]
    return picked or ["CPUExecutionProvider"]


class ParakeetTranscriber:
    """faster-whisper `Transcriber` look-alike backed by onnx-asr."""

    def __init__(self, cfg: SttConfig):
        self.cfg = cfg
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                import onnx_asr
            except ImportError as exc:
                raise RuntimeError(
                    "engine='parakeet' needs onnx-asr — run: "
                    "pip install -r requirements-parakeet.txt"
                ) from exc
            provs = _providers()
            self._model = onnx_asr.load_model(self.cfg.parakeet_model, providers=provs)
            log.info("Parakeet model '%s' loaded (providers: %s)",
                     self.cfg.parakeet_model, provs)
        return self._model

    def warmup(self) -> None:
        try:
            self._load()
        except Exception as exc:
            log.warning("Parakeet warmup failed (%s); will retry on first use", exc)

    def transcribe(self, audio, initial_prompt: str | None = None) -> Transcript:
        """Transcribe a mono float32 array (16 kHz) or a wav path. Returns a
        Transcript with the plain text as its single part (no pause boundaries —
        Parakeet self-punctuates and the holistic cleanup path consumes the text
        directly)."""
        if isinstance(audio, np.ndarray):
            if audio.size == 0:
                return Transcript(parts=[], boundaries=[])
            text = self._load().recognize(audio, sample_rate=16000)
        else:
            text = self._load().recognize(str(audio))  # wav path (e.g. --dry-run)
        text = (text or "").strip()
        return Transcript(parts=[text] if text else [], boundaries=[])
```

In `app/stt.py`, add the factory (near the bottom, after `Transcriber`):

```python
def make_transcriber(cfg: SttConfig):
    """Return the STT backend for the configured engine. Parakeet is imported
    lazily so the base app never imports onnx-asr."""
    if cfg.engine == "parakeet":
        from .parakeet import ParakeetTranscriber
        return ParakeetTranscriber(cfg)
    return Transcriber(cfg)
```

- [ ] **Step 4: Run** `.venv/Scripts/python.exe -m pytest tests/test_parakeet.py -v` → PASS; full suite → PASS.
- [ ] **Step 5: Commit** `git add app/parakeet.py app/stt.py tests/test_parakeet.py` / `feat(stt): Parakeet (onnx-asr) backend + engine factory`.

---

## Task 3: Wire hotkey + CLI to the engine

**Files:** Modify `app/hotkey.py`, `app/__main__.py`; Test `tests/test_smoke.py` (or `tests/test_training.py`) for the streaming-skip.

**Interfaces:** Consumes `make_transcriber`. When `engine == "parakeet"`, no `StreamingSession`/`LiveCleanup`.

- [ ] **Step 1: Failing test** — append to `tests/test_smoke.py`:

```python
def test_parakeet_engine_skips_streaming_session():
    # With engine=parakeet, _on_press must NOT build a StreamingSession
    # (Parakeet is non-streaming); the full clip is transcribed at release.
    from unittest.mock import MagicMock
    from app.hotkey import PushToTalkApp
    from app.config import Config
    cfg = Config()
    cfg.stt.engine = "parakeet"
    cfg.stt.streaming = True  # even if streaming is on, Parakeet ignores it
    fake = MagicMock()
    fake.cfg = cfg
    fake.recorder.recording = False
    fake._busy.locked.return_value = False
    PushToTalkApp._on_press(fake)
    assert fake._session is None
    fake.recorder.start.assert_called_once()
```

- [ ] **Step 2: Run → FAIL** (current `_on_press` builds a session whenever `stt.streaming`).

- [ ] **Step 3: Implement.**

In `app/hotkey.py`, add the import: `from .stt import Transcriber, make_transcriber` (replace the existing `from .stt import Transcriber`).

In `__init__`, change `self.transcriber = Transcriber(cfg.stt)` to:

```python
        self.transcriber = make_transcriber(cfg.stt)
```

In `_on_press`, change the streaming guard so Parakeet never streams:

```python
        if self.cfg.stt.streaming and self.cfg.stt.engine == "whisper":
            self._session = StreamingSession(
```

(the rest of that block — the `LiveCleanup` creation inside it — is unchanged).

In `run()`, generalize the model-load log + load (Parakeet may raise if onnx-asr is missing; don't crash startup):

```python
        log.info("Loading STT engine '%s'...", self.cfg.stt.engine)
        try:
            self.transcriber._load()
        except Exception as exc:
            log.error("STT engine failed to load (%s)", exc)
```

In `_reload_stt`, use the factory:

```python
    def _reload_stt(self) -> None:
        log.info("Loading STT engine '%s'...", self.cfg.stt.engine)
        new = make_transcriber(self.cfg.stt)
        try:
            new._load()
        except Exception as exc:
            log.error("STT engine failed to load (%s); keeping current", exc)
            return
        self.transcriber = new
        log.info("Speech engine switched to '%s'", self.cfg.stt.engine)
```

In `_settings_snapshot`, add `"engine": self.cfg.stt.engine,`.

In `_apply_settings`, handle the engine like the model (rebuild STT on change). After the `old_model` capture, add `old_engine = self.cfg.stt.engine`; set `self.cfg.stt.engine = values.get("engine", old_engine)`; persist it in `save_config_updates` under `"stt"` (`"engine": self.cfg.stt.engine` alongside `"model"`); and trigger `_reload_stt` when **either** the model or the engine changed:

```python
        if self.cfg.stt.model != old_model or self.cfg.stt.engine != old_engine:
            threading.Thread(target=self._reload_stt, daemon=True).start()
```

In `app/__main__.py` `_dry_run`, replace `raw = Transcriber(cfg.stt).transcribe(wav)`-style construction with the factory (keep the existing `.transcribe(...)`/views usage):

```python
    from .stt import make_transcriber
    tr = make_transcriber(cfg.stt).transcribe(wav)
```

(Adjust the surrounding lines to keep the current `tr.model_text(source)` / `fallback_text` usage — only the transcriber construction changes.)

- [ ] **Step 4: Run** full suite → PASS.
- [ ] **Step 5: Commit** `git add app/hotkey.py app/__main__.py tests/test_smoke.py` / `feat(hotkey): select STT engine; Parakeet runs non-streaming`.

---

## Task 4: Settings dialog — Engine dropdown

**Files:** Modify `app/overlay_ui.py` (`SettingsDialog`).

**Interfaces:** Settings emit/consume `"engine"`.

- [ ] **Step 1:** Add near `STT_MODELS`:

```python
STT_ENGINES = ["whisper", "parakeet"]
```

- [ ] **Step 2:** In `SettingsDialog.__init__`, create the combo and place it above the Speech-model field:

```python
        self.engine = QtWidgets.QComboBox()
        self.engine.addItems(STT_ENGINES)
        self.engine.setToolTip(
            "whisper = faster-whisper; parakeet = NVIDIA Parakeet TDT via onnx-asr "
            "(pip install -r requirements-parakeet.txt). Speech model below applies to whisper.")
```

Add it to the DICTATION section before the Speech-model field:

```python
        field("STT engine", self.engine)
        field("Speech model", self.stt_model)
```

- [ ] **Step 3:** In `load(values)`, add:

```python
        self.engine.setCurrentText(str(values.get("engine", "whisper")))
```

In `_save`'s emitted `values`, add:

```python
                    "engine": self.engine.currentText(),
```

- [ ] **Step 4: Run** `.venv/Scripts/python.exe -m pytest tests/test_overlay_ui.py -q` → PASS (the `selftest` builds `SettingsDialog`, exercising the new widget). Full suite → PASS.
- [ ] **Step 5: Commit** `git add app/overlay_ui.py` / `feat(overlay): STT engine dropdown in Settings`.

---

## Task 5: Packaging + docs

**Files:** Create `requirements-parakeet.txt`; Modify `README.md`.

- [ ] **Step 1:** Create `requirements-parakeet.txt`:

```
# Optional: NVIDIA Parakeet TDT speech-to-text engine (set [stt].engine = "parakeet").
# Lightweight — no NeMo/PyTorch. onnxruntime (CPU) is already a base dependency.
onnx-asr>=0.12
# For GPU (NVIDIA CUDA): swap the CPU runtime for the GPU one —
#   pip uninstall onnxruntime && pip install onnxruntime-gpu
# The CUDA/cuDNN libraries come from requirements-gpu.txt.
```

- [ ] **Step 2:** In `README.md`, add a short "Choosing an STT engine" subsection under the config/setup area: Whisper (default, streaming, `large-v3-turbo` for speed) vs Parakeet TDT (`pip install -r requirements-parakeet.txt`, non-streaming, self-punctuating, ~1 s, auto-GPU with `onnxruntime-gpu`), selectable via `[stt].engine` or the Settings Engine dropdown.

- [ ] **Step 3: Commit** `git add requirements-parakeet.txt README.md` / `docs: Parakeet engine install + STT engine choice`.

---

## Task 6: Verification

- [ ] **Step 1:** Full suite `.venv/Scripts/python.exe -m pytest -q` green.
- [ ] **Step 2 (scripted):** `.venv/Scripts/python.exe -m app --dry-run assets/sample.wav` with `[stt].engine = "parakeet"` set → prints the Parakeet transcript + cleaned output (the model is already cached). Revert engine to whisper after.
- [ ] **Step 3 (user, live):** In Settings, switch Engine to `parakeet`, dictate, confirm it transcribes; switch back to `whisper` + `large-v3-turbo` and compare.

---

## Self-Review (plan author)

- **Spec coverage:** engine/parakeet_model config (T1); ParakeetTranscriber auto-GPU + factory (T2); hotkey engine selection + non-streaming Parakeet + settings/reload + CLI (T3); Settings dropdown (T4); requirements-parakeet + README (T5); verification (T6). All spec sections mapped.
- **Type consistency:** `make_transcriber(cfg) -> Transcriber|ParakeetTranscriber`; `ParakeetTranscriber.transcribe -> Transcript`; `SttConfig.engine/parakeet_model`; settings `"engine"` key threaded through snapshot/load/save/apply. Consistent.
- **No cycle:** `parakeet.py` imports `stt.Transcript`; `stt.make_transcriber` imports `parakeet` lazily (call-time) — no import cycle.
- **Placeholders:** none — full code/tests given; Parakeet tests mock `onnx_asr` (no download/runtime in CI).
