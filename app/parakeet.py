"""Parakeet TDT speech-to-text via onnx-asr (optional engine).

Runs NVIDIA Parakeet TDT 0.6b through ONNX Runtime — no NeMo/PyTorch. Auto-uses
CUDA when onnxruntime-gpu is installed, else CPU. Non-streaming: the whole clip is
transcribed at release (Parakeet is fast enough not to need windowing).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

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


def _register_cuda_libs(base_dir: str) -> None:
    """Put the CUDA-13 library bin dirs under `base_dir` on the DLL search path
    so onnxruntime-gpu's CUDA provider finds cuDNN/cuBLAS 13. `base_dir` is a
    `pip install --target <dir>` location for the nvidia-*-cu13 wheels, kept
    isolated from faster-whisper's CUDA-12 libs in site-packages (their cuDNN
    DLLs share names and would otherwise cross-contaminate). No-op if absent."""
    base = Path(base_dir)
    if not base.is_dir():
        log.warning("parakeet_cuda_dir %r not found; CUDA provider may fail to "
                    "load and Parakeet will fall back to CPU", base_dir)
        return
    nvidia = base / "nvidia"
    root = nvidia if nvidia.is_dir() else base
    found = False
    for bin_dir in sorted(root.glob("*/bin")):
        try:
            os.add_dll_directory(str(bin_dir))
        except OSError:
            pass
        # ctranslate2/onnxruntime load CUDA libs with plain LoadLibrary, which
        # ignores add_dll_directory but does search PATH — so prepend there too.
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        found = True
    log.info("Registered CUDA-13 libs from %s (%s)", base_dir,
             "ok" if found else "no */bin dirs found")


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
            if "CUDAExecutionProvider" in provs and self.cfg.parakeet_cuda_dir:
                # onnxruntime-gpu's CUDA provider loads cuDNN/cuBLAS via plain
                # LoadLibrary (PATH only). Point it at the ISOLATED CUDA-13 libs
                # (parakeet_cuda_dir) — deliberately NOT faster-whisper's CUDA-12
                # libs in site-packages, whose cuDNN shares the same DLL names and
                # would break one engine or the other.
                _register_cuda_libs(self.cfg.parakeet_cuda_dir)
            self._model = onnx_asr.load_model(self.cfg.parakeet_model, providers=provs)
            log.info("Parakeet model '%s' loaded (providers: %s)",
                     self.cfg.parakeet_model, provs)
        return self._model

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
