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
