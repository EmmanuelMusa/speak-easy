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


def test_register_cuda_libs_adds_bin_dirs_to_path(tmp_path, monkeypatch):
    import os
    from app.parakeet import _register_cuda_libs
    (tmp_path / "nvidia" / "cublas" / "bin").mkdir(parents=True)
    (tmp_path / "nvidia" / "cudnn" / "bin").mkdir(parents=True)
    monkeypatch.setenv("PATH", "EXISTING")
    _register_cuda_libs(str(tmp_path))
    path = os.environ["PATH"]
    assert str(tmp_path / "nvidia" / "cublas" / "bin") in path
    assert str(tmp_path / "nvidia" / "cudnn" / "bin") in path
    assert "EXISTING" in path  # existing PATH preserved


def test_register_cuda_libs_missing_dir_is_noop(tmp_path):
    from app.parakeet import _register_cuda_libs
    _register_cuda_libs(str(tmp_path / "nope"))  # must not raise
