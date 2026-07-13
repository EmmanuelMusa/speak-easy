"""Speech-to-text via faster-whisper (CTranslate2) with Silero VAD gating.

The model is loaded once and reused. device="auto" tries CUDA and falls back
to CPU (int8) if CUDA libraries are unavailable.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from .config import SttConfig

log = logging.getLogger(__name__)

# Pause-based punctuation: the silence between Whisper segments is the
# speaker's own punctuation. Only applied when the segment doesn't already
# end with punctuation, so Whisper's prosody judgement wins when it has one.
#
# Thresholds are on the VISIBLE gap after VAD padding: each speech chunk is
# padded by _VAD_PAD_MS on both sides, so a real pause of P seconds shows as
# roughly P - 2*pad. 0.20s visible ≈ a 0.45s spoken beat (comma); 0.65s
# visible ≈ a 0.9s real stop (full stop).
COMMA_GAP_S = 0.20
PERIOD_GAP_S = 0.65

# VAD tuning that makes short pauses observable at all: by default chunks
# only split on ~2s of silence, so a mid-sentence pause is invisible in the
# restored timestamps.
_VAD_MIN_SILENCE_MS = 300
_VAD_PAD_MS = 120


# A pause right after one of these is (almost) never punctuation — clauses
# don't end on articles, conjunctions, prepositions, auxiliaries, or subject
# pronouns. It's the speaker thinking ("we should ... move the meeting").
# Deliberately absent: object-capable pronouns (it, you, them, this), which
# legitimately end sentences ("I fixed it").
_THINKING_WORDS = frozenset(
    "a an the and or but nor so to of in on at for with by from as than "
    "that which who whose because if while when where i we he she they "
    "my our your their his is are was were be been being am do does did "
    "have has had will would can could should shall may might must "
    "don't doesn't didn't isn't aren't wasn't weren't won't wouldn't "
    "couldn't shouldn't can't very quite really just about".split()
)

_LAST_WORD_RE = re.compile(r"[a-z']+$")


def _pause_is_thinking(text: str) -> bool:
    m = _LAST_WORD_RE.search(text.lower())
    return bool(m) and m.group() in _THINKING_WORDS


def classify_gap(prev_text: str, gap: float) -> str:
    """Classify the pause that followed `prev_text`: 'none' | 'comma' |
    'period'. Mirrors append_gap_punctuation's decision so the two never
    diverge: a pause after a function word is the speaker thinking (none); a
    Whisper terminal already present needs nothing; a long stop is a period
    (upgrading a trailing comma); a short stop after a word is a comma."""
    if not prev_text or _pause_is_thinking(prev_text):
        return "none"
    last = prev_text[-1]
    if gap >= PERIOD_GAP_S:
        if last.isalnum() or last == ",":
            return "period"
        return "none"
    if gap >= COMMA_GAP_S and last.isalnum():
        return "comma"
    return "none"


def append_gap_punctuation(text: str, gap: float) -> str:
    """Punctuate `text` for a `gap`-second pause that followed it. Thin
    wrapper over classify_gap so a single place owns the decision."""
    kind = classify_gap(text, gap)
    if kind == "period":
        if text[-1].isalnum():
            return text + "."
        if text[-1] == ",":
            return text[:-1] + "."
        return text
    if kind == "comma":
        return text + ","
    return text


def stitch_segments(segs: "list[tuple[float, float, str]]") -> str:
    """Join segments, inserting pause-based punctuation at the gaps."""
    parts: list[str] = []
    prev_end: float | None = None
    for start, end, text in segs:
        if text:
            if parts and prev_end is not None:
                before = parts[-1]
                parts[-1] = append_gap_punctuation(before, start - prev_end)
                if parts[-1] is not before and parts[-1].endswith("."):
                    text = text[0].upper() + text[1:]
            parts.append(text)
        prev_end = end
    return " ".join(parts).strip()


_dlls_registered = False


def _register_nvidia_dlls() -> None:
    """Make pip-installed cuBLAS/cuDNN DLLs findable by ctranslate2.

    The nvidia-cublas-cu12 / nvidia-cudnn-cu12 wheels drop their DLLs in
    site-packages/nvidia/*/bin, which is not on the default DLL search path.
    No-op if the packages aren't installed (CPU fallback still applies).
    """
    global _dlls_registered
    if _dlls_registered:
        return
    _dlls_registered = True
    import os
    import site

    for base in site.getsitepackages():
        nvidia = Path(base) / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in nvidia.glob("*/bin"):
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:
                continue
            # ctranslate2 loads CUDA libs with plain LoadLibrary, which
            # ignores add_dll_directory but does search PATH.
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
            log.debug("Registered DLL dir: %s", bin_dir)


class Transcriber:
    def __init__(self, cfg: SttConfig):
        self.cfg = cfg
        self._model = None
        self._device = None
        self._force_cpu = False

    def _load(self):
        if self._model is not None:
            return self._model
        _register_nvidia_dlls()
        from faster_whisper import WhisperModel  # lazy: heavy import

        attempts = []
        if not self._force_cpu and self.cfg.device in ("auto", "cuda"):
            attempts.append(("cuda", self.cfg.compute_type))
        if self._force_cpu or self.cfg.device in ("auto", "cpu"):
            attempts.append(("cpu", "int8"))

        last_err = None
        for device, compute_type in attempts:
            try:
                self._model = WhisperModel(
                    self.cfg.model, device=device, compute_type=compute_type
                )
                self._device = device
                log.info("Loaded %s on %s (%s)", self.cfg.model, device, compute_type)
                return self._model
            except Exception as exc:  # CUDA missing/unsupported -> try next
                last_err = exc
                log.warning("Could not load on %s: %s", device, exc)
        raise RuntimeError(f"Failed to load Whisper model: {last_err}")

    def transcribe(
        self,
        audio: "np.ndarray | str | Path",
        initial_prompt: str | None = None,
    ) -> str:
        """Transcribe a mono float32 array (16 kHz) or an audio file path.

        vad_filter=True runs Silero VAD inside faster-whisper to drop
        silence/non-speech before decoding. Pauses between segments become
        punctuation (see stitch_segments).
        """
        segs = self.transcribe_segments(audio, initial_prompt=initial_prompt)
        return stitch_segments(segs)

    def transcribe_segments(
        self,
        audio: "np.ndarray | str | Path",
        initial_prompt: str | None = None,
    ) -> list[tuple[float, float, str]]:
        """Transcribe and return (start, end, text) per segment.

        Timestamps are seconds relative to the given audio. `initial_prompt`
        conditions the decoder on preceding text (used by streaming passes
        so later chunks see what was already said).
        """
        if isinstance(audio, np.ndarray) and audio.size == 0:
            return []
        try:
            return self._run(audio, initial_prompt)
        except RuntimeError as exc:
            # CUDA can fail lazily at encode time (e.g. missing cuBLAS/cuDNN
            # DLLs) even though the model constructed fine. Fall back to CPU.
            if self._device == "cuda" and self.cfg.device == "auto":
                log.warning("CUDA failed at runtime (%s); retrying on CPU", exc)
                self._model = None
                self._force_cpu = True
                return self._run(audio, initial_prompt)
            raise

    def _run(
        self,
        audio: "np.ndarray | str | Path",
        initial_prompt: str | None = None,
    ) -> list[tuple[float, float, str]]:
        model = self._load()
        segments, _info = model.transcribe(
            str(audio) if isinstance(audio, Path) else audio,
            language=self.cfg.language or None,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": _VAD_MIN_SILENCE_MS,
                "speech_pad_ms": _VAD_PAD_MS,
            },
            beam_size=self.cfg.beam_size,
            initial_prompt=initial_prompt,
            # Word timings expose the speaker's pauses. VAD only splits on
            # ~2s of silence, so without these a mid-utterance pause is
            # invisible: Whisper spans it with one segment and the pause
            # punctuation in stitch_segments never fires.
            word_timestamps=True,
        )
        out: list[tuple[float, float, str]] = []
        for seg in segments:
            words = seg.words or []
            if not words:
                out.append((seg.start, seg.end, seg.text.strip()))
                continue
            # Split the segment wherever the speaker paused, so every real
            # pause is a boundary that stitch_segments can punctuate.
            group = [words[0]]
            for w in words[1:]:
                if w.start - group[-1].end >= COMMA_GAP_S:
                    out.append((group[0].start, group[-1].end,
                                "".join(x.word for x in group).strip()))
                    group = [w]
                else:
                    group.append(w)
            out.append((group[0].start, group[-1].end,
                        "".join(x.word for x in group).strip()))
        return out
