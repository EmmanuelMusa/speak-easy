"""Configuration loading for Speak Easy.

Reads config.toml (stdlib tomllib) and exposes a typed Config object with
defaults matching the shipped config file, so the app still runs if keys
are missing.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


@dataclass
class SttConfig:
    model: str = "small.en"
    device: str = "auto"          # auto | cuda | cpu
    compute_type: str = "float16"  # used on CUDA; CPU forces int8
    language: str = "en"
    beam_size: int = 2             # 1 = fastest, 5 = most accurate
    # Transcribe while the hotkey is held (commit settled segments early)
    # so release only pays for the last ~second of audio.
    streaming: bool = True


@dataclass
class CleanupConfig:
    enabled: bool = True
    # Clean finished sentences with the LLM while you're still dictating,
    # so release only pays for the last unfinished sentence.
    streaming: bool = True
    # 127.0.0.1, NOT localhost: on Windows, "localhost" tries IPv6 first and
    # wastes ~2s per request when the server only listens on IPv4.
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b"
    timeout_seconds: float = 15.0
    # On battery the GPU downclocks and generation is several times slower;
    # the timeout is multiplied by this so responses complete instead of
    # being abandoned for the dumb local fallback.
    battery_timeout_multiplier: float = 4.0
    # Where sentence punctuation/casing comes from. "model": the cleanup LLM
    # receives words + Whisper punctuation only and punctuates from context.
    # "pauses": the LLM also gets the deterministic pause-derived punctuation
    # (legacy behavior / A-B baseline). The offline fallback always keeps the
    # pause punctuation regardless of this setting.
    punctuation_source: str = "model"
    custom_vocabulary: list[str] = field(default_factory=list)


@dataclass
class InjectionConfig:
    delivery_method: str = "clipboard"  # clipboard | sendinput
    paste_delay: float = 0.05
    # After pasting, verify via UI Automation that the text landed; on a
    # confirmed failure keep it on the clipboard and notify.
    verify_paste: bool = True
    # Which chord to paste with in a detected terminal. Modern Windows 11
    # terminals (Windows Terminal, cmd, PowerShell) and TUI apps like Claude
    # Code accept Ctrl+V, so that's the default; set "shift_insert" for older
    # consoles that only take Shift+Insert.
    terminal_paste: str = "ctrl_v"  # ctrl_v | shift_insert


@dataclass
class HotkeyConfig:
    binding: str = "f9"


@dataclass
class OverlayConfig:
    enabled: bool = True


@dataclass
class TrainingConfig:
    enabled: bool = False
    # How many recent corrections get injected into the LLM prompt.
    max_examples: int = 5
    # After a thumbs-down correction, also replace the text already typed
    # into the focused app (guarded: skipped if you've edited it since).
    replace_on_correction: bool = True
    # When a correction includes what you actually said, also save that
    # dictation's audio (into training_audio/, gitignored) as a training pair
    # for a future speech-model fine-tune. Off = text-only training / privacy.
    save_correction_audio: bool = True


@dataclass
class ContextConfig:
    # Carry recent dictations into the next one as cleanup-LLM reference
    # (names/terms/casing consistency). Deliberately never fed to Whisper:
    # prompt text can leak into the transcript. In-memory only.
    enabled: bool = True
    # Context older than this is ignored (a new train of thought).
    expiry_seconds: float = 300.0
    # Cap on carried text; keeps Whisper's 224-token prompt budget safe.
    max_chars: int = 600
    max_utterances: int = 5
    # Also read the text around the caret in the focused app (UI Automation)
    # at hotkey press: it becomes Whisper/cleanup context and drives
    # continuation casing, spacing, and punctuation. Password fields are
    # never read; apps that expose no text are silently skipped.
    surrounding: bool = True
    surrounding_before_chars: int = 400
    surrounding_after_chars: int = 200


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    # Play a subtle low tone when recording starts (hotkey pressed).
    start_sound: bool = True
    start_sound_volume: float = 0.18  # 0.0-1.0, kept low on purpose


@dataclass
class Config:
    stt: SttConfig = field(default_factory=SttConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    injection: InjectionConfig = field(default_factory=InjectionConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)


def _section(data: dict, name: str, cls):
    """Build a dataclass section from a TOML table, ignoring unknown keys."""
    raw = data.get(name, {})
    known = {f for f in cls.__dataclass_fields__}
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return Config()
    # Tolerate a UTF-8 BOM (Notepad and PowerShell often add one).
    data = tomllib.loads(path.read_bytes().decode("utf-8-sig"))
    cfg = Config(
        stt=_section(data, "stt", SttConfig),
        cleanup=_section(data, "cleanup", CleanupConfig),
        injection=_section(data, "injection", InjectionConfig),
        hotkey=_section(data, "hotkey", HotkeyConfig),
        overlay=_section(data, "overlay", OverlayConfig),
        training=_section(data, "training", TrainingConfig),
        context=_section(data, "context", ContextConfig),
        audio=_section(data, "audio", AudioConfig),
    )
    if cfg.cleanup.punctuation_source not in ("model", "pauses"):
        log.warning(
            "Unrecognized cleanup.punctuation_source %r; 'model' behavior will apply",
            cfg.cleanup.punctuation_source,
        )
    return cfg


def _fmt_toml(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def save_config_updates(
    updates: dict[str, dict[str, object]], path: str | Path | None = None
) -> None:
    """Persist setting changes into config.toml, preserving comments/layout.

    `updates` maps section name -> {key: new_value}. Existing key lines are
    rewritten in place; missing keys are appended to their section; missing
    sections are appended to the file.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    lines = path.read_text(encoding="utf-8-sig").splitlines() if path.exists() else []
    pending = {s: dict(kv) for s, kv in updates.items()}

    out: list[str] = []
    section = None

    def flush_section(name: str | None) -> None:
        # Append keys that weren't found in the section we're leaving.
        if name and pending.get(name):
            for k, v in pending.pop(name).items():
                out.append(f"{k} = {_fmt_toml(v)}")

    import re as _re

    for line in lines:
        m = _re.match(r"^\s*\[(.+?)\]\s*$", line)
        if m:
            flush_section(section)
            section = m.group(1)
            out.append(line)
            continue
        km = _re.match(r"^(\s*)([A-Za-z0-9_]+)(\s*=\s*)", line)
        if km and section in pending and km.group(2) in pending[section]:
            value = pending[section].pop(km.group(2))
            out.append(f"{km.group(1)}{km.group(2)}{km.group(3)}{_fmt_toml(value)}")
            continue
        out.append(line)
    flush_section(section)
    for name, kv in pending.items():
        if not kv:
            continue
        out.append("")
        out.append(f"[{name}]")
        for k, v in kv.items():
            out.append(f"{k} = {_fmt_toml(v)}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
