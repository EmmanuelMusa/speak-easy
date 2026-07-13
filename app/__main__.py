"""Entry point: `python -m app` runs the push-to-talk loop.

`python -m app --dry-run path/to.wav` runs the offline pipeline proof:
WAV -> Silero VAD -> faster-whisper -> cleanup -> print (no injection).
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description=(
            "Speak Easy: fully local push-to-talk dictation "
            "(faster-whisper + Ollama cleanup + cursor injection)."
        ),
    )
    parser.add_argument(
        "--config", default=None, help="path to config.toml (default: repo root)"
    )
    parser.add_argument(
        "--dry-run",
        metavar="WAV",
        default=None,
        help="transcribe+clean a WAV file and print the result instead of "
        "running the hotkey loop",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="disable the Ollama LLM cleanup pass (local filler strip only)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from .config import load_config

    cfg = load_config(args.config)
    if args.no_cleanup:
        cfg.cleanup.enabled = False

    if args.dry_run:
        return _dry_run(cfg, args.dry_run)

    # Dry-run is exempt: it only prints, so it can run alongside the app.
    from . import single_instance

    if not single_instance.acquire():
        single_instance.notify_already_running()
        return 1

    from .hotkey import PushToTalkApp

    PushToTalkApp(cfg).run()
    return 0


def _dry_run(cfg, wav_path: str) -> int:
    from pathlib import Path

    from .cleanup import Cleaner
    from .stt import Transcriber

    wav = Path(wav_path)
    if not wav.exists():
        print(f"error: {wav} not found", file=sys.stderr)
        return 2

    print(f"[dry-run] transcribing {wav} (model={cfg.stt.model}, VAD on)...")
    raw = Transcriber(cfg.stt).transcribe(wav)
    print(f"[dry-run] raw transcript : {raw!r}")
    cleaned = Cleaner(cfg.cleanup).clean(raw)
    print(f"[dry-run] cleaned output : {cleaned!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
