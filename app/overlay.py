"""Status bar controller: drives the Qt overlay subprocess (overlay_ui.py).

Streams commands to the child over stdin and receives events (settings saved,
feedback verdicts) as JSON lines on the child's stdout:

    parent -> child : state / level / settings / feedback   (see overlay_ui)
    child -> parent : {"type": "settings_saved", ...} / {"type": "feedback", ...}
                      / {"type": "quit"} / {"type": "restart"}

Assign `on_settings` and `on_feedback` callables to receive events. If
PySide6 isn't installed the overlay disables itself gracefully — dictation
works without it.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
import threading
import time

log = logging.getLogger(__name__)


class Overlay:
    def __init__(self, enabled: bool = True, level_source=None):
        self.enabled = enabled
        self._level = level_source or (lambda: 0.0)
        self._proc: subprocess.Popen | None = None
        self._state = "idle"
        self._lock = threading.Lock()
        self._settings: dict = {}
        self._feedback_id = 0
        self._pending: tuple[int, str, str] | None = None  # (id, raw, output)
        #: callbacks the app assigns
        self.on_settings = None   # fn(values: dict)
        self.on_feedback = None   # fn(raw, output, rating, transcript, ideal, tags)
        self.on_quit = None       # fn() — user hit Quit in settings or the tray
        self.on_restart = None    # fn() — user hit Restart in the tray

    # -- public API (thread-safe) -----------------------------------------

    def show_idle(self) -> None:
        self._post("idle")

    def show_recording(self) -> None:
        self._post("recording")

    def show_processing(self) -> None:
        self._post("processing")

    def hide(self) -> None:
        self._post("hide")

    def close(self) -> None:
        """Shut the overlay child down (it exits on stdin EOF)."""
        self.enabled = False
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=2)
        except Exception:
            proc.terminate()

    def send_settings(self, values: dict) -> None:
        """Give the child the current settings (populates the dialog)."""
        self._settings = values
        if self.enabled and self._proc is not None:
            self._send("settings " + json.dumps(values))

    def request_feedback(self, raw: str, output: str) -> None:
        """Show the training-mode feedback panel for the latest dictation.
        Only one panel is live at a time, so a single pending slot suffices."""
        if not self.enabled:
            return
        self._ensure_proc()
        self._feedback_id += 1
        self._pending = (self._feedback_id, raw, output)
        self._send(
            "feedback "
            + json.dumps({"id": self._feedback_id, "raw": raw, "cleaned": output})
        )

    # -- internals ----------------------------------------------------------

    def _post(self, state: str) -> None:
        if not self.enabled:
            return
        self._state = state
        self._ensure_proc()
        self._send(f"state {state}")

    def _ensure_proc(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            if importlib.util.find_spec("PySide6") is None:
                log.warning("PySide6 not installed; overlay disabled")
                self.enabled = False
                return
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "app.overlay_ui"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                creationflags=flags,
            )
            threading.Thread(target=self._pump_levels, daemon=True).start()
            threading.Thread(target=self._read_events, daemon=True).start()
        if self._settings:
            self._send("settings " + json.dumps(self._settings))

    def _send(self, line: str) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
        except OSError as exc:
            log.warning("Overlay pipe broke (%s); disabling", exc)
            self.enabled = False

    def _pump_levels(self) -> None:
        """Stream the mic level to the bar while recording (~30 Hz)."""
        while self.enabled and self._proc and self._proc.poll() is None:
            if self._state == "recording":
                self._send(f"level {self._level():.4f}")
            time.sleep(0.033)

    def _read_events(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                self._dispatch(event)
            except Exception:
                log.exception("Overlay event handler failed: %r", event)

    def _dispatch(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "quit" and self.on_quit:
            self.on_quit()
        elif kind == "restart" and self.on_restart:
            self.on_restart()
        elif kind == "settings_saved" and self.on_settings:
            self.on_settings(event.get("values", {}))
        elif kind == "feedback":
            pending = self._pending
            if pending is None or pending[0] != event.get("id"):
                return  # stale/superseded id — ignore
            self._pending = None
            # An answer carries a rating or an ideal correction; a dismiss
            # (Cancel / closed unanswered) carries neither and is not recorded.
            if (event.get("rating") is not None or event.get("ideal")
                    or event.get("transcript") or event.get("tags")) and self.on_feedback:
                _id, raw, output = pending
                self.on_feedback(
                    raw, output, event.get("rating"), event.get("transcript"),
                    event.get("ideal"), event.get("tags") or [],
                )
