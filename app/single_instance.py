"""Single-instance guard via a named Win32 mutex.

Two Speak Easy instances would both react to the hotkey and double-type
every dictation. The first instance creates a named mutex that lives until
its process exits (Windows cleans it up even on a crash); later launches
see ERROR_ALREADY_EXISTS and bow out.
"""

from __future__ import annotations

import ctypes
import sys

_MUTEX_NAME = "Local\\SpeakEasy.SingleInstance"
_ERROR_ALREADY_EXISTS = 183

_handle: int | None = None  # keeps the mutex alive for the process lifetime


def acquire() -> bool:
    """True if this is the only running instance (mutex acquired)."""
    global _handle
    if _handle is not None:
        return True
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, False, _MUTEX_NAME)
    if not handle:
        return True  # can't create the mutex at all: don't block startup
    if kernel32.GetLastError() == _ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(handle)
        return False
    _handle = handle
    return True


def release() -> None:
    """Release the mutex (tests; the OS handles process exit itself)."""
    global _handle
    if _handle is not None:
        ctypes.windll.kernel32.CloseHandle(_handle)
        _handle = None


def notify_already_running() -> None:
    """Tell the user why this launch is exiting, console or not."""
    msg = (
        "SpeakEasy is already running — check the pill at the bottom "
        "of your screen. Only one instance can own the hotkey."
    )
    if sys.stderr is not None and sys.stderr.isatty():
        print(msg, file=sys.stderr)
    else:
        # pythonw launch: no console, so use a message box (0x40 = info icon).
        ctypes.windll.user32.MessageBoxW(0, msg, "SpeakEasy", 0x40)
