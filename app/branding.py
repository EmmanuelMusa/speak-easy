"""App identity: name, icon, and the Windows calls that make both stick.

Running as `python -m app` means every window the app owns is, as far as
Windows is concerned, a Python window: the console gets python.exe's icon, and
the taskbar groups our windows under Python because that is the executable.
Two calls fix it, and both have to happen in *every* process that shows UI
(the hotkey loop and the Qt overlay child are separate processes):

  set_app_id()   — an explicit AppUserModelID, which is what the taskbar
                   actually groups and labels by.
  brand_console() — the console window's own title and icon, which the
                   AppUserModelID does not touch.

Everything here is best-effort and silent on failure. Branding must never be
the reason the app won't start, and none of it exists off Windows.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

APP_NAME = "SpeakEasy"
#: Taskbar identity. Any pinned shortcut must carry the same string to group
#: with the running app, so treat this as a stable ID, not a display name.
APP_ID = "SpeakEasy.Dictation"

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
ICO_PATH = _ASSETS / "speakeasy.ico"     # multi-size, for Win32 + the tray
PNG_PATH = _ASSETS / "icon-256.png"      # single size, for Qt windows


def set_app_id(app_id: str = APP_ID) -> None:
    """Give this process its own taskbar identity instead of inheriting
    python.exe's. Must run before any window is created."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception as exc:
        log.debug("Could not set AppUserModelID (%s)", exc)


def brand_console(title: str = APP_NAME) -> None:
    """Retitle the console window and give it the app icon. No-op when there is
    no console (launched via pythonw, a shortcut, or an IDE)."""
    if sys.platform != "win32":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32
        kernel32.SetConsoleTitleW(title)
        kernel32.GetConsoleWindow.restype = ctypes.c_void_p
        hwnd = kernel32.GetConsoleWindow()
        if not hwnd or not ICO_PATH.exists():
            return
        IMAGE_ICON, LR_LOADFROMFILE, WM_SETICON = 1, 0x0010, 0x0080
        user32.LoadImageW.restype = ctypes.c_void_p
        user32.LoadImageW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p,
                                      ctypes.c_uint, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_uint]
        user32.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint,
                                        ctypes.c_void_p, ctypes.c_void_p]
        # ICON_SMALL is the title bar and taskbar; ICON_BIG is Alt+Tab.
        for wparam, px in ((0, 16), (1, 32)):
            hicon = user32.LoadImageW(None, str(ICO_PATH), IMAGE_ICON, px, px,
                                      LR_LOADFROMFILE)
            if hicon:
                user32.SendMessageW(ctypes.c_void_p(hwnd), WM_SETICON,
                                    ctypes.c_void_p(wparam),
                                    ctypes.c_void_p(hicon))
    except Exception as exc:
        log.debug("Could not brand the console window (%s)", exc)


def apply(console: bool = True) -> None:
    """Everything at once, for a process that is starting up."""
    set_app_id()
    if console:
        brand_console()
