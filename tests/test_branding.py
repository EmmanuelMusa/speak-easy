"""App identity: the icon asset, the Win32 branding calls, and tray Restart."""

from __future__ import annotations

import struct

from app import branding
from app.overlay import Overlay


def test_app_name_matches_the_logo():
    """The logo, the dictation output ('speak easy' -> 'SpeakEasy') and the
    windows all have to agree on one spelling."""
    assert branding.APP_NAME == "SpeakEasy"


def test_icon_is_a_multi_size_ico():
    """Windows picks a different bitmap for the tray (16px), the taskbar (32px)
    and Alt+Tab (256px). A single-size .ico gets scaled and looks soft."""
    assert branding.ICO_PATH.exists(), f"{branding.ICO_PATH} is missing"
    data = branding.ICO_PATH.read_bytes()
    reserved, kind, count = struct.unpack("<HHH", data[:6])
    assert (reserved, kind) == (0, 1), "not an ICO header"
    assert count >= 5, f"only {count} sizes in the icon"
    sizes = set()
    for i in range(count):
        entry = data[6 + 16 * i: 22 + 16 * i]
        w = entry[0] or 256  # 0 encodes 256 in the ICO format
        length, offset = struct.unpack("<II", entry[8:16])
        assert offset + length <= len(data), f"entry {i} runs past end of file"
        sizes.add(w)
    assert {16, 32, 256} <= sizes, f"missing a needed size, has {sorted(sizes)}"


def test_branding_calls_are_safe_without_a_console():
    """Both run in the overlay child, which is spawned with CREATE_NO_WINDOW.
    Branding must never be the reason the app fails to start."""
    branding.set_app_id()
    branding.brand_console()
    branding.apply(console=False)


def test_overlay_restart_event_fires_callback():
    ov = Overlay(enabled=False)
    fired = []
    ov.on_restart = lambda: fired.append(True)
    ov._dispatch({"type": "restart"})
    assert fired == [True]


def test_restart_request_stops_the_app_before_relaunching():
    """Restart must go through the normal shutdown: the replacement claims the
    hotkey, the overlay pipe and the single-instance mutex, so the old process
    has to let go of them first."""
    from unittest.mock import patch

    with patch("app.hotkey.PushToTalkApp.__init__", return_value=None):
        from app.hotkey import PushToTalkApp

        app = PushToTalkApp.__new__(PushToTalkApp)
    import threading

    app._quit = threading.Event()
    app._restart = False
    app._request_restart()
    assert app._restart is True
    assert app._quit.is_set(), "restart did not end the run loop"
