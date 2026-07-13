"""Power handling: throttling opt-out, battery detection, battery-scaled
Ollama timeout, and the settings-dialog quit path."""

import sys
from unittest.mock import MagicMock, patch

from app import power
from app.cleanup import Cleaner
from app.config import CleanupConfig
from app.overlay import Overlay


def test_opt_out_of_power_throttling_succeeds_on_windows():
    if sys.platform != "win32":
        assert power.opt_out_of_power_throttling() is False
        return
    assert power.opt_out_of_power_throttling() is True


def test_on_battery_returns_bool():
    assert power.on_battery() in (True, False)


def test_cleanup_timeout_scales_on_battery():
    cfg = CleanupConfig(enabled=True, timeout_seconds=15.0,
                        battery_timeout_multiplier=4.0)
    fake = MagicMock()
    fake.json.return_value = {"response": "Is it ready?"}
    fake.raise_for_status.return_value = None
    with patch("app.cleanup.power.on_battery", return_value=True), \
         patch("app.cleanup.requests.post", return_value=fake) as mock_post:
        Cleaner(cfg).clean("is it ready")
    assert mock_post.call_args.kwargs["timeout"] == 60.0

    with patch("app.cleanup.power.on_battery", return_value=False), \
         patch("app.cleanup.requests.post", return_value=fake) as mock_post:
        Cleaner(cfg).clean("is it ready")
    assert mock_post.call_args.kwargs["timeout"] == 15.0


def test_overlay_quit_event_fires_callback():
    ov = Overlay(enabled=False)
    fired = []
    ov.on_quit = lambda: fired.append(True)
    ov._dispatch({"type": "quit"})
    assert fired == [True]


def test_overlay_close_without_child_is_noop():
    ov = Overlay(enabled=True)
    ov.close()  # no child process was ever spawned
    assert ov.enabled is False
