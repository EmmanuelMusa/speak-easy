"""Tests for the in-place correction guard, replace_last dispatch, and the
deferred clipboard restore (regression: slow-pasting apps must read the
current dictation, never a stale clipboard)."""

import time
from unittest.mock import patch

from app import injection
from app.config import InjectionConfig
from app.injection import Injector, _ClipboardRestorer, inject_clipboard, selection_matches


def test_selection_matches_normalizes_whitespace_not_case():
    assert selection_matches("Hello   world.", "Hello world.")  # collapsed spaces
    assert selection_matches("a\nb", "a\r\nb")          # newline normalization
    assert not selection_matches("Go by 2pm.", "Go by 3pm.")  # user edited
    assert not selection_matches("hello", "Hello")       # case-sensitive (safe)
    assert not selection_matches("", "anything")         # nothing injected


def test_replace_last_applies_when_selection_unchanged():
    inj = Injector(InjectionConfig(delivery_method="clipboard"))
    inj.last_text = "Go by 2pm."
    with patch("app.injection._set_foreground_window"), \
         patch("app.injection._select_back"), \
         patch("app.injection._press_ctrl_c"), \
         patch("app.injection._get_clipboard_text", return_value="Go by 2pm."), \
         patch("app.injection._set_clipboard_text"), \
         patch.object(injection, "_restorer") as restorer, \
         patch.object(inj, "_deliver") as deliver, \
         patch("app.injection._tap") as tap:
        result = inj.replace_last("Go by 3pm.")
    assert result is True
    deliver.assert_called_once_with("Go by 3pm.")
    restorer.schedule.assert_called_once()  # restore deferred, not inline
    tap.assert_not_called()          # no deselect — we replaced
    assert inj.last_text == "Go by 3pm."


def test_replace_last_aborts_when_user_edited():
    inj = Injector(InjectionConfig(delivery_method="clipboard"))
    inj.last_text = "Go by 2pm."
    with patch("app.injection._set_foreground_window"), \
         patch("app.injection._select_back"), \
         patch("app.injection._press_ctrl_c"), \
         patch("app.injection._get_clipboard_text",
               return_value="Go by 2pm. and more typing"), \
         patch("app.injection._set_clipboard_text"), \
         patch.object(inj, "_deliver") as deliver, \
         patch("app.injection._tap") as tap:
        result = inj.replace_last("Go by 3pm.")
    assert result is False
    deliver.assert_not_called()      # user's text left untouched
    tap.assert_called_once()         # deselected instead
    assert inj.last_text == "Go by 2pm."


def test_replace_last_noop_without_prior_injection():
    inj = Injector(InjectionConfig())
    assert inj.replace_last("anything") is False


def test_replace_last_logs_expected_and_got_on_mismatch(caplog):
    import logging

    inj = Injector(InjectionConfig(delivery_method="clipboard"))
    inj.last_text = "Go by 2pm."
    with patch("app.injection._set_foreground_window"), \
         patch("app.injection._select_back"), \
         patch("app.injection._press_ctrl_c"), \
         patch("app.injection._get_clipboard_text",
               return_value="Go by 2pm. and more typing"), \
         patch("app.injection._set_clipboard_text"), \
         patch.object(inj, "_deliver") as deliver, \
         patch("app.injection._tap") as tap:
        with caplog.at_level(logging.INFO, logger="app.injection"):
            result = inj.replace_last("Go by 3pm.")
    assert result is False
    deliver.assert_not_called()
    tap.assert_called_once()
    messages = [r.message for r in caplog.records]
    assert any("Go by 2pm." in m and "Go by 2pm. and more typing" in m
               for m in messages)


# --- deferred clipboard restore (the stale-paste race) ------------------------

def _fake_clipboard(initial: str):
    clip = {"v": initial}
    return (
        clip,
        patch("app.injection._get_clipboard_text", side_effect=lambda: clip["v"]),
        patch("app.injection._set_clipboard_text",
              side_effect=lambda t: clip.__setitem__("v", t)),
        patch("app.injection._press_ctrl_v"),
    )


def test_slow_pasting_app_reads_current_dictation():
    """Regression: an app that processes Ctrl+V after the old inline 150ms
    restore window used to paste the stale clipboard (previous dictation)."""
    clip, get_p, set_p, v_p = _fake_clipboard("previous dictation text")
    with get_p, set_p, v_p, \
         patch.object(injection, "_restorer", _ClipboardRestorer(delay=0.25)):
        inject_clipboard("current words only", paste_delay=0)
        time.sleep(0.18)  # slow app finally reads the clipboard to paste
        assert clip["v"] == "current words only"
        time.sleep(0.25)  # after the window, the user's clipboard comes back
        assert clip["v"] == "previous dictation text"


def test_burst_dictations_restore_original_user_clipboard():
    clip, get_p, set_p, v_p = _fake_clipboard("user's own copied text")
    with get_p, set_p, v_p, \
         patch.object(injection, "_restorer", _ClipboardRestorer(delay=0.2)):
        inject_clipboard("dictation one", paste_delay=0)
        inject_clipboard("dictation two", paste_delay=0)  # within the window
        time.sleep(0.35)
        # Never restores dictation one — the original user clip survives.
        assert clip["v"] == "user's own copied text"


def test_cancel_keeps_dictated_text_on_clipboard():
    clip, get_p, set_p, v_p = _fake_clipboard("old clipboard")
    with get_p, set_p, v_p:
        restorer = _ClipboardRestorer(delay=0.15)
        with patch.object(injection, "_restorer", restorer):
            inject_clipboard("dictated text", paste_delay=0)
        restorer.cancel()
        time.sleep(0.3)
        assert clip["v"] == "dictated text"  # restore abandoned


# --- terminal-aware paste and delivery verification ---------------------------

def test_terminal_paste_defaults_to_ctrl_v():
    # Default: terminals get Ctrl+V (shift_insert=False), matching modern
    # Windows Terminal / cmd / TUI apps like Claude Code.
    inj = Injector(InjectionConfig(delivery_method="clipboard", verify_paste=False))
    inj.last_hwnd = 1234
    with patch("app.injection.is_terminal_window", return_value=True), \
         patch("app.injection.inject_clipboard") as clip:
        inj._deliver("ls -la")
    assert clip.call_args.kwargs["shift_insert"] is False


def test_terminal_paste_shift_insert_opt_in():
    # Older consoles: opt into Shift+Insert, and only in terminals.
    inj = Injector(InjectionConfig(
        delivery_method="clipboard", verify_paste=False, terminal_paste="shift_insert"))
    inj.last_hwnd = 1234
    with patch("app.injection.is_terminal_window", return_value=True), \
         patch("app.injection.inject_clipboard") as clip:
        inj._deliver("ls -la")
    assert clip.call_args.kwargs["shift_insert"] is True
    with patch("app.injection.is_terminal_window", return_value=False), \
         patch("app.injection.inject_clipboard") as clip:
        inj._deliver("hello")
    assert clip.call_args.kwargs["shift_insert"] is False


def test_is_terminal_window_matches_class_and_process():
    with patch("app.injection._window_class", return_value="CASCADIA_HOSTING_WINDOW_CLASS"), \
         patch("app.injection._window_process", return_value="whatever.exe"):
        assert injection.is_terminal_window(1)
    with patch("app.injection._window_class", return_value="Chrome_WidgetWin_1"), \
         patch("app.injection._window_process", return_value="pwsh.exe"):
        assert injection.is_terminal_window(1)
    with patch("app.injection._window_class", return_value="Chrome_WidgetWin_1"), \
         patch("app.injection._window_process", return_value="chrome.exe"):
        assert not injection.is_terminal_window(1)
    assert not injection.is_terminal_window(None)


def _surrounding(before):
    from app.focus import Surrounding
    return Surrounding(before=before)


def test_verify_confirmed_failure_keeps_clipboard_and_notifies():
    inj = Injector(InjectionConfig())
    inj.last_hwnd = 4321
    pre = _surrounding("The field says this")
    with patch("app.focus.read_surrounding",
               return_value=_surrounding("The field says this")), \
         patch("app.injection._get_foreground_window", return_value=4321), \
         patch("app.injection._set_clipboard_text") as set_clip, \
         patch.object(injection._restorer, "cancel") as cancel, \
         patch("app.injection.ctypes") as fake_ctypes, \
         patch("app.injection.time.sleep"):
        inj._verify_delivery("dictated words", pre)
    set_clip.assert_called_once_with("dictated words")
    cancel.assert_called_once()
    fake_ctypes.windll.user32.MessageBoxW.assert_called_once()


def test_verify_success_and_unknown_do_nothing():
    inj = Injector(InjectionConfig())
    inj.last_hwnd = 4321
    pre = _surrounding("The field says this")
    with patch("app.injection._get_foreground_window", return_value=4321), \
         patch("app.focus.read_surrounding",
               return_value=_surrounding("The field says this dictated words")), \
         patch("app.injection._set_clipboard_text") as set_clip, \
         patch("app.injection.time.sleep"):
        inj._verify_delivery("dictated words", pre)   # landed
    set_clip.assert_not_called()
    with patch("app.injection._get_foreground_window", return_value=4321), \
         patch("app.focus.read_surrounding", return_value=None), \
         patch("app.injection._set_clipboard_text") as set_clip, \
         patch("app.injection.time.sleep"):
        inj._verify_delivery("dictated words", pre)   # unobservable
    set_clip.assert_not_called()


def test_verify_aborts_when_focus_leaves_target_window():
    # A slow app that eventually accepts the paste, but the user has clicked
    # elsewhere by the time we check: we must NOT nag about a field we can no
    # longer see. Focus mismatch => bail, never notify.
    inj = Injector(InjectionConfig())
    inj.last_hwnd = 4321
    pre = _surrounding("The field says this")
    with patch("app.injection._get_foreground_window", return_value=9999), \
         patch("app.focus.read_surrounding",
               return_value=_surrounding("The field says this")), \
         patch("app.injection._set_clipboard_text") as set_clip, \
         patch("app.injection.ctypes") as fake_ctypes, \
         patch("app.injection.time.sleep"):
        inj._verify_delivery("dictated words", pre)
    set_clip.assert_not_called()
    fake_ctypes.windll.user32.MessageBoxW.assert_not_called()


def test_verify_confirms_on_a_later_poll():
    # First readback still shows the old field (app hasn't processed the paste
    # yet); a later one shows the text. Must confirm delivery, not nag.
    inj = Injector(InjectionConfig())
    inj.last_hwnd = 4321
    pre = _surrounding("The field says this")
    reads = [
        _surrounding("The field says this"),               # not yet
        _surrounding("The field says this dictated words"),  # landed
    ]
    with patch("app.injection._get_foreground_window", return_value=4321), \
         patch("app.focus.read_surrounding", side_effect=reads), \
         patch("app.injection._set_clipboard_text") as set_clip, \
         patch("app.injection.ctypes") as fake_ctypes, \
         patch("app.injection.time.sleep"):
        inj._verify_delivery("dictated words", pre)
    set_clip.assert_not_called()
    fake_ctypes.windll.user32.MessageBoxW.assert_not_called()


# --- clipboard robustness ----------------------------------------------------

def test_open_clipboard_retries_then_succeeds():
    import app.injection as inj_mod
    calls = {"n": 0}

    class FakeWin32:
        error = RuntimeError

        def OpenClipboard(self):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("clipboard busy")

    fake = FakeWin32()
    with patch.dict("sys.modules", {"win32clipboard": fake}), \
         patch("app.injection.time.sleep"):
        inj_mod._open_clipboard()
    assert calls["n"] == 3  # failed twice, succeeded on the third try


def test_clipboard_delivery_falls_back_to_typing():
    # If the clipboard is unreachable, the dictation must still land by
    # typing rather than vanish silently.
    inj = Injector(InjectionConfig(delivery_method="clipboard", verify_paste=False))
    inj.last_hwnd = 1234
    with patch("app.injection.is_terminal_window", return_value=False), \
         patch("app.injection.inject_clipboard",
               side_effect=RuntimeError("OpenClipboard failed")), \
         patch("app.injection.inject_sendinput") as typed:
        inj._deliver("hello world")
    typed.assert_called_once_with("hello world")


def test_user_copy_after_dictation_is_never_clobbered():
    clip, get_p, set_p, v_p = _fake_clipboard("old clipboard")
    with get_p, set_p, v_p, \
         patch.object(injection, "_restorer", _ClipboardRestorer(delay=0.2)):
        inject_clipboard("dictated text", paste_delay=0)
        clip["v"] = "user copied something new"  # user hits Ctrl+C meanwhile
        time.sleep(0.3)
        assert clip["v"] == "user copied something new"
