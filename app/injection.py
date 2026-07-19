"""Inject text at the cursor of whichever window is focused.

Two delivery methods, chosen by config:
  - "clipboard":  set clipboard -> synthesize Ctrl+V -> restore clipboard.
    Fast and reliable for long text; handles all Unicode.
  - "sendinput":  ctypes SendInput with KEYEVENTF_UNICODE, typing the text
    character by character. No clipboard side effects.

Win32 imports are lazy so tests can import and mock this module anywhere.
"""

from __future__ import annotations

import ctypes
import logging
import re
import threading
import time

from .config import InjectionConfig

log = logging.getLogger(__name__)

# --- SendInput structures (KEYEVENTF_UNICODE) -------------------------------

KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002
INPUT_KEYBOARD = 1

_ULONG_PTR = ctypes.POINTER(ctypes.c_ulong)


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _MOUSEINPUT(ctypes.Structure):
    # Never sent, but it must be in the union: Win32's INPUT union is sized
    # by its largest member (MOUSEINPUT, 32 bytes on x64). Without it,
    # sizeof(INPUT) is 32 instead of 40 and SendInput rejects EVERY call
    # with ERROR_INVALID_PARAMETER — silently typing nothing.
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("union", _INPUTUNION)]


def inject_sendinput(text: str) -> None:
    """Type `text` into the focused window via one batched SendInput call.

    One call for the whole text: the OS queues every key event atomically,
    so real keyboard/mouse input can't interleave mid-dictation, and it is
    much faster than per-character calls. Non-BMP characters (emoji) are
    sent as UTF-16 surrogate pairs, which KEYEVENTF_UNICODE expects.
    """
    if not text:
        return
    units = text.encode("utf-16-le")
    n = len(units)  # bytes / 2 = code units; 2 events per unit
    events = (_INPUT * n)()
    i = 0
    for off in range(0, len(units), 2):
        code = units[off] | (units[off + 1] << 8)
        for flags in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP):
            events[i] = _INPUT(
                type=INPUT_KEYBOARD,
                union=_INPUTUNION(
                    ki=_KEYBDINPUT(wVk=0, wScan=code, dwFlags=flags, time=0,
                                   dwExtraInfo=None)
                ),
            )
            i += 1
    sent = ctypes.windll.user32.SendInput(n, events, ctypes.sizeof(_INPUT))
    if sent != n:
        err = ctypes.windll.kernel32.GetLastError()
        log.warning("SendInput delivered %d/%d events (error %d)", sent, n, err)


# --- Clipboard + Ctrl+V ------------------------------------------------------

def _open_clipboard(retries: int = 12, delay: float = 0.02) -> None:
    """OpenClipboard with retry. Only one process may hold the clipboard at a
    time, so OpenClipboard fails transiently whenever a clipboard-history
    tool, the pasting app, or our own deferred restore timer has it open. A
    single attempt was the difference between the dictated text landing and
    silently vanishing — retry for ~250ms before giving up."""
    import win32clipboard  # lazy

    last: Exception | None = None
    for _ in range(retries):
        try:
            win32clipboard.OpenClipboard()
            return
        except Exception as exc:  # pywintypes.error: access denied (busy)
            last = exc
            time.sleep(delay)
    raise last if last else RuntimeError("OpenClipboard failed")


def _get_clipboard_text() -> str | None:
    import win32clipboard  # lazy

    _open_clipboard()
    try:
        if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
            return win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
        return None
    finally:
        win32clipboard.CloseClipboard()


_NUM_LINE = re.compile(r"^[ \t]*\d+[.)]\s+(.*\S)\s*$")
_BUL_LINE = re.compile(r"^[ \t]*[-*•]\s+(.*\S)\s*$")


def _looks_listy(text: str) -> bool:
    return any(_NUM_LINE.match(ln) or _BUL_LINE.match(ln)
               for ln in text.split("\n"))


def _text_to_html(text: str) -> str:
    """Render `text` as an HTML fragment structured like the rich-list HTML that
    editors treat as a NATIVE, auto-exiting list: runs of numbered/bulleted
    lines become an <ol>/<ul> of bare <li> items, and lead-in / trailing lines
    are bare text (NOT <p>-wrapped) — all inside a font-inheriting <div>. The
    <p> wrapping we used before made editors treat the list as a separate block;
    this shape (matched to a reference dictation app) pastes as one native flow."""
    import html as _html
    lines = text.split("\n")
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        num, bul = _NUM_LINE.match(lines[i]), _BUL_LINE.match(lines[i])
        if num or bul:
            pat, tag = (_NUM_LINE, "ol") if num else (_BUL_LINE, "ul")
            items = []
            while i < len(lines) and pat.match(lines[i]):
                items.append("<li>" + _html.escape(pat.match(lines[i]).group(1))
                             + "</li>")
                i += 1
            blocks.append(f"<{tag}>\r\n\r\n" + "\r\n\r\n".join(items)
                          + f"\r\n\r\n</{tag}>")
        else:
            if lines[i].strip():
                blocks.append(_html.escape(lines[i].strip()))
            i += 1
    return '<div style="font: inherit;">' + "\r\n\r\n".join(blocks) + "</div>"


def _cf_html(fragment: str) -> bytes:
    """Wrap an HTML fragment in the CF_HTML clipboard format (header of byte
    offsets, then the HTML). Offsets are computed on the UTF-8 encoding."""
    prefix = "<html><body><!--StartFragment-->"
    suffix = "<!--EndFragment--></body></html>"
    tmpl = ("Version:0.9\r\nStartHTML:{0:08d}\r\nEndHTML:{1:08d}\r\n"
            "StartFragment:{2:08d}\r\nEndFragment:{3:08d}\r\n")
    header_len = len(tmpl.format(0, 0, 0, 0).encode("utf-8"))
    frag_b = fragment.encode("utf-8")
    start_frag = header_len + len(prefix.encode("utf-8"))
    end_frag = start_frag + len(frag_b)
    end_html = end_frag + len(suffix.encode("utf-8"))
    header = tmpl.format(header_len, end_html, start_frag, end_frag)
    return (header + prefix).encode("utf-8") + frag_b + suffix.encode("utf-8")


def _set_clipboard_rich(text: str, html_fragment: str) -> None:
    """Put BOTH plain text and an HTML version on the clipboard, so rich editors
    take the HTML (native list) and everything else takes the plain text."""
    import win32clipboard  # lazy
    cf_html = win32clipboard.RegisterClipboardFormat("HTML Format")
    data = _cf_html(html_fragment)
    _open_clipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, text)
        win32clipboard.SetClipboardData(cf_html, data)
    finally:
        win32clipboard.CloseClipboard()


def _set_clipboard_text(text: str) -> None:
    import win32clipboard  # lazy

    _open_clipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, text)
    finally:
        win32clipboard.CloseClipboard()


def _press_ctrl_key(vk: int) -> None:
    VK_CONTROL = 0x11
    user32 = ctypes.windll.user32
    user32.keybd_event(VK_CONTROL, 0, 0, 0)
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
    user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)


def _press_ctrl_v() -> None:
    _press_ctrl_key(0x56)  # V


def _press_ctrl_c() -> None:
    _press_ctrl_key(0x43)  # C


def _press_shift_insert() -> None:
    VK_SHIFT_, VK_INSERT = 0x10, 0x2D
    user32 = ctypes.windll.user32
    user32.keybd_event(VK_SHIFT_, 0, 0, 0)
    user32.keybd_event(VK_INSERT, 0, 0, 0)
    user32.keybd_event(VK_INSERT, 0, KEYEVENTF_KEYUP, 0)
    user32.keybd_event(VK_SHIFT_, 0, KEYEVENTF_KEYUP, 0)


# --- terminal detection (terminals want Shift+Insert, not Ctrl+V) ------------

_TERMINAL_CLASSES = {
    "ConsoleWindowClass",              # classic conhost (cmd, PowerShell 5)
    "CASCADIA_HOSTING_WINDOW_CLASS",   # Windows Terminal
}
_TERMINAL_PROCS = {
    "windowsterminal.exe", "wt.exe", "cmd.exe", "powershell.exe", "pwsh.exe",
    "conhost.exe", "openconsole.exe", "alacritty.exe", "wezterm-gui.exe",
    "mintty.exe", "hyper.exe", "conemu64.exe", "conemu.exe", "tabby.exe",
}


def _window_class(hwnd) -> str:
    buf = ctypes.create_unicode_buffer(256)
    ctypes.windll.user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def _window_process(hwnd) -> str:
    pid = ctypes.c_ulong()
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = ctypes.windll.kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value
    )
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(512)
        size = ctypes.c_ulong(512)
        if ctypes.windll.kernel32.QueryFullProcessImageNameW(
            h, 0, buf, ctypes.byref(size)
        ):
            return buf.value.rsplit("\\", 1)[-1].lower()
        return ""
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def is_terminal_window(hwnd) -> bool:
    """True when the focused window is a terminal, where Ctrl+V is
    unreliable (shells and TUI apps often bind it) but Shift+Insert pastes."""
    if not hwnd:
        return False
    try:
        return (_window_class(hwnd) in _TERMINAL_CLASSES
                or _window_process(hwnd) in _TERMINAL_PROCS)
    except Exception:
        return False


# --- Selection + focus (for in-place correction) -----------------------------

VK_SHIFT, VK_LEFT, VK_RIGHT = 0x10, 0x25, 0x27


def _tap(vk: int) -> None:
    user32 = ctypes.windll.user32
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)


def _select_back(n: int) -> None:
    """Select the previous `n` characters (Shift+Left x n)."""
    user32 = ctypes.windll.user32
    user32.keybd_event(VK_SHIFT, 0, 0, 0)
    for _ in range(n):
        _tap(VK_LEFT)
    user32.keybd_event(VK_SHIFT, 0, KEYEVENTF_KEYUP, 0)


def _get_foreground_window():
    try:
        import win32gui
        return win32gui.GetForegroundWindow()
    except Exception:
        return None


def _set_foreground_window(hwnd) -> None:
    if not hwnd:
        return
    try:
        import win32gui
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass  # foreground lock may refuse; the guard below still protects us


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\r\n", "\n")).strip()


def selection_matches(expected: str, selected: str) -> bool:
    """True if the currently-selected text is (still) what we injected —
    the guard that prevents clobbering text the user has since edited."""
    return bool(expected) and _norm(expected) == _norm(selected)


class _ClipboardRestorer:
    """Deferred, verified clipboard restore.

    Apps process a synthesized Ctrl+V asynchronously: browsers/Electron can
    read the clipboard hundreds of ms after the keystroke, especially while
    the GPU/CPU are busy transcribing. Restoring the old clipboard on a
    fixed short delay therefore made slow apps paste the OLD content —
    previous dictations leaked into the current one.

    Instead the restore runs on a background timer after `delay` seconds
    and only fires if the clipboard still holds our text (a copy made by
    the user in the meantime always wins). Back-to-back dictations cancel
    the pending timer but keep the ORIGINAL user clipboard, so it is what
    gets restored at the end of a burst — never an older dictation.
    """

    def __init__(self, delay: float = 2.0):
        self._delay = delay
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._user_clip: str | None = None
        self._our_text: str | None = None

    def schedule(self, our_text: str, previous: str | None) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()  # burst: keep the original user clip
            else:
                self._user_clip = previous
            self._our_text = our_text
            self._timer = threading.Timer(self._delay, self._restore)
            self._timer.daemon = True
            self._timer.start()

    def cancel(self) -> None:
        """Abandon the pending restore — the dictated text should STAY on
        the clipboard (paste failed; it's the user's only copy)."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._user_clip = None
            self._our_text = None

    def _restore(self) -> None:
        with self._lock:
            self._timer = None
            our, user = self._our_text, self._user_clip
            self._our_text = None
            self._user_clip = None
        if user is None:
            return
        try:
            if _get_clipboard_text() == our:
                _set_clipboard_text(user)
        except Exception:
            pass


_restorer = _ClipboardRestorer()

# Delivery verification poll: check up to _VERIFY_ATTEMPTS times, waiting
# _VERIFY_INTERVAL between checks (~2s total). Long enough that a slow app on
# battery still confirms before we declare failure; short enough not to nag.
_VERIFY_ATTEMPTS = 7
_VERIFY_INTERVAL = 0.3

_AUTO = object()


def inject_clipboard(
    text: str,
    paste_delay: float = 0.05,
    previous=_AUTO,
    shift_insert: bool = False,
    rich: bool = True,
) -> None:
    """Paste `text` at the cursor via clipboard; restore the clipboard later.

    The restore is deferred and verified (see _ClipboardRestorer) so a
    slow-pasting app still reads OUR text, never a stale clipboard.
    `shift_insert` pastes with Shift+Insert (terminals) instead of Ctrl+V.
    `rich` also puts an HTML version on the clipboard for lists, so rich editors
    paste a native list; plain text is always included as the fallback.
    """
    if previous is _AUTO:
        try:
            previous = _get_clipboard_text()
        except Exception:
            previous = None
    placed = False
    if rich and _looks_listy(text):
        try:
            _set_clipboard_rich(text, _text_to_html(text))
            placed = True
        except Exception as exc:  # HTML clipboard is best-effort; fall back
            log.debug("rich clipboard failed (%s); using plain text", exc)
    if not placed:
        _set_clipboard_text(text)
    time.sleep(paste_delay)
    if shift_insert:
        _press_shift_insert()
    else:
        _press_ctrl_v()
    _restorer.schedule(text, previous)


# --- Dispatcher ---------------------------------------------------------------

class Injector:
    def __init__(self, cfg: InjectionConfig):
        self.cfg = cfg
        #: what we last typed, and the window we typed it into — used by
        #: replace_last() for in-place correction.
        self.last_text: str = ""
        self.last_hwnd = None

    def _deliver(self, text: str) -> None:
        if self.cfg.delivery_method == "sendinput":
            inject_sendinput(text)
            return
        # Terminals only need Shift+Insert on older consoles; modern ones
        # (and TUI apps like Claude Code) take Ctrl+V, which is the default.
        shift_insert = (
            self.cfg.terminal_paste == "shift_insert"
            and is_terminal_window(self.last_hwnd)
        )
        try:  # "clipboard" (default)
            inject_clipboard(
                text,
                paste_delay=self.cfg.paste_delay,
                shift_insert=shift_insert,
                rich=self.cfg.rich_paste,
            )
        except Exception as exc:
            # The clipboard was unreachable even after retries. Rather than
            # drop the dictation silently, type it directly so it still lands.
            log.warning("Clipboard delivery failed (%s); typing instead", exc)
            inject_sendinput(text)

    def inject(self, text: str) -> None:
        if not text:
            return
        self.last_hwnd = _get_foreground_window()  # the target app, pre-paste
        self.last_text = text
        pre = None
        if self.cfg.verify_paste:
            from . import focus  # lazy: keeps this module import-light
            pre = focus.read_surrounding(300, 0)
        self._deliver(text)
        log.info("Injected %d chars via %s", len(text), self.cfg.delivery_method)
        if self.cfg.verify_paste and pre is not None:
            threading.Thread(
                target=self._verify_delivery, args=(text, pre), daemon=True
            ).start()

    def _verify_delivery(self, text: str, pre) -> None:
        """Confirm the text landed; if it demonstrably didn't, keep it on
        the clipboard and tell the user (Wispr-style fallback).

        Polls for up to ~2s rather than checking once: a slow app (especially
        on battery, where the GPU/CPU are busy and the paste is processed
        late) used to fail a single 0.6s check and pop a false "paste failed"
        dialog even though the text arrived a moment later. Only a SUSTAINED,
        confirmed failure acts — the field stayed readable and byte-identical
        the whole time — and the check aborts the instant focus leaves the
        target window (we can no longer judge that field).
        """
        from . import focus

        want = _norm(text)
        tail = want[-min(len(want), 120):]
        pre_norm = _norm(pre.before)
        for _ in range(_VERIFY_ATTEMPTS):  # ~2s total at 0.3s per attempt
            time.sleep(_VERIFY_INTERVAL)  # let the target app process the paste
            if _get_foreground_window() != self.last_hwnd:
                return  # focus moved: can't judge the original field anymore
            post = focus.read_surrounding(300, 0)
            if post is None:
                return  # can't observe: assume delivered
            got = _norm(post.before)
            if want and got.endswith(tail):
                return  # delivered
            if pre_norm != got:
                return  # field changed some other way: not a confirmed failure
        log.warning("Paste did not land after 2s; leaving text on the clipboard")
        _restorer.cancel()
        try:
            _set_clipboard_text(text)
        except Exception:
            return
        msg = (
            "The app didn't accept the dictated text. It's on your "
            "clipboard — click where you want it and press Ctrl+V."
        )
        ctypes.windll.user32.MessageBoxW(0, msg, "Speak Easy", 0x40)

    def replace_last(self, new_text: str) -> bool:
        """Replace the text we last injected with `new_text`, but only if it
        is still unchanged at the cursor. Returns True if replaced, False if
        the guard aborted (user edited/moved) or there was nothing to replace.
        """
        old = self.last_text
        if not old or not new_text:
            return False
        _set_foreground_window(self.last_hwnd)
        time.sleep(0.2)

        previous_clip = None
        try:
            previous_clip = _get_clipboard_text()
        except Exception:
            pass
        try:
            _select_back(len(old))
            time.sleep(0.05)
            _press_ctrl_c()
            time.sleep(0.1)
            selected = _get_clipboard_text() or ""
        except Exception as exc:
            log.warning("Replace: could not read selection (%s)", exc)
            return False

        if not selection_matches(old, selected):
            _tap(VK_RIGHT)  # deselect, leave the user's text untouched
            log.info("Replace aborted: text changed since injection "
                     "(expected %r, selection %r)", old, selected)
            # No paste was sent, so an immediate restore is race-free.
            if previous_clip is not None:
                try:
                    _set_clipboard_text(previous_clip)
                except Exception:
                    pass
            return False

        # The clipboard currently holds our Ctrl+C verification copy, not
        # user content — hand the restorer the clip captured before that.
        _restorer.schedule(selected, previous_clip)
        self._deliver(new_text)  # typing/paste replaces the selection
        self.last_text = new_text
        log.info("Replaced injected text in place (%d -> %d chars)",
                 len(old), len(new_text))
        return True
