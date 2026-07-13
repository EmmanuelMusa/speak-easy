"""Surrounding-text context: read what's around the caret in the focused app.

Uses UI Automation's TextPattern to grab a few hundred characters before and
after the caret at hotkey press. The before-caret text is the strongest
possible Whisper initial_prompt (it is literally the text the dictation
continues), tells the cleanup LLM whether to capitalize the first word, and
lets the injector decide on a leading space.

Best-effort by design: works in Word, Outlook, browsers, Notepad and most
native apps; returns None in terminals, canvas-rendered editors (Google
Docs), and anything not exposing TextPattern. Password fields are always
skipped. All reads happen off the hot path (a worker thread at key press),
so a slow or silent app costs nothing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

# UIA constants (UIAutomationClient.h)
_UIA_TextPatternId = 10014
_UIA_IsPasswordPropertyId = 30019
_RangeEndpoint_Start = 0
_RangeEndpoint_End = 1
_TextUnit_Character = 0

# Secret-shaped content is scrubbed before surrounding text enters the
# context pipeline: PEM blocks and long unbroken base64/hex runs (keys,
# tokens). Everything is local-only regardless, but secrets add nothing as
# dictation context and don't belong in prompts or logs.
_SECRET_RE = re.compile(
    r"-----BEGIN [A-Z ]+-----.*?(?:-----END [A-Z ]+-----|\Z)"
    r"|[A-Za-z0-9+/_\-]{40,}={0,2}",
    re.DOTALL,
)

# Chars that end a sentence; anything else before the caret means the
# dictation continues mid-sentence.
_SENTENCE_END = tuple(".!?\n")
# Openers after which no leading space is wanted.
_NO_SPACE_AFTER = tuple("([{‘“'\"\n\t —–-/")


@dataclass
class Surrounding:
    before: str = ""
    after: str = ""
    app: str = ""

    @property
    def mid_sentence(self) -> bool:
        """True when the caret sits mid-sentence (continue in lowercase)."""
        t = self.before.rstrip(" \t")
        return bool(t) and not t.endswith(_SENTENCE_END)

    @property
    def continues_after(self) -> bool:
        """True when text follows the caret on the same line."""
        return bool(self.after.split("\n", 1)[0].strip())


def needs_leading_space(before: str, text: str) -> bool:
    """Should a space be prepended so `text` doesn't glue to `before`?"""
    if not before or not text:
        return False
    last = before[-1]
    return last not in _NO_SPACE_AFTER and not last.isspace() \
        and bool(re.match(r"[\w\"'(‘“]", text))


def warmup() -> None:
    """Generate/load the comtypes UIA wrapper off the startup path.

    The first GetModule call code-generates a Python wrapper for
    UIAutomationCore.dll (a couple of seconds); doing it at app start means
    the first hotkey press doesn't pay it.
    """
    try:
        _uia_module()
    except Exception as exc:
        log.warning("UIA warmup failed (%s); surrounding text disabled", exc)


def _uia_module():
    import comtypes
    import comtypes.client

    try:
        comtypes.CoInitialize()  # per-thread; harmless if already done
    except OSError:
        pass
    comtypes.client.GetModule("UIAutomationCore.dll")
    from comtypes.gen import UIAutomationClient

    return UIAutomationClient


def read_surrounding(before_chars: int = 400, after_chars: int = 200) -> Surrounding | None:
    """Read text around the caret in the focused control, or None.

    Call from a worker thread — the UIA round trip can take tens of
    milliseconds and some apps answer slowly.
    """
    try:
        return _read(before_chars, after_chars)
    except Exception as exc:
        log.debug("Surrounding-text read failed: %s", exc)
        return None


def _read(before_chars: int, after_chars: int) -> Surrounding | None:
    import comtypes.client

    UIAC = _uia_module()
    uia = comtypes.client.CreateObject(
        UIAC.CUIAutomation, interface=UIAC.IUIAutomation
    )
    el = uia.GetFocusedElement()
    if el is None:
        return None
    if el.GetCurrentPropertyValue(_UIA_IsPasswordPropertyId):
        return None  # never read password fields
    pat = el.GetCurrentPattern(_UIA_TextPatternId)
    if pat is None:
        return None
    tp = pat.QueryInterface(UIAC.IUIAutomationTextPattern)
    sel = tp.GetSelection()
    if sel is None or sel.Length == 0:
        return None  # no caret exposed
    caret = sel.GetElement(0)

    before_rng = caret.Clone()
    before_rng.MoveEndpointByRange(_RangeEndpoint_End, caret, _RangeEndpoint_Start)
    before_rng.MoveEndpointByUnit(
        _RangeEndpoint_Start, _TextUnit_Character, -before_chars
    )
    after_rng = caret.Clone()
    after_rng.MoveEndpointByRange(_RangeEndpoint_Start, caret, _RangeEndpoint_End)
    after_rng.MoveEndpointByUnit(_RangeEndpoint_End, _TextUnit_Character, after_chars)

    return Surrounding(
        before=_SECRET_RE.sub(" ", before_rng.GetText(-1) or ""),
        after=_SECRET_RE.sub(" ", after_rng.GetText(-1) or ""),
        app=_foreground_app(),
    )


def _foreground_app() -> str:
    import ctypes

    hwnd = ctypes.windll.user32.GetForegroundWindow()
    buf = ctypes.create_unicode_buffer(256)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, 256)
    return buf.value
