"""Qt overlay subprocess: dark pill + waveform, hover settings, feedback UI.

Runs as its own process (`python -m app.overlay_ui`). Line protocol:

  parent -> child (stdin):
    state idle|recording|processing|hide
    level 0.42
    settings {"training_enabled": true, "hotkey": "f9", ...}
    feedback {"id": 3, "raw": "spoken text", "cleaned": "text that was typed"}

  child -> parent (stdout, JSON per line):
    {"type": "settings_saved", "values": {...}}
    {"type": "feedback", "id": 3, "rating": 1..5|null, "transcript": "..."|null,
     "ideal": "..."|null, "tags": ["misheard word", ...]}
    {"type": "quit"}   (user hit Quit in settings; parent exits, closing stdin)

Exits when stdin closes. All UI runs on the Qt main thread; stdin is read on
a helper thread and drained via a queue inside the render timer.
"""

from __future__ import annotations

import json
import math
import pathlib
import queue
import sys
import threading
import time

PILL_RGBA = (20, 20, 23)          # dark charcoal body
BAR_RGBA = (237, 237, 240, 240)   # near-white waveform / icon / text
ACCENT = (110, 150, 255)          # soft blue accent (pill hover only)

# --- Brand ------------------------------------------------------------------
# The mark and its colour sweep live in app/logo.py, which tools/make_icon.py
# also draws from, so the tray icon and these windows can never drift apart.
# It is imported inside main() with the rest of the Qt-dependent code.
# The pill and its waveform keep their own neutral charcoal above: the
# always-on-screen element stays quiet.
IDLE_ALPHA = 120
ACTIVE_ALPHA = 216
CHIP_ALPHA = 238

# Pill footprint per state (width, height). Compact, tighter visualizer.
DIMS = {"idle": (44, 5), "recording": (94, 18), "processing": (94, 18),
        "hide": (44, 5)}
# The canvas leaves headroom ABOVE the pill so the shortcut hint can rise over
# it on mic-hover; the pill/chip itself sits near the canvas bottom.
CANVAS_W, CANVAS_H = 300, 104
PILL_FROM_BOTTOM = 30              # pill centre, measured up from canvas bottom
PILL_CY = CANVAS_H - PILL_FROM_BOTTOM
SCREEN_MARGIN = 12                # gap between the pill and the screen edge

# On hover the pill reveals two SEPARATE circular buttons — a settings gear and
# a mic — side by side. Each raises its own tag above it only while hovered.
CIRCLE_D = 34                      # diameter of each circular button
CIRCLE_GAP = 16                    # space between the two circles
GEAR_D = 16                        # settings-gear glyph inside its circle
MIC_D = 17                         # microphone glyph inside its circle
HOVER_GRACE_S = 0.5               # buttons linger briefly after the mouse leaves
BARS = 15                          # finer resolution than before (was 11)
BAR_W = 2.4                        # crisp pill-shaped bar width

# Per-icon hint tag (rises above a circle while it is hovered).
HINT_H = 30
CIRCLE_RGBA = (32, 32, 37)         # circular-button face
CIRCLE_EDGE = (70, 71, 80)         # circular-button border
KEYCAP_RGBA = (44, 45, 52)         # raised keycap face
KEYCAP_EDGE = (86, 88, 98)         # keycap top edge / border

STT_MODELS = ["tiny.en", "base", "small.en", "medium", "large-v3",
              "large-v3-turbo"]
STT_ENGINES = ["whisper", "parakeet"]
PARAKEET_MODELS = ["nemo-parakeet-tdt-0.6b-v2", "nemo-parakeet-tdt-0.6b-v3"]
OLLAMA_MODELS = ["qwen2.5:3b", "llama3.2:3b", "llama3.1:8b", "phi3:mini",
                 "mistral:7b"]
DELIVERY = ["clipboard", "sendinput"]

FEEDBACK_TAGS = ["misheard word", "wrong punctuation", "over-deleted",
                 "wrong casing", "bad list"]

FEEDBACK_QSS = """
QWidget#root { background:#12141c; border:1px solid #262b3b; border-radius:12px; }
QLabel { color:#e6e8ef; font-size:12px; }
QLabel#preview { color:#d3d7e2; font-size:12px; }
QLabel#title { color:#ffffff; font-size:13px; font-weight:700; }
QLabel[role="flabel"] { color:#7c85a0; font-size:10px; font-weight:700;
    letter-spacing:1.2px; }
QLabel#ro { background:#0d0f16; border:1px solid #232838; border-radius:8px;
    padding:7px 9px; color:#949cb0; }
QPlainTextEdit { background:#1b1f2d; color:#e6e8ef; border:1px solid #333a4e;
    border-radius:8px; padding:6px 8px; font-size:12px; }
QPlainTextEdit:focus { border:1px solid #5f4ff3; }
QPushButton#link { background:transparent; color:#8b93ff; border:none;
    font-size:12px; font-weight:600; }
QPushButton#tag { background:#1b1f2d; color:#b6bccd; border:1px solid #333a4e;
    border-radius:11px; padding:4px 10px; font-size:11px; }
QPushButton#tag:checked { background:#2b2a63; color:#d5d2ff;
    border:1px solid #5f4ff3; }
QPushButton#ghost { background:#242a3a; color:#e6e8ef; border:none;
    border-radius:8px; padding:7px 13px; font-size:12px; }
QPushButton#save { border:none; color:#ffffff; border-radius:8px;
    padding:7px 13px; font-size:12px; font-weight:700;
    background:qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2a6df4, stop:1 #8b35e8); }
QPushButton#save:hover { background:qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #3d7dff, stop:1 #a049f5); }
"""

# Shared theme for the settings/review windows, keyed to the logo: near-black
# navy surfaces, indigo as the single flat accent, and the cyan->pink sweep
# reserved for the brand mark, the progress ring and the primary button.
DIALOG_QSS = """
QWidget#root { background: #101219; }
QLabel { color: #e6e8ef; font-size: 12px; }
QLabel#title { color: #ffffff; font-size: 16px; font-weight: 700; }
QLabel#subtitle { color: #838ba2; font-size: 11px; }
QLabel[role="section"] {
    color: #7c85a0; font-size: 10px; font-weight: 700; letter-spacing: 1.4px;
    padding-top: 4px;
}
QLabel[role="field"] { color: #b6bccd; font-size: 12px; }
QCheckBox { color: #e6e8ef; font-size: 12px; spacing: 9px; }
QCheckBox::indicator {
    width: 17px; height: 17px; border-radius: 5px;
    border: 1px solid #333a4e; background: #1b1f2d;
}
QCheckBox::indicator:hover { border: 1px solid #5f4ff3; }
QCheckBox::indicator:checked {
    background: #5f4ff3; border: 1px solid #7a66ff;
}
QLineEdit, QComboBox {
    background: #1b1f2d; color: #e6e8ef; border: 1px solid #2b3145;
    border-radius: 8px; padding: 7px 10px; font-size: 12px; min-height: 16px;
}
QLineEdit:focus, QComboBox:focus, QComboBox:on { border: 1px solid #5f4ff3; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background: #1b1f2d; color: #e6e8ef; border: 1px solid #2b3145;
    border-radius: 8px; padding: 4px; outline: none;
    selection-background-color: #5f4ff3; selection-color: #ffffff;
}
QListWidget {
    background: #161926; color: #e6e8ef; border: 1px solid #262b3b;
    border-radius: 8px; padding: 2px; outline: none;
}
QPushButton {
    background: #242a3a; color: #e6e8ef; border: none; border-radius: 8px;
    padding: 8px 15px; font-size: 12px;
}
QPushButton:hover { background: #2e3549; }
QPushButton#save {
    color: #ffffff; font-weight: 700;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2a6df4, stop:1 #8b35e8);
}
QPushButton#save:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #3d7dff, stop:1 #a049f5);
}
QPushButton#quit { background: transparent; color: #e0687f; padding-left: 4px; }
QPushButton#quit:hover { background: #2a1c26; color: #f5809a; }
QPushButton#link {
    background: transparent; color: #8b93ff; text-align: left; padding: 4px 2px;
}
QPushButton#link:hover { color: #a9aeff; }
QFrame#sep { background: #222735; max-height: 1px; border: none; }
QFrame#card {
    background: #161926; border: 1px solid #252a3a; border-radius: 13px;
}
QLabel[role="card"] {
    color: #7c85a0; font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
}
QLabel#brand { color: #ffffff; font-size: 17px; font-weight: 700; }
QLabel#brandsub { color: #838ba2; font-size: 11px; }
QPushButton#change {
    background: #222839; color: #c3caff; border: 1px solid #333a4e;
    border-radius: 8px; padding: 6px 12px; font-size: 11px; font-weight: 600;
}
QPushButton#change:hover { background: #2b3247; border: 1px solid #5f4ff3; }
"""

# The tray's right-click menu, matched to the windows above so the app looks
# like one thing wherever you meet it.
TRAY_QSS = """
QMenu {
    background: #161926; color: #e6e8ef; border: 1px solid #262b3b;
    border-radius: 8px; padding: 6px; font-size: 12px;
}
QMenu::item { padding: 7px 28px 7px 14px; border-radius: 6px; }
QMenu::item:selected { background: #5f4ff3; color: #ffffff; }
QMenu::separator { height: 1px; background: #262b3b; margin: 5px 8px; }
"""


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    from PySide6 import QtCore, QtGui, QtWidgets

    # This is a separate process from the hotkey loop, so it needs the taskbar
    # identity set here too — without it Windows files our windows under Python.
    # No console of our own (we're spawned with CREATE_NO_WINDOW).
    from . import branding, logo

    # The mark and its sweep come from app/logo.py, which tools/make_icon.py
    # draws the tray icon from too, so the two can never drift apart.
    BRAND_STOPS = logo.BRAND_STOPS
    _brand_gradient = logo.gradient
    _draw_brand_mark = logo.draw_mark

    branding.set_app_id()
    app = QtWidgets.QApplication([])
    app.setApplicationName(branding.APP_NAME)
    # Prefer the .ico: it carries every size Windows asks for, so the tray and
    # Alt+Tab each get a purpose-made bitmap instead of one downscaled PNG.
    _icon_path = (branding.ICO_PATH if branding.ICO_PATH.exists()
                  else branding.PNG_PATH)
    APP_ICON = QtGui.QIcon(str(_icon_path)) if _icon_path.exists() else QtGui.QIcon()
    app.setWindowIcon(APP_ICON)
    # The tray icon is the app's only persistent handle once the console is
    # gone, so closing the last window must not end the process.
    app.setQuitOnLastWindowClosed(False)
    commands: "queue.Queue[tuple[str, str]]" = queue.Queue()

    def _gear_path(center, r_out: float, r_in: float, hole: float,
                   teeth: int = 8):
        """A crisp vector settings gear as a QPainterPath: a cog silhouette
        (alternating tip/valley radii) with a hollow centre. Scales cleanly
        at any size — no emoji, no bitmap."""
        path = QtGui.QPainterPath()
        path.setFillRule(QtCore.Qt.OddEvenFill)
        half = math.pi / teeth * 0.42  # angular half-width of a tooth tip
        for i in range(teeth):
            a = 2 * math.pi * i / teeth
            pts = (
                (a - math.pi / teeth + half, r_in),   # valley before the tooth
                (a - half, r_out),                     # tooth rises
                (a + half, r_out),                     # tooth top
                (a + math.pi / teeth - half, r_in),    # valley after the tooth
            )
            for j, (ang, rad) in enumerate(pts):
                x = center.x() + rad * math.cos(ang)
                y = center.y() + rad * math.sin(ang)
                if i == 0 and j == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
        path.closeSubpath()
        path.addEllipse(center, hole, hole)  # OddEven fill -> hollow hub
        return path

    def _star_path(cx, cy, r_out, r_in, points=5):
        """A crisp 5-point star as a QPainterPath (vector, no emoji)."""
        path = QtGui.QPainterPath()
        for i in range(points * 2):
            r = r_out if i % 2 == 0 else r_in
            ang = -math.pi / 2 + i * math.pi / points
            x = cx + r * math.cos(ang)
            y = cy + r * math.sin(ang)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        path.closeSubpath()
        return path

    def _draw_mic(p, cx, cy, d, color, pen_w=1.7):
        """Draw a crisp vector microphone centred on (cx, cy), `d` tall: a
        filled capsule head, a stroked cradle arc around its lower half, and a
        short stem to a base line. No emoji, scales cleanly."""
        head_w = d * 0.42
        head_h = d * 0.60
        head = QtCore.QRectF(cx - head_w / 2, cy - d / 2, head_w, head_h)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(color)
        p.drawRoundedRect(head, head_w / 2, head_w / 2)
        pen = QtGui.QPen(color)
        pen.setWidthF(pen_w)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.NoBrush)
        cradle_r = head_w * 0.92
        cradle = QtCore.QRectF(cx - cradle_r, cy - d / 2 + head_h - cradle_r,
                               cradle_r * 2, cradle_r * 2)
        p.drawArc(cradle, 180 * 16, 180 * 16)   # lower semicircle
        stem_top = cy - d / 2 + head_h + cradle_r * 0.15
        base_y = cy + d / 2
        p.drawLine(QtCore.QPointF(cx, stem_top), QtCore.QPointF(cx, base_y))
        p.drawLine(QtCore.QPointF(cx - d * 0.22, base_y),
                   QtCore.QPointF(cx + d * 0.22, base_y))

    def _shortcut_keys(binding: str):
        """Split a hotkey binding ("Control + Shift + Space") into display key
        names for keycaps (["Ctrl", "Shift", "Space"])."""
        disp = {"control": "Ctrl", "ctrl": "Ctrl", "shift": "Shift",
                "alt": "Alt", "super": "Win", "cmd": "Win", "win": "Win",
                "space": "Space", "escape": "Esc", "enter": "Enter"}
        keys = []
        for raw in binding.replace(",", "+").split("+"):
            k = raw.strip()
            if not k:
                continue
            keys.append(disp.get(k.lower(), k[:1].upper() + k[1:]))
        return keys or ["F9"]

    def _draw_keycap(p, x, cy, text, font):
        """Draw a small raised keycap containing `text`, its left edge at x and
        vertically centred on cy. Returns its right edge x (for laying a row)."""
        p.setFont(font)
        fm = QtGui.QFontMetrics(font)
        pad = 7
        w = fm.horizontalAdvance(text) + pad * 2
        h = 19
        rect = QtCore.QRectF(x, cy - h / 2, w, h)
        p.setPen(QtGui.QPen(QtGui.QColor(*KEYCAP_EDGE), 1.0))
        p.setBrush(QtGui.QColor(*KEYCAP_RGBA))
        p.drawRoundedRect(rect, 5, 5)
        p.setPen(QtGui.QColor(*BAR_RGBA[:3]))
        p.drawText(rect, QtCore.Qt.AlignCenter, text)
        return x + w

    # ------------------------------------------------------------------ pill

    class Bar(QtWidgets.QWidget):
        def __init__(self):
            super().__init__(
                None,
                QtCore.Qt.FramelessWindowHint
                | QtCore.Qt.WindowStaysOnTopHint
                | QtCore.Qt.Tool
                | QtCore.Qt.WindowDoesNotAcceptFocus,
            )
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
            self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
            self.setMouseTracking(True)
            self.setFixedSize(CANVAS_W, CANVAS_H)
            geo = app.primaryScreen().availableGeometry()
            # Canvas bottom sits SCREEN_MARGIN above the screen edge; PILL_CY
            # then puts the pill near the bottom with headroom above for the hint.
            self.move(geo.center().x() - CANVAS_W // 2,
                      geo.bottom() - SCREEN_MARGIN - CANVAS_H)

            self.state = "idle"
            self.level = 0.0
            self._floor = 0.0          # tracked quiet baseline (auto-gain)
            self._peak = 0.0           # tracked recent loud peak (auto-gain)
            self.w, self.h = float(DIMS["idle"][0]), float(DIMS["idle"][1])
            self.t = 0.0
            self.history = [0.0] * (BARS * 2)
            self.heights = [0.0] * BARS
            self.reveal = 0.0          # 0 = pill, 1 = the two circular buttons
            self.chip_hovered = False  # cursor over either button (pointer cursor)
            self.gear_hovered = False  # cursor over the settings circle
            self.mic_hovered = False   # cursor over the mic circle
            self.hint_gear = 0.0       # 0..1 reveal of the "Settings" tag
            self.hint_mic = 0.0        # 0..1 reveal of the shortcut tag
            self._hover_until = 0.0
            self._pin = None           # debug: force a reveal value for shots
            self._pin_hint = None      # debug: force a tag open ("gear"/"mic")
            self.settings: dict = {}
            self.settings_dialog = None
            self.review_dialog = None
            self.feedback_panel = None
            self.on_settings_changed = None  # set by the tray, to refresh its tip

            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._tick)
            self.timer.start(16)

        # -- geometry helpers ------------------------------------------

        def _pill_geo(self):
            """The slim pill rect (idle bar / live waveform), from self.w/h."""
            cx, cy = CANVAS_W / 2, PILL_CY
            rect = QtCore.QRectF(cx - self.w / 2, cy - self.h / 2,
                                 self.w, self.h)
            return rect, self.h / 2

        def _circles(self):
            """Return (gear_rect, mic_rect): the two circular buttons revealed on
            hover, side by side and centred on the pill. Full-size once revealed
            so hit-testing matches the painted glyphs."""
            cx, cy = CANVAS_W / 2, PILL_CY
            off = (CIRCLE_D + CIRCLE_GAP) / 2
            gear = QtCore.QRectF(cx - off - CIRCLE_D / 2, cy - CIRCLE_D / 2,
                                 CIRCLE_D, CIRCLE_D)
            mic = QtCore.QRectF(cx + off - CIRCLE_D / 2, cy - CIRCLE_D / 2,
                                CIRCLE_D, CIRCLE_D)
            return gear, mic

        def _chip_active(self) -> bool:
            return self.reveal > 0.5

        def _hover_pad(self):
            """Bounding region over both circles (plus slack) that keeps the
            buttons revealed while the cursor moves between/around them."""
            cx, cy = CANVAS_W / 2, PILL_CY
            w = 2 * (CIRCLE_D + CIRCLE_GAP) + CIRCLE_D
            h = CIRCLE_D + 16
            return QtCore.QRectF(cx - w / 2, cy - h / 2, w, h)

        # -- events ------------------------------------------------------

        def mousePressEvent(self, event):
            if not self._chip_active():
                return
            gear, mic = self._circles()
            # The mic opens Settings on the hotkey field, so the shortcut it just
            # previewed is right there to change.
            if gear.contains(event.position()):
                self._open_settings()
            elif mic.contains(event.position()):
                self._open_settings(focus_hotkey=True)

        def _open_settings(self, focus_hotkey: bool = False):
            if self.settings_dialog is None:
                self.settings_dialog = SettingsDialog(self.settings)
            self.settings_dialog.load(self.settings)
            self.settings_dialog.show()
            self.settings_dialog.raise_()
            self.settings_dialog.activateWindow()
            if focus_hotkey:
                self.settings_dialog.hotkey.setFocus()
                self.settings_dialog.hotkey.selectAll()

        def _open_review(self):
            """Review learnings without going through Settings first (the tray
            offers it directly)."""
            if self.review_dialog is None:
                self.review_dialog = ReviewDialog(
                    int(self.settings.get("target_pairs", 200)))
            self.review_dialog.target = max(
                1, int(self.settings.get("target_pairs", 200)))
            self.review_dialog.refresh()
            self.review_dialog.show()
            self.review_dialog.raise_()
            self.review_dialog.activateWindow()

        # -- protocol ------------------------------------------------------

        def handle(self, cmd: str, arg: str):
            if cmd == "__eof__":
                QtWidgets.QApplication.quit()
            elif cmd == "state":
                self.state = arg
            elif cmd == "level":
                try:
                    self.level = float(arg)
                except ValueError:
                    pass
            elif cmd == "settings":
                try:
                    self.settings = json.loads(arg)
                except json.JSONDecodeError:
                    pass
                else:
                    if self.on_settings_changed:
                        self.on_settings_changed(self.settings)
            elif cmd == "feedback":
                try:
                    req = json.loads(arg)
                except json.JSONDecodeError:
                    return
                if self.feedback_panel is not None:
                    self.feedback_panel.close()
                self.feedback_panel = FeedbackPanel(self, req)
                self.feedback_panel.show()
            elif cmd == "selftest":
                # Build every dialog AND drive the real interaction paths — key
                # capture, engine sync, the pill's circular buttons — so an
                # automated test exercises them, not just the render paths.
                try:
                    self._run_selftest()
                    emit({"type": "selftest_ok"})
                except Exception as exc:  # pragma: no cover - reported to parent
                    import traceback
                    emit({"type": "selftest_err",
                          "error": f"{exc!r}\n{traceback.format_exc()}"})
            elif cmd == "pin":  # debug/visual-QA: pin the chip reveal (0..1)
                self._pin = None if arg == "none" else float(arg)
            elif cmd == "pinhint":  # debug/visual-QA: force a tag ("gear"/"mic")
                self._pin_hint = None if arg in ("none", "0", "") else arg
            elif cmd == "shot":  # debug: render current frame over gray -> PNG
                pm = QtGui.QPixmap(self.size() * 3)  # 3x for a crisp look
                pm.setDevicePixelRatio(3)
                pm.fill(QtGui.QColor(74, 76, 82))
                self.render(pm)
                pm.save(arg)
                emit({"type": "shot_ok"})
            elif cmd == "shotdialog":  # debug: screenshot the settings popup
                d = SettingsDialog(self.settings)
                d.ensurePolished()
                d.resize(d.sizeHint())
                d.grab().save(arg)
                d.deleteLater()
                emit({"type": "shot_ok"})
            elif cmd == "shotfeedback":  # debug: screenshot the feedback popup
                d = FeedbackPanel(self, {
                    "id": 1, "raw": "so lets ship the emoji thing today",
                    "cleaned": "So let's ship the emoji thing today."})
                d.ensurePolished()
                d.resize(d.sizeHint())
                QtWidgets.QApplication.processEvents()
                d.grab().save(arg)
                d.deleteLater()
                emit({"type": "shot_ok"})
            elif cmd == "shotreview":  # debug: screenshot the review dashboard
                d = ReviewDialog(int(self.settings.get("target_pairs", 200)))
                d.refresh()
                d.ensurePolished()
                d.resize(d.sizeHint())
                QtWidgets.QApplication.processEvents()  # flush resize -> reflow
                d.grab().save(arg)
                d.deleteLater()
                emit({"type": "shot_ok"})

        def _run_selftest(self):
            """Drive the interactive UI paths a test can't click: shortcut
            capture, engine-field sync, and the pill's circular buttons. Raises
            AssertionError on the first failure (reported to the parent)."""
            def check(cond, msg):
                if not cond:
                    raise AssertionError(msg)

            def press(widget, key):
                widget.keyPressEvent(QtGui.QKeyEvent(
                    QtCore.QEvent.KeyPress, key, QtCore.Qt.NoModifier))

            class _Click:
                def __init__(self, pos):
                    self._pos = pos

                def position(self):
                    return self._pos

            # Render paths for every dialog + the feedback panel.
            dlg = SettingsDialog(self.settings)
            ReviewDialog().refresh()
            fp = FeedbackPanel(self, {"id": 0, "raw": "hello wrld",
                                      "cleaned": "Hello world."})
            fp._expand()
            fp.close()

            # Shortcut capture formats the binding; Esc cancels it.
            sf = dlg.hotkey
            sf.setText("f9")
            sf.start_capture()
            check(sf._capturing, "capture did not start")
            for k in (QtCore.Qt.Key_Control, QtCore.Qt.Key_Shift,
                      QtCore.Qt.Key_Space):
                press(sf, k)
            check(not sf._capturing, "capture did not end on the final key")
            check(sf.text() == "control + shift + space",
                  f"unexpected binding {sf.text()!r}")
            sf.start_capture()
            press(sf, QtCore.Qt.Key_Escape)
            check(not sf._capturing and sf.text() == "control + shift + space",
                  "Escape did not cancel the capture cleanly")

            # Engine sync shows exactly one model field.
            dlg.engine.setCurrentText("whisper")
            check(dlg._whisper_field.isVisibleTo(dlg)
                  and not dlg._parakeet_field.isVisibleTo(dlg),
                  "whisper engine did not show only the whisper model field")
            dlg.engine.setCurrentText("parakeet")
            check(dlg._parakeet_field.isVisibleTo(dlg)
                  and not dlg._whisper_field.isVisibleTo(dlg),
                  "parakeet engine did not show only the parakeet model field")
            dlg.close()

            # The pill's circular buttons open Settings (mic in rebind mode).
            self.settings_dialog = None
            self.reveal = 1.0
            gear, mic = self._circles()
            self.mousePressEvent(_Click(gear.center()))
            check(self.settings_dialog is not None,
                  "clicking the gear did not open Settings")
            self.settings_dialog.hide()
            self.mousePressEvent(_Click(mic.center()))
            check(self.settings_dialog.hotkey._capturing,
                  "clicking the mic did not open Settings in rebind mode")
            self.settings_dialog.hotkey._stop_capture()
            self.settings_dialog.hide()
            self.reveal = 0.0

            # Review opens straight from the tray, bypassing Settings.
            self._open_review()
            check(self.review_dialog is not None and self.review_dialog.isVisible(),
                  "the tray's Review entry did not open the dialog")
            self.review_dialog.hide()

            # The tray icon: present, carries a real icon, and offers the four
            # entries. Settings and Review are triggered for real; Quit and
            # Restart only checked for presence, since firing them would tear
            # the app down mid-test.
            qapp = QtWidgets.QApplication.instance()
            tray = getattr(qapp, "_tray", None)
            if tray is None:
                check(not QtWidgets.QSystemTrayIcon.isSystemTrayAvailable(),
                      "a system tray is available but no tray icon was created")
            else:
                check(not tray.icon().isNull(), "the tray icon has no image")
                check(branding.APP_NAME in tray.toolTip(),
                      f"unbranded tray tooltip {tray.toolTip()!r}")
                labels = [a.text() for a in tray.contextMenu().actions()
                          if a.text()]
                for wanted in ("Settings…", "Review learnings…", "Restart", "Quit"):
                    check(wanted in labels,
                          f"tray menu is missing {wanted!r} (has {labels})")
                self.settings_dialog = None
                tray.contextMenu().actions()[0].trigger()
                check(self.settings_dialog is not None,
                      "the tray's Settings entry did not open Settings")
                self.settings_dialog.hide()
                self.review_dialog.hide()

        # -- animation ------------------------------------------------------

        def _tick(self):
            try:
                while True:
                    self.handle(*commands.get_nowait())
            except queue.Empty:
                pass
            tw, th = DIMS.get(self.state, DIMS["idle"])
            self.w += (tw - self.w) * 0.28
            self.h += (th - self.h) * 0.28
            self.t += 0.016
            # The settings chip only appears from the idle pill (never over a
            # live waveform). A short grace keeps it from flickering while the
            # cursor crosses it.
            now = time.monotonic()
            # The buttons only appear from the idle pill (never over a live
            # waveform). A short grace keeps them from flickering as the cursor
            # crosses the gap between the two circles.
            local = QtCore.QPointF(self.mapFromGlobal(QtGui.QCursor.pos()))
            can_reveal = self.state == "idle"
            if can_reveal and self._hover_pad().contains(local):
                self._hover_until = now + HOVER_GRACE_S
            want = 1.0 if (can_reveal and now < self._hover_until) else 0.0
            if self._pin is not None:
                want = self._pin
            self.reveal += (want - self.reveal) * 0.30
            if self.reveal < 0.004:
                self.reveal = 0.0
            active = self._chip_active()
            gear, mic = self._circles()
            self.gear_hovered = active and gear.contains(local)
            self.mic_hovered = active and mic.contains(local)
            self.chip_hovered = self.gear_hovered or self.mic_hovered
            # Each tag rises only while its own circle is hovered.
            want_gear = 1.0 if (self.gear_hovered or self._pin_hint == "gear") else 0.0
            want_mic = 1.0 if (self.mic_hovered or self._pin_hint == "mic") else 0.0
            self.hint_gear += (want_gear - self.hint_gear) * 0.28
            self.hint_mic += (want_mic - self.hint_mic) * 0.28
            for attr in ("hint_gear", "hint_mic"):
                if getattr(self, attr) < 0.004:
                    setattr(self, attr, 0.0)
            self.setCursor(
                QtCore.Qt.PointingHandCursor if self.chip_hovered
                else QtCore.Qt.ArrowCursor
            )
            # Auto-gain: raw mic RMS is tiny (~1e-4 quiet, a small multiple when
            # speaking) and its magnitude varies hugely by mic and speaker, so
            # the old fixed `level * 9` mapping left the bars pinned at the
            # baseline — the waveform looked frozen while you spoke. Instead we
            # track the quiet floor (slow to rise, fast to fall) and the recent
            # peak (fast attack, slow release) and show where the level sits
            # between them: silence rests, speech fills the bars, on any device.
            lvl = self.level
            self._floor += (lvl - self._floor) * (0.02 if lvl > self._floor else 0.3)
            self._peak = max(lvl, self._peak * 0.985)
            span = self._peak - self._floor
            disp = 0.0 if span < 1e-5 else max(0.0, min(1.0, (lvl - self._floor) / span))
            self.history.pop(0)
            self.history.append(disp)
            mid = (BARS - 1) / 2
            for i in range(BARS):
                if self.state == "recording":
                    # Centre bars react most (bell weight) for a tidy shape.
                    weight = 1.0 - abs(i - mid) / (mid + 1.0) * 0.5
                    target = 0.10 + 0.90 * self.history[-1 - i] * weight
                elif self.state == "processing":
                    # Gentle, slow travelling wave — calm and symmetric.
                    target = 0.28 + 0.24 * math.sin(self.t * 3.0 + i * 0.45)
                    target *= 1.0 - abs(i - mid) / (mid + 1.0) * 0.40
                else:
                    target = 0.0
                # Slightly gentler easing than before -> smoother, less jittery.
                self.heights[i] += (target - self.heights[i]) * 0.22
            self.setVisible(self.state != "hide")
            self.update()

        def paintEvent(self, _event):
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            p.setRenderHint(QtGui.QPainter.TextAntialiasing)
            cx, cy = CANVAS_W / 2, PILL_CY
            r = self.reveal

            # Alpha-1 hover pad over where the buttons appear, so the region
            # gets mouse events even before the circles fade in (fully
            # transparent pixels of a translucent window are click-through).
            if self.state == "idle":
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(QtGui.QColor(0, 0, 0, 1))
                pad = self._hover_pad()
                p.drawRoundedRect(pad, pad.height() / 2, pad.height() / 2)

            # Slim pill / live-waveform container. Fades out as the two circular
            # buttons take over on hover.
            rect, radius = self._pill_geo()
            pill_op = (1.0 - r) if self.state == "idle" else 1.0
            if pill_op > 0.01:
                p.save()
                p.setOpacity(pill_op)
                base_alpha = IDLE_ALPHA if self.state == "idle" else ACTIVE_ALPHA
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(QtGui.QColor(*PILL_RGBA, base_alpha))
                p.drawRoundedRect(rect, radius, radius)
                p.restore()

            # Waveform: crisp pill-shaped bars (filled rounded rects) rather than
            # soft stroked lines — sharper, more solid, and reads like a hi-res
            # level meter. Fades out as the settings chip takes over.
            if self.state in ("recording", "processing") and self.h > 11 \
                    and r < 0.35:
                p.save()
                p.setOpacity(1.0 - r / 0.35)
                p.setPen(QtCore.Qt.NoPen)
                inner = self.w - self.h - 6
                step = inner / (BARS - 1)
                max_bar = self.h - 6
                x0 = cx - inner / 2
                mid = (BARS - 1) / 2
                for i in range(BARS):
                    bar_h = max(BAR_W, self.heights[i] * max_bar)
                    x = x0 + i * step
                    edge = 1.0 - abs(i - mid) / mid * 0.30  # gentle end taper
                    c = QtGui.QColor(*BAR_RGBA)
                    c.setAlphaF(c.alphaF() * edge)
                    p.setBrush(c)
                    p.drawRoundedRect(
                        QtCore.QRectF(x - BAR_W / 2, cy - bar_h / 2, BAR_W, bar_h),
                        BAR_W / 2, BAR_W / 2,
                    )
                p.restore()

            # Two separate circular buttons, fading + scaling in on hover.
            if r > 0.02:
                ease = r * r * (3 - 2 * r)  # smoothstep
                gear, mic = self._circles()
                self._paint_circle(p, gear, ease, self.gear_hovered, "gear")
                self._paint_circle(p, mic, ease, self.mic_hovered, "mic")

            # Each icon's tag rises above it only while that icon is hovered.
            gear, mic = self._circles()
            if self.hint_gear > 0.01:
                self._paint_tag(p, gear.center().x(), self.hint_gear, "gear")
            if self.hint_mic > 0.01:
                self._paint_tag(p, mic.center().x(), self.hint_mic, "mic")
            p.end()

        def _paint_circle(self, p, circ, ease, hovered, kind):
            """Draw one circular button (gear or mic), scaling in with `ease`."""
            p.save()
            p.setOpacity(ease)
            s = 0.82 + 0.18 * ease
            c = circ.center()
            p.translate(c)
            p.scale(s, s)
            p.translate(-c.x(), -c.y())
            face = [v + 10 for v in CIRCLE_RGBA] if hovered else list(CIRCLE_RGBA)
            p.setPen(QtGui.QPen(QtGui.QColor(*CIRCLE_EDGE, 210 if hovered else 150),
                                1.0))
            p.setBrush(QtGui.QColor(*face, 242))
            p.drawEllipse(circ)
            col = QtGui.QColor(*ACCENT) if hovered else QtGui.QColor(*BAR_RGBA)
            ccx, ccy = c.x(), c.y()
            if kind == "gear":
                p.save()
                p.translate(ccx, ccy)
                p.rotate((1.0 - ease) * -70.0)  # settles into place
                p.fillPath(
                    _gear_path(QtCore.QPointF(0, 0), GEAR_D / 2,
                               GEAR_D / 2 * 0.66, GEAR_D / 2 * 0.34, teeth=8),
                    col,
                )
                p.restore()
            else:
                _draw_mic(p, ccx, ccy, MIC_D, col)
            p.restore()

        def _paint_tag(self, p, icon_cx, amount, kind):
            """Draw a floating tag above a circle: "Settings" for the gear, or
            "Hold <keys> to dictate" (keys as keycaps) for the mic. A small notch
            points down at the icon."""
            p.save()
            p.setOpacity(min(1.0, amount))
            label_font = p.font()
            label_font.setPointSizeF(9.0)
            fm_l = QtGui.QFontMetrics(label_font)
            gap = 6
            if kind == "gear":
                content_w = fm_l.horizontalAdvance("Settings")
            else:
                keys = _shortcut_keys(str(self.settings.get("hotkey", "f9")))
                cap_font = p.font()
                cap_font.setPointSizeF(8.0)
                cap_font.setBold(True)
                fm_c = QtGui.QFontMetrics(cap_font)
                lead_w = fm_l.horizontalAdvance("Hold")
                tail_w = fm_l.horizontalAdvance("to dictate")
                cap_ws = [fm_c.horizontalAdvance(k) + 14 for k in keys]
                content_w = lead_w + gap + sum(cap_ws) + gap * len(keys) + tail_w
            pad = 13
            tag_w = content_w + pad * 2
            tag_h = HINT_H
            bottom = (PILL_CY - CIRCLE_D / 2) - 9 + (1.0 - amount) * 6
            left = max(4, min(icon_cx - tag_w / 2, CANVAS_W - tag_w - 4))
            tag = QtCore.QRectF(left, bottom - tag_h, tag_w, tag_h)

            p.setPen(QtGui.QPen(QtGui.QColor(*BAR_RGBA[:3], 44), 1.0))
            p.setBrush(QtGui.QColor(15, 15, 18, 246))
            p.drawRoundedRect(tag, 10, 10)
            nx = max(tag.left() + 10, min(icon_cx, tag.right() - 10))
            notch = QtGui.QPainterPath()
            notch.moveTo(nx - 5, tag.bottom() - 1)
            notch.lineTo(nx + 5, tag.bottom() - 1)
            notch.lineTo(nx, tag.bottom() + 5)
            notch.closeSubpath()
            p.fillPath(notch, QtGui.QColor(15, 15, 18, 246))

            tcy = tag.center().y()
            p.setPen(QtGui.QColor(*BAR_RGBA[:3]))
            p.setFont(label_font)
            if kind == "gear":
                p.drawText(tag, QtCore.Qt.AlignCenter, "Settings")
            else:
                x = tag.left() + pad
                p.drawText(QtCore.QRectF(x, tcy - 9, lead_w, 18),
                           QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, "Hold")
                x += lead_w + gap
                for k in keys:
                    x = _draw_keycap(p, x, tcy, k, cap_font) + gap
                p.setPen(QtGui.QColor(*BAR_RGBA[:3]))
                p.setFont(label_font)
                p.drawText(QtCore.QRectF(x, tcy - 9, tail_w + 4, 18),
                           QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft,
                           "to dictate")
            p.restore()

    # ------------------------------------------------------------- settings

    def _qt_key_name(key):
        """Map a Qt key code to the global-hotkeys name used in a binding
        string. Returns None for keys we don't bind (the caller ignores them).
        Anything odd that slips through is caught by the apply-time validator."""
        Q = QtCore.Qt
        mods = {Q.Key_Control: "control", Q.Key_Shift: "shift",
                Q.Key_Alt: "alt", Q.Key_Meta: "window"}
        if key in mods:
            return mods[key]
        if Q.Key_F1 <= key <= Q.Key_F24:
            return "f%d" % (key - Q.Key_F1 + 1)
        if Q.Key_A <= key <= Q.Key_Z:
            return chr(key).lower()
        if Q.Key_0 <= key <= Q.Key_9:
            return chr(key)
        return {Q.Key_Space: "space", Q.Key_Tab: "tab",
                Q.Key_CapsLock: "caps_lock", Q.Key_Insert: "insert",
                Q.Key_Home: "home", Q.Key_End: "end",
                Q.Key_PageUp: "page_up", Q.Key_PageDown: "page_down"}.get(key)

    class BrandMark(QtWidgets.QWidget):
        """The app logo at header size — the same gradient ring and waveform as
        assets/logo-dark.png, so a window reads as SpeakEasy at a glance."""

        def __init__(self, d: int = 40):
            super().__init__()
            self._d = d
            self.setFixedSize(d, d)

        def paintEvent(self, _e):
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            _draw_brand_mark(p, self._d / 2, self._d / 2, self._d)
            p.end()

    class ShortcutField(QtWidgets.QWidget):
        """The push-to-talk binding, shown as keycaps. Click to capture a new
        chord: hold your modifiers and press the final key; Esc cancels. Exposes
        text()/setText() so it drops into the existing load/save path."""

        def __init__(self):
            super().__init__()
            self._binding = "f9"
            self._capturing = False
            self._mods = set()
            self.setMinimumHeight(48)
            self.setCursor(QtCore.Qt.PointingHandCursor)
            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.setToolTip("Click, then press your push-to-talk keys")

        # compat with the QLineEdit the dialog used to hold here
        def text(self):
            return self._binding

        def setText(self, b):
            self._binding = (b or "f9").strip() or "f9"
            self.update()

        def selectAll(self):
            self.start_capture()

        def start_capture(self):
            self._capturing = True
            self._mods = set()
            self.grabKeyboard()
            self.setFocus()
            self.update()

        def _stop_capture(self):
            self._capturing = False
            self.releaseKeyboard()
            self.update()

        def mousePressEvent(self, _e):
            self.start_capture() if not self._capturing else self._stop_capture()

        def keyPressEvent(self, e):
            if not self._capturing:
                return super().keyPressEvent(e)
            if e.key() == QtCore.Qt.Key_Escape:
                self._stop_capture()
                return
            name = _qt_key_name(e.key())
            if name in ("control", "shift", "alt", "window"):
                self._mods.add(name)
                self.update()
                return
            if name:
                order = ("control", "shift", "alt", "window")
                mods = [m for m in order if m in self._mods]
                self._binding = " + ".join(mods + [name])
                self._stop_capture()

        def focusOutEvent(self, e):
            if self._capturing:
                self._stop_capture()
            super().focusOutEvent(e)

        def paintEvent(self, _e):
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            p.setRenderHint(QtGui.QPainter.TextAntialiasing)
            rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            focused = self._capturing
            p.setPen(QtGui.QPen(QtGui.QColor(0x6e, 0x96, 0xff) if focused
                                else QtGui.QColor(0x33, 0x33, 0x3a), 1.0))
            p.setBrush(QtGui.QColor(0x23, 0x23, 0x27))
            p.drawRoundedRect(rect, 9, 9)
            cy = rect.center().y()
            cap_font = p.font()
            cap_font.setPointSizeF(9.0)
            cap_font.setBold(True)
            if self._capturing:
                keys = [k.title() for k in
                        sorted(self._mods, key="control shift alt window".split().index)]
                if keys:
                    x = rect.left() + 12
                    for k in keys:
                        x = _draw_keycap(p, x, cy, k, cap_font) + 6
                    p.setPen(QtGui.QColor(0x83, 0x83, 0x8c))
                    lbl = p.font(); lbl.setPointSizeF(9.5); p.setFont(lbl)
                    p.drawText(QtCore.QRectF(x, cy - 9, rect.right() - x - 8, 18),
                               QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft,
                               "…press a key")
                else:
                    p.setPen(QtGui.QColor(0x9a, 0x9a, 0xa2))
                    lbl = p.font(); lbl.setPointSizeF(10.0); p.setFont(lbl)
                    p.drawText(rect, QtCore.Qt.AlignCenter,
                               "Press your push-to-talk keys…")
            else:
                keys = _shortcut_keys(self._binding)
                x = rect.left() + 12
                for k in keys:
                    x = _draw_keycap(p, x, cy, k, cap_font) + 6
            p.end()

    class SettingsDialog(QtWidgets.QWidget):
        def __init__(self, values: dict):
            super().__init__(None, QtCore.Qt.WindowStaysOnTopHint)
            self.setWindowTitle("SpeakEasy Settings")
            self.setObjectName("root")
            self.setStyleSheet(DIALOG_QSS)
            self.setFixedWidth(384)
            self._review_dialog = None

            self.training = QtWidgets.QCheckBox("Ask for feedback after dictations")
            self.replace = QtWidgets.QCheckBox(
                "Fix the typed text in place on correction"
            )
            self.hotkey = ShortcutField()
            self.engine = QtWidgets.QComboBox()
            self.engine.addItems(STT_ENGINES)
            self.engine.setToolTip(
                "whisper = faster-whisper (streaming); parakeet = NVIDIA Parakeet "
                "TDT via onnx-asr (pip install -r requirements-parakeet.txt).")
            self.stt_model = QtWidgets.QComboBox()
            self.stt_model.addItems(STT_MODELS)
            self.parakeet_model = QtWidgets.QComboBox()
            self.parakeet_model.addItems(PARAKEET_MODELS)
            self.parakeet_model.setToolTip("v2 = English (fastest/most accurate EN); "
                                           "v3 = multilingual")
            self.ollama_model = QtWidgets.QComboBox()
            self.ollama_model.setEditable(True)
            self.ollama_model.addItems(OLLAMA_MODELS)
            self.cleanup = QtWidgets.QCheckBox("Clean up with the local AI model")
            self.keep_warm = QtWidgets.QCheckBox("Keep the models warm while idle")
            self.keep_warm.setToolTip(
                "Every few minutes, nudge the speech + cleanup models so they "
                "don't unload while idle — so the first dictation after a pause "
                "isn't slow. Uses a little VRAM and power.")
            self.spoken_emoji = QtWidgets.QCheckBox(
                "Insert emoji when you say their name")
            self.spoken_emoji.setToolTip(
                "Say \"fire emoji\" and get 🔥. The word \"emoji\" is required, "
                "so ordinary speech is never changed — \"call the fire "
                "department\" stays as it is.")
            self.delivery = QtWidgets.QComboBox()
            self.delivery.addItems(DELIVERY)

            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(20, 18, 20, 16)
            root.setSpacing(11)

            # Header: brand mark + name.
            header = QtWidgets.QHBoxLayout()
            header.setSpacing(12)
            header.addWidget(BrandMark())
            brand = QtWidgets.QVBoxLayout()
            brand.setSpacing(1)
            name = QtWidgets.QLabel("SpeakEasy")
            name.setObjectName("brand")
            sub = QtWidgets.QLabel("Dictation settings")
            sub.setObjectName("brandsub")
            brand.addWidget(name)
            brand.addWidget(sub)
            header.addLayout(brand)
            header.addStretch(1)
            root.addLayout(header)
            root.addSpacing(2)

            def card(title: str) -> QtWidgets.QVBoxLayout:
                fr = QtWidgets.QFrame()
                fr.setObjectName("card")
                lay = QtWidgets.QVBoxLayout(fr)
                lay.setContentsMargins(15, 12, 15, 14)
                lay.setSpacing(9)
                t = QtWidgets.QLabel(title)
                t.setProperty("role", "card")
                lay.addWidget(t)
                root.addWidget(fr)
                return lay

            def field(lay, label: str, widget):
                """A label+widget row in its own container, so it can be shown
                or hidden as a unit (space collapses cleanly when hidden)."""
                c = QtWidgets.QWidget()
                inner = QtWidgets.QVBoxLayout(c)
                inner.setContentsMargins(0, 0, 0, 0)
                inner.setSpacing(5)
                lbl = QtWidgets.QLabel(label)
                lbl.setProperty("role", "field")
                inner.addWidget(lbl)
                inner.addWidget(widget)
                lay.addWidget(c)
                return c

            ptt = card("PUSH-TO-TALK")
            ptt.addWidget(self.hotkey)
            ptt_hint = QtWidgets.QLabel("Hold to dictate · click the keys to rebind")
            ptt_hint.setProperty("role", "field")
            ptt.addWidget(ptt_hint)

            speech = card("SPEECH")
            field(speech, "Engine", self.engine)
            # Engine-specific model choice: only the active engine's field shows.
            self._whisper_field = field(speech, "Whisper model", self.stt_model)
            self._parakeet_field = field(speech, "Parakeet model", self.parakeet_model)
            self.engine.currentTextChanged.connect(self._sync_engine_fields)

            clean = card("CLEANUP")
            clean.addWidget(self.cleanup)
            field(clean, "Cleanup model", self.ollama_model)

            behave = card("BEHAVIOR")
            behave.addWidget(self.training)
            behave.addWidget(self.replace)
            behave.addWidget(self.keep_warm)
            behave.addWidget(self.spoken_emoji)
            field(behave, "Text delivery", self.delivery)
            self.review_btn = QtWidgets.QPushButton("Review what it has learned…")
            self.review_btn.setObjectName("link")
            self.review_btn.setCursor(QtCore.Qt.PointingHandCursor)
            self.review_btn.clicked.connect(self._open_review)
            behave.addWidget(self.review_btn)

            root.addSpacing(4)
            buttons = QtWidgets.QHBoxLayout()
            buttons.setSpacing(8)
            quit_btn = QtWidgets.QPushButton("Quit")
            quit_btn.setObjectName("quit")
            quit_btn.setCursor(QtCore.Qt.PointingHandCursor)
            quit_btn.setToolTip(
                "Fully close SpeakEasy (frees the hotkey and the "
                "single-instance lock so you can start it again)"
            )
            save = QtWidgets.QPushButton("Save")
            save.setObjectName("save")
            save.setCursor(QtCore.Qt.PointingHandCursor)
            cancel = QtWidgets.QPushButton("Cancel")
            cancel.setCursor(QtCore.Qt.PointingHandCursor)
            quit_btn.clicked.connect(self._quit)
            save.clicked.connect(self._save)
            cancel.clicked.connect(self.hide)
            buttons.addWidget(quit_btn)
            buttons.addStretch(1)
            buttons.addWidget(cancel)
            buttons.addWidget(save)
            root.addLayout(buttons)
            self.load(values)

        def _quit(self):
            # The parent app exits and closes our stdin; read_stdin's EOF
            # handler then shuts this process down too.
            emit({"type": "quit"})
            self.hide()

        def _open_review(self):
            target = getattr(self, "_target", 200)
            if self._review_dialog is None:
                self._review_dialog = ReviewDialog(target)
            self._review_dialog.target = max(1, int(target))
            self._review_dialog.refresh()
            self._review_dialog.show()
            self._review_dialog.raise_()
            self._review_dialog.activateWindow()

        def _sync_engine_fields(self, *_):
            """Show only the active engine's model field; hide the other."""
            whisper = self.engine.currentText() == "whisper"
            self._whisper_field.setVisible(whisper)
            self._parakeet_field.setVisible(not whisper)
            self.adjustSize()  # collapse/expand the dialog to fit

        def load(self, values: dict):
            self.training.setChecked(bool(values.get("training_enabled", False)))
            self.replace.setChecked(bool(values.get("replace_on_correction", True)))
            self.hotkey.setText(str(values.get("hotkey", "f9")))
            self.engine.setCurrentText(str(values.get("engine", "whisper")))
            self.stt_model.setCurrentText(str(values.get("stt_model", "small.en")))
            self.parakeet_model.setCurrentText(
                str(values.get("parakeet_model", "nemo-parakeet-tdt-0.6b-v2")))
            self.ollama_model.setCurrentText(str(values.get("ollama_model", "llama3.1:8b")))
            self.cleanup.setChecked(bool(values.get("cleanup_enabled", True)))
            self.keep_warm.setChecked(bool(values.get("keep_warm", False)))
            self.spoken_emoji.setChecked(bool(values.get("spoken_emoji", True)))
            self.delivery.setCurrentText(str(values.get("delivery_method", "clipboard")))
            self._target = int(values.get("target_pairs", 200))
            self._sync_engine_fields()

        def _save(self):
            emit({
                "type": "settings_saved",
                "values": {
                    "training_enabled": self.training.isChecked(),
                    "replace_on_correction": self.replace.isChecked(),
                    "hotkey": self.hotkey.text().strip() or "f9",
                    "engine": self.engine.currentText(),
                    "stt_model": self.stt_model.currentText(),
                    "parakeet_model": self.parakeet_model.currentText(),
                    "ollama_model": self.ollama_model.currentText().strip(),
                    "cleanup_enabled": self.cleanup.isChecked(),
                    "keep_warm": self.keep_warm.isChecked(),
                    "spoken_emoji": self.spoken_emoji.isChecked(),
                    "delivery_method": self.delivery.currentText(),
                },
            })
            self.hide()

    # ------------------------------------------------------------- review

    class RingProgress(QtWidgets.QWidget):
        """Circular progress ring with the current count in its centre — the
        hero stat for voice-training data, echoing the app's circular motifs."""

        def __init__(self, d: int = 96):
            super().__init__()
            self._d = d
            self._val = 0
            self._max = 1
            self.setFixedSize(d, d)

        def set_values(self, val: int, mx: int):
            self._val = max(0, int(val))
            self._max = max(1, int(mx))
            self.update()

        def paintEvent(self, _e):
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            p.setRenderHint(QtGui.QPainter.TextAntialiasing)
            w = 9.0
            ring = QtCore.QRectF(w / 2 + 1, w / 2 + 1,
                                 self._d - w - 2, self._d - w - 2)
            track = QtGui.QPen(QtGui.QColor(0x22, 0x27, 0x35), w)
            track.setCapStyle(QtCore.Qt.RoundCap)
            p.setPen(track)
            p.drawArc(ring, 0, 360 * 16)
            frac = min(1.0, self._val / self._max)
            done = frac >= 1.0
            # The filled arc carries the logo's sweep, conical so the colour
            # follows the angle the way the logo's ring does. The stops are
            # squeezed onto the drawn arc (scaled by `frac`) rather than around
            # the whole circle, so the ring shows the complete cyan -> pink
            # gradient at any fill level. Positions run backwards from 1.0
            # because a conical gradient advances anticlockwise in maths terms,
            # which is clockwise on screen, and the arc is drawn clockwise from
            # 12 o'clock. A finished ring goes green so "complete" reads
            # without having to count.
            if done:
                brush = QtGui.QBrush(QtGui.QColor(0x74, 0xd0, 0x9a))
            else:
                cone = QtGui.QConicalGradient(self._d / 2, self._d / 2, 90.0)
                first = QtGui.QColor(BRAND_STOPS[0][1])
                cone.setColorAt(0.0, first)  # kills the seam at 12 o'clock
                for pos, hexc in BRAND_STOPS:
                    cone.setColorAt(max(0.0, 1.0 - pos * frac),
                                    QtGui.QColor(hexc))
                brush = QtGui.QBrush(cone)
            arc = QtGui.QPen(brush, w)
            arc.setCapStyle(QtCore.Qt.RoundCap)
            p.setPen(arc)
            p.drawArc(ring, 90 * 16, -int(round(360 * frac)) * 16)
            p.setPen(QtGui.QColor(0xff, 0xff, 0xff))
            f = p.font()
            f.setPointSizeF(19)
            f.setBold(True)
            p.setFont(f)
            p.drawText(QtCore.QRectF(0, self._d * 0.24, self._d, self._d * 0.4),
                       QtCore.Qt.AlignCenter, str(self._val))
            p.setPen(QtGui.QColor(0x83, 0x83, 0x8c))
            f2 = p.font()
            f2.setPointSizeF(9)
            f2.setBold(False)
            p.setFont(f2)
            p.drawText(QtCore.QRectF(0, self._d * 0.54, self._d, self._d * 0.24),
                       QtCore.Qt.AlignCenter, f"of {self._max}")
            p.end()

    class ReviewDialog(QtWidgets.QWidget):
        """Browse and prune what training mode has learned: voice-training
        progress and correction examples. Edits the same files the app reads,
        so changes take effect on the next dictation."""

        def __init__(self, target_pairs: int = 200):
            super().__init__(None, QtCore.Qt.WindowStaysOnTopHint)
            self.setWindowTitle("Review learnings")
            self.setObjectName("root")
            self.setStyleSheet(DIALOG_QSS)
            self.setMinimumSize(480, 520)
            self.target = max(1, int(target_pairs))
            from .training import TrainingStore
            self.store = TrainingStore()

            outer = QtWidgets.QVBoxLayout(self)
            outer.setContentsMargins(20, 18, 20, 16)
            outer.setSpacing(11)

            # Header: brand mark + name.
            header = QtWidgets.QHBoxLayout()
            header.setSpacing(12)
            header.addWidget(BrandMark())
            brand = QtWidgets.QVBoxLayout()
            brand.setSpacing(1)
            t = QtWidgets.QLabel("Review learnings")
            t.setObjectName("brand")
            s = QtWidgets.QLabel("What SpeakEasy has picked up from you")
            s.setObjectName("brandsub")
            brand.addWidget(t)
            brand.addWidget(s)
            header.addLayout(brand)
            header.addStretch(1)
            outer.addLayout(header)

            def card(title):
                fr = QtWidgets.QFrame()
                fr.setObjectName("card")
                lay = QtWidgets.QVBoxLayout(fr)
                lay.setContentsMargins(15, 12, 15, 14)
                lay.setSpacing(9)
                if title:
                    tl = QtWidgets.QLabel(title)
                    tl.setProperty("role", "card")
                    lay.addWidget(tl)
                outer.addWidget(fr)
                return fr, lay

            # Voice-training hero: ring + count + hint.
            fr, lay = card("VOICE TRAINING")
            fr.setToolTip("(audio, what-you-actually-said) pairs saved for a "
                          "future voice fine-tune")
            hrow = QtWidgets.QHBoxLayout()
            hrow.setSpacing(16)
            self.ring = RingProgress()
            hrow.addWidget(self.ring)
            col = QtWidgets.QVBoxLayout()
            col.setSpacing(4)
            col.addStretch(1)
            self.pairs_count = QtWidgets.QLabel()
            self.pairs_count.setStyleSheet(
                "color:#ffffff;font-size:16px;font-weight:700;")
            self.pairs_hint = QtWidgets.QLabel()
            self.pairs_hint.setObjectName("brandsub")
            self.pairs_hint.setWordWrap(True)
            col.addWidget(self.pairs_count)
            col.addWidget(self.pairs_hint)
            col.addStretch(1)
            hrow.addLayout(col, 1)
            lay.addLayout(hrow)

            # Corrections list.
            fr_c, lay_c = card("CORRECTIONS")
            fr_c.setToolTip("Taught to the cleanup model as few-shot examples")
            self.corr_list = QtWidgets.QListWidget()
            lay_c.addWidget(self.corr_list)
            outer.setStretchFactor(fr_c, 1)

            self.corr_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            self.corr_list.setWordWrap(True)
            self.corr_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

            close = QtWidgets.QPushButton("Close")
            close.setCursor(QtCore.Qt.PointingHandCursor)
            close.clicked.connect(self.hide)
            outer.addWidget(close, alignment=QtCore.Qt.AlignRight)

        def _row(self, text: str, on_delete) -> QtWidgets.QWidget:
            w = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(6, 4, 6, 4)
            lay.setSpacing(8)
            label = QtWidgets.QLabel(text)
            label.setWordWrap(True)
            btn = QtWidgets.QPushButton("Forget")
            btn.setObjectName("change")
            btn.setFixedWidth(68)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.clicked.connect(on_delete)
            lay.addWidget(label, 1)
            lay.addWidget(btn, 0, QtCore.Qt.AlignTop)
            return w

        def _add(self, listw, text, on_delete):
            item = QtWidgets.QListWidgetItem(listw)
            row = self._row(text, on_delete)
            listw.addItem(item)
            listw.setItemWidget(item, row)

        def _reflow(self):
            """Size each list row to the viewport so its text wraps instead of
            being clipped, and its item grows to the wrapped height."""
            listw = self.corr_list
            w = max(60, listw.viewport().width())
            for i in range(listw.count()):
                row = listw.itemWidget(listw.item(i))
                if row is None:
                    continue  # plain empty-state item
                row.setFixedWidth(w)
                # sizeHint() ignores word-wrap, so a long correction reports the
                # height of a single line and the rows clip into each other.
                # heightForWidth is the only figure that accounts for wrapping.
                lay = row.layout()
                h = lay.heightForWidth(w) if lay.hasHeightForWidth() else -1
                if h <= 0:
                    row.adjustSize()
                    h = row.sizeHint().height()
                listw.item(i).setSizeHint(QtCore.QSize(w, h))
                row.setFixedHeight(h)

        def resizeEvent(self, e):
            super().resizeEvent(e)
            self._reflow()

        def refresh(self):
            n = self.store.trainable_pair_count()
            self.ring.set_values(n, self.target)
            if n >= self.target:
                self.pairs_count.setText("Ready to fine-tune")
                self.pairs_hint.setText(
                    "You've collected enough voice data to fine-tune the speech "
                    "model to your voice.")
            else:
                self.pairs_count.setText(f"{n} of {self.target} pairs")
                self.pairs_hint.setText(
                    f"{self.target - n} more corrections with “what you actually "
                    "said” to reach the fine-tuning target.")
            self.corr_list.clear()

            def short(s, n=150):
                s = " ".join(str(s).split())
                return s if len(s) <= n else s[: n - 1].rstrip() + "…"

            for e in reversed(self.store.corrections(n=None)):
                ts = e.get("ts")
                text = f"“{short(e.get('raw',''))}”  →  “{short(e.get('ideal',''))}”"
                self._add(self.corr_list, text,
                          lambda _=False, ts=ts: self._forget_corr(ts))
            if self.corr_list.count() == 0:
                self.corr_list.addItem("No corrections yet.")
            self._reflow()

        def _forget_corr(self, ts):
            self.store.delete_correction(ts)
            self.refresh()

    # ------------------------------------------------------------- feedback

    class StarBar(QtWidgets.QWidget):
        """A row of `count` vector stars; click sets the rating (1..count)."""
        rated = QtCore.Signal(int)

        def __init__(self, count=5, size=20, interactive=True):
            super().__init__()
            self._count = count
            self._size = size
            self._rating = 0
            self._interactive = interactive
            self._cell = size + 5
            self.setFixedSize(count * self._cell, size + 4)
            if interactive:
                self.setCursor(QtCore.Qt.PointingHandCursor)

        def rating(self) -> int:
            return self._rating

        def setRating(self, n: int) -> None:
            self._rating = max(0, min(self._count, int(n)))
            self.update()

        def _star_at(self, x: float) -> int:
            return max(1, min(self._count, int(x // self._cell) + 1))

        def mousePressEvent(self, event):
            if self._interactive:
                self.setRating(self._star_at(event.position().x()))
                self.rated.emit(self._rating)

        def paintEvent(self, _event):
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            s = self._size
            for i in range(self._count):
                cx = i * self._cell + self._cell / 2
                cy = (s + 4) / 2
                path = _star_path(cx, cy, s / 2, s / 4.4)
                if i < self._rating:
                    # Filled stars ride the brand sweep across the whole row,
                    # so five stars reads as the logo's gradient.
                    p.fillPath(path, QtGui.QBrush(_brand_gradient(
                        0, 0, self._count * self._cell, 0)))
                else:
                    pen = QtGui.QPen(QtGui.QColor(58, 64, 84))
                    pen.setWidthF(1.4)
                    p.setPen(pen)
                    p.setBrush(QtCore.Qt.NoBrush)
                    p.drawPath(path)
            p.end()

    class FeedbackPanel(QtWidgets.QWidget):
        """Progressive training feedback: a collapsed rating strip that expands
        into a four-field teaching form. Never steals keyboard focus until the
        user clicks 'Correct it'. No timeout — it waits until answered or is
        superseded by the next dictation's panel."""

        def __init__(self, bar: "Bar", req: dict):
            super().__init__(
                None,
                QtCore.Qt.FramelessWindowHint
                | QtCore.Qt.WindowStaysOnTopHint
                | QtCore.Qt.Tool
                | QtCore.Qt.WindowDoesNotAcceptFocus,
            )
            self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
            self.setObjectName("root")
            self.setStyleSheet(FEEDBACK_QSS)
            self.setFixedWidth(430)
            self._bar = bar
            self.req = req
            self.raw = str(req.get("raw", ""))
            self.cleaned = str(req.get("cleaned", ""))
            self.answered = False
            self._expanded = False
            self._drag_from = None       # cursor→top-left offset while dragging
            self._user_moved = False     # once dragged, keep the user's spot

            self._root = QtWidgets.QVBoxLayout(self)
            self._root.setContentsMargins(15, 13, 15, 13)
            self._root.setSpacing(9)

            self._preview = QtWidgets.QLabel(f"“{self.cleaned}”")
            self._preview.setObjectName("preview")
            self._preview.setWordWrap(True)
            self._root.addWidget(self._preview)

            self._strip = QtWidgets.QHBoxLayout()
            self._stars = StarBar()
            self._stars.rated.connect(self._quick_rate)
            self._correct = QtWidgets.QPushButton("Correct it ›")
            self._correct.setObjectName("link")
            self._correct.setCursor(QtCore.Qt.PointingHandCursor)
            self._correct.setFocusPolicy(QtCore.Qt.NoFocus)
            self._correct.clicked.connect(self._expand)
            self._strip.addWidget(self._stars)
            self._strip.addStretch(1)
            self._strip.addWidget(self._correct)
            self._root.addLayout(self._strip)

            self.adjustSize()
            self._place()

        # -- geometry / drag -----------------------------------------------
        def _avail(self):
            scr = self.screen() or QtWidgets.QApplication.primaryScreen()
            return scr.availableGeometry()

        def _place(self):
            """Sit in the bottom-right (out of the centre text area) and clamp
            fully on-screen. Once the user drags the panel, keep their spot and
            only clamp so it can't run off-screen as it grows/shrinks."""
            self.adjustSize()
            if not self._user_moved:
                a = self._avail()
                m = 16
                self.move(a.right() - self.width() - m,
                          a.bottom() - self.height() - m)
            self._clamp()

        def _clamp(self):
            a = self._avail()
            m = 8
            x = min(max(self.x(), a.left() + m), a.right() - self.width() - m)
            y = min(max(self.y(), a.top() + m), a.bottom() - self.height() - m)
            self.move(int(x), int(y))

        # Drag the panel anywhere: press on its body (not on the stars, tags,
        # buttons or text fields, which handle their own clicks) and move.
        def mousePressEvent(self, event):
            if event.button() == QtCore.Qt.LeftButton:
                self._drag_from = (event.globalPosition().toPoint()
                                   - self.frameGeometry().topLeft())
                event.accept()

        def mouseMoveEvent(self, event):
            if self._drag_from is not None and (
                    event.buttons() & QtCore.Qt.LeftButton):
                self._user_moved = True
                self.move(event.globalPosition().toPoint() - self._drag_from)
                event.accept()

        def mouseReleaseEvent(self, event):
            self._drag_from = None

        # -- collapsed fast path -------------------------------------------
        def _quick_rate(self, n: int):
            self._submit(rating=n, transcript=None, ideal=None, tags=[])

        # -- expand into the teaching form ---------------------------------
        def _flabel(self, text: str) -> QtWidgets.QLabel:
            lbl = QtWidgets.QLabel(text)
            lbl.setProperty("role", "flabel")
            return lbl

        def _readonly(self, text: str) -> QtWidgets.QLabel:
            lbl = QtWidgets.QLabel(text)
            lbl.setObjectName("ro")
            lbl.setWordWrap(True)
            return lbl

        def _editor(self, text: str) -> QtWidgets.QPlainTextEdit:
            ed = QtWidgets.QPlainTextEdit(text)
            ed.setTabChangesFocus(True)
            fm = ed.fontMetrics()
            ed.setFixedHeight(fm.lineSpacing() * 4 + 16)  # ~4 lines, roomier
            return ed

        def _expand(self):
            if self._expanded:
                return
            self._expanded = True
            # Hide the collapsed strip's "Correct it" link; keep the stars idea
            # in the form instead.
            self._correct.hide()
            self._stars.hide()
            self._preview.hide()

            title_row = QtWidgets.QHBoxLayout()
            title = QtWidgets.QLabel("Teach SpeakEasy")
            title.setObjectName("title")
            self._form_stars = StarBar()
            self._form_stars.setRating(self._stars.rating())
            title_row.addWidget(title)
            title_row.addStretch(1)
            title_row.addWidget(self._form_stars)
            self._root.addLayout(title_row)

            self._root.addWidget(self._flabel("HEARD · SPEECH → TEXT"))
            self._root.addWidget(self._readonly(self.raw))
            self._root.addWidget(self._flabel("CLEANED · WHAT IT TYPED"))
            self._root.addWidget(self._readonly(self.cleaned))

            self._root.addWidget(self._flabel("WHAT YOU ACTUALLY SAID"))
            self._actual = self._editor(self.raw)
            self._root.addWidget(self._actual)
            self._root.addWidget(self._flabel("IDEAL CLEANUP"))
            self._ideal = self._editor(self.cleaned)
            self._root.addWidget(self._ideal)

            self._root.addWidget(self._flabel("WHAT WENT WRONG"))
            # 3-per-row grid so all five chips fit the fixed-width panel with
            # uniform widths (a single row overflows / clips at 360px).
            tag_grid = QtWidgets.QGridLayout()
            tag_grid.setSpacing(6)
            self._tag_btns = {}
            for idx, t in enumerate(FEEDBACK_TAGS):
                b = QtWidgets.QPushButton(t)
                b.setObjectName("tag")
                b.setCheckable(True)
                b.setCursor(QtCore.Qt.PointingHandCursor)
                b.setFocusPolicy(QtCore.Qt.NoFocus)
                self._tag_btns[t] = b
                tag_grid.addWidget(b, idx // 3, idx % 3)
            self._root.addLayout(tag_grid)

            btns = QtWidgets.QHBoxLayout()
            cancel = QtWidgets.QPushButton("Cancel")
            cancel.setObjectName("ghost")
            cancel.setCursor(QtCore.Qt.PointingHandCursor)
            cancel.clicked.connect(self.close)
            save = QtWidgets.QPushButton("Save lesson")
            save.setObjectName("save")
            save.setCursor(QtCore.Qt.PointingHandCursor)
            save.clicked.connect(self._save)
            btns.addStretch(1)
            btns.addWidget(cancel)
            btns.addWidget(save)
            self._root.addLayout(btns)

            # Now the panel may take focus so the fields are editable.
            self.setWindowFlag(QtCore.Qt.WindowDoesNotAcceptFocus, False)
            self._place()
            self.show()
            self.activateWindow()
            self._actual.setFocus()

        def _save(self):
            actual = self._actual.toPlainText().strip()
            ideal_txt = self._ideal.toPlainText().strip()
            transcript = actual if actual and actual != self.raw.strip() else None
            ideal = ideal_txt if ideal_txt and ideal_txt != self.cleaned.strip() else None
            tags = [t for t, b in self._tag_btns.items() if b.isChecked()]
            rating = self._form_stars.rating() or None
            self._submit(rating=rating, transcript=transcript, ideal=ideal, tags=tags)

        # -- submit / close ------------------------------------------------
        def _submit(self, rating, transcript, ideal, tags):
            if self.answered:
                return
            self.answered = True
            emit({"type": "feedback", "id": self.req.get("id"),
                  "rating": rating, "transcript": transcript,
                  "ideal": ideal, "tags": tags})
            self.close()

    # ------------------------------------------------------------------ io

    bar = Bar()
    bar.show()

    def _build_tray():
        """Windows tray icon: the app's handle when the pill is easy to miss and
        there may be no console to Ctrl+C. Returns None where no tray exists
        (some Linux desktops), which is not an error — the pill still works."""
        if not QtWidgets.QSystemTrayIcon.isSystemTrayAvailable():
            return None
        tray = QtWidgets.QSystemTrayIcon(APP_ICON)
        menu = QtWidgets.QMenu()
        menu.setStyleSheet(TRAY_QSS)

        def act(text, slot):
            a = menu.addAction(text)
            a.triggered.connect(slot)
            return a

        act("Settings…", lambda: bar._open_settings())
        act("Review learnings…", bar._open_review)
        menu.addSeparator()
        # Restart re-launches the whole app, hotkey loop included, so a changed
        # speech engine or a wedged model gets a clean process without the user
        # hunting for the terminal.
        act("Restart", lambda: emit({"type": "restart"}))
        act("Quit", lambda: emit({"type": "quit"}))
        tray.setContextMenu(menu)
        tray.setToolTip(_tray_tip(bar.settings))
        # Left-click opens settings; the context menu is the right-click.
        tray.activated.connect(
            lambda reason: bar._open_settings()
            if reason == QtWidgets.QSystemTrayIcon.Trigger else None
        )
        tray.show()
        # Keep a reference on the app or the menu is garbage-collected and the
        # icon goes dead on right-click.
        app._tray, app._tray_menu = tray, menu
        return tray

    def _tray_tip(settings: dict) -> str:
        key = str(settings.get("hotkey", "") or "").strip()
        return (f"{branding.APP_NAME} — hold {key} to dictate" if key
                else branding.APP_NAME)

    tray = _build_tray()
    if tray is not None:
        # The tip names the current push-to-talk key, so it has to follow a
        # rebind rather than being fixed at startup.
        bar.on_settings_changed = lambda s: tray.setToolTip(_tray_tip(s))

    def read_stdin():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            cmd, _, arg = line.partition(" ")
            commands.put((cmd, arg))
        # EOF: parent is gone. Quit must run on the Qt main thread —
        # app.quit() from this thread is unreliable — so queue it like any
        # other command; the render timer picks it up within a frame.
        commands.put(("__eof__", ""))

    threading.Thread(target=read_stdin, daemon=True).start()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
