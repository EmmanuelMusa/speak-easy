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
import queue
import sys
import threading
import time

PILL_RGBA = (20, 20, 23)          # dark charcoal body
BAR_RGBA = (237, 237, 240, 240)   # near-white waveform / icon / text
ACCENT = (110, 150, 255)          # soft blue accent (hover, focus, Save)
IDLE_ALPHA = 120
ACTIVE_ALPHA = 216
CHIP_ALPHA = 238

# Pill footprint per state (width, height). Compact, tighter visualizer.
DIMS = {"idle": (44, 5), "recording": (94, 18), "processing": (94, 18),
        "hide": (44, 5)}
CANVAS_W, CANVAS_H = 150, 44
PILL_CY = CANVAS_H / 2             # everything is vertically centered now

# On hover the pill morphs into this settings chip (icon + "Settings" label)
# in place — it replaces the pill rather than floating above it.
CHIP_W, CHIP_H = 122, 30
GEAR_D = 16                        # settings-gear diameter inside the chip
HOVER_GRACE_S = 0.5               # chip lingers briefly after the mouse leaves
BARS = 15                          # finer resolution than before (was 11)
BAR_W = 2.4                        # crisp pill-shaped bar width

STT_MODELS = ["tiny.en", "base", "small.en", "medium", "large-v3"]
OLLAMA_MODELS = ["llama3.1:8b", "llama3.2:3b", "phi3:mini", "mistral:7b"]
DELIVERY = ["clipboard", "sendinput"]

FEEDBACK_TAGS = ["misheard word", "wrong punctuation", "over-deleted",
                 "wrong casing", "bad list"]

FEEDBACK_QSS = """
QWidget#root { background:#1b1b1f; border:1px solid #2e2e34; border-radius:12px; }
QLabel { color:#e7e7ea; font-size:12px; }
QLabel#preview { color:#d8d8de; font-size:12px; }
QLabel#title { color:#ffffff; font-size:13px; font-weight:700; }
QLabel[role="flabel"] { color:#7f8695; font-size:10px; font-weight:700;
    letter-spacing:1.2px; }
QLabel#ro { background:#141417; border:1px solid #27272d; border-radius:8px;
    padding:7px 9px; color:#9a9aa2; }
QPlainTextEdit { background:#232327; color:#e7e7ea; border:1px solid #3a3a44;
    border-radius:8px; padding:6px 8px; font-size:12px; }
QPlainTextEdit:focus { border:1px solid #6e96ff; }
QPushButton#link { background:transparent; color:#6e96ff; border:none;
    font-size:12px; font-weight:600; }
QPushButton#tag { background:#232327; color:#b9b9c0; border:1px solid #3b3b42;
    border-radius:11px; padding:4px 10px; font-size:11px; }
QPushButton#tag:checked { background:#2a3350; color:#cdd8ff; border:1px solid #6e96ff; }
QPushButton#ghost { background:#2a2a31; color:#e7e7ea; border:none;
    border-radius:8px; padding:7px 13px; font-size:12px; }
QPushButton#save { background:#6e96ff; color:#0f1220; border:none;
    border-radius:8px; padding:7px 13px; font-size:12px; font-weight:700; }
"""

# Shared dark theme for the settings/review windows — matches the pill.
DIALOG_QSS = """
QWidget#root { background: #17171a; }
QLabel { color: #e7e7ea; font-size: 12px; }
QLabel#title { color: #ffffff; font-size: 16px; font-weight: 700; }
QLabel#subtitle { color: #83838c; font-size: 11px; }
QLabel[role="section"] {
    color: #7f8695; font-size: 10px; font-weight: 700; letter-spacing: 1.4px;
    padding-top: 4px;
}
QLabel[role="field"] { color: #b9b9c0; font-size: 12px; }
QCheckBox { color: #e7e7ea; font-size: 12px; spacing: 9px; }
QCheckBox::indicator {
    width: 17px; height: 17px; border-radius: 5px;
    border: 1px solid #3b3b42; background: #232327;
}
QCheckBox::indicator:hover { border: 1px solid #6e96ff; }
QCheckBox::indicator:checked {
    background: #6e96ff; border: 1px solid #6e96ff;
}
QLineEdit, QComboBox {
    background: #232327; color: #e7e7ea; border: 1px solid #33333a;
    border-radius: 8px; padding: 7px 10px; font-size: 12px; min-height: 16px;
}
QLineEdit:focus, QComboBox:focus, QComboBox:on { border: 1px solid #6e96ff; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background: #232327; color: #e7e7ea; border: 1px solid #33333a;
    border-radius: 8px; padding: 4px; outline: none;
    selection-background-color: #6e96ff; selection-color: #10121a;
}
QListWidget {
    background: #202024; color: #e7e7ea; border: 1px solid #2e2e34;
    border-radius: 8px; padding: 2px; outline: none;
}
QPushButton {
    background: #2a2a31; color: #e7e7ea; border: none; border-radius: 8px;
    padding: 8px 15px; font-size: 12px;
}
QPushButton:hover { background: #34343d; }
QPushButton#save {
    background: #6e96ff; color: #0f1220; font-weight: 700;
}
QPushButton#save:hover { background: #85a6ff; }
QPushButton#quit { background: transparent; color: #d97070; padding-left: 4px; }
QPushButton#quit:hover { background: #2c2022; color: #e88a8a; }
QPushButton#link {
    background: transparent; color: #6e96ff; text-align: left; padding: 4px 2px;
}
QPushButton#link:hover { color: #90adff; }
QFrame#sep { background: #26262c; max-height: 1px; border: none; }
"""


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    from PySide6 import QtCore, QtGui, QtWidgets

    app = QtWidgets.QApplication([])
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
            self.move(geo.center().x() - CANVAS_W // 2, geo.bottom() - CANVAS_H - 14)

            self.state = "idle"
            self.level = 0.0
            self.w, self.h = float(DIMS["idle"][0]), float(DIMS["idle"][1])
            self.t = 0.0
            self.history = [0.0] * (BARS * 2)
            self.heights = [0.0] * BARS
            self.reveal = 0.0          # 0 = pill, 1 = settings chip (animated)
            self.chip_hovered = False
            self._hover_until = 0.0
            self._pin = None           # debug: force a reveal value for shots
            self.settings: dict = {}
            self.settings_dialog = None
            self.feedback_panel = None

            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._tick)
            self.timer.start(16)

        # -- geometry helpers ------------------------------------------

        def _morph_geo(self):
            """Rect + corner radius interpolated between the current pill and
            the settings chip by self.reveal. One shape, so paint and hit-test
            agree and there are no click-through gaps."""
            cx, cy = CANVAS_W / 2, PILL_CY
            r = self.reveal
            w = self.w + (CHIP_W - self.w) * r
            h = self.h + (CHIP_H - self.h) * r
            rect = QtCore.QRectF(cx - w / 2, cy - h / 2, w, h)
            return rect, h / 2  # fully rounded (pill-shaped chip)

        def _chip_active(self) -> bool:
            return self.reveal > 0.5

        # -- events ------------------------------------------------------

        def mousePressEvent(self, event):
            rect, _ = self._morph_geo()
            if self._chip_active() and rect.contains(event.position()):
                self._open_settings()

        def _open_settings(self):
            if self.settings_dialog is None:
                self.settings_dialog = SettingsDialog(self.settings)
            self.settings_dialog.load(self.settings)
            self.settings_dialog.show()
            self.settings_dialog.raise_()
            self.settings_dialog.activateWindow()

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
                # Instantiate the dialogs + feedback panel so their render paths
                # are exercised by automated tests (they can't click).
                try:
                    SettingsDialog(self.settings)
                    ReviewDialog().refresh()
                    fp = FeedbackPanel(self, {"id": 0, "raw": "hello wrld",
                                              "cleaned": "Hello world."})
                    fp._expand()
                    fp.close()
                    emit({"type": "selftest_ok"})
                except Exception as exc:  # pragma: no cover - reported to parent
                    emit({"type": "selftest_err", "error": repr(exc)})
            elif cmd == "pin":  # debug/visual-QA: pin the chip reveal (0..1)
                self._pin = None if arg == "none" else float(arg)
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
            elif cmd == "shotreview":  # debug: screenshot the review dashboard
                d = ReviewDialog(int(self.settings.get("target_pairs", 200)))
                d.refresh()
                d.ensurePolished()
                d.resize(d.sizeHint())
                d.grab().save(arg)
                d.deleteLater()
                emit({"type": "shot_ok"})

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
            can_reveal = self.state == "idle"
            if can_reveal and self.underMouse():
                self._hover_until = now + HOVER_GRACE_S
            want = 1.0 if (can_reveal and now < self._hover_until) else 0.0
            if self._pin is not None:
                want = self._pin
            self.reveal += (want - self.reveal) * 0.30
            if self.reveal < 0.004:
                self.reveal = 0.0
            local = self.mapFromGlobal(QtGui.QCursor.pos())
            rect, _ = self._morph_geo()
            self.chip_hovered = self._chip_active() and rect.contains(
                QtCore.QPointF(local)
            )
            self.setCursor(
                QtCore.Qt.PointingHandCursor if self.chip_hovered
                else QtCore.Qt.ArrowCursor
            )
            self.history.pop(0)
            self.history.append(min(1.0, self.level * 9.0))
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

            # Alpha-1 hover pad around the slim idle bar so the small target is
            # easy to reach (fully transparent pixels of a translucent window
            # are click-through on Windows, so they'd get no mouse events).
            if self.state == "idle":
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(QtGui.QColor(0, 0, 0, 1))
                p.drawRoundedRect(
                    QtCore.QRectF(cx - CHIP_W / 2, cy - CHIP_H / 2, CHIP_W, CHIP_H),
                    CHIP_H / 2, CHIP_H / 2,
                )

            rect, radius = self._morph_geo()
            base_alpha = IDLE_ALPHA if self.state == "idle" else ACTIVE_ALPHA
            alpha = int(base_alpha + (CHIP_ALPHA - base_alpha) * r)
            body = list(PILL_RGBA)
            if self.chip_hovered:
                body = [c + 12 for c in PILL_RGBA]  # lift on hover
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(body[0], body[1], body[2], alpha))
            p.drawRoundedRect(rect, radius, radius)
            if r > 0.05:  # hairline border defines the chip edge crisply
                pen = QtGui.QPen(QtGui.QColor(*BAR_RGBA[:3], int(46 * r)))
                pen.setWidthF(1.0)
                p.setPen(pen)
                p.setBrush(QtCore.Qt.NoBrush)
                p.drawRoundedRect(rect, radius, radius)

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

            # Settings chip contents: vector gear + "Settings" label.
            if r > 0.02:
                ease = r * r * (3 - 2 * r)  # smoothstep
                p.save()
                p.setOpacity(ease)
                icon_c = QtGui.QColor(*ACCENT) if self.chip_hovered \
                    else QtGui.QColor(*BAR_RGBA)
                gx = rect.left() + 17
                # Gear spins into place as the chip opens — a small, satisfying
                # settle rather than a distracting continuous spin.
                p.save()
                p.translate(gx, cy)
                p.rotate((1.0 - ease) * -80.0)
                p.fillPath(
                    _gear_path(QtCore.QPointF(0, 0), GEAR_D / 2,
                               GEAR_D / 2 * 0.66, GEAR_D / 2 * 0.34, teeth=8),
                    icon_c,
                )
                p.restore()
                p.setPen(QtGui.QColor(*BAR_RGBA))
                font = p.font()
                font.setPointSizeF(9.5)
                font.setLetterSpacing(QtGui.QFont.PercentageSpacing, 104)
                p.setFont(font)
                lx = gx + GEAR_D / 2 + 9 + (1.0 - ease) * 6  # slides in slightly
                p.drawText(
                    QtCore.QRectF(lx, cy - 9, rect.right() - lx - 8, 18),
                    QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, "Settings",
                )
                p.restore()
            p.end()

    # ------------------------------------------------------------- settings

    class SettingsDialog(QtWidgets.QWidget):
        def __init__(self, values: dict):
            super().__init__(None, QtCore.Qt.WindowStaysOnTopHint)
            self.setWindowTitle("Speak Easy Settings")
            self.setObjectName("root")
            self.setStyleSheet(DIALOG_QSS)
            self.setFixedWidth(360)
            self._review_dialog = None

            self.training = QtWidgets.QCheckBox("Ask for feedback after dictations")
            self.replace = QtWidgets.QCheckBox(
                "Fix the typed text in place on correction"
            )
            self.hotkey = QtWidgets.QLineEdit()
            self.stt_model = QtWidgets.QComboBox()
            self.stt_model.addItems(STT_MODELS)
            self.ollama_model = QtWidgets.QComboBox()
            self.ollama_model.setEditable(True)
            self.ollama_model.addItems(OLLAMA_MODELS)
            self.cleanup = QtWidgets.QCheckBox("Clean up with the local AI model")
            self.delivery = QtWidgets.QComboBox()
            self.delivery.addItems(DELIVERY)

            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(22, 20, 22, 18)
            root.setSpacing(4)

            title = QtWidgets.QLabel("Speak Easy")
            title.setObjectName("title")
            subtitle = QtWidgets.QLabel("Dictation settings")
            subtitle.setObjectName("subtitle")
            root.addWidget(title)
            root.addWidget(subtitle)
            root.addSpacing(14)

            def section(text: str) -> None:
                lbl = QtWidgets.QLabel(text)
                lbl.setProperty("role", "section")
                root.addWidget(lbl)

            def field(label: str, widget) -> None:
                lbl = QtWidgets.QLabel(label)
                lbl.setProperty("role", "field")
                root.addWidget(lbl)
                root.addWidget(widget)
                root.addSpacing(8)

            section("DICTATION")
            root.addSpacing(6)
            field("Push-to-talk key", self.hotkey)
            field("Speech model", self.stt_model)
            field("Cleanup model", self.ollama_model)
            root.addWidget(self.cleanup)
            root.addSpacing(12)

            sep = QtWidgets.QFrame()
            sep.setObjectName("sep")
            sep.setFixedHeight(1)
            root.addWidget(sep)
            root.addSpacing(12)

            section("BEHAVIOR")
            root.addSpacing(6)
            root.addWidget(self.training)
            root.addSpacing(4)
            root.addWidget(self.replace)
            root.addSpacing(10)
            field("Text delivery", self.delivery)

            self.review_btn = QtWidgets.QPushButton("Review what it has learned…")
            self.review_btn.setObjectName("link")
            self.review_btn.setCursor(QtCore.Qt.PointingHandCursor)
            self.review_btn.clicked.connect(self._open_review)
            root.addWidget(self.review_btn)
            root.addSpacing(16)

            buttons = QtWidgets.QHBoxLayout()
            buttons.setSpacing(8)
            quit_btn = QtWidgets.QPushButton("Quit")
            quit_btn.setObjectName("quit")
            quit_btn.setCursor(QtCore.Qt.PointingHandCursor)
            quit_btn.setToolTip(
                "Fully close Speak Easy (frees the hotkey and the "
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

        def load(self, values: dict):
            self.training.setChecked(bool(values.get("training_enabled", False)))
            self.replace.setChecked(bool(values.get("replace_on_correction", True)))
            self.hotkey.setText(str(values.get("hotkey", "f9")))
            self.stt_model.setCurrentText(str(values.get("stt_model", "small.en")))
            self.ollama_model.setCurrentText(str(values.get("ollama_model", "llama3.1:8b")))
            self.cleanup.setChecked(bool(values.get("cleanup_enabled", True)))
            self.delivery.setCurrentText(str(values.get("delivery_method", "clipboard")))
            self._target = int(values.get("target_pairs", 200))

        def _save(self):
            emit({
                "type": "settings_saved",
                "values": {
                    "training_enabled": self.training.isChecked(),
                    "replace_on_correction": self.replace.isChecked(),
                    "hotkey": self.hotkey.text().strip() or "f9",
                    "stt_model": self.stt_model.currentText(),
                    "ollama_model": self.ollama_model.currentText().strip(),
                    "cleanup_enabled": self.cleanup.isChecked(),
                    "delivery_method": self.delivery.currentText(),
                },
            })
            self.hide()

    # ------------------------------------------------------------- review

    class ReviewDialog(QtWidgets.QWidget):
        """Browse and prune what training mode has learned: correction
        examples and auto-learned vocabulary. Edits the same files the app
        reads, so changes take effect on the next dictation."""

        def __init__(self, target_pairs: int = 200):
            super().__init__(None, QtCore.Qt.WindowStaysOnTopHint)
            self.setWindowTitle("Review learnings")
            self.setObjectName("root")
            self.setStyleSheet(DIALOG_QSS)
            self.setMinimumSize(470, 470)
            self.target = max(1, int(target_pairs))
            from .training import TrainingStore
            self.store = TrainingStore()

            outer = QtWidgets.QVBoxLayout(self)
            outer.setContentsMargins(20, 18, 20, 16)
            outer.setSpacing(8)

            title = QtWidgets.QLabel("Review learnings")
            title.setObjectName("title")
            outer.addWidget(title)

            pairs_lbl = QtWidgets.QLabel("VOICE TRAINING DATA")
            pairs_lbl.setProperty("role", "section")
            pairs_lbl.setToolTip(
                "(audio, what-you-actually-said) pairs saved for a future "
                "voice fine-tune")
            outer.addWidget(pairs_lbl)
            self.pairs_count = QtWidgets.QLabel()
            self.pairs_count.setStyleSheet(
                "color:#ffffff;font-size:15px;font-weight:700;")
            outer.addWidget(self.pairs_count)
            self.pairs_bar = QtWidgets.QProgressBar()
            self.pairs_bar.setTextVisible(False)
            self.pairs_bar.setFixedHeight(8)
            self.pairs_bar.setStyleSheet(
                "QProgressBar{background:#232327;border:1px solid #33333a;"
                "border-radius:5px;}"
                "QProgressBar::chunk{background:#6e96ff;border-radius:5px;}")
            outer.addWidget(self.pairs_bar)
            self.pairs_hint = QtWidgets.QLabel()
            self.pairs_hint.setObjectName("subtitle")
            self.pairs_hint.setWordWrap(True)
            outer.addWidget(self.pairs_hint)
            outer.addSpacing(4)

            corr_lbl = QtWidgets.QLabel("CORRECTION EXAMPLES")
            corr_lbl.setProperty("role", "section")
            corr_lbl.setToolTip("Taught to the cleanup model as few-shot examples")
            outer.addWidget(corr_lbl)
            self.corr_list = QtWidgets.QListWidget()
            outer.addWidget(self.corr_list, 3)

            vocab_lbl = QtWidgets.QLabel("LEARNED VOCABULARY")
            vocab_lbl.setProperty("role", "section")
            vocab_lbl.setToolTip("Words the model must preserve exactly")
            outer.addWidget(vocab_lbl)
            self.vocab_list = QtWidgets.QListWidget()
            outer.addWidget(self.vocab_list, 1)

            close = QtWidgets.QPushButton("Close")
            close.setCursor(QtCore.Qt.PointingHandCursor)
            close.clicked.connect(self.hide)
            outer.addWidget(close, alignment=QtCore.Qt.AlignRight)

        def _row(self, text: str, on_delete) -> QtWidgets.QWidget:
            w = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(4, 2, 4, 2)
            label = QtWidgets.QLabel(text)
            label.setWordWrap(True)
            btn = QtWidgets.QPushButton("Forget")
            btn.setFixedWidth(70)
            btn.clicked.connect(on_delete)
            lay.addWidget(label, 1)
            lay.addWidget(btn)
            return w

        def _add(self, listw, text, on_delete):
            item = QtWidgets.QListWidgetItem(listw)
            row = self._row(text, on_delete)
            item.setSizeHint(row.sizeHint())
            listw.addItem(item)
            listw.setItemWidget(item, row)

        def refresh(self):
            n = self.store.trainable_pair_count()
            self.pairs_count.setText(f"{n} / {self.target} pairs")
            self.pairs_bar.setMaximum(self.target)
            self.pairs_bar.setValue(min(n, self.target))
            if n >= self.target:
                self.pairs_hint.setText("Ready to fine-tune your voice.")
            else:
                self.pairs_hint.setText(
                    f"{self.target - n} more corrections with “what you actually "
                    "said” to reach the fine-tuning target.")
            self.corr_list.clear()
            self.vocab_list.clear()
            for e in reversed(self.store.corrections(n=None)):
                ts = e.get("ts")
                text = f"“{e.get('raw','')}”  →  “{e.get('ideal','')}”"
                self._add(self.corr_list, text,
                          lambda _=False, ts=ts: self._forget_corr(ts))
            for term in self.store.learned_vocab():
                self._add(self.vocab_list, term,
                          lambda _=False, t=term: self._forget_vocab(t))
            if self.corr_list.count() == 0:
                self.corr_list.addItem("No corrections yet.")
            if self.vocab_list.count() == 0:
                self.vocab_list.addItem("No learned words yet.")

        def _forget_corr(self, ts):
            self.store.delete_correction(ts)
            self.refresh()

        def _forget_vocab(self, term):
            self.store.remove_vocab(term)
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
                    p.fillPath(path, QtGui.QColor(*ACCENT))
                else:
                    pen = QtGui.QPen(QtGui.QColor(74, 74, 84))
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
            self.setFixedWidth(360)
            self._bar = bar
            self.req = req
            self.raw = str(req.get("raw", ""))
            self.cleaned = str(req.get("cleaned", ""))
            self.answered = False
            self._expanded = False

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
            self._reposition()

        # -- geometry ------------------------------------------------------
        def _reposition(self):
            self.adjustSize()
            g = self._bar.geometry()
            self.move(g.center().x() - self.width() // 2,
                      g.top() - self.height() - 8)

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
            ed.setFixedHeight(fm.lineSpacing() * 2 + 18)
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
            title = QtWidgets.QLabel("Teach Speak Easy")
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
            tag_row = QtWidgets.QHBoxLayout()
            tag_row.setSpacing(6)
            self._tag_btns = {}
            for t in FEEDBACK_TAGS:
                b = QtWidgets.QPushButton(t)
                b.setObjectName("tag")
                b.setCheckable(True)
                b.setCursor(QtCore.Qt.PointingHandCursor)
                b.setFocusPolicy(QtCore.Qt.NoFocus)
                self._tag_btns[t] = b
                tag_row.addWidget(b)
            tag_row.addStretch(1)
            self._root.addLayout(tag_row)

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
            self._reposition()
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
