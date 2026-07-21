"""The SpeakEasy mark, drawn as vectors.

assets/logo-dark.png is the logo of record, but it carries the wordmark inside
the ring and a wide margin, so scaled to a 16px tray icon it is a smudge. This
redraws the same motif — a gradient ring broken at 3 and 9 o'clock by a
waveform running through it — at whatever size is asked for, dropping detail as
it shrinks.

Imported by the Qt overlay (which already depends on PySide6) and by
tools/make_icon.py, which bakes assets/speakeasy.ico from it.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui

#: The logo's colour sweep, sampled from the ring of assets/logo-dark.png:
#: cyan at 9 o'clock through blue and violet to pink at 3 o'clock.
BRAND_STOPS = (
    (0.00, "#00c8ff"),   # cyan
    (0.22, "#0088f8"),   # azure
    (0.42, "#2a6df4"),   # blue
    (0.58, "#5f4ff3"),   # indigo
    (0.74, "#9d2dec"),   # violet
    (0.88, "#da31ca"),   # magenta
    (1.00, "#f51c6c"),   # pink
)

# The waveform as (x fraction, amplitude in -1..1). One tall spike just left of
# centre, decaying wiggles either side, flat where it leaves through the ring's
# gaps. Three versions: below roughly 64px the fine wiggles pack closer than the
# stroke is wide and smear into a blob, and by 24px only the one spike survives.
WAVE_FULL = ((0.00, 0.00), (0.15, 0.00), (0.20, 0.20), (0.25, -0.30),
             (0.30, 0.45), (0.35, -0.25), (0.40, 0.32), (0.44, -0.55),
             (0.48, 1.00), (0.53, -0.88), (0.58, 0.38), (0.63, -0.32),
             (0.67, 0.58), (0.72, -0.42), (0.77, 0.28), (0.82, -0.16),
             (0.87, 0.10), (0.92, 0.00), (1.00, 0.00))
WAVE_MID = ((0.00, 0.00), (0.18, 0.00), (0.28, 0.34), (0.36, -0.40),
            (0.48, 1.00), (0.57, -0.80), (0.66, 0.52), (0.74, -0.30),
            (0.84, 0.00), (1.00, 0.00))
WAVE_TINY = ((0.00, 0.00), (0.26, 0.00), (0.36, -0.42), (0.48, 1.00),
             (0.60, -0.62), (0.72, 0.30), (0.82, 0.00), (1.00, 0.00))


def gradient(x0: float, y0: float, x1: float, y1: float) -> QtGui.QLinearGradient:
    """The logo's sweep as a paintable gradient between two points."""
    g = QtGui.QLinearGradient(x0, y0, x1, y1)
    for pos, hexc in BRAND_STOPS:
        g.setColorAt(pos, QtGui.QColor(hexc))
    return g


def _wave_for(d: float):
    if d >= 64:
        return WAVE_FULL
    return WAVE_MID if d >= 28 else WAVE_TINY


def draw_mark(p: QtGui.QPainter, cx: float, cy: float, d: float) -> None:
    """Paint the mark centred on (cx, cy), `d` across, on whatever is behind."""
    stroke = max(1.15, d * (0.085 if d >= 28 else 0.105))
    pen = QtGui.QPen(QtGui.QBrush(gradient(cx - d / 2, cy, cx + d / 2, cy)),
                     stroke)
    pen.setCapStyle(QtCore.Qt.RoundCap)
    pen.setJoinStyle(QtCore.Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(QtCore.Qt.NoBrush)
    r = d * 0.42
    ring = QtCore.QRectF(cx - r, cy - r, r * 2, r * 2)
    # Gaps at 3 and 9 o'clock, where the waveform runs out of the ring. They
    # widen as the mark shrinks so the join stays legible.
    gap = 12 if d >= 28 else 20
    p.drawArc(ring, gap * 16, (180 - 2 * gap) * 16)
    p.drawArc(ring, (180 + gap) * 16, (180 - 2 * gap) * 16)
    path = QtGui.QPainterPath()
    w, amp = d * 0.98, d * 0.30
    for i, (fx, fa) in enumerate(_wave_for(d)):
        pt = QtCore.QPointF(cx - w / 2 + fx * w, cy - fa * amp)
        path.moveTo(pt) if i == 0 else path.lineTo(pt)
    p.drawPath(path)


def render(size: int) -> QtGui.QImage:
    """The mark alone on transparency, at `size` px square — what an icon
    wants, with only the margin the round stroke caps need."""
    img = QtGui.QImage(size, size, QtGui.QImage.Format_ARGB32)
    img.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    draw_mark(p, size / 2, size / 2, size * 0.94)
    p.end()
    return img
