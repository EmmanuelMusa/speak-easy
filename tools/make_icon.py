"""Bake assets/speakeasy.ico from the vector mark in app/logo.py.

Run after changing the mark:  python tools/make_icon.py

Windows picks a different bitmap per context — 16px in the tray, 32px in the
taskbar, 256px in the file dialog — so the icon carries a purpose-drawn image
at each size rather than one big one it has to scale down.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6 import QtCore, QtGui  # noqa: E402

from app import logo  # noqa: E402

SIZES = (16, 20, 24, 32, 48, 64, 128, 256)
OUT = Path(__file__).resolve().parent.parent / "assets" / "speakeasy.ico"


def main() -> int:
    QtGui.QGuiApplication([])  # a QPainter needs an application instance
    pngs = []
    for size in SIZES:
        buf = QtCore.QBuffer()
        buf.open(QtCore.QIODevice.WriteOnly)
        logo.render(size).save(buf, "PNG")
        pngs.append(bytes(buf.data()))

    # ICONDIR: reserved=0, type=1 (icon), image count.
    out = struct.pack("<HHH", 0, 1, len(pngs))
    offset = 6 + 16 * len(pngs)
    for size, data in zip(SIZES, pngs):
        px = size if size < 256 else 0  # 0 encodes 256 in this format
        out += struct.pack("<BBBBHHII", px, px, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    out += b"".join(pngs)

    OUT.write_bytes(out)
    print(f"wrote {OUT.relative_to(OUT.parent.parent)} "
          f"({len(out):,} bytes, sizes {', '.join(map(str, SIZES))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
