"""Create or repair the Start Menu shortcut.

    python tools/install_shortcut.py            # install / repair
    python tools/install_shortcut.py --console  # launch with a visible console

A shortcut stores its own name, icon and taskbar identity, so none of the
branding the app does at runtime reaches one that already exists. A shortcut
made by hand off pythonw.exe shows the Python logo and whatever the .lnk file
happens to be called. This writes all three:

  filename  -> what Start Menu search displays ("SpeakEasy")
  icon      -> assets/speakeasy.ico rather than the interpreter's
  AppUserModelID -> matches app.branding.APP_ID, so a pinned shortcut groups
                    with the running app instead of docking beside it

Idempotent: run it again after moving the project and it repoints everything.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import branding  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
START_MENU = Path(os.path.expandvars(
    r"%APPDATA%\Microsoft\Windows\Start Menu\Programs"))
LINK = START_MENU / f"{branding.APP_NAME}.lnk"


def _venv_python(console: bool) -> Path:
    """pythonw.exe runs with no console — right for a tray app. python.exe
    keeps one, for when you want to watch the logs."""
    exe = "python.exe" if console else "pythonw.exe"
    venv = ROOT / ".venv" / "Scripts" / exe
    return venv if venv.exists() else Path(sys.executable).with_name(exe)


def _write_shortcut(target: Path, console: bool) -> None:
    from win32com.client import Dispatch

    shell = Dispatch("WScript.Shell")
    link = shell.CreateShortCut(str(LINK))
    link.TargetPath = str(target)
    link.Arguments = "-m app"
    link.WorkingDirectory = str(ROOT)
    link.IconLocation = f"{branding.ICO_PATH},0"
    link.Description = f"{branding.APP_NAME} - local push-to-talk dictation"
    link.WindowStyle = 7 if console else 1  # 7 = start minimised
    link.Save()


def _set_app_id() -> bool:
    """Stamp the taskbar identity onto the .lnk. Explorer indexes a shortcut
    the moment it is written and briefly holds it open, so this retries."""
    from win32com.propsys import propsys, pscon

    GPS_READWRITE = 2
    for _ in range(8):
        try:
            store = propsys.SHGetPropertyStoreFromParsingName(
                str(LINK), None, GPS_READWRITE, propsys.IID_IPropertyStore)
            store.SetValue(pscon.PKEY_AppUserModel_ID,
                           propsys.PROPVARIANTType(branding.APP_ID))
            store.Commit()
            return True
        except Exception:
            time.sleep(1.0)
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--console", action="store_true",
                    help="launch with a visible console instead of silently")
    args = ap.parse_args(argv)

    if sys.platform != "win32":
        print("Windows only.", file=sys.stderr)
        return 1
    if not branding.ICO_PATH.exists():
        print(f"error: {branding.ICO_PATH} missing - run tools/make_icon.py first",
              file=sys.stderr)
        return 2

    # Clear out any older shortcut under a different name, or Start Menu search
    # keeps offering both.
    for stale in START_MENU.glob("*.lnk"):
        if stale != LINK and stale.stem.replace(" ", "").lower() == "speakeasy":
            stale.unlink()
            print(f"removed stale shortcut: {stale.name}")

    target = _venv_python(args.console)
    if not target.exists():
        print(f"error: no interpreter at {target}", file=sys.stderr)
        return 2

    START_MENU.mkdir(parents=True, exist_ok=True)
    _write_shortcut(target, args.console)
    ok = _set_app_id()

    print(f"shortcut : {LINK}")
    print(f"target   : {target} -m app")
    print(f"icon     : {branding.ICO_PATH}")
    print(f"app id   : {branding.APP_ID}" if ok else
          "app id   : could not be set (a pinned copy may not group; harmless)")
    # Plain ASCII: a Windows console defaults to cp1252 and mangles anything else.
    print("\nStart Menu search caches aggressively - the old name and icon can "
          "linger for a few minutes, or until Explorer restarts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
