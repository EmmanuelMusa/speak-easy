"""Headless render smoke: the overlay subprocess builds every dialog —
including the redesigned FeedbackPanel (collapsed + expanded) — without error."""

import importlib.util
import json
import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(importlib.util.find_spec("PySide6") is None,
                    reason="PySide6 not installed")
def test_overlay_ui_selftest_renders_feedback_panel():
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.overlay_ui"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, encoding="utf-8", env=env,
    )
    # selftest instantiates the dialogs + feedback panel; closing stdin (EOF)
    # then lets the child quit, so communicate returns all stdout.
    out, _ = proc.communicate(input="selftest\n", timeout=60)
    msgs = [json.loads(ln) for ln in out.splitlines()
            if ln.strip().startswith("{")]
    types = [m.get("type") for m in msgs]
    assert "selftest_ok" in types, out
    assert "selftest_err" not in types, out
