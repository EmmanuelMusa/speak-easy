# Feedback Panel Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the thin thumbs-up/down feedback panel with a progressive rating-plus-correction UI that captures richer training data (1–5 stars, failure tags, true-transcript, ideal-cleanup) and wires the new true-transcript into vocabulary learning.

**Architecture:** A collapsed rating strip (full text, vector stars, no timeout) that expands into a four-field teaching form on "Correct it". The overlay child (`overlay_ui.py`) renders it and emits `{rating, transcript, ideal, tags}`; the parent (`overlay.py`) forwards to `hotkey.py` which stores it via `TrainingStore.record`, now mining STT-misheard terms from the true transcript.

**Tech Stack:** Python 3 (stdlib `json`, `math`), PySide6 (Qt overlay subprocess), pytest.

## Global Constraints

- No new third-party dependencies — stdlib + existing packages only.
- Tests run with `.venv/Scripts/python.exe -m pytest` (Windows, Git Bash).
- This IS a git repository on branch `dev`. Commit each task (targeted `git add` of only the files changed). Commit messages: NO `Co-Authored-By` trailer, NO AI-authorship attribution.
- There is an unrelated uncommitted `config.toml` change (a hotkey binding) — never stage or commit it.
- Backward compatibility: existing `training_data.jsonl` lines (old `{ts, raw, output, verdict, ideal}` schema) must still load; the new schema is a superset.
- Vector icons only — no emoji anywhere in the UI.
- Baseline before Task 1: full suite **124 passing**.

**Reference spec:** `docs/superpowers/specs/2026-07-14-feedback-panel-design.md`

---

## Task 1: TrainingStore — rich `record`, STT-mishear vocab, `corrections` by ideal

**Files:**
- Modify: `app/training.py` (`record`, `corrections`)
- Test: `tests/test_training.py`

**Interfaces:**
- Produces: `TrainingStore.record(raw, output, verdict, ideal=None, *, rating=None, transcript=None, tags=None)` — writes `{ts, raw, output, rating, transcript, ideal, tags, verdict}`. Mines cleanup terms from `(output → ideal)` and STT-misheard terms from `(raw → transcript)`.
- `TrainingStore.corrections(n=5)` filters entries that carry an `ideal`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_training.py`:

```python
def test_record_persists_rating_transcript_tags(tmp_path):
    store = make_store(tmp_path)
    store.record("raw truth", "Out.", "bad", "Ideal.", rating=2,
                 transcript="raw truth fixed", tags=["wrong punctuation", "misheard word"])
    e = store._all_entries()[-1]
    assert e["rating"] == 2
    assert e["transcript"] == "raw truth fixed"
    assert e["tags"] == ["wrong punctuation", "misheard word"]
    assert e["ideal"] == "Ideal."


def test_tags_default_to_empty_and_optional_fields_null(tmp_path):
    store = make_store(tmp_path)
    store.record("raw", "Out.", "ok", rating=5)
    e = store._all_entries()[-1]
    assert e["tags"] == []
    assert e["rating"] == 5
    assert e["transcript"] is None


def test_stt_mishear_vocab_mined_from_transcript(tmp_path):
    store = make_store(tmp_path)
    # STT misheard the name; the true transcript carries the right spelling.
    store.record("meet with mr ogi up", "Meet with Mr Ogi up.", "bad",
                 ideal=None, transcript="meet with Mr Ogiop")
    assert "Ogiop" in store.learned_vocab()


def test_corrections_filter_by_ideal_not_verdict(tmp_path):
    store = make_store(tmp_path)
    store.record("a", "A.", "ok", rating=5)              # no ideal -> not a correction
    store.record("b", "B.", "bad", "Bravo.", rating=2)   # ideal -> correction
    corr = store.corrections(n=None)
    assert len(corr) == 1
    assert corr[0]["ideal"] == "Bravo."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_training.py::test_record_persists_rating_transcript_tags tests/test_training.py::test_stt_mishear_vocab_mined_from_transcript -v`
Expected: FAIL (`record()` got an unexpected keyword argument `rating`).

- [ ] **Step 3: Rewrite `record` and `corrections`**

In `app/training.py`, replace the `record` method with:

```python
    def record(self, raw: str, output: str, verdict: str, ideal: str | None = None,
               *, rating: int | None = None, transcript: str | None = None,
               tags: list[str] | None = None) -> None:
        """Append one feedback entry and mine vocabulary. `verdict` is kept for
        backward compatibility; `rating` (1-5), `transcript` (what the user
        actually said) and `tags` are the richer signal. Mines TWO diffs: the
        cleanup fix (output -> ideal) and the STT mishear (raw -> transcript),
        both through the gated _mine_vocab so only genuine terms are learned."""
        entry = {
            "ts": time.time(),
            "raw": raw,
            "output": output,
            "rating": rating,
            "transcript": transcript or None,
            "ideal": ideal or None,
            "tags": list(tags) if tags else [],
            "verdict": verdict,          # "ok" | "bad" (derived, kept for compat)
        }
        with self._lock:
            with open(self.data_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        learned: list[str] = []
        if ideal and ideal.strip() and ideal.strip() != output.strip():
            learned += self._mine_vocab(output, ideal)
        if transcript and transcript.strip() and transcript.strip() != raw.strip():
            learned += self._mine_vocab(raw, transcript)
        if learned:
            self._add_vocab(learned)
            log.info("Learned vocabulary: %s", learned)
```

Replace the `corrections` method body's filter:

```python
    def corrections(self, n: int | None = 5) -> list[dict]:
        """Feedback entries that carry an ideal-cleanup correction (most recent
        `n`, or all if n is None)."""
        found = [e for e in self._all_entries() if e.get("ideal")]
        return found if n is None else found[-n:]
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_training.py -v`
Expected: PASS (4 new tests plus existing storage/vocab/few-shot tests — `test_record_and_corrections`, `test_delete_correction_removes_lesson`, `test_vocab_mining_learns_substituted_words`, etc. stay green: they pass `ideal` on their "bad" entries, so the ideal-based filter keeps them as corrections).

- [ ] **Step 5: Commit**

```bash
git add app/training.py tests/test_training.py
git commit -m "feat(training): rich record schema + STT-mishear vocab mining"
```

---

## Task 2: Overlay — send raw+cleaned, single-slot pending, new on_feedback shape

**Files:**
- Modify: `app/overlay.py` (`__init__`, `request_feedback`, `_dispatch`)
- Test: `tests/test_training.py` (the overlay-dispatch tests)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Overlay.on_feedback(raw, output, rating, transcript, ideal, tags)` callback shape; child protocol `feedback {"id", "raw", "cleaned"}` (parent→child) and `{"type":"feedback","id","rating","transcript","ideal","tags"}` (child→parent).

- [ ] **Step 1: Rewrite the failing tests**

In `tests/test_training.py`, replace `test_feedback_event_dispatch` and `test_feedback_timeout_not_recorded` with:

```python
def test_feedback_event_dispatch():
    ov = Overlay(enabled=False)
    got = {}
    ov.on_feedback = lambda raw, out, rating, transcript, ideal, tags: got.update(
        raw=raw, out=out, rating=rating, transcript=transcript, ideal=ideal, tags=tags
    )
    ov._pending = (1, "raw text", "typed text")
    ov._dispatch({"type": "feedback", "id": 1, "rating": 2,
                  "transcript": "raw truth", "ideal": "better text",
                  "tags": ["wrong punctuation"]})
    assert got == {"raw": "raw text", "out": "typed text", "rating": 2,
                   "transcript": "raw truth", "ideal": "better text",
                   "tags": ["wrong punctuation"]}
    assert ov._pending is None


def test_feedback_rating_only_dispatch():
    ov = Overlay(enabled=False)
    seen = []
    ov.on_feedback = lambda *a: seen.append(a)
    ov._pending = (3, "raw", "out")
    ov._dispatch({"type": "feedback", "id": 3, "rating": 5,
                  "transcript": None, "ideal": None, "tags": []})
    assert seen == [("raw", "out", 5, None, None, [])]


def test_feedback_dismiss_not_recorded():
    ov = Overlay(enabled=False)
    calls = []
    ov.on_feedback = lambda *a: calls.append(a)
    ov._pending = (2, "raw", "out")
    ov._dispatch({"type": "feedback", "id": 2, "rating": None,
                  "transcript": None, "ideal": None, "tags": []})
    assert calls == []
    assert ov._pending is None  # slot cleared on the matching id


def test_feedback_stale_id_ignored():
    ov = Overlay(enabled=False)
    calls = []
    ov.on_feedback = lambda *a: calls.append(a)
    ov._pending = (5, "raw", "out")
    ov._dispatch({"type": "feedback", "id": 4, "rating": 3})  # mismatched id
    assert calls == []
    assert ov._pending == (5, "raw", "out")  # untouched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_training.py -k feedback -v`
Expected: FAIL (`_pending` is a dict; `on_feedback` called with 4 args).

- [ ] **Step 3: Update `app/overlay.py`**

In `__init__`, replace the pending line and the callback comment:

```python
        self._feedback_id = 0
        self._pending: tuple[int, str, str] | None = None  # (id, raw, output)
        #: callbacks the app assigns
        self.on_settings = None   # fn(values: dict)
        self.on_feedback = None   # fn(raw, output, rating, transcript, ideal, tags)
```

Replace `request_feedback`:

```python
    def request_feedback(self, raw: str, output: str) -> None:
        """Show the training-mode feedback panel for the latest dictation.
        Only one panel is live at a time, so a single pending slot suffices."""
        if not self.enabled:
            return
        self._ensure_proc()
        self._feedback_id += 1
        self._pending = (self._feedback_id, raw, output)
        self._send(
            "feedback "
            + json.dumps({"id": self._feedback_id, "raw": raw, "cleaned": output})
        )
```

Replace the `feedback` branch of `_dispatch`:

```python
        elif kind == "feedback":
            pending = self._pending
            if pending is None or pending[0] != event.get("id"):
                return  # stale/superseded id — ignore
            self._pending = None
            # An answer carries a rating or an ideal correction; a dismiss
            # (Cancel / closed unanswered) carries neither and is not recorded.
            if (event.get("rating") is not None or event.get("ideal")) and self.on_feedback:
                _id, raw, output = pending
                self.on_feedback(
                    raw, output, event.get("rating"), event.get("transcript"),
                    event.get("ideal"), event.get("tags") or [],
                )
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_training.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/overlay.py tests/test_training.py
git commit -m "feat(overlay): richer feedback protocol + single-slot pending"
```

---

## Task 3: Hotkey — new `_record_feedback` signature

**Files:**
- Modify: `app/hotkey.py` (`_record_feedback`)

**Interfaces:**
- Consumes: `Overlay.on_feedback(raw, output, rating, transcript, ideal, tags)`, `TrainingStore.record(..., rating=, transcript=, tags=)`.

This task wires the parent callback to storage. It has no unit test (constructing `PushToTalkApp` pulls in audio/STT hardware); the protocol is covered by Task 2's dispatch tests and the Task 5 manual run.

- [ ] **Step 1: Replace `_record_feedback`**

In `app/hotkey.py`, replace the method:

```python
    def _record_feedback(self, raw: str, output: str, rating, transcript,
                         ideal, tags) -> None:
        # verdict retained only for the stored schema / legacy few-shot.
        verdict = "ok" if (rating == 5 and not ideal) else "bad"
        self.training.record(
            raw, output, verdict, ideal,
            rating=rating, transcript=transcript, tags=tags,
        )
        if not ideal:
            log.info("Feedback: rating %s%s", rating,
                     f", tags {tags}" if tags else "")
            return
        log.info("Correction saved (rating %s)", rating)
        # The corrected text is what should inform the next dictation.
        self.context.replace_last(ideal)
        if self.cfg.training.replace_on_correction:
            replaced = self.injector.replace_last(ideal)
            log.info(
                "In-place correction: %s",
                "applied" if replaced else "skipped (text changed)",
            )
```

- [ ] **Step 2: Run the full suite (nothing should regress)**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS (128 tests — 124 baseline + Task 1's 4 new, with Task 2 net-neutral on count after replacing 2 tests with 4). Confirm no failures; the exact count is informational.

- [ ] **Step 3: Commit**

```bash
git add app/hotkey.py
git commit -m "feat(hotkey): store rating/tags/transcript from feedback"
```

---

## Task 4: Overlay UI — progressive FeedbackPanel with vector stars + tags

**Files:**
- Modify: `app/overlay_ui.py` (remove `FEEDBACK_TIMEOUT_MS`; add `FEEDBACK_TAGS`, `FEEDBACK_QSS`; add `_star_path` + `StarBar`; rewrite `FeedbackPanel`; extend the `selftest` command)
- Test: `tests/test_overlay_ui.py` (create)

**Interfaces:**
- Consumes: parent sends `feedback {"id","raw","cleaned"}`.
- Produces: emits `{"type":"feedback","id","rating","transcript","ideal","tags"}`.

Note: the widget classes are nested inside `main()` in this file (existing pattern); keep them there. `import math` and `emit()` already exist at module scope.

- [ ] **Step 1: Write the failing test**

Create `tests/test_overlay_ui.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_overlay_ui.py -v`
Expected: FAIL — the current `selftest` does not build `FeedbackPanel`, and once it does (Step 3) the new panel must render. (Before Step 3 this passes trivially, so this test is meaningful only after Step 3 extends `selftest`; if it passes now, proceed — Step 3 makes it exercise the new panel.)

- [ ] **Step 3: Implement in `app/overlay_ui.py`**

(a) Remove the timeout constant. Delete this line near the top:

```python
FEEDBACK_TIMEOUT_MS = 6000
```

(b) Add module-level constants after `DELIVERY = [...]`:

```python
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
```

(c) Inside `main()`, next to `_gear_path`, add the star path helper:

```python
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
```

(d) Inside `main()`, before the `FeedbackPanel` class, add the star widget:

```python
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
```

(e) Replace the entire `FeedbackPanel` class with:

```python
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
```

(f) Extend the `selftest` command handler (inside `Bar.handle`) to also build the feedback panel in both states:

```python
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
```

- [ ] **Step 4: Run the test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_overlay_ui.py -v`
Expected: PASS (`selftest_ok` emitted; no `selftest_err`). If PySide6 is unavailable the test skips.

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/overlay_ui.py tests/test_overlay_ui.py
git commit -m "feat(overlay-ui): progressive feedback panel, vector stars, tags, no timeout"
```

---

## Task 5: Manual verification

**Files:** none (verification only).

- [ ] **Step 1: Confirm training mode is on**

Check `config.toml` has `[training] enabled = true`. If not, set it (do not commit that change).

- [ ] **Step 2: Run the app and exercise the panel**

Run: `.venv/Scripts/python.exe -m app`
Then, holding the push-to-talk key, dictate a short phrase. When the collapsed strip appears:
- Confirm: full cleaned text shown (not truncated), five vector **stars** (no emoji), a "Correct it ›" link, and it does **not** disappear on its own (no timeout).
- Tap a star → confirm the panel closes and the log prints `Feedback: rating N`.
- Dictate again; click **Correct it ›** → confirm it expands to Heard / Cleaned (read-only) / What you actually said / Ideal cleanup (editable) / the five tag chips (several selectable at once) / Save. Edit the ideal, pick two tags, set stars, **Save**.

- [ ] **Step 3: Confirm the data landed**

Run: `tail -n 2 training_data.jsonl`
Expected: the last line has `rating`, `tags` (your two tags), and `ideal` (your edit). If you also edited "what you actually said", `transcript` is set and any genuinely misheard distinctive term appears in `learned_vocab.json`.

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS.

---

## Self-Review (completed by plan author)

- **Spec coverage:** progressive collapsed→expanded panel (Task 4); 1–5 vector stars (Task 4 `StarBar`/`_star_path`); multi-select tags (Task 4 checkable `#tag` buttons); four fields Heard/Cleaned/Actual/Ideal (Task 4 `_expand`); full text / no truncation (Task 4 word-wrapped `preview`/`ro`); no timeout (Task 4 removes `FEEDBACK_TIMEOUT_MS`); no-focus-steal until expand (Task 4 flags); data schema rating/transcript/tags + derived verdict (Task 1 `record`); STT-mishear vocab from transcript (Task 1); corrections by ideal (Task 1); protocol raw+cleaned + single-slot pending + on_feedback shape (Task 2); hotkey wiring + derived verdict (Task 3); tests (Tasks 1, 2, 4). All spec sections mapped.
- **Type consistency:** `record(raw, output, verdict, ideal=None, *, rating, transcript, tags)`, `corrections(n)`, `on_feedback(raw, output, rating, transcript, ideal, tags)`, `request_feedback(raw, output)` → `{"id","raw","cleaned"}`, child emits `{"rating","transcript","ideal","tags"}`, `StarBar.rating()/setRating()/rated`, `FEEDBACK_TAGS` — consistent across Tasks 1–4.
- **Placeholder scan:** none — every code and test step is complete.
- **Known non-tested surface:** `_record_feedback` (Task 3) and the pixel-level panel look are not unit-tested (Qt UI nested in `main()`, matching the existing codebase pattern); they are covered by Task 2's dispatch tests, Task 4's headless `selftest` render smoke, and Task 5's manual run.
