# Audio Capture Implementation Plan (Sub-project A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a feedback correction includes a verbatim `transcript`, save that dictation's audio as a WAV and link it to the training-log entry, producing verified `(audio, transcript)` acoustic-training pairs.

**Architecture:** `TrainingStore` gains a `save_audio` that writes a 16-bit PCM mono WAV (stdlib `wave`) into a gitignored `training_audio/` dir and a `record(audio_path=…)` field; the hotkey stashes each dictation's audio and writes it only when the submitted feedback carries a `transcript`.

**Tech Stack:** Python 3 stdlib (`wave`); numpy ndarray methods (already the audio type); no new dependencies.

## Global Constraints

- No new third-party dependencies — stdlib + the existing numpy ndarray only.
- Tests run with `.venv/Scripts/python.exe -m pytest` (Windows, Git Bash).
- This IS a git repository on branch `dev`. Commit each task (targeted `git add` of only the files changed). Commit messages: NO `Co-Authored-By` trailer, NO AI-authorship attribution.
- Backward compatibility: old `training_data.jsonl` lines (no `audio` field) still load.
- Audio is saved ONLY for dictations corrected with a non-empty `transcript`; a failed save must never break dictation.
- Baseline before Task 1: full suite **141 passing**.

**Reference spec:** `docs/superpowers/specs/2026-07-14-audio-capture-design.md`

---

## Task 1: TrainingStore — `save_audio` + `record(audio_path)`

**Files:**
- Modify: `app/training.py` (imports; `AUDIO_DIR`; `__init__`; `record`; add `save_audio`)
- Test: `tests/test_training.py` (update `make_store`; add tests)

**Interfaces:**
- Produces: `TrainingStore(data_path=…, vocab_path=…, audio_dir=AUDIO_DIR)`;
  `save_audio(audio, sample_rate) -> str | None` (project-root-relative POSIX path like `training_audio/<ts>.wav`, or `None`);
  `record(..., audio_path: str | None = None)` writing an `"audio"` field.

- [ ] **Step 1: Write the failing tests**

At the top of `tests/test_training.py`, add these imports (after the existing `import json`):

```python
import wave

import numpy as np
```

Update the `make_store` helper to use a temp audio dir:

```python
def make_store(tmp_path):
    return TrainingStore(
        data_path=tmp_path / "data.jsonl", vocab_path=tmp_path / "vocab.json",
        audio_dir=tmp_path / "training_audio",
    )
```

Append the new tests:

```python
def test_save_audio_writes_valid_wav(tmp_path):
    store = make_store(tmp_path)
    audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 0.25], dtype=np.float32)
    rel = store.save_audio(audio, 16000)
    assert rel and rel.startswith("training_audio/") and rel.endswith(".wav")
    wav_path = store.audio_dir.parent / rel
    assert wav_path.exists()
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() == len(audio)


def test_save_audio_empty_returns_none(tmp_path):
    store = make_store(tmp_path)
    assert store.save_audio(np.array([], dtype=np.float32), 16000) is None


def test_record_stores_audio_path(tmp_path):
    store = make_store(tmp_path)
    store.record("raw", "Out.", "bad", "Ideal.", transcript="raw truth",
                 audio_path="training_audio/123.wav")
    assert store._all_entries()[-1]["audio"] == "training_audio/123.wav"


def test_record_without_audio_path_is_null(tmp_path):
    store = make_store(tmp_path)
    store.record("raw", "Out.", "ok", rating=5)
    assert store._all_entries()[-1]["audio"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_training.py -k "save_audio or audio_path or without_audio" -v`
Expected: FAIL (`TrainingStore.__init__` has no `audio_dir`; no `save_audio`; no `audio` field).

- [ ] **Step 3: Implement in `app/training.py`**

Add `import wave` to the stdlib imports (e.g. right after `import time`).

Add the audio-dir constant next to the other path constants (after `VOCAB_PATH`):

```python
AUDIO_DIR = _ROOT / "training_audio"
```

Replace `TrainingStore.__init__` with:

```python
    def __init__(self, data_path: Path = DATA_PATH, vocab_path: Path = VOCAB_PATH,
                 audio_dir: Path = AUDIO_DIR):
        self.data_path = data_path
        self.vocab_path = vocab_path
        self.audio_dir = audio_dir
        self._lock = threading.Lock()
```

Add the `"audio"` field to the entry dict inside `record` (between `"tags"` and
`"verdict"`), and add the `audio_path` keyword parameter to its signature:

```python
    def record(self, raw: str, output: str, verdict: str, ideal: str | None = None,
               *, rating: int | None = None, transcript: str | None = None,
               tags: list[str] | None = None, audio_path: str | None = None) -> None:
```

```python
            "tags": list(tags) if tags else [],
            "audio": audio_path,
            "verdict": verdict,
```

Add the `save_audio` method (near `record`):

```python
    def save_audio(self, audio, sample_rate: int) -> str | None:
        """Write `audio` (a float32 mono ndarray in [-1, 1]) as a 16-bit PCM mono
        WAV under audio_dir; return its path relative to audio_dir.parent (the
        project root in production), e.g. "training_audio/<ts>.wav" — or None for
        empty audio / on any error. Never raises: a failed save must not break
        dictation."""
        try:
            if audio is None or len(audio) == 0:
                return None
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            name = f"{int(time.time() * 1000)}.wav"
            path = self.audio_dir / name
            pcm = (audio.clip(-1.0, 1.0) * 32767.0).astype("<i2")
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(int(sample_rate))
                wf.writeframes(pcm.tobytes())
            return f"{self.audio_dir.name}/{name}"
        except Exception as exc:
            log.warning("Could not save correction audio (%s)", exc)
            return None
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_training.py -v`
Expected: PASS (4 new tests plus all existing — `record`'s new `audio` field is additive and the `make_store` change is transparent to tests that don't use audio).

- [ ] **Step 5: Commit**

```bash
git add app/training.py tests/test_training.py
git commit -m "feat(training): save_audio + audio path on correction entries"
```

---

## Task 2: Config toggle + hotkey capture + gitignore

**Files:**
- Modify: `app/config.py` (`TrainingConfig`)
- Modify: `config.toml` (`[training]`)
- Modify: `app/hotkey.py` (`__init__`, `_process`, `_record_feedback`)
- Modify: `.gitignore`
- Test: `tests/test_training.py` (hotkey wiring)

**Interfaces:**
- Consumes: `TrainingStore.save_audio(audio, sample_rate)`, `record(..., audio_path=)`.
- Produces: `TrainingConfig.save_correction_audio: bool = True`; `PushToTalkApp._last_audio` stash.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_training.py`:

```python
def test_record_feedback_saves_audio_on_transcript():
    from app.hotkey import PushToTalkApp
    fake = MagicMock()
    fake.cfg.training.save_correction_audio = True
    fake.cfg.training.replace_on_correction = False
    fake._last_audio = ("AUDIO", 16000)
    fake.training.save_audio.return_value = "training_audio/1.wav"
    PushToTalkApp._record_feedback(fake, "raw", "out", 3, "raw true", None, [])
    fake.training.save_audio.assert_called_once_with("AUDIO", 16000)
    assert fake.training.record.call_args.kwargs["audio_path"] == "training_audio/1.wav"
    assert fake._last_audio is None  # stash cleared


def test_record_feedback_no_audio_without_transcript():
    from app.hotkey import PushToTalkApp
    fake = MagicMock()
    fake.cfg.training.save_correction_audio = True
    fake.cfg.training.replace_on_correction = False
    fake._last_audio = ("AUDIO", 16000)
    PushToTalkApp._record_feedback(fake, "raw", "out", 5, None, None, [])
    fake.training.save_audio.assert_not_called()
    assert fake.training.record.call_args.kwargs["audio_path"] is None
    assert fake._last_audio is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_training.py -k record_feedback -v`
Expected: FAIL — the current `_record_feedback` neither reads `_last_audio`/`save_correction_audio` nor passes `audio_path` to `record`.

- [ ] **Step 3: Add the config field**

In `app/config.py`, inside `@dataclass class TrainingConfig`, add after `replace_on_correction`:

```python
    # When a correction includes what you actually said, also save that
    # dictation's audio (into training_audio/, gitignored) as a training pair
    # for a future speech-model fine-tune. Off = text-only training / privacy.
    save_correction_audio: bool = True
```

In `config.toml`, inside `[training]`, add after the `replace_on_correction` line:

```toml
# When a correction includes what you actually said, also save that dictation's
# audio (into training_audio/, gitignored) so a later run can fine-tune the
# speech model to your voice. Turn off for text-only training / privacy.
save_correction_audio = true
```

- [ ] **Step 4: Add the gitignore entry**

In `.gitignore`, add a line:

```
training_audio/
```

- [ ] **Step 5: Wire the hotkey**

In `app/hotkey.py` `PushToTalkApp.__init__`, add the stash field (next to `self._surrounding`):

```python
        self._last_audio: tuple | None = None  # (audio, sr) held for correction capture
```

In `_process`, replace the feedback request:

```python
                if self.cfg.training.enabled:
                    self.overlay.request_feedback(raw, cleaned)
```

with (stash the audio first so a later correction can save it):

```python
                if self.cfg.training.enabled:
                    self._last_audio = (audio, self.cfg.audio.sample_rate)
                    self.overlay.request_feedback(raw, cleaned)
```

Replace `_record_feedback` with (save audio BEFORE the `if not ideal` return, so a
transcript-only correction still captures its pair):

```python
    def _record_feedback(self, raw: str, output: str, rating, transcript,
                         ideal, tags) -> None:
        audio_path = None
        stash, self._last_audio = self._last_audio, None
        if transcript and self.cfg.training.save_correction_audio and stash is not None:
            audio, sr = stash
            audio_path = self.training.save_audio(audio, sr)
        # verdict retained only for the stored schema / legacy few-shot.
        verdict = "ok" if (rating == 5 and not ideal) else "bad"
        self.training.record(
            raw, output, verdict, ideal,
            rating=rating, transcript=transcript, tags=tags, audio_path=audio_path,
        )
        if not ideal:
            log.info("Feedback: rating %s%s%s", rating,
                     f", tags {tags}" if tags else "",
                     " + audio" if audio_path else "")
            return
        log.info("Correction saved (rating %s)%s", rating,
                 " + audio" if audio_path else "")
        # The corrected text is what should inform the next dictation.
        self.context.replace_last(ideal)
        if self.cfg.training.replace_on_correction:
            replaced = self.injector.replace_last(ideal)
            log.info(
                "In-place correction: %s",
                "applied" if replaced else "skipped (text changed)",
            )
```

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS (the two new hotkey tests plus everything else; the config field is additive with a default, so config/load tests are unaffected).

- [ ] **Step 7: Commit**

```bash
git add app/config.py config.toml app/hotkey.py .gitignore tests/test_training.py
git commit -m "feat(hotkey): capture correction audio for acoustic-training pairs"
```

---

## Task 3: Verification

**Files:** none (verification only).

- [ ] **Step 1: End-to-end store check (scriptable)**

Confirm the store writes a real WAV and links it, resolving the pair the way Sub-project B will:

```bash
.venv/Scripts/python.exe - <<'PY'
from app.training import TrainingStore
from pathlib import Path
import tempfile, wave, json
import numpy as np
d = Path(tempfile.mkdtemp())
s = TrainingStore(data_path=d/"data.jsonl", vocab_path=d/"vocab.json", audio_dir=d/"training_audio")
audio = np.linspace(-0.5, 0.5, 8000, dtype=np.float32)  # 0.5s @ 16k
rel = s.save_audio(audio, 16000)
s.record("ogi up", "Ogi up.", "bad", ideal=None, transcript="Ogiop", audio_path=rel)
e = s._all_entries()[-1]
print("entry audio:", e["audio"], "transcript:", e["transcript"])
wav = s.audio_dir.parent / e["audio"]         # resolves like _ROOT / entry["audio"]
with wave.open(str(wav), "rb") as wf:
    print("wav ok:", wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes())
PY
```
Expected: prints `entry audio: training_audio/<ts>.wav transcript: Ogiop` and `wav ok: 1 2 16000 8000` — a resolvable `(audio, transcript)` pair.

- [ ] **Step 2: Full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS.

- [ ] **Step 3: (User, live) GUI check**

With `[training] enabled = true` and `save_correction_audio = true`, run the app, dictate, click "Correct it", edit **What you actually said**, and Save. Confirm a new WAV appears in `training_audio/` and the last line of `training_data.jsonl` has both `audio` and `transcript` set. (Requires a mic + click-through — not automatable here.)

---

## Self-Review (completed by plan author)

- **Spec coverage:** `save_audio` (16-bit PCM mono WAV, stdlib `wave`, returns path relative to `audio_dir.parent`, `None` on empty/error) — Task 1; `record(audio_path)` `"audio"` field — Task 1; `audio_dir` param for tests — Task 1; save only when `transcript` present — Task 2 `_record_feedback`; `_last_audio` stash in `_process` (single slot) — Task 2; `save_correction_audio` config default true — Task 2; `training_audio/` gitignored — Task 2; the `(audio, transcript)` resolution contract — Task 3 check. All spec sections mapped.
- **Type consistency:** `save_audio(audio, sample_rate) -> str | None`, `record(..., audio_path=None)` with `"audio"` field, `TrainingStore(..., audio_dir=)`, `_last_audio: tuple | None`, `save_correction_audio: bool` — consistent across Tasks 1–2. `_record_feedback` saves before the `if not ideal` return so transcript-only corrections are captured.
- **Placeholder scan:** none — every code and test step is complete.
- **Non-unit-tested surface:** the live GUI capture (Task 3 Step 3) needs a mic; it's covered structurally by the `save_audio`/`record` unit tests and the `_record_feedback` MagicMock wiring tests.
