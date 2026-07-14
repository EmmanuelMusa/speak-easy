"""StreamingSession: commit logic, tail stitching, and graceful degradation.

Passes are driven by calling _pass_once() directly (no background thread)
so the tests are deterministic; finish() is exercised with the thread
never started running passes, which mirrors the no-commit path, and with
pre-seeded commits for the stitching path.
"""

import numpy as np

from app.audio import Recorder
from app.config import AudioConfig
from app.streaming import StreamingSession

SR = 16000


def _audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


class FakeTranscriber:
    """Scripted segment responses; records every call it receives."""

    def __init__(self, segment_script=None):
        self.segment_script = list(segment_script or [])
        self.segment_calls = []   # (n_samples, initial_prompt)

    def transcribe_segments(self, audio, initial_prompt=None):
        self.segment_calls.append((len(audio), initial_prompt))
        return self.segment_script.pop(0) if self.segment_script else []


def _session(fake, snapshot, **kw):
    defaults = dict(window_seconds=3.0, margin_seconds=1.2)
    defaults.update(kw)
    return StreamingSession(fake, snapshot, SR, **defaults)


def test_no_pass_until_window_filled():
    fake = FakeTranscriber()
    s = _session(fake, lambda: _audio(2.0))  # < 3s window
    s._pass_once()
    assert fake.segment_calls == []


def test_commits_only_settled_segments():
    fake = FakeTranscriber(
        segment_script=[[(0.0, 2.0, "hello there"), (2.5, 4.6, "world")]]
    )
    s = _session(fake, lambda: _audio(5.0))
    s._pass_once()
    assert s._parts == ["hello there"]
    assert s._committed_samples == int(2.5 * SR)


def test_later_pass_sees_committed_text_as_prompt():
    fake = FakeTranscriber(
        segment_script=[
            [(0.0, 2.0, "first part")],
            [(0.0, 2.0, "second part")],
        ]
    )
    buf = {"dur": 5.0}
    s = _session(fake, lambda: _audio(buf["dur"]))
    s._pass_once()
    buf["dur"] = 8.0  # more speech arrives
    s._pass_once()
    # Second pass starts after the committed 2.0s and is prompted with it.
    n_samples, prompt = fake.segment_calls[1]
    assert n_samples == int(6.0 * SR)
    assert prompt == "first part"


def test_finish_stitches_commits_and_tail():
    fake = FakeTranscriber(
        segment_script=[
            [(0.0, 2.0, "hello there"), (2.5, 4.6, "world")],
            [(0.0, 1.5, "world again")],
        ]
    )
    s = _session(fake, lambda: _audio(5.0))
    s._pass_once()
    s._thread.start()
    # Fallback view keeps the 0.5s pause as a comma; model view drops it.
    assert s.finish(_audio(6.0), "pauses") == "hello there, world again"
    n_samples, prompt = fake.segment_calls[1]
    assert n_samples == int(3.5 * SR)
    assert prompt == "hello there"


def test_long_pause_becomes_full_stop_across_commits():
    fake = FakeTranscriber(
        segment_script=[
            [(0.0, 2.0, "we should ship it")],
            [(1.5, 3.0, "also the docs need a pass")],
        ]
    )
    buf = {"dur": 5.0}
    s = _session(fake, lambda: _audio(buf["dur"]))
    s._pass_once()
    buf["dur"] = 8.0
    s._pass_once()
    assert s._parts == ["we should ship it", "also the docs need a pass"]
    assert s._boundaries == ["period"]


def test_finish_without_commits_degrades_to_batch():
    fake = FakeTranscriber(segment_script=[[(0.0, 1.0, "all of it")]])
    s = _session(fake, lambda: _audio(1.0))
    s._thread.start()
    out = s.finish(_audio(1.0))
    assert out == "all of it"
    assert fake.segment_calls[0] == (int(1.0 * SR), None)


def test_failed_pass_does_not_lose_audio():
    class Flaky(FakeTranscriber):
        def transcribe_segments(self, audio, initial_prompt=None):
            if not self.segment_calls:  # the mid-hold pass dies...
                self.segment_calls.append((len(audio), initial_prompt))
                raise RuntimeError("transient")
            return super().transcribe_segments(audio, initial_prompt)

    import pytest

    fake = Flaky(segment_script=[[(0.0, 5.0, "recovered")]])
    s = _session(fake, lambda: _audio(5.0))
    with pytest.raises(RuntimeError):  # the loop swallows this in production
        s._pass_once()
    s._thread.start()
    out = s.finish(_audio(5.0))  # ...but finish still covers all the audio
    assert out == "recovered"
    assert s._committed_samples == 0  # nothing was skipped


def test_finish_survives_tail_transcription_failure():
    class DeadModel(FakeTranscriber):
        def transcribe_segments(self, audio, initial_prompt=None):
            raise RuntimeError("model gone")

    s = _session(DeadModel(), lambda: _audio(1.0))
    s._parts = ["what we already have"]
    s._thread.start()
    assert s.finish(_audio(2.0)) == "what we already have"


def test_stable_sentences_and_remaining():
    fake = FakeTranscriber()
    s = _session(fake, lambda: _audio(1.0))
    s._parts = ["we shipped it", "the docs are next", "and then"]
    s._boundaries = ["period", "none"]
    # "we shipped it" is a complete, stable sentence (period pause; not the
    # still-mutable last part). The rest is not yet stable.
    assert s.stable_sentences("model") == ["we shipped it"]
    assert s.stable_sentences("model") == []          # cursor advanced
    # finish-time: everything remaining, including the final partial part.
    assert s.remaining_sentences("model") == ["the docs are next and then"]


def test_recorder_snapshot_is_nondestructive():
    rec = Recorder(AudioConfig())
    rec._chunks = [np.ones((100, 1), dtype=np.float32)] * 3
    snap = rec.snapshot()
    assert snap.shape == (300,)
    assert len(rec._chunks) == 3  # chunks still there for stop()
