"""Integration seam: real StreamingSession driving a real LiveCleanup.

The unit tests for LiveCleanup exercise it against a hand-written
FakeSession; this test wires the REAL StreamingSession (which owns the
_emitted cursor and the stable/remaining split) to the REAL LiveCleanup, with
only the Transcriber and Cleaner faked. It proves the mid-hold commit ->
stable_sentences -> finish -> remaining_sentences handoff loses and
duplicates nothing.
"""

import numpy as np

from app.config import CleanupConfig
from app.live_cleanup import LiveCleanup
from app.streaming import StreamingSession

SR = 16000


def _audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


class FakeTranscriber:
    """Scripted segment responses (see tests/test_streaming.py)."""

    def __init__(self, segment_script=None):
        self.segment_script = list(segment_script or [])
        self.segment_calls = []

    def transcribe_segments(self, audio, initial_prompt=None):
        self.segment_calls.append((len(audio), initial_prompt))
        return self.segment_script.pop(0) if self.segment_script else []


class EchoCleaner:
    """Echoes the model text back unchanged (stripped), so the assembled
    output reflects the raw session text exactly and any loss/duplication
    in the streaming<->live-cleanup seam is directly visible."""

    def __init__(self):
        self.cfg = CleanupConfig()
        self.calls = []

    def clean(self, model_text, fallback_text=None, context=None,
              surrounding=None, reformat=True):
        self.calls.append(model_text)
        return model_text.strip()


def test_streaming_session_to_live_cleanup_seam_loses_nothing():
    # Mid-hold pass: two segments with a 0.9s gap between them, which is a
    # 'period' boundary (>= PERIOD_GAP_S) -> the first sentence becomes
    # complete and stable while still holding the key; the second segment
    # is the still-mutable last part.
    fake = FakeTranscriber(
        segment_script=[
            [(0.0, 2.0, "we shipped the release"), (2.9, 4.4, "docs are next week")],
            # Tail (after finish()): continues the second sentence with a
            # short (comma-range) pause, so it merges rather than splitting.
            [(0.3, 1.0, "before the deadline")],
        ]
    )
    session = StreamingSession(
        fake, lambda: _audio(6.0), SR, window_seconds=3.0, margin_seconds=1.2
    )

    session._pass_once()  # commits both segments (pass1 buffer ends at 4.4s)
    assert session._parts == ["we shipped the release", "docs are next week"]
    assert session._boundaries == ["period"]

    session._thread.start()

    cleaner = EchoCleaner()
    live = LiveCleanup(
        session, cleaner,
        context_provider=lambda: None,
        surrounding_provider=lambda: None,
    )
    live._poll_once()  # cleans the mid-hold-committed first sentence
    assert cleaner.calls == ["we shipped the release"]
    live._thread.start()

    raw = session.finish(_audio(7.0), "model")
    out = live.finalize(raw)

    expected = "we shipped the release docs are next week before the deadline"
    assert out == expected
    # No loss or duplication across the handoff.
    assert out.count("we shipped the release") == 1
    assert out.count("docs are next week") == 1
    assert out.count("before the deadline") == 1
    assert out.index("we shipped the release") < out.index("docs are next week")
    assert out.index("docs are next week") < out.index("before the deadline")
