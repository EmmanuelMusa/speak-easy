"""ContextStore recency/capping and context plumbing into cleanup + streaming."""

from unittest.mock import MagicMock, patch

import numpy as np

from app.cleanup import Cleaner
from app.config import CleanupConfig, ContextConfig
from app.context import ContextStore
from app.streaming import StreamingSession


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_recent_dictations_become_cleanup_context():
    store = ContextStore(ContextConfig(), clock=FakeClock())
    store.add("We migrated the Kubernetes cluster.")
    store.add("The rollout finishes Thursday.")
    assert store.cleanup_context() == (
        "We migrated the Kubernetes cluster. The rollout finishes Thursday."
    )


def test_disabled_store_returns_none():
    store = ContextStore(ContextConfig(enabled=False))
    store.add("something")
    assert store.cleanup_context() is None


def test_context_expires():
    clock = FakeClock()
    store = ContextStore(ContextConfig(expiry_seconds=300), clock=clock)
    store.add("old thought")
    clock.now += 301
    store.add("new thought")
    assert store.cleanup_context() == "new thought"


def test_caps_chars_and_utterances():
    store = ContextStore(
        ContextConfig(max_chars=20, max_utterances=2), clock=FakeClock()
    )
    for text in ["one alpha", "two bravo", "three charlie"]:
        store.add(text)
    prompt = store.cleanup_context()
    assert "one alpha" not in prompt  # trimmed by max_utterances
    assert len(prompt) <= 20          # trimmed by max_chars, keeps the tail
    assert prompt.endswith("three charlie")


def test_correction_replaces_last_entry():
    store = ContextStore(ContextConfig(), clock=FakeClock())
    store.add("the meeting is at 9")
    store.replace_last("The meeting is at 3pm.")
    assert store.cleanup_context() == "The meeting is at 3pm."


def test_cleaner_puts_context_in_system_prompt_only():
    cfg = CleanupConfig(enabled=True)
    raw = "um the rollout finishes on thursday afternoon this week"
    fake = MagicMock()
    fake.json.return_value = {
        "response": "The rollout finishes on Thursday afternoon this week."
    }
    fake.raise_for_status.return_value = None
    with patch("app.cleanup.requests.post", return_value=fake) as mock_post:
        out = Cleaner(cfg).clean(raw, context="We migrated the Kubernetes cluster.")
    assert out == "The rollout finishes on Thursday afternoon this week."
    payload = mock_post.call_args.kwargs["json"]
    assert "Kubernetes cluster" in payload["system"]
    assert "Kubernetes" not in payload["prompt"]  # transcript stays clean


def test_streaming_prompts_contain_only_current_utterance():
    """Regression: no cross-utterance text may reach Whisper's prompt —
    prompt words can leak into the raw transcript, upstream of the guard."""
    calls = []
    script = [[(0.0, 2.0, "and the deploy")], [(0.0, 1.0, "went fine")]]

    class Fake:
        def transcribe_segments(self, audio, initial_prompt=None):
            calls.append(initial_prompt)
            return script.pop(0)

    audio = np.zeros(5 * 16000, dtype=np.float32)
    s = StreamingSession(Fake(), lambda: audio, 16000)
    s._pass_once()
    assert calls[0] is None  # first pass decodes cold: no history, ever
    s._thread.start()
    out = s.finish(np.zeros(6 * 16000, dtype=np.float32))
    # Tail pass sees only text committed within THIS utterance.
    assert calls[-1] == "and the deploy"
    assert out == "and the deploy went fine"
