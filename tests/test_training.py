"""Tests for training mode: feedback storage, few-shot injection, vocabulary
mining, overlay event dispatch, and settings persistence."""

import json
import wave

import numpy as np

from app.config import load_config, save_config_updates
from app.cleanup import Cleaner
from app.config import CleanupConfig
from app.overlay import Overlay
from app.training import TrainingStore

from unittest.mock import MagicMock, patch


def make_store(tmp_path):
    return TrainingStore(
        data_path=tmp_path / "data.jsonl", vocab_path=tmp_path / "vocab.json",
        audio_dir=tmp_path / "training_audio",
    )


# --- storage + few-shot -------------------------------------------------------

def test_record_and_corrections(tmp_path):
    store = make_store(tmp_path)
    store.record("raw one", "out one", "ok")
    store.record("walking through stats in ogi up", "Walking through stats in Ogi up.",
                 "bad", "Walking through setting it up.")
    pairs = store.corrections()
    assert len(pairs) == 1
    assert pairs[0]["ideal"] == "Walking through setting it up."


def test_few_shot_block_appears_in_prompt(tmp_path):
    store = make_store(tmp_path)
    store.record("go by 2pm no sorry 3pm", "Go by 2pm, no sorry, 3pm.",
                 "bad", "Go by 3pm.")
    cfg = CleanupConfig(enabled=True)
    cleaner = Cleaner(cfg, training=store)
    fake = MagicMock()
    fake.json.return_value = {"response": "Whatever."}
    fake.raise_for_status.return_value = None
    with patch("app.cleanup.requests.post", return_value=fake) as mock_post:
        # query shares distinctive terms with the stored correction -> retrieved
        cleaner.clean("go by 4pm no sorry 5pm")
    system = mock_post.call_args.kwargs["json"]["system"]
    assert "Go by 3pm." in system  # relevant correction injected as example


def test_unrelated_utterance_injects_no_correction(tmp_path):
    store = make_store(tmp_path)
    store.record("go by 2pm no sorry 3pm", "Go by 2pm, no sorry, 3pm.",
                 "bad", "Go by 3pm.")
    cfg = CleanupConfig(enabled=True)
    cleaner = Cleaner(cfg, training=store)
    fake = MagicMock()
    fake.json.return_value = {"response": "Whatever."}
    fake.raise_for_status.return_value = None
    with patch("app.cleanup.requests.post", return_value=fake) as mock_post:
        cleaner.clean("the weather is nice today and the sky is clear")
    system = mock_post.call_args.kwargs["json"]["system"]
    assert "Go by 3pm." not in system  # nothing similar -> no example injected


def test_delete_correction_removes_lesson(tmp_path):
    store = make_store(tmp_path)
    store.record("a", "A.", "bad", "Alpha.")
    store.record("b", "B.", "bad", "Bravo.")
    corr = store.corrections(n=None)
    assert len(corr) == 2
    store.delete_correction(corr[0]["ts"])
    remaining = store.corrections(n=None)
    assert len(remaining) == 1
    assert remaining[0]["ideal"] == "Bravo."


def test_remove_vocab_forgets_term(tmp_path):
    store = make_store(tmp_path)
    store._add_vocab(["Ogiop", "Kubernetes"])
    store.remove_vocab("ogiop")  # case-insensitive
    assert store.learned_vocab() == ["Kubernetes"]


def test_vocab_mining_learns_substituted_words(tmp_path):
    store = make_store(tmp_path)
    store.record("meet with mr ogi up tomorrow", "Meet with Mr Ogi up tomorrow.",
                 "bad", "Meet with Mr Ogiop tomorrow.")
    assert "Ogiop" in store.learned_vocab()


def test_vocab_mining_ignores_grammar_corrections(tmp_path):
    store = make_store(tmp_path)
    # A real correction from training_data.jsonl: the model prepended "But",
    # the user wanted "What". That is a rewrite, not a misheard term, and must
    # never become preserve-forever vocabulary.
    store.record("But exactly send input", "But exactly send input.",
                 "bad", "What exactly is send input?")
    assert store.learned_vocab() == []


def test_vocab_mining_ignores_plain_lowercase_reword(tmp_path):
    store = make_store(tmp_path)
    # Ordinary lowercase word swap (tense/word-choice) — not vocabulary.
    store.record("we should handle it", "We should handle it.",
                 "bad", "We should manage it.")
    assert store.learned_vocab() == []


def test_vocab_mining_learns_phonetic_mishear_with_caps(tmp_path):
    store = make_store(tmp_path)
    # Distinct term the model misheard phonetically -> keep it.
    store.record("use web sockets here", "Use web sockets here.",
                 "bad", "Use WebSockets here.")
    assert "WebSockets" in store.learned_vocab()


def test_learned_vocab_merged_into_prompt(tmp_path):
    store = make_store(tmp_path)
    store._add_vocab(["Ogiop"])
    cfg = CleanupConfig(enabled=True)
    cleaner = Cleaner(cfg, training=store)
    fake = MagicMock()
    fake.json.return_value = {"response": "Whatever."}
    fake.raise_for_status.return_value = None
    with patch("app.cleanup.requests.post", return_value=fake) as mock_post:
        cleaner.clean("another long sentence that goes to the language model")
    prompt = mock_post.call_args.kwargs["json"]["prompt"]
    assert "Ogiop" in prompt


# --- overlay event dispatch ---------------------------------------------------

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


def test_feedback_transcript_only_dispatch():
    ov = Overlay(enabled=False)
    seen = []
    ov.on_feedback = lambda *a: seen.append(a)
    ov._pending = (7, "meet mr ogi up", "Meet Mr Ogi up.")
    ov._dispatch({"type": "feedback", "id": 7, "rating": None,
                  "transcript": "meet Mr Ogiop", "ideal": None, "tags": []})
    assert seen == [("meet mr ogi up", "Meet Mr Ogi up.", None,
                     "meet Mr Ogiop", None, [])]
    assert ov._pending is None


def test_feedback_tags_only_dispatch():
    ov = Overlay(enabled=False)
    seen = []
    ov.on_feedback = lambda *a: seen.append(a)
    ov._pending = (8, "raw", "out")
    ov._dispatch({"type": "feedback", "id": 8, "rating": None,
                  "transcript": None, "ideal": None, "tags": ["bad list"]})
    assert seen == [("raw", "out", None, None, None, ["bad list"])]
    assert ov._pending is None


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


def test_settings_event_dispatch():
    ov = Overlay(enabled=False)
    got = {}
    ov.on_settings = got.update
    ov._dispatch({"type": "settings_saved", "values": {"hotkey": "f8"}})
    assert got == {"hotkey": "f8"}


# --- settings persistence -----------------------------------------------------

def test_save_config_updates_preserves_comments(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        "# my comment\n[stt]\nmodel = \"small.en\"\n\n[training]\nenabled = false\n",
        encoding="utf-8",
    )
    save_config_updates(
        {"stt": {"model": "large-v3"}, "training": {"enabled": True},
         "hotkey": {"binding": "f8"}},
        path=cfg_file,
    )
    text = cfg_file.read_text(encoding="utf-8")
    assert "# my comment" in text
    cfg = load_config(cfg_file)
    assert cfg.stt.model == "large-v3"
    assert cfg.training.enabled is True
    assert cfg.hotkey.binding == "f8"  # new section appended


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


def test_record_feedback_derives_verdict_and_replaces():
    from unittest.mock import MagicMock
    from app.hotkey import PushToTalkApp
    # rating 5, no ideal -> verdict "ok", no in-place replace
    fake = MagicMock()
    fake.cfg.training.replace_on_correction = True
    PushToTalkApp._record_feedback(fake, "raw", "out", 5, None, None, [])
    assert fake.training.record.call_args.args[2] == "ok"     # derived verdict
    fake.context.replace_last.assert_not_called()
    fake.injector.replace_last.assert_not_called()
    # rating 2 with ideal -> verdict "bad", context + injector replace
    fake = MagicMock()
    fake.cfg.training.replace_on_correction = True
    PushToTalkApp._record_feedback(fake, "raw", "out", 2, None, "Ideal.", ["x"])
    assert fake.training.record.call_args.args[2] == "bad"
    fake.context.replace_last.assert_called_once_with("Ideal.")
    fake.injector.replace_last.assert_called_once_with("Ideal.")


# --- relevant_corrections TF-IDF retriever ---------------------------------

def test_relevant_corrections_ranks_shared_distinctive_term_first(tmp_path):
    store = make_store(tmp_path)
    store.record("schedule the meeting", "Schedule the meeting.", "bad", "Schedule the sync.")
    store.record("deploy the kubernetes cluster", "Deploy the kubernetes cluster.",
                 "bad", "Deploy the Kubernetes cluster.")
    store.record("send the email", "Send the email.", "bad", "Send the mail.")
    got = store.relevant_corrections("redeploy the kubernetes cluster now")
    assert got, "expected a match"
    assert got[0]["ideal"] == "Deploy the Kubernetes cluster."


def test_relevant_corrections_gate_returns_empty_when_nothing_similar(tmp_path):
    store = make_store(tmp_path)
    store.record("deploy the kubernetes cluster", "Deploy the kubernetes cluster.",
                 "bad", "Deploy the Kubernetes cluster.")
    assert store.relevant_corrections("what time is lunch tomorrow") == []


def test_relevant_corrections_surfaces_old_lesson_over_recent_noise(tmp_path):
    store = make_store(tmp_path)
    # one distinctive OLD correction, then 6 unrelated NEWER ones
    store.record("call ogiop about it", "Call Ogiop about it.", "bad", "Call Ogiop about it.")
    for i in range(6):
        store.record(f"unrelated thing number {i}", f"Unrelated thing number {i}.",
                     "bad", f"Totally different {i}.")
    got = store.relevant_corrections("please call ogiop again")
    assert any(e["ideal"] == "Call Ogiop about it." for e in got)


def test_relevant_corrections_empty_inputs(tmp_path):
    store = make_store(tmp_path)
    assert store.relevant_corrections("anything") == []      # no corrections
    store.record("call ogiop", "Call Ogiop.", "bad", "Call Ogiop.")
    assert store.relevant_corrections("") == []               # empty query
    assert store.relevant_corrections("   ") == []


def test_relevant_corrections_ignores_short_stopword_query(tmp_path):
    store = make_store(tmp_path)
    store.record("yes I think we should ship it", "Yes, I think we should ship it.",
                 "bad", "Yes, we should ship it.")
    # A one-word streamed chunk must NOT match on a shared common word.
    assert store.relevant_corrections("yes") == []
    assert store.relevant_corrections("yes.") == []
    assert store.relevant_corrections("okay") == []


def test_relevant_corrections_tie_break_prefers_recent(tmp_path):
    import time as _t
    store = make_store(tmp_path)
    store.record("deploy the ogiop service", "Deploy the ogiop service.",
                 "bad", "Deploy the Ogiop service OLD.")
    _t.sleep(0.01)  # ensure a later ts
    store.record("deploy the ogiop service", "Deploy the ogiop service.",
                 "bad", "Deploy the Ogiop service NEW.")
    # Identical raw -> identical similarity; the more recent must win the tie.
    got = store.relevant_corrections("redeploy the ogiop service", n=1)
    assert got and got[0]["ideal"] == "Deploy the Ogiop service NEW."


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
