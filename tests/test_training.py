"""Tests for training mode: feedback storage, few-shot injection, vocabulary
mining, overlay event dispatch, and settings persistence."""

import json

from app.config import load_config, save_config_updates
from app.cleanup import Cleaner
from app.config import CleanupConfig
from app.overlay import Overlay
from app.training import TrainingStore

from unittest.mock import MagicMock, patch


def make_store(tmp_path):
    return TrainingStore(
        data_path=tmp_path / "data.jsonl", vocab_path=tmp_path / "vocab.json"
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
        cleaner.clean("this is a long enough sentence to reach the model")
    system = mock_post.call_args.kwargs["json"]["system"]
    assert "Go by 3pm." in system  # correction injected as example


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
