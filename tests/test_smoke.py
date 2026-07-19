"""Smoke tests: cleanup filler-stripping, the always-LLM cleanup path, and
injection dispatch — with all OS keystroke and Ollama calls mocked."""

from unittest.mock import MagicMock, patch

from app.cleanup import Cleaner, strip_fillers, too_divergent
from app.config import CleanupConfig, InjectionConfig
from app.injection import Injector
from app.overlay import Overlay


# --- 1. cleanup strips filler words -----------------------------------------

def test_strip_fillers_removes_noise_and_capitalizes():
    assert strip_fillers("um hello there") == "Hello there."


def test_strip_fillers_noise_variants():
    out = strip_fillers("ehmm the deploy is uhh kind of stuck")
    assert out == "The deploy is kind of stuck."
    assert strip_fillers("Mm, I think, umm, we can push the deadline.") == \
        "I think, we can push the deadline."


def test_strip_fillers_keeps_speech_phrases_and_stutters():
    # "you know", "I mean", and stutters are the speaker's words — kept.
    out = strip_fillers(
        "So basically, um, the the quarterly numbers look good, you know, "
        "better than last year."
    )
    assert out == (
        "So basically, the the quarterly numbers look good, you know, "
        "better than last year."
    )
    assert strip_fillers("you know what I mean it's just not working") == \
        "You know what I mean it's just not working."


def test_cleaner_strips_fillers_without_llm():
    cfg = CleanupConfig(enabled=False)
    assert Cleaner(cfg).clean("um this is uh a test") == "This is a test."


# --- 2. every utterance goes through the LLM, however short -----------------

def test_short_utterance_uses_llm():
    # Wispr Flow behavior: only the LLM can turn "is it ready" into
    # "Is it ready?" — the local strip would type "Is it ready."
    cfg = CleanupConfig(enabled=True)
    fake = MagicMock()
    fake.json.return_value = {"response": "Is it ready?"}
    fake.raise_for_status.return_value = None
    with patch("app.cleanup.requests.post", return_value=fake) as mock_post:
        result = Cleaner(cfg).clean("is it ready")
    mock_post.assert_called_once()
    assert result == "Is it ready?"
    # The transcript is sent as tagged data so the model formats the
    # question instead of answering it.
    payload = mock_post.call_args.kwargs["json"]
    assert "<transcript>" in payload["prompt"]


def test_short_utterance_with_correction_cue_resolved_by_llm():
    cfg = CleanupConfig(enabled=True)
    fake = MagicMock()
    fake.json.return_value = {"response": "I have a meeting by 3pm."}
    fake.raise_for_status.return_value = None
    with patch("app.cleanup.requests.post", return_value=fake) as mock_post:
        result = Cleaner(cfg).clean("I have a meeting by 9am no sorry by 3pm")
    mock_post.assert_called_once()
    assert result == "I have a meeting by 3pm."


def test_long_utterance_calls_llm():
    cfg = CleanupConfig(enabled=True)
    cleaner = Cleaner(cfg)
    long_text = (
        "um so basically what i wanted to say is that we should definitely "
        "move the meeting to thursday afternoon you know"
    )
    # Spec-compliant LLM output: noise stripped, punctuation added, every
    # real word kept (a response that dropped words would be rejected by
    # the divergence guard).
    fake = MagicMock()
    fake.json.return_value = {
        "response": "So basically what I wanted to say is that we should "
        "definitely move the meeting to Thursday afternoon, you know."
    }
    fake.raise_for_status.return_value = None
    with patch("app.cleanup.requests.post", return_value=fake) as mock_post:
        result = cleaner.clean(long_text)
    mock_post.assert_called_once()
    assert result.startswith("So basically what I wanted to say")


def test_warmup_skipped_when_disabled():
    cleaner = Cleaner(CleanupConfig(enabled=False))
    with patch("app.cleanup.requests.post") as mock_post:
        cleaner.warmup()
    mock_post.assert_not_called()


def test_warmup_pings_ollama_when_enabled():
    cleaner = Cleaner(CleanupConfig(enabled=True))
    fake = MagicMock()
    fake.raise_for_status.return_value = None
    with patch("app.cleanup.requests.post", return_value=fake) as mock_post:
        cleaner.warmup()
    mock_post.assert_called_once()


def test_llm_failure_falls_back_to_local_strip():
    cfg = CleanupConfig(enabled=True, timeout_seconds=0.01)
    cleaner = Cleaner(cfg)
    with patch("app.cleanup.requests.post", side_effect=ConnectionError("down")):
        result = cleaner.clean("um the quick brown fox jumps over the lazy dog")
    assert result == "The quick brown fox jumps over the lazy dog."


# --- divergence guard: paraphrased LLM output is rejected --------------------

def test_paraphrased_llm_output_is_rejected():
    raw = "um I plan to go by 2 p.m. no sorry by 3 p.m."
    paraphrase = "I've changed the meeting time from 2 p.m. to 3 p.m."
    assert too_divergent(raw, paraphrase)

    cfg = CleanupConfig(enabled=True)
    fake = MagicMock()
    fake.json.return_value = {"response": paraphrase}
    fake.raise_for_status.return_value = None
    with patch("app.cleanup.requests.post", return_value=fake):
        result = Cleaner(cfg).clean(raw)
    # Falls back to the local strip of the speaker's actual words.
    assert result == "I plan to go by 2 p.m. no sorry by 3 p.m."  # noise-only strip


def test_faithful_llm_output_is_accepted():
    raw = "um I plan to go by 2 p.m. no sorry by 3 p.m."
    faithful = "I plan to go by 3 p.m."
    assert not too_divergent(raw, faithful)

    cfg = CleanupConfig(enabled=True)
    fake = MagicMock()
    fake.json.return_value = {"response": faithful}
    fake.raise_for_status.return_value = None
    with patch("app.cleanup.requests.post", return_value=fake):
        assert Cleaner(cfg).clean(raw) == faithful


# --- divergence guard: dropped speaker words are rejected --------------------

def test_dropped_speaker_words_are_rejected():
    raw = (
        "um so basically I think we should uh schedule the product review "
        "meeting for Thursday afternoon you know right after lunch"
    )
    # LLM silently deleted "you know" — that's the speaker's phrasing.
    dropped = (
        "So basically, I think we should schedule the product review "
        "meeting for Thursday afternoon, right after lunch."
    )
    assert too_divergent(raw, dropped)


def test_noise_and_command_word_removal_is_accepted():
    # Vocal noises are the LLM's job to remove — never counted as dropped.
    raw = "um so basically I think we should uh schedule the meeting for Thursday"
    clean = "So basically, I think we should schedule the meeting for Thursday."
    assert not too_divergent(raw, clean)
    # Spoken punctuation commands become punctuation; the command words vanish.
    raw = "make the bar smaller in bracket less wide also close bracket and dark"
    clean = "Make the bar smaller (less wide also) and dark."
    assert not too_divergent(raw, clean)


def test_enumeration_reformat_is_accepted():
    raw = (
        "we need three things first the budget second the timeline "
        "and third the staffing plan"
    )
    clean = "We need three things:\n1. The budget\n2. The timeline\n3. The staffing plan"
    assert not too_divergent(raw, clean)


def test_number_word_enumeration_reformat_is_accepted():
    # Regression: "number one/two/three" -> a numbered list dropped the words
    # "number", "one", "two", "three", which the guard used to reject, so the
    # user got the unformatted local strip instead of the list.
    raw = "the steps are number one open the file number two edit it number three save it"
    clean = "The steps are:\n1. Open the file\n2. Edit it\n3. Save it"
    assert not too_divergent(raw, clean)


def test_then_finally_enumeration_reformat_is_accepted():
    raw = "my priorities are ship the release then fix the login bug and finally update the docs"
    clean = "My priorities are:\n1. Ship the release\n2. Fix the login bug\n3. Update the docs"
    assert not too_divergent(raw, clean)


def test_list_scaffold_leniency_does_not_apply_to_prose():
    # The scaffold-word leniency is scoped to list OUTPUT only. In prose, a
    # dropped "number"/"then" must still be caught as a real deletion.
    raw = "call the number then wait for the tone and then press one"
    dropped = "Call the wait for the tone and press."  # 'number','then','one' gone
    assert too_divergent(raw, dropped)


# --- deterministic enumeration -> numbered list -----------------------------

def test_reformat_ordinal_sentences_into_list():
    from app.cleanup import reformat_enumeration
    # Mixed period/comma separators, with a repeated "I'll" subject kept.
    text = ("So I'll do it in three ways. Firstly, I'll look at what I've "
            "learned. Secondly, I'll improve on it, thirdly, I'll find better "
            "ways to do it.")
    assert reformat_enumeration(text) == (
        "So I'll do it in three ways:\n"
        "1. I'll look at what I've learned.\n"
        "2. I'll improve on it.\n"
        "3. I'll find better ways to do it."
    )


def test_reformat_number_word_sentences_into_list():
    from app.cleanup import reformat_enumeration
    text = "First, grow the team. Second, ship the app. Third, cut costs."
    assert reformat_enumeration(text) == (
        "1. Grow the team.\n2. Ship the app.\n3. Cut costs."
    )


def test_reformat_leaves_prose_and_existing_lists_untouched():
    from app.cleanup import reformat_enumeration
    # Not an enumeration (one boundary ordinal only).
    assert reformat_enumeration("I went to the store, then I came home.") == \
        "I went to the store, then I came home."
    # "first aid" / "second thoughts": ordinals not at item boundaries.
    prose = "I had first aid training, and second thoughts about it."
    assert reformat_enumeration(prose) == prose
    # Already a list: unchanged.
    already = "Things:\n1. Budget\n2. Timeline"
    assert reformat_enumeration(already) == already


# --- overlay ------------------------------------------------------------------

def test_overlay_disabled_is_noop():
    ov = Overlay(enabled=False)
    ov.show_idle()
    ov.show_recording()
    ov.show_processing()
    ov.hide()
    assert ov._proc is None  # never spawned the UI process


# --- 3. injection dispatches via the configured method ----------------------

def test_injector_dispatches_clipboard():
    inj = Injector(InjectionConfig(delivery_method="clipboard", verify_paste=False))
    with patch("app.injection.inject_clipboard") as clip, \
         patch("app.injection.inject_sendinput") as send, \
         patch("app.injection.is_terminal_window", return_value=False):
        inj.inject("hello world")
    clip.assert_called_once_with(
        "hello world", paste_delay=0.05, shift_insert=False
    )
    send.assert_not_called()


def test_injector_dispatches_sendinput():
    inj = Injector(InjectionConfig(delivery_method="sendinput", verify_paste=False))
    with patch("app.injection.inject_clipboard") as clip, \
         patch("app.injection.inject_sendinput") as send:
        inj.inject("hello world")
    send.assert_called_once_with("hello world")
    clip.assert_not_called()


def test_injector_skips_empty_text():
    inj = Injector(InjectionConfig())
    with patch("app.injection.inject_clipboard") as clip:
        inj.inject("")
    clip.assert_not_called()


def test_clean_collapses_ellipses_without_llm():
    cfg = CleanupConfig(enabled=False)
    assert Cleaner(cfg).clean("so I was thinking...") == "So I was thinking."


def test_clean_uses_fallback_text_for_local_strip():
    # LLM off: the local strip runs on fallback_text (pause punctuation),
    # not the clean model_text.
    cfg = CleanupConfig(enabled=False)
    out = Cleaner(cfg).clean("we shipped it the docs are next",
                             fallback_text="we shipped it. the docs are next")
    assert out == "We shipped it. The docs are next."


def test_cleanup_streaming_defaults_off_for_holistic_quality():
    # Holistic (whole-utterance) cleanup is the default; per-chunk streaming
    # is opt-in. Chunking at pauses was what broke punctuation/lists.
    from app.config import CleanupConfig
    assert CleanupConfig().streaming is False


def test_number_words_to_digits_not_rejected():
    from app.cleanup import too_divergent
    # "ten million" spoken -> "10 million" cleaned drops the WORD "ten"; the
    # guard must treat spoken numbers as legitimately convertible to digits.
    raw = "the budget is ten million to twenty million naira"
    clean = "The budget is 10 million to 20 million Naira."
    assert not too_divergent(raw, clean)


def test_parallel_item_list_not_rejected():
    from app.cleanup import too_divergent
    raw = ("the registration value is value ten million to twenty million value "
           "twenty one million to one hundred million value above five hundred million")
    clean = ("The registration value is:\n"
             "- Value: 10 million to 20 million\n"
             "- Value: 21 million to 100 million\n"
             "- Value: above 500 million")
    assert not too_divergent(raw, clean)


def test_prompt_covers_parallel_item_lists():
    from app.cleanup import SYSTEM_PROMPT
    p = SYSTEM_PROMPT.lower()
    assert "parallel" in p or "same shape" in p  # parallel-item list rule present
    assert "- " in SYSTEM_PROMPT                  # a bulleted-list example present


def test_prose_paraphrase_dropping_and_plus_word_is_rejected():
    from app.cleanup import too_divergent
    # Dropping a conjunction AND a content word in prose is real content loss —
    # the guard must still catch it (regression: "and" was wrongly always-droppable).
    raw = "please send the invoice and receipt to accounting"
    clean = "Please send the invoice to accounting."
    assert too_divergent(raw, clean)


def test_drop_noise_catches_fillers_the_model_missed():
    from app.cleanup import drop_noise
    # A filler the LLM left in mid-sentence is removed, spacing tidied, casing
    # preserved (no forced period, no mid-text recap).
    assert drop_noise("The um quarterly numbers look good") == \
        "The quarterly numbers look good"
    # Leading noise the model capitalized -> re-capitalize the new first word.
    assert drop_noise("Um, hello there") == "Hello there"
    # Clean text is returned unchanged.
    assert drop_noise("Ship the release today") == "Ship the release today"
    # Real words that merely contain the letters are NOT touched.
    assert drop_noise("The summary is short") == "The summary is short"


def test_capitalize_sentences_fixes_model_lowercase_starts():
    from app.cleanup import capitalize_sentences
    # The whole reason this exists: models return lowercase sentence starts.
    assert capitalize_sentences("hello there. how are you? fine.") == \
        "Hello there. How are you? Fine."
    # Already-correct text is unchanged (idempotent).
    assert capitalize_sentences("Ship it. Then rest.") == "Ship it. Then rest."
    # Mid-sentence insertion: keep the first letter lowercase, still fix the rest.
    assert capitalize_sentences("the rest. another point.", cap_first=False) == \
        "the rest. Another point."
    # Abbreviations aren't treated as sentence ends.
    assert capitalize_sentences("meet at 3 p.m. tomorrow") == \
        "Meet at 3 p.m. tomorrow"


def test_flow_edit_mid_sentence_and_continuation():
    from app.cleanup import flow_edit
    # Mid-sentence: a plain continuation word is lowercased to flow on.
    assert flow_edit("The rest follows.", mid_sentence=True,
                     continues_after=False) == "the rest follows."
    # continues_after: a trailing period the model added is dropped.
    assert flow_edit("adding a few words.", mid_sentence=True,
                     continues_after=True) == "adding a few words"
    # A proper noun at the insertion point is NEVER lowercased.
    assert flow_edit("John said yes.", mid_sentence=True,
                     continues_after=False) == "John said yes."
    # A sentence-starter adverb keeps its capital even mid-caret ("Also…" is far
    # more often a new sentence than a continuation).
    assert flow_edit("Also we should go.", mid_sentence=True,
                     continues_after=False) == "Also we should go."
    # Not mid-sentence: untouched (normal standalone sentence).
    assert flow_edit("The rest follows.", mid_sentence=False,
                     continues_after=False) == "The rest follows."


def test_stt_engine_defaults():
    from app.config import SttConfig
    assert SttConfig().engine == "whisper"
    assert SttConfig().parakeet_model == "nemo-parakeet-tdt-0.6b-v2"


def test_parakeet_engine_skips_streaming_session():
    # With engine=parakeet, _on_press must NOT build a StreamingSession
    # (Parakeet is non-streaming); the full clip is transcribed at release.
    from unittest.mock import MagicMock
    from app.hotkey import PushToTalkApp
    from app.config import Config
    cfg = Config()
    cfg.stt.engine = "parakeet"
    cfg.stt.streaming = True  # even if streaming is on, Parakeet ignores it
    fake = MagicMock()
    fake.cfg = cfg
    fake.recorder.recording = False
    fake._busy.locked.return_value = False
    PushToTalkApp._on_press(fake)
    assert fake._session is None
    fake.recorder.start.assert_called_once()


def test_reload_stt_rolls_back_and_warns_on_failure(monkeypatch):
    # A live engine switch that fails to load must NOT install the broken
    # engine, must roll the (already-persisted) config back to the working one
    # so the next start isn't bricked, and must surface a visible warning.
    from unittest.mock import MagicMock
    import app.hotkey as hk
    from app.config import Config

    cfg = Config()
    cfg.stt.engine = "parakeet"           # _apply_settings already switched...
    cfg.stt.parakeet_model = "nemo-parakeet-tdt-0.6b-v2"
    previous = {"model": cfg.stt.model, "engine": "whisper",
                "parakeet_model": cfg.stt.parakeet_model}

    boom = MagicMock()
    boom._load.side_effect = RuntimeError("no onnx-asr")
    monkeypatch.setattr(hk, "make_transcriber", lambda _c: boom)
    saved: dict = {}
    monkeypatch.setattr(hk, "save_config_updates", lambda d: saved.update(d))
    notes: list = []
    monkeypatch.setattr(hk, "_notify_user", lambda m: notes.append(m))

    fake = MagicMock()
    fake.cfg = cfg
    working = fake.transcriber
    hk.PushToTalkApp._reload_stt(fake, previous)

    assert cfg.stt.engine == "whisper"           # rolled back in memory
    assert saved["stt"]["engine"] == "whisper"   # ...and re-persisted
    assert fake.transcriber is working           # broken engine not installed
    assert notes and "parakeet" in notes[0]      # user was warned


def test_on_press_skips_streaming_when_transcriber_cant_segment():
    # Race guard: even with engine=whisper + streaming, if the current
    # transcriber lacks transcribe_segments (mid engine-switch), no StreamingSession.
    from unittest.mock import MagicMock
    from app.hotkey import PushToTalkApp
    from app.config import Config
    cfg = Config()
    cfg.stt.engine = "whisper"
    cfg.stt.streaming = True
    fake = MagicMock()
    fake.cfg = cfg
    fake.recorder.recording = False
    fake._busy.locked.return_value = False
    fake.transcriber = MagicMock(spec=[])   # no transcribe_segments attribute
    PushToTalkApp._on_press(fake)
    assert fake._session is None
