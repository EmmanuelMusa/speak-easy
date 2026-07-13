"""Surrounding-text logic: continuation flags, spacing, and cleanup plumbing.

The UIA read itself is environment-dependent and exercised manually; these
tests cover everything downstream of it (pure logic + prompt wiring).
"""

from unittest.mock import MagicMock, patch

from app.cleanup import Cleaner, strip_fillers
from app.config import CleanupConfig
from app.focus import Surrounding, needs_leading_space


# --- continuation flags -------------------------------------------------------

def test_mid_sentence_detection():
    assert Surrounding(before="I think we should").mid_sentence
    assert Surrounding(before="items: ").mid_sentence
    assert not Surrounding(before="Sounds good. ").mid_sentence
    assert not Surrounding(before="New paragraph\n").mid_sentence
    assert not Surrounding(before="").mid_sentence


def test_continues_after_only_counts_same_line():
    assert Surrounding(after=" and then some").continues_after
    assert not Surrounding(after="\nNext paragraph here").continues_after
    assert not Surrounding(after="   ").continues_after


def test_needs_leading_space():
    assert needs_leading_space("Hello world", "next")
    assert needs_leading_space("done.", "Next")
    assert not needs_leading_space("Hello world ", "next")
    assert not needs_leading_space("(", "aside")
    assert not needs_leading_space("", "anything")
    assert not needs_leading_space("Hello", ", punctuation first")


# --- secret scrubbing ----------------------------------------------------------

def test_secret_shaped_text_is_scrubbed():
    from app.focus import _SECRET_RE

    pem = (
        "config here\n-----BEGIN PRIVATE KEY-----\n"
        "MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg\n"
        "-----END PRIVATE KEY-----\nand the deploy is set for"
    )
    scrubbed = _SECRET_RE.sub(" ", pem)
    assert "PRIVATE KEY" not in scrubbed
    assert "MIGHAgEAMBMG" not in scrubbed
    assert "and the deploy is set for" in scrubbed
    # Long unbroken token (API key shaped) goes; normal prose stays.
    token = "ghp_" + "a1B2" * 12
    assert _SECRET_RE.sub(" ", f"the token {token} expired") == "the token   expired"
    assert _SECRET_RE.sub(" ", "we ship on Thursday afternoon") == \
        "we ship on Thursday afternoon"


# --- local cleanup path respects continuation ---------------------------------

def test_strip_fillers_continuation_flags():
    assert strip_fillers("um the numbers look good", capitalize=False) == \
        "the numbers look good."
    assert strip_fillers("um the numbers look good", ensure_period=False) == \
        "The numbers look good"


def test_local_clean_mid_sentence_stays_lowercase():
    cleaner = Cleaner(CleanupConfig(enabled=False))
    out = cleaner.clean(
        "um and also the deploy",
        surrounding=Surrounding(before="We fixed the tests", after=" tomorrow."),
    )
    assert out == "and also the deploy"  # no capital, no period


# --- LLM path gets the continuation block --------------------------------------

def test_cleaner_llm_sees_surrounding_block():
    cfg = CleanupConfig(enabled=True)
    fake = MagicMock()
    fake.json.return_value = {"response": "and the rollout finishes Thursday"}
    fake.raise_for_status.return_value = None
    surrounding = Surrounding(
        before="The migration is done", after=" as planned.", app="Notepad"
    )
    with patch("app.cleanup.requests.post", return_value=fake) as mock_post:
        out = Cleaner(cfg).clean(
            "um and the rollout finishes thursday", surrounding=surrounding
        )
    assert out == "and the rollout finishes Thursday"
    system = mock_post.call_args.kwargs["json"]["system"]
    assert "The migration is done" in system
    assert "lowercase the first word" in system
    assert "do not add a trailing period" in system
