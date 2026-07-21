# Spoken emoji — design

**Date:** 2026-07-21
**Status:** approved

## Problem

Dictating into chat, there is no way to produce an emoji. You have to stop, reach
for the mouse or an emoji picker, and break the flow that dictation exists to
protect.

## Goal

Saying the name of an emoji inserts the emoji character.

    "Great work on this fire emoji"   ->   Great work on this 🔥
    "ship it rocket emoji"            ->   ship it 🚀

## Decisions

### Trigger: the marker word is required

An emoji appears only when the utterance contains the word `emoji` (or `emojis`)
adjacent to a known name. Both orders are accepted: `fire emoji` and
`emoji fire`.

This is what makes the feature safe. Ordinary speech is never rewritten, because
ordinary speech does not say "emoji":

    "Call the fire department"   ->   Call the fire department

A bare-name trigger (`thumbs up` -> 👍 with no marker) was rejected: the same
phrases occur in normal prose, and a dictation tool that silently corrupts your
words is worse than one that makes you say an extra syllable.

### Scope: everywhere, no app gating

Because the trigger is explicit, no per-application allowlist is needed — you
only get an emoji when you asked for one. The feature therefore behaves
identically in Slack, a browser, Word, and an editor.

Per-app gating was rejected on predictability grounds. Chat done in a browser
tab (WhatsApp Web, Slack web, Gmail) is indistinguishable from any other browser
window at the process level, so an allowlist would make the same app behave two
different ways depending on how it was opened.

### Vocabulary: curated, alias-rich, fail-open

Roughly 150 commonly used emoji, each answering to several spoken names, so
there is no single exact phrase to memorise:

| Emoji | Aliases |
|---|---|
| 😂 | crying laughing, laughing crying, tears of joy, joy |
| 👍 | thumbs up, thumbs-up, plus one |
| 🔥 | fire, lit |
| ❤️ | heart, red heart, love |
| 🚀 | rocket, rocket ship |
| 🙏 | pray, prayer hands, please, thank you |

The full Unicode set (~1900) was rejected: its official names are not things
anyone says aloud ("face with tears of joy"), and short names collide across
many emoji. A fuzzy fallback to that set was also rejected — it converts a clean
miss into a confidently wrong substitution.

**An unrecognised name is left untouched.** `"banana emoji"` stays as the words
"banana emoji". A missing lexicon entry produces a mildly awkward sentence, never
a mangled one.

## Architecture

### New module: `app/emoji.py`

`cleanup.py` is already large and carries several responsibilities. The lexicon
and its matcher live in their own module exposing one function:

```python
def apply_spoken_emoji(text: str) -> str
```

and one data structure, a `dict[str, str]` mapping alias -> emoji character.
Nothing outside the module needs to know how matching works.

### Pipeline placement: deterministic post-pass

`apply_spoken_emoji` is called from `Cleaner._finish`, immediately after
`_normalize_app_name` and **before** `capitalize_sentences` (so casing is
computed on the final text, and an emoji at a sentence start does not leave the
following word lowercase).

Consequences of this placement:

- The cleanup LLM never sees a Unicode emoji. It sees the words "fire emoji" and
  passes them through as ordinary text. Small models can drop, duplicate, or
  mangle emoji in a prompt; this sidesteps the question.
- The divergence guard (`too_divergent`) runs earlier, comparing plain words on
  both sides. There is no interaction with it.
- One insertion point covers all three paths — LLM output, the offline fallback,
  and streaming mode (`LiveCleanup.finalize` also routes through `Cleaner.clean`).

The alternative — a pre-pass beside `_apply_spoken_punctuation` — is the right
call for brackets and quotes, because the model actively rewrites punctuation
and would undo them. Emoji are under no such pressure. If testing later shows the
model swallowing the marker word, adding the pre-pass is a two-line change, and
substitution is idempotent (after the swap no `X emoji` remains to match), so
running both passes is harmless.

### Matching rules

- One regex alternation over all aliases, **longest alias first**, so
  "crying laughing emoji" resolves to 😂 rather than matching a shorter
  "crying" entry.
- Both orders: `<name> emoji` and `emoji <name>`; plural `emojis` accepted.
- Case-insensitive, word-boundary anchored, tolerant of an STT-inserted comma
  ("rocket, emoji").
- Aliases are regex-escaped when the pattern is built.
- Trailing punctuation is preserved: "ship it rocket emoji." -> "ship it 🚀."

## Configuration

`CleanupConfig.spoken_emoji: bool = True`, surfaced as a checkbox in the Settings
dialog's Behavior group beside the existing keep-warm toggle:

> Insert emoji when you say their name

When off, the pass is skipped and the words come through unchanged.

## Testing

New `tests/test_emoji.py`:

- alias resolves to the right character
- both word orders work
- longest match wins over a shorter overlapping alias
- case-insensitive
- surrounding punctuation preserved
- unknown name passes through unchanged
- **false-positive guard:** "call the fire department" is unchanged

In `tests/test_cleanup.py`:

- the pass is wired into `_finish`
- `spoken_emoji = False` disables it

## Out of scope

Deliberately excluded, each easy to add later, none worth the surface area now:

- fuzzy fallback to the full Unicode set
- skin-tone modifiers
- user-defined aliases in config
- per-application gating
