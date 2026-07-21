"""Spoken emoji: saying the name of an emoji inserts the emoji.

    "Great work on this fire emoji"   ->   Great work on this 🔥
    "ship it rocket emoji"            ->   ship it 🚀

The word "emoji" is required. That is the whole safety story: ordinary speech
does not say it, so ordinary speech is never rewritten.

    "Call the fire department"        ->   Call the fire department

A bare-name trigger was considered and rejected — the same phrases turn up in
normal prose, and a dictation tool that silently corrupts your words is worse
than one that costs you a syllable.

The vocabulary is curated rather than the full ~1900-emoji Unicode set, whose
official names ("face with tears of joy") are not things anyone says out loud
and whose short names collide. An unrecognised name is left alone: "banana
emoji" stays as those two words, which is a mildly awkward sentence rather than
a mangled one.

Runs as a deterministic pass in Cleaner._finish, after the model rather than
before it: handing a small model raw Unicode emoji got them silently deleted.
The model is left looking at the words "rocket emoji", which it often converts
by itself — so the divergence guard normalizes both sides through this module
before comparing, and Cleaner separately rejects output that dropped an emoji
the speaker asked for.
"""

from __future__ import annotations

import re

#: emoji -> the things a person might call it. Aliases must be unique across
#: the whole table; _build_aliases raises if two emoji claim the same one.
#:
#: Deliberately absent are words that are common in ordinary speech *next to*
#: the marker, because the reversed "emoji <name>" order would then fire on
#: real sentences: "like" ("use an emoji like a thumbs up"), "look" ("does the
#: emoji look right"), and "please" ("no emoji please").
EMOJI: dict[str, tuple[str, ...]] = {
    # -- faces ---------------------------------------------------------------
    "😀": ("grinning", "grin"),
    "😄": ("smile", "smiling", "happy"),
    "😁": ("beaming",),
    "😂": ("crying laughing", "laughing crying", "tears of joy", "laughing",
           "joy", "lol"),
    "🤣": ("rolling on the floor", "rofl", "rolling laughing"),
    "😊": ("blush", "blushing"),
    "🙂": ("slight smile", "slightly smiling"),
    "🙃": ("upside down",),
    "😉": ("wink", "winking"),
    "😍": ("heart eyes", "hearts in eyes"),
    "🥰": ("smiling with hearts", "in love"),
    "😘": ("blowing a kiss", "kiss", "kissing"),
    "😋": ("yum", "delicious", "tasty"),
    "😜": ("winking tongue", "cheeky"),
    "🤪": ("zany", "goofy"),
    "🤑": ("money mouth", "money face"),
    "🤗": ("hugging", "hug"),
    "🤭": ("hand over mouth", "oops"),
    "🤫": ("shushing", "shush", "quiet"),
    "🤔": ("thinking", "think", "hmm"),
    "🤐": ("zipper mouth", "zip it"),
    "😐": ("neutral", "straight face"),
    "😑": ("expressionless",),
    "😶": ("no mouth", "speechless"),
    "😏": ("smirk", "smirking"),
    "😒": ("unamused", "unimpressed"),
    "🙄": ("eye roll", "rolling eyes"),
    "😬": ("grimace", "grimacing", "awkward"),
    "😴": ("sleeping", "asleep", "zzz"),
    "😷": ("mask", "masked face"),
    "🤒": ("sick", "thermometer face"),
    "🤢": ("nauseated", "queasy"),
    "🤮": ("vomiting", "throwing up"),
    "🥵": ("hot face", "overheated"),
    "🥶": ("cold face", "freezing"),
    "🤯": ("mind blown", "exploding head"),
    "🥳": ("partying", "party face"),
    "😎": ("sunglasses", "cool"),
    "🤓": ("nerd", "nerdy"),
    "🧐": ("monocle", "scrutinizing"),
    "😕": ("confused",),
    "😟": ("worried",),
    "🙁": ("slight frown", "frowning"),
    "😮": ("open mouth", "surprised"),
    "😲": ("astonished", "shocked"),
    "🥺": ("pleading", "puppy eyes"),
    "😢": ("crying", "sad", "cry"),
    "😭": ("sobbing", "loudly crying", "bawling"),
    "😱": ("screaming", "scream", "horror"),
    "😞": ("disappointed",),
    "😩": ("weary",),
    "😫": ("tired",),
    "🥱": ("yawning", "yawn"),
    "😤": ("triumph", "huffing", "frustrated"),
    "😡": ("angry", "mad", "rage"),
    "🤬": ("cursing", "swearing"),
    "😈": ("devil", "mischievous"),
    "💀": ("skull", "dead"),
    "🤡": ("clown",),
    # -- hands and people ----------------------------------------------------
    "👍": ("thumbs up", "plus one"),
    "👎": ("thumbs down", "dislike"),
    "👌": ("ok hand", "okay hand", "ok", "okay"),
    "🤌": ("pinched fingers", "chefs kiss", "chef kiss"),
    "✌️": ("peace", "peace sign", "victory"),
    "🤞": ("fingers crossed", "crossed fingers"),
    "🤘": ("rock on", "horns"),
    "🤙": ("call me", "shaka"),
    "👈": ("pointing left",),
    "👉": ("pointing right",),
    "👆": ("pointing up",),
    "👇": ("pointing down",),
    "✋": ("raised hand",),
    "🖖": ("vulcan", "live long"),
    "👋": ("wave", "waving", "hello", "hi", "bye"),
    "🤝": ("handshake", "shaking hands", "deal"),
    "🙏": ("pray", "praying", "prayer hands", "thank you", "thanks"),
    "👏": ("clap", "clapping", "applause"),
    "🙌": ("raising hands", "praise", "hooray"),
    "👊": ("fist bump",),
    "✊": ("raised fist", "fist"),
    "💪": ("muscle", "flex", "strong", "biceps"),
    "🤷": ("shrug", "shrugging"),
    "🤦": ("facepalm", "face palm"),
    "🙇": ("bowing", "bow"),
    "👀": ("eyes",),
    "🧠": ("brain",),
    # -- hearts and marks ----------------------------------------------------
    "❤️": ("heart", "red heart", "love"),
    "🧡": ("orange heart",),
    "💛": ("yellow heart",),
    "💚": ("green heart",),
    "💙": ("blue heart",),
    "💜": ("purple heart",),
    "🖤": ("black heart",),
    "🤍": ("white heart",),
    "💔": ("broken heart", "heartbreak"),
    "💕": ("two hearts",),
    "💖": ("sparkling heart",),
    "💯": ("hundred", "one hundred", "hundred points", "perfect", "100"),
    "✅": ("check", "check mark", "tick", "done"),
    "☑️": ("ballot check",),
    "❌": ("cross mark", "x mark", "wrong"),
    "⚠️": ("warning",),
    "❗": ("exclamation",),
    "❓": ("question mark",),
    # -- energy and celebration ----------------------------------------------
    "🔥": ("fire", "lit", "flame"),
    "✨": ("sparkles", "sparkle"),
    "⭐": ("star",),
    "🌟": ("glowing star",),
    "💫": ("dizzy",),
    "⚡": ("lightning", "zap", "high voltage"),
    "💥": ("boom", "collision", "explosion"),
    "🎉": ("party popper", "tada", "celebration", "celebrate", "congrats"),
    "🎊": ("confetti",),
    "🎈": ("balloon",),
    "🎁": ("gift", "present"),
    "🏆": ("trophy",),
    "🥇": ("gold medal", "first place"),
    "🎯": ("bullseye", "target", "direct hit"),
    "🚀": ("rocket", "rocket ship", "launch"),
    # -- work and objects ----------------------------------------------------
    "💡": ("light bulb", "idea", "bulb"),
    "🔔": ("bell", "notification"),
    "🔒": ("lock", "locked"),
    "🔑": ("key",),
    "🔍": ("magnifying glass", "search"),
    "📌": ("pin", "pushpin"),
    "📎": ("paperclip",),
    "✏️": ("pencil",),
    "📝": ("memo", "note", "writing"),
    "📅": ("calendar", "date"),
    "⏰": ("alarm clock", "alarm"),
    "⏳": ("hourglass", "waiting"),
    "💻": ("laptop", "computer"),
    "🖥️": ("desktop computer", "monitor"),
    "⌨️": ("keyboard",),
    "🖱️": ("computer mouse",),
    "📱": ("phone", "mobile", "smartphone"),
    "☎️": ("telephone",),
    "📧": ("email", "e-mail", "mail"),
    "💬": ("speech balloon", "comment", "chat"),
    "💭": ("thought bubble",),
    "🔗": ("link", "chain"),
    "🐛": ("bug",),
    "🛠️": ("tools", "hammer and wrench"),
    "⚙️": ("gear", "settings", "cog"),
    "🗑️": ("trash", "wastebasket", "bin"),
    "📸": ("camera", "photo"),
    "🎬": ("clapper board", "movie", "action"),
    "🎮": ("video game", "gaming", "controller"),
    "🎵": ("musical note", "music"),
    "🎧": ("headphones",),
    # -- food and drink ------------------------------------------------------
    "☕": ("coffee", "hot beverage"),
    "🍵": ("tea",),
    "🍺": ("beer",),
    "🍻": ("cheers", "clinking beers"),
    "🥂": ("champagne", "toast"),
    "🍕": ("pizza",),
    "🍔": ("burger", "hamburger"),
    "🍟": ("fries", "french fries"),
    "🍦": ("ice cream",),
    "🍰": ("cake", "slice of cake"),
    "🎂": ("birthday cake", "birthday"),
    "🍪": ("cookie",),
    "🍎": ("apple",),
    # -- world ---------------------------------------------------------------
    "🐶": ("dog", "puppy"),
    "🐱": ("cat", "kitten"),
    "🐻": ("bear",),
    "🐼": ("panda",),
    "🦄": ("unicorn",),
    "🐍": ("snake", "python"),
    "🦋": ("butterfly",),
    "🌸": ("cherry blossom",),
    "🌹": ("rose",),
    "🌻": ("sunflower",),
    "🌈": ("rainbow",),
    "☀️": ("sun", "sunny"),
    "🌙": ("moon", "crescent moon"),
    "☁️": ("cloud",),
    "❄️": ("snowflake", "snow"),
    "🌊": ("ocean wave", "ocean"),
    "🏠": ("house", "home"),
    "🚗": ("car",),
    "✈️": ("airplane", "plane", "flight"),
    "🚲": ("bicycle", "bike"),
    "🌍": ("earth", "globe", "world"),
    "🗺️": ("map",),
    "⚽": ("soccer ball", "football"),
    "🏀": ("basketball",),
}


def _key(alias: str) -> str:
    """The lookup form of an alias: lowercase, with any run of spaces or
    hyphens flattened to one space. Used when building the table AND when
    looking up what was matched, so "e-mail", "e mail" and "E-Mail" are one
    entry — and so two spellings of the same name collide loudly below rather
    than one of them silently never matching."""
    return re.sub(r"[\s\-]+", " ", alias).strip().lower()


def _build_aliases() -> dict[str, str]:
    """alias -> emoji. Raises on a duplicate: two emoji answering to the same
    name is a typo in the table above, and silently letting the last one win
    would make the winner depend on dict order."""
    out: dict[str, str] = {}
    for char, names in EMOJI.items():
        for name in names:
            key = _key(name)
            if key in out:
                raise ValueError(
                    f"emoji alias {name!r} is claimed by both "
                    f"{out[key]} and {char}"
                )
            out[key] = char
    return out


ALIASES = _build_aliases()


def _alias_pattern(alias: str) -> str:
    """One alias as a regex: words separated by any run of space or hyphen, so
    a single "thumbs up" entry also matches "thumbs-up" and "thumbs  up"."""
    return r"[\s\-]+".join(re.escape(word) for word in alias.split())


# Longest alias first: regex alternation takes the first branch that matches at
# a position, so without this "crying laughing emoji" would stop at "crying".
_ALTERNATION = "|".join(
    _alias_pattern(a) for a in sorted(ALIASES, key=len, reverse=True)
)
_MARKER = r"emojis?"
# Either order, with the marker adjacent. [\s,]+ tolerates the comma the STT
# sometimes drops in ("rocket, emoji").
_SPOKEN_EMOJI_RE = re.compile(
    rf"\b(?:(?P<before>{_ALTERNATION})[\s,]+{_MARKER}"
    rf"|{_MARKER}[\s,]+(?P<after>{_ALTERNATION}))\b",
    re.IGNORECASE,
)


def apply_spoken_emoji(text: str) -> str:
    """Replace spoken emoji commands with the emoji. Text containing no
    "<name> emoji" / "emoji <name>" pair comes back unchanged."""
    if not text or "emoji" not in text.lower():
        return text  # cheap reject: the marker is mandatory

    def swap(m: re.Match) -> str:
        alias = m.group("before") or m.group("after")
        return ALIASES.get(_key(alias), m.group(0))

    return _SPOKEN_EMOJI_RE.sub(swap, text)


#: Every emoji this module can produce — used by the cleanup guard to notice a
#: model that deleted one the speaker asked for.
EMOJI_CHARS = frozenset(EMOJI)

# Longest first again: "❤️" is two code points (heart + variation selector) and
# must be tried before any single-code-point emoji that starts the same way.
_EMOJI_ALT = "|".join(re.escape(c) for c in sorted(EMOJI, key=len, reverse=True))
# An emoji, then the separator before the next one. The next emoji is a
# lookahead so it is not consumed and can start the following match, which
# collapses a whole run in a single pass.
_EMOJI_SEPARATOR_RE = re.compile(
    rf"({_EMOJI_ALT})[,\s]+(?=(?:{_EMOJI_ALT}))"
)
# The comma leading INTO a run goes as well: "I like food, 😂" is a pause you
# made while speaking, not punctuation you want to read. Only a comma or
# semicolon — a period is a real sentence end and stays ("Ship it. 🚀").
# (?<=\S) keeps a run at the very start of the text from gaining a leading
# space.
_EMOJI_LEADIN_RE = re.compile(
    rf"(?<=\S)[ \t]*[,;]+[ \t]*(?=(?:{_EMOJI_ALT}))"
)
# Cheap reject: scanning the 187-branch alternation over a whole dictation costs
# ~285µs even when there is no emoji in it. A regex character class is NOT the
# way to rule that out — Python's re falls off a fast path when a class is full
# of non-BMP code points and takes ~260µs, barely better. frozenset.isdisjoint
# walks the string in C against a hash set: ~33µs for the same text.
_EMOJI_LEAD_CHARS = frozenset(char[0] for char in EMOJI)


def collapse_emoji_runs(text: str) -> str:
    """Butt consecutive emoji together and drop the comma leading into them:
    "I like food, 😂, 👍, 🔥, 🚀." -> "I like food 😂👍🔥🚀."

    Dictating emoji produces separators that are pauses in speech but read as
    punctuation on screen — nobody types "food, 😂, 👍". A period before a run
    survives, because that is a real sentence ending.

    Separate from apply_spoken_emoji, and not gated on the marker word, because
    the cleanup model often emits the emoji itself — by then "emoji" is gone
    from the text and the marker-gated pass would skip it.
    """
    if not text or _EMOJI_LEAD_CHARS.isdisjoint(text):
        return text
    text = _EMOJI_SEPARATOR_RE.sub(r"\1", text)
    return _EMOJI_LEADIN_RE.sub(" ", text)
