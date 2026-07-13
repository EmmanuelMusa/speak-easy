# Speak Easy

A fully local, offline [Wispr Flow](https://wisprflow.ai)-style dictation app for Windows.

**Hold a hotkey → speak → release → cleaned text appears at your cursor, in whatever app is focused.** No cloud calls anywhere in the pipeline.

## How it works

```
hold F9 ──► mic capture ──► Silero VAD + faster-whisper (CUDA/CPU)
                                        │ raw transcript
                                        ▼
                    < 10 words?  ──yes──► local filler strip only  (fast path)
                          │no
                          ▼
                Ollama LLM cleanup (fillers, casing, punctuation, grammar)
                          │
                          ▼
            inject at cursor: clipboard Ctrl+V  or  SendInput Unicode typing
```

- **STT:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) with `vad_filter=True` (Silero VAD gates silence). Batch-after-silence — no streaming.
- **Cleanup:** near-verbatim by design. Only vocal noises (`um, uh, mm, ehmm, oouu...`) are removed — conversational phrases ("you know", "I mean"), stutters ("the the"), and your word choice are untouched. A local LLM served by [Ollama](https://ollama.com) adds capitalization/punctuation, formats dictated enumerations as lists, and resolves explicit self-corrections ("no, sorry, by 3 p.m." → keeps only 3 p.m., in your words). A divergence guard rejects any LLM output that paraphrases (too many words you never said) and delivers the locally-stripped verbatim text instead. Utterances **under 10 words skip the LLM entirely** (~0 ms cleanup). If Ollama is down or slow, we fall back to the local strip — dictation never blocks.
- **Injection:** two methods (see below).

## Setup

### 1. Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### 2. Ollama (the cleanup LLM)

Install Ollama from <https://ollama.com/download> (or `winget install Ollama.Ollama`), then pull a model:

```powershell
ollama pull llama3.1:8b     # default — best quality
# lighter alternatives (set [cleanup].ollama_model in config.toml):
ollama pull phi3:mini       # ~2.2 GB, fastest
ollama pull mistral:7b
```

Ollama runs as a local server on `http://localhost:11434`. Nothing leaves your machine.

### 3. Hotkey setup

The push-to-talk key is set in `config.toml`:

```toml
[hotkey]
binding = "f9"    # hold to record, release to dictate
```

Syntax follows the [global-hotkeys](https://pypi.org/project/global-hotkeys/) package: single keys (`"f9"`) or combos (`"control + shift + space"`). Pick a key you don't use elsewhere — it's captured system-wide.

## Run

```powershell
.\.venv\Scripts\python -m app            # start the push-to-talk loop
.\.venv\Scripts\python -m app --help     # all options
.\.venv\Scripts\python -m app --dry-run assets\sample.wav   # offline pipeline demo
.\.venv\Scripts\python -m app --no-cleanup                  # skip the LLM pass
```

First run downloads the Whisper model (`tiny.en` ≈ 75 MB) to your Hugging Face cache; after that everything is offline.

Only one instance runs at a time (two would double-type every dictation): a second launch shows "already running" and exits. `--dry-run` is exempt.

## Settings & training mode

Hover the pill — a gear icon fades in on its right; click it for the settings window (training mode, hotkey, speech model, cleanup model, cleanup on/off, text delivery). Changes apply live and save to `config.toml`; the hotkey rebinds instantly and a speech-model change reloads in the background.

**Training mode** (off by default): when on, a small 👍/👎 panel appears above the pill after each dictation. It never steals your keyboard focus — keep typing, or ignore it and it fades after 6 s.
- 👍 (or ignore) → logged as good.
- 👎 → a field opens for what it *should* have said. That correction is saved to `training_data.jsonl` and used two ways: recent corrections are injected into the cleanup prompt as few-shot examples (style/formatting improves immediately), and misheard words (names, jargon) are auto-added to a learned vocabulary (`learned_vocab.json`) so they stop being mistranscribed.

**Fix-in-place:** when you submit a correction, the app also replaces the text it already typed into your app — *but only if you haven't touched it since*. It re-selects exactly what it typed, verifies it's unchanged (clipboard compare), and either replaces it or, if you've kept typing, leaves your text alone. Toggle with "Fix the typed text in place on correction" in Settings (`[training] replace_on_correction`).

**Review learnings:** Settings → "Review learnings…" opens a panel listing every correction example and learned word, each with a **Forget** button. Removing a lesson takes effect on your next dictation. Undo a bad correction anytime.

Nothing is fine-tuned — learning is instant and fully local. The collected JSONL is also exactly the dataset you'd need for a real offline LoRA fine-tune later.

## Spoken commands

While dictating you can say:

| You say | You get |
|---|---|
| "...by 9am, **no, sorry**, by 3pm" (or "scratch that", "wait, no") | only the corrected version: "...by 3pm" |
| "**in bracket** less wide also" / "open bracket ... close bracket" | (less wide also) |
| "she said **quote** I'll handle it **unquote**" | she said "I'll handle it" |
| "he told me I will be there by noon" (reported speech) | he told me, "I will be there by noon." |
| "first the budget second the timeline third the staffing" | a numbered list |

Correction cues always go through the LLM, even on short utterances.

## The two injection methods

Set `[injection].delivery_method` in `config.toml`:

| Method | How | Trade-off |
|---|---|---|
| `clipboard` (default) | Copies the text, synthesizes **Ctrl+V**, then restores your previous clipboard. | Fastest for long text; works in virtually every app. Briefly touches the clipboard. |
| `sendinput` | Types the text character-by-character via Win32 `SendInput` with `KEYEVENTF_UNICODE`. | No clipboard side effects; slower on long text; a few apps ignore synthetic keystrokes. |

## Configuration reference

Everything lives in `config.toml`:

- `[stt]` — `model` (`tiny.en` default; `base`/`small`/`medium`/`large-v3` for more accuracy), `device` (`auto` tries CUDA, falls back to CPU int8), `compute_type`.
- `[cleanup]` — `enabled`, `ollama_model`, `timeout_seconds`, `custom_vocabulary` (names/jargon the LLM must preserve exactly). Every utterance goes through the LLM, however short — "is it ready" comes out "Is it ready?"; the local strip is only the failure fallback.
- `[cleanup].battery_timeout_multiplier` — on battery the GPU downclocks and the LLM runs several times slower; the Ollama timeout is multiplied by this (default 4) so cleanup completes at full quality instead of timing out. The app also opts itself out of Windows Power Throttling (EcoQoS) at startup, so audio capture and transcription keep full CPU speed on battery. For the fastest on-battery dictation, additionally set Windows Power mode to "Best performance" and the NVIDIA setting "Battery Boost" off — GPU clocks are driver policy and can't be raised from the app.
- **Quitting** — hover the pill; it morphs into a **Settings** chip. Click it, then "Quit" in the popup. This fully closes the app (hotkey and single-instance lock released) so you can start it again — e.g. after pulling new features.
- `[injection]` — `delivery_method`, `paste_delay`, `verify_paste`, `terminal_paste`. In detected terminals (Windows Terminal, cmd, PowerShell, Alacritty, WezTerm…) the paste chord is `terminal_paste`: `"ctrl_v"` (default — works in modern consoles and TUI apps like Claude Code) or `"shift_insert"` for older consoles. Clipboard `OpenClipboard` is retried (it fails transiently when a clipboard-history tool or the pasting app holds it); if the clipboard is unreachable even after retries, the text is typed directly instead of vanishing. With `verify_paste` on, the app polls the field for up to ~2s after pasting and only on a *sustained* confirmed failure (field readable and unchanged the whole time, focus still on the target) keeps the text on the clipboard and notifies you — a slow app on battery no longer triggers a false "Ctrl+V it yourself" prompt.
- `[hotkey]` — `binding`.
- `[overlay]` — `enabled`. A subtle dark pill at the bottom center (antialiased Qt rendering, click-through): a faint slim bar when idle, expanding into a compact voice-reactive waveform (centered bell shape, edge-faded bars) while you dictate, and a gentle traveling wave while transcribing/cleaning. Hovering the idle pill morphs it in place into a **Settings** chip — a vector gear icon that spins into place beside a "Settings" label — which opens the settings popup. Renders in its own small subprocess; if PySide6 isn't installed the app runs fine without it.
- `[audio].start_sound` / `start_sound_volume` — a subtle low tone (soft G3+fifth, ~150 ms, Hann-enveloped so there's no click) plays the instant recording starts, so you get non-visual confirmation the hotkey registered. Set `start_sound = false` to silence it. Windows-only; no-ops elsewhere.
- `[stt].beam_size` — decoding beam (1 = fastest, 5 = most accurate, default 2).
- `[context]` — `enabled`, `expiry_seconds`, `max_chars`. Cross-utterance context: your recent dictations (in-memory only, recency-capped) become a reference block for the cleanup LLM, so names, jargon, and casing stay consistent across consecutive dictations. Context is deliberately **never** fed to Whisper — decoder prompt text can leak into the raw transcript on ambiguous audio, upstream of the divergence guard. A training-mode correction replaces the context too, so the fix propagates forward.
- `[context].surrounding` — surrounding-text context (default on). At hotkey press the text around the caret is read via UI Automation (off the hot path, in a worker thread): it drives cleanup continuation — dictating mid-sentence continues in lowercase without a stray trailing period — and a leading space is added when pasting directly after a word. Like dictation history, it is never fed to Whisper. Best-effort: works in Word/Outlook/browsers/most native apps; terminals and canvas editors (Google Docs) silently get the old behavior. Password fields are never read.
- **Pause punctuation** — your pauses become punctuation: a spoken beat (~½s) inserts a comma, a real stop (~1s) a full stop with the next word capitalized. Whisper's own `?` `.` `!` always win; a trailing comma before a long stop is upgraded to a full stop, because the voice outranks the language model on where you actually stopped.
- `[cleanup].streaming` — live cleanup (default on, requires `[stt].streaming`). Finished sentences are cleaned by the LLM while you're still dictating; at release only the last unfinished sentence is cleaned, so the wait stays near-constant regardless of dictation length. Self-corrections that reference the previous sentence merge and re-clean it.
- **Lists** — a dictated enumeration becomes a numbered list, built **deterministically** in code (not left to the model, which is unreliable at restructuring already-punctuated sentences at this size). When the text carries 2+ ordinal cues at sentence/clause starts ("Firstly, I'll… Secondly, I'll… Thirdly, I'll…", "First, … Second, …", "Number one, … Number two, …", mixing commas and periods), the lead-in becomes a "…:" line and each ordinal-led clause becomes a numbered item with the ordinal word dropped and everything else kept. Ordinary prose that merely contains "first" (e.g. "first aid", "I went first then came home") is never turned into a list.
- **Thinking pauses** — a pause after a function word ("we should …", "move it to the …") is treated as the speaker thinking, never as punctuation, and the cleanup LLM is instructed to drop pause-punctuation that interrupts a grammatical unit.
- `[stt].streaming` — transcribe while the hotkey is held (default on). Settled speech is committed by background passes, and each pass sees the text so far as decoder context; on release only the last ~second of audio remains to transcribe, so the wait stays ~0.1s no matter how long you dictated. Turn off to restore single-pass transcription at release.

## Tests

```powershell
.\.venv\Scripts\python -m pytest
```

Covers filler stripping, the <10-word skip-LLM branch, LLM-failure fallback, and injection dispatch (OS keystrokes and Ollama are mocked).

## Notes

- Windows-first: injection and hotkeys use Win32 APIs.
- CUDA: with an NVIDIA GPU, `device = "auto"` uses CUDA when available and falls back to CPU otherwise (including lazy failures like a missing `cublas64_12.dll`). To enable GPU STT: `pip install -r requirements-gpu.txt` — the app finds and registers the DLLs automatically at startup. `tiny.en`/`base` are faster than real-time even on CPU, so this is optional; with the GPU active, `small.en` or `large-v3` become practical for higher accuracy. (Ollama handles its own CUDA — the LLM runs on GPU out of the box.)
- No cloud STT/LLM in the core path, by design.
