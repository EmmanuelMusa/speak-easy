"""Push-to-talk main loop: hold the hotkey to record, release to dictate.

Uses the global-hotkeys package (Win32 hooks) for system-wide press/release
callbacks. Transcription + cleanup + injection run on a worker thread so the
hotkey hook is never blocked. Also owns training mode (feedback -> learning)
and applies settings changes coming back from the overlay's settings dialog.
"""

from __future__ import annotations

import logging
import threading
import time

from .audio import Recorder
from .cleanup import Cleaner
from .config import Config, save_config_updates
from .context import ContextStore
from .live_cleanup import LiveCleanup
from . import focus, power, sound
from .injection import Injector
from .overlay import Overlay
from .streaming import StreamingSession
from .stt import Transcriber
from .training import TrainingStore

log = logging.getLogger(__name__)


def _binding_is_valid(binding: str) -> bool:
    """True if every key in `binding` is a name global_hotkeys recognises.

    Mirrors the library's parsing — chords split on ',', keys on '+' — and
    uses its own name→virtual-key table, so we can reject a bad key before the
    live rebind tears the working hotkey down. If the library can't be
    imported (validation impossible) we optimistically allow it and let the
    rebind attempt be the guard.
    """
    try:
        from global_hotkeys.hotkey_checker import _to_virtualkey
    except Exception:
        return True
    stripped = binding.replace(" ", "")
    if not stripped:
        return False
    # Mirror the library: chords on ',', keys on '+', with NO empty-token
    # filtering — a dangling separator ("control +", "f9,") yields an empty
    # key the library maps to None and rejects, so we must too. Its number
    # parser can raise on odd input, so any failure here means "not valid".
    try:
        for chord in stripped.split(","):
            keys = chord.split("+")
            if any(not k or _to_virtualkey(k) is None for k in keys):
                return False
    except Exception:
        return False
    return True


class PushToTalkApp:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.recorder = Recorder(cfg.audio)
        self.transcriber = Transcriber(cfg.stt)
        self.training = TrainingStore()
        self.context = ContextStore(cfg.context)
        self.cleaner = Cleaner(cfg.cleanup, training=self.training)
        self.injector = Injector(cfg.injection)
        self.overlay = Overlay(
            cfg.overlay.enabled, level_source=lambda: self.recorder.level
        )
        self.overlay.on_settings = self._apply_settings
        self.overlay.on_feedback = self._record_feedback
        self._quit = threading.Event()
        self.overlay.on_quit = self._quit.set
        self._busy = threading.Lock()
        self._session: StreamingSession | None = None
        self._live: LiveCleanup | None = None
        self._surrounding: focus.Surrounding | None = None
        self._press_gen = 0  # invalidates surrounding reads from older presses
        self._last_audio: tuple | None = None  # (audio, sr) held for correction capture

    # -- hotkey callbacks ------------------------------------------------

    def _on_press(self) -> None:
        if self._busy.locked():
            # Otherwise this drop is invisible: you speak, nothing appears.
            log.info("Hotkey ignored: still processing the previous utterance")
            return
        if self.recorder.recording:
            return  # key autorepeat: the hold is already in progress
        log.info("Recording... (release %s to dictate)", self.cfg.hotkey.binding)
        if self.cfg.audio.start_sound:
            sound.play_start_cue(self.cfg.audio.start_sound_volume)
        self.recorder.start()
        if self.cfg.stt.streaming:
            self._session = StreamingSession(
                self.transcriber,
                self.recorder.snapshot,
                self.cfg.audio.sample_rate,
            ).start()
            if self.cfg.cleanup.enabled and self.cfg.cleanup.streaming:
                self._live = LiveCleanup(
                    self._session,
                    self.cleaner,
                    context_provider=self.context.cleanup_context,
                    surrounding_provider=lambda: self._surrounding,
                ).start()
        self._surrounding = None
        if self.cfg.context.surrounding:
            self._press_gen += 1
            gen = self._press_gen

            def _read_surrounding() -> None:
                # Used for cleanup casing/continuation and inject spacing
                # only — never fed to Whisper (prompt text can leak into
                # the transcript, upstream of every guard).
                s = focus.read_surrounding(
                    self.cfg.context.surrounding_before_chars,
                    self.cfg.context.surrounding_after_chars,
                )
                if gen == self._press_gen:
                    self._surrounding = s

            threading.Thread(target=_read_surrounding, daemon=True).start()
        self.overlay.show_recording()

    def _on_release(self) -> None:
        if not self.recorder.recording:
            return
        audio = self.recorder.stop()
        session, self._session = self._session, None
        live, self._live = self._live, None
        surrounding = self._surrounding
        self.overlay.show_processing()
        threading.Thread(
            target=self._process, args=(audio, session, surrounding, live),
            daemon=True,
        ).start()

    def _process(
        self,
        audio,
        session: StreamingSession | None = None,
        surrounding: focus.Surrounding | None = None,
        live: LiveCleanup | None = None,
    ) -> None:
        with self._busy:
            try:
                t0 = time.perf_counter()
                has_before = surrounding is not None and surrounding.before.strip()
                source = self.cfg.cleanup.punctuation_source
                fallback_full = None
                if session is not None:
                    raw = session.finish(audio, source)
                    fallback_full = session.fallback_text()
                else:
                    tr = self.transcriber.transcribe(audio)
                    raw = tr.model_text(source)
                    fallback_full = tr.fallback_text
                t_stt = time.perf_counter()
                if not raw:
                    log.info("No speech detected.")
                    return
                # Surrounding text subsumes dictation history (the last
                # dictation was typed into that very field), so don't send
                # both to the LLM.
                if live is not None:
                    cleaned = live.finalize(raw)
                else:
                    cleaned = self.cleaner.clean(
                        raw,
                        fallback_text=fallback_full,
                        context=None if has_before else self.context.cleanup_context(),
                        surrounding=surrounding,
                    )
                t_clean = time.perf_counter()
                self.context.add(cleaned)
                if surrounding is not None and focus.needs_leading_space(
                    surrounding.before, cleaned
                ):
                    cleaned = " " + cleaned
                self.injector.inject(cleaned)
                t_end = time.perf_counter()
                log.info(
                    "Done in %.2fs (stt %.2fs, clean %.2fs, inject %.2fs): %r",
                    t_end - t0, t_stt - t0, t_clean - t_stt, t_end - t_clean,
                    cleaned,
                )
                if self.cfg.training.enabled:
                    self._last_audio = (audio, self.cfg.audio.sample_rate)
                    self.overlay.request_feedback(raw, cleaned)
            finally:
                self.overlay.show_idle()

    # -- training feedback ---------------------------------------------------

    def _record_feedback(self, raw: str, output: str, rating, transcript,
                         ideal, tags) -> None:
        audio_path = None
        stash, self._last_audio = self._last_audio, None
        if transcript and self.cfg.training.save_correction_audio and stash is not None:
            audio, sr = stash
            audio_path = self.training.save_audio(audio, sr)
        # verdict retained only for the stored schema / legacy few-shot.
        verdict = "ok" if (rating == 5 and not ideal) else "bad"
        self.training.record(
            raw, output, verdict, ideal,
            rating=rating, transcript=transcript, tags=tags, audio_path=audio_path,
        )
        if not ideal:
            log.info("Feedback: rating %s%s%s", rating,
                     f", tags {tags}" if tags else "",
                     " + audio" if audio_path else "")
            return
        log.info("Correction saved (rating %s)%s", rating,
                 " + audio" if audio_path else "")
        # The corrected text is what should inform the next dictation.
        self.context.replace_last(ideal)
        if self.cfg.training.replace_on_correction:
            replaced = self.injector.replace_last(ideal)
            log.info(
                "In-place correction: %s",
                "applied" if replaced else "skipped (text changed)",
            )

    # -- settings ---------------------------------------------------------------

    def _settings_snapshot(self) -> dict:
        return {
            "training_enabled": self.cfg.training.enabled,
            "replace_on_correction": self.cfg.training.replace_on_correction,
            "hotkey": self.cfg.hotkey.binding,
            "stt_model": self.cfg.stt.model,
            "ollama_model": self.cfg.cleanup.ollama_model,
            "cleanup_enabled": self.cfg.cleanup.enabled,
            "delivery_method": self.cfg.injection.delivery_method,
            "target_pairs": self.cfg.training.target_pairs,
        }

    def _apply_settings(self, values: dict) -> None:
        """Apply settings from the dialog live, then persist to config.toml."""
        log.info("Applying settings: %s", values)
        old_hotkey = self.cfg.hotkey.binding
        old_model = self.cfg.stt.model

        self.cfg.training.enabled = bool(values.get("training_enabled", False))
        self.cfg.training.replace_on_correction = bool(
            values.get("replace_on_correction", True)
        )
        # Validate the key name BEFORE anything is torn down or persisted. An
        # unrecognised name ("ctrl", "esc", "cmd + q") makes global_hotkeys
        # raise on register; if that reached the live rebind it would leave NO
        # working hotkey and write the bad value to disk (bricking the next
        # start). So reject it here and keep the current key instead.
        new_hotkey = values.get("hotkey", old_hotkey)
        if new_hotkey != old_hotkey and not _binding_is_valid(new_hotkey):
            log.warning(
                "Ignoring invalid push-to-talk key %r; keeping %r. Use "
                "global-hotkeys names joined by '+', e.g. 'f9', 'control + "
                "shift + space' (note: 'control' not 'ctrl', 'escape' not "
                "'esc', 'enter' not 'return').",
                new_hotkey, old_hotkey,
            )
            new_hotkey = old_hotkey
        self.cfg.hotkey.binding = new_hotkey
        self.cfg.stt.model = values.get("stt_model", old_model)
        self.cfg.cleanup.ollama_model = values.get(
            "ollama_model", self.cfg.cleanup.ollama_model
        )
        self.cfg.cleanup.enabled = bool(values.get("cleanup_enabled", True))
        self.cfg.injection.delivery_method = values.get(
            "delivery_method", self.cfg.injection.delivery_method
        )

        save_config_updates({
            "training": {
                "enabled": self.cfg.training.enabled,
                "replace_on_correction": self.cfg.training.replace_on_correction,
            },
            "hotkey": {"binding": self.cfg.hotkey.binding},
            "stt": {"model": self.cfg.stt.model},
            "cleanup": {
                "ollama_model": self.cfg.cleanup.ollama_model,
                "enabled": self.cfg.cleanup.enabled,
            },
            "injection": {"delivery_method": self.cfg.injection.delivery_method},
        })

        if self.cfg.hotkey.binding != old_hotkey:
            if not self._rebind_hotkey(previous=old_hotkey):
                # Rebind failed at the library level even though the name
                # validated; the previous key was restored, so reflect that.
                self.cfg.hotkey.binding = old_hotkey
        if self.cfg.stt.model != old_model:
            threading.Thread(target=self._reload_stt, daemon=True).start()
        self.overlay.send_settings(self._settings_snapshot())

    def _rebind_hotkey(self, previous: str | None = None) -> bool:
        """Re-register the push-to-talk key. Returns True on success. On
        failure the `previous` binding is re-registered so the user is never
        left without a working hotkey."""
        import time

        import global_hotkeys as gh

        def _register(binding: str) -> None:
            gh.stop_checking_hotkeys()
            try:
                gh.clear_hotkeys()
            except AttributeError:
                pass
            # global_hotkeys' own restart path sleeps here to let the previous
            # checker thread die before a fresh one starts; skipping it lets
            # the old thread linger and double-fire callbacks.
            time.sleep(0.5)
            gh.register_hotkeys([
                [binding, self._on_press, self._on_release, False],
            ])
            gh.start_checking_hotkeys()

        try:
            _register(self.cfg.hotkey.binding)
            log.info("Hotkey rebound to '%s'", self.cfg.hotkey.binding)
            return True
        except Exception as exc:
            log.warning(
                "Hotkey rebind to '%s' failed (%s)", self.cfg.hotkey.binding, exc
            )
            if previous and previous != self.cfg.hotkey.binding:
                try:
                    _register(previous)
                    log.info("Restored previous hotkey '%s'", previous)
                except Exception:
                    log.exception(
                        "Could not restore previous hotkey; press may be dead "
                        "until restart"
                    )
            return False

    def _reload_stt(self) -> None:
        log.info("Loading Whisper model '%s'...", self.cfg.stt.model)
        new = Transcriber(self.cfg.stt)
        new._load()
        self.transcriber = new
        log.info("Speech model switched to '%s'", self.cfg.stt.model)

    # -- main loop ---------------------------------------------------------

    def run(self) -> None:
        import global_hotkeys as gh  # lazy: installs Win32 hooks

        # On battery, Windows throttles background processes (EcoQoS) hard
        # enough to drop audio frames and stall the pipeline — opt out.
        power.opt_out_of_power_throttling()
        if power.on_battery():
            log.info(
                "Running on battery: the GPU is downclocked, so dictation "
                "will be slower (cleanup waits longer instead of degrading)."
            )
        # Warm up both models so the first dictation isn't slow. Ollama's
        # cold load (~40s for an 8B model) runs in the background so the
        # hotkey is usable immediately.
        threading.Thread(target=self.cleaner.warmup, daemon=True).start()
        if self.cfg.context.surrounding:
            # comtypes code-generates its UIA wrapper on first use (~2s);
            # pay that at startup, not on the first hotkey press.
            threading.Thread(target=focus.warmup, daemon=True).start()
        self.overlay.show_idle()  # slim ready-bar at the bottom of the screen
        self.overlay.send_settings(self._settings_snapshot())
        log.info("Loading Whisper model '%s'...", self.cfg.stt.model)
        self.transcriber._load()

        gh.register_hotkeys([
            [self.cfg.hotkey.binding, self._on_press, self._on_release, False],
        ])
        gh.start_checking_hotkeys()
        log.info(
            "Speak Easy ready. Hold '%s' to dictate; Ctrl+C here or the "
            "Quit button in settings to quit.",
            self.cfg.hotkey.binding,
        )
        if self.cfg.training.enabled and self.cfg.training.save_correction_audio:
            n, t = self.training.trainable_pair_count(), self.cfg.training.target_pairs
            if n >= t:
                log.info("Acoustic training data: %d/%d pairs — ready to fine-tune.", n, t)
            else:
                log.info("Acoustic training data: %d/%d pairs (%d more to start "
                         "fine-tuning your voice).", n, t, t - n)
        try:
            while not self._quit.wait(0.5):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            gh.stop_checking_hotkeys()
            self.overlay.close()
            log.info("Stopped.")
