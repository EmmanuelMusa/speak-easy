"""Speak Easy — a fully local Wispr Flow clone for Windows.

Pipeline: push-to-talk hotkey -> mic capture -> faster-whisper (Silero VAD)
-> local Ollama cleanup -> text injection at the cursor.
"""

__version__ = "0.1.0"
