"""MusicTool: AI transcription of electronic music audio into multi-track MIDI."""

from .pipeline import TranscriptionOptions, TranscriptionResult, Track, Note, transcribe

__all__ = ["TranscriptionOptions", "TranscriptionResult", "Track", "Note", "transcribe"]
__version__ = "0.1.0"
