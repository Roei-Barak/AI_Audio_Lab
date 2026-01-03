"""File-related helpers for loading/saving audio files."""
from __future__ import annotations

from typing import Optional

from pydub import AudioSegment


def load_audio(path: str) -> AudioSegment:
    """Load an audio file into a pydub.AudioSegment.

    Raises RuntimeError on failure.
    """
    try:
        return AudioSegment.from_file(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio file {path}: {exc}") from exc


def save_audio(segment: AudioSegment, path: str, format: Optional[str] = None) -> None:
    """Save an AudioSegment to disk."""
    try:
        segment.export(path, format=format)
    except Exception as exc:
        raise RuntimeError(f"Failed to save audio file {path}: {exc}") from exc
