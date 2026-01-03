"""Placeholder for source separation module (e.g., Spleeter/Demucs integration)."""
from typing import Any


class SeparationModule:
    """Stub for future separation models like Spleeter or Demucs."""

    def __init__(self, model_name: str = "demucs") -> None:
        self.model_name = model_name

    def separate(self, audio: Any) -> Any:
        """Placeholder separation method.

        Replace with actual model inference code later.
        """
        raise NotImplementedError
