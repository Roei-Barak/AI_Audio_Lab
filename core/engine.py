"""Audio engine responsible for real-time audio I/O using sounddevice.

This module exposes a singleton `AudioEngine` that manages a sounddevice
stream and offers start/stop controls.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import sounddevice as sd


class AudioEngine:
    """Singleton audio engine managing sounddevice streams.

    Usage:
        engine = AudioEngine.instance()
        engine.start()
        engine.stop()
    """

    _instance_lock = threading.Lock()
    _instance: Optional["AudioEngine"] = None

    def __new__(cls) -> "AudioEngine":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # avoid reinitialization on repeated calls
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._stream: Optional[sd.Stream] = None
        self.sample_rate: int = 44100
        self.channels: int = 2
        self.device: Optional[int] = None
        self._running: bool = False

    @classmethod
    def instance(cls) -> "AudioEngine":
        """Return the singleton instance."""
        return cls()

    def _callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, time, status) -> None:
        """Very small pass-through callback (placeholder).

        This is intentionally simple; replace with DSP chain later.
        """
        if status:
            # print status for initial debugging
            print(f"Stream status: {status}")

        try:
            # Simple pass-through if channels match
            outdata[:] = indata
        except Exception:
            # Silence on error to avoid loud artifacts
            outdata.fill(0)

    def start(self) -> None:
        """Start the audio stream.

        Raises:
            RuntimeError: if the device or stream fails to initialize.
        """
        if self._running:
            return

        try:
            self._stream = sd.Stream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self._callback,
                device=self.device,
            )
            self._stream.start()
            self._running = True
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(f"Failed to start audio stream: {exc}") from exc

    def stop(self) -> None:
        """Stop and close the audio stream."""
        if not self._running or self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
            self._running = False

    def is_running(self) -> bool:
        """Return whether the engine is currently running."""
        return self._running

    def list_devices(self) -> list:
        """Return a list of available audio devices for debugging.

        This is a convenience for use from the UI.
        """
        try:
            return sd.query_devices()
        except Exception as exc:
            raise RuntimeError(f"Unable to query devices: {exc}") from exc
