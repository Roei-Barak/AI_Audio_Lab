# Copilot / Agent Instructions for AI_AUDIO_LAB

Purpose: Give short, high-value guidance so an AI coding agent can be immediately productive in this repository.

- Big picture
  - Project is a modular Python Smart DAW.
  - Major packages:
    - `core/` — real-time audio engine & DSP (see `core/engine.py`). AudioEngine is a singleton managing a `sounddevice` Stream.
    - `ui/` — PyQt6 UI (see `ui/main_window.py`). MainWindow owns an `AudioEngine` instance and toggles Start/Stop.
    - `ai_modules/` — placeholder modules (e.g., `spleeter.py`, `pitch_detection.py`, `chord_gen.py`) exposing small class APIs that currently raise `NotImplementedError`.
    - `integrations/` — external integrations (e.g., `youtube.py`, `lyrics.py`) are stubs.
    - `utils/` — project helpers (add file utilities here).
  - Entry point: `main.py` (creates QApplication and shows `MainWindow`).

- Critical developer workflows
  - Run locally: create a Python 3.10+ venv -> `pip install -r requirements.txt` -> `python main.py`.
  - Audio device debugging: use `AudioEngine.instance().list_devices()` to inspect devices before starting the stream.
  - Expect device-dependent behavior when touching `sounddevice` (PortAudio). In CI, runs may fail unless headless or mocked.
  - Headless UI testing: set `QT_QPA_PLATFORM=offscreen` for non-GUI environments.

- Project-specific conventions and patterns
  - Use typed Python and module-level docstrings — follow existing style (type hints, docstrings, small functions).
  - Singleton pattern in `core/engine.py` uses `__new__` + `_initialized` to avoid reinitialization; preserve that pattern when extending the engine.
  - UI -> engine interaction: `MainWindow` holds the engine object and calls `start/stop`. Keep UI logic separated from audio processing.
  - AI modules are thin adapters (small classes with single-responsibility methods). Implementations should preserve these simple APIs (e.g., `SeparationModule.separate(audio)`).

- Integration & external dependency notes
  - `sounddevice` requires PortAudio and may throw at runtime — handle errors and surface them in the UI (see `on_play`/`on_stop` in `ui/main_window.py`).
  - `pydub` requires `ffmpeg` installed on the system path for file manipulations.
  - `librosa`, `numpy`, `scipy` are for offline analysis; heavy GPU-based models (PyTorch) are planned and listed in `requirements.txt`.

- Cross-component communication patterns
  - Audio flow is intended to be pushed/pulled via the engine callback. For visualizers, route audio frames to a thread-safe buffer and let the UI poll/update via a QTimer.
  - Avoid doing heavy work directly in the `sounddevice` callback — buffer and process on worker threads.

- Examples & quick patterns to follow
  - Start/Stop pattern:
    - engine = AudioEngine.instance();
    - engine.start()  # may raise RuntimeError
    - engine.stop()
  - Device list check:
    - devices = AudioEngine.instance().list_devices()
  - UI error handling: prefer `QMessageBox.critical(self, "Audio Error", str(exc))` for visible errors in the main thread.

- Tests and CI
  - No tests or CI config yet — when adding tests, mock `sounddevice` and set `QT_QPA_PLATFORM=offscreen`. Keep unit tests fast and avoid requiring audio hardware.

- When editing this file
  - Merge with any future `.github/copilot-instructions.md` content: preserve any human-written policies and example workflows.
  - Keep guidance short and code-linked: use file paths and concrete API examples.

If anything here is unclear or you'd like more detail (examples for visualization buffering, a mock-friendly `AudioEngine` wrapper, or a suggested unit-test scaffold), tell me which area to expand and I will iterate.