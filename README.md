# AI_AUDIO_LAB

## Lyra Studio (karaoke studio): start here
- `app/`: the web UI (React + Vite + TS). Simple Mode by default, plus the full studio.
- `server/`: the processing server (FastAPI): YouTube/file import, separation, Whisper transcription, alignment, MP4 render, accounts. It serves the UI at http://localhost:8443.
- On Windows: `run_lyra.bat`. Details: `server/README.md`, `app/README.md`. CI: `.github/workflows/lyra.yml`.
- The earlier Gradio app (`app.py`, `web_ui.py`, `logic.py`, `backend.py`) stays as it was; the server reuses the same model stack. `legacy/app_from_broken_filename.py` is an older `app.py` that was uploaded with its code as the file name, which broke `git clone`.

---


AI_AUDIO_LAB is a modular, Python-based Smart DAW focused on research and experimental
AI-assisted audio workflows. The project is organized to separate the real-time audio
engine, UI, AI modules, and integrations while keeping a clear, testable architecture.

---

## Project Structure ✅

```
AI_AUDIO_LAB/
├─ core/                 # Audio engine, DSP building blocks
│  ├─ __init__.py
│  └─ engine.py          # AudioEngine singleton (sounddevice wrapper)
├─ ui/                   # PyQt6 UI widgets and main window
│  ├─ __init__.py
│  └─ main_window.py     # MainWindow + waveform placeholder
├─ ai_modules/           # Placeholders for Spleeter/Demucs, pitch, chords
│  ├─ __init__.py
│  └─ (spleeter, pitch_detection, chord_gen)
├─ integrations/         # YouTube downloads, lyrics, metadata
│  ├─ __init__.py
│  └─ (youtube, lyrics)
├─ utils/                # File helpers and formatting utilities
│  ├─ __init__.py
│  └─ file_utils.py
├─ requirements.txt      # Project dependencies
└─ main.py               # Application entry point
```

---

## Quickstart 🚀

1. Create a Python 3.10+ virtual environment and activate it.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python main.py
```

> Note: The current waveform viewer and AI modules are placeholders. The
> `AudioEngine` uses `sounddevice` with a simple pass-through callback; expect
> device-dependent behavior when starting the stream.

---

## Next Steps / Roadmap 💡

- Replace waveform placeholder with a GPU-accelerated canvas for real-time rendering
- Add unit tests and CI
- Integrate separation (Demucs), transcription (Whisper), and advanced plugins
- Add robust audio routing, plugin host, and project/session management

---

If you'd like, I can run a quick smoke test or add a sample audio file and a minimal test harness next.
