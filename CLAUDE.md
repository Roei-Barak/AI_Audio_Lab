# CLAUDE.md — AI_Audio_Lab

Comprehensive guide for AI assistants working in this repository.

---

## Project Overview

**AI_Audio_Lab** is a modular Python Karaoke Studio and Smart DAW (Digital Audio Workstation). Its primary purpose is to take a YouTube link and produce a karaoke video with synchronized subtitles and separated audio. The system has Hebrew-language UI throughout and full BIDI text support.

**Current state:** Two complete, working karaoke pipeline implementations exist (`V39`, `V64`) alongside a Gradio web UI (`app.py`). A planned modular refactor will extract each processing stage into standalone, independently runnable modules.

---

## Repository Structure

```
AI_Audio_Lab/
├── CLAUDE.md                      # This file
├── README.md                      # High-level project documentation
├── .github/
│   └── copilot-instructions.md    # Legacy AI agent guidance (superseded by this file)
├── .gitignore
│
├── app.py                         # Main Gradio web UI (Hebrew interface, V71)
├── backend.py                     # Large backend file — mostly commented-out legacy code
├── V39                            # Working karaoke pipeline v39 (Python script, no .py)
├── V64                            # Working karaoke pipeline v64 (Python script, no .py)
├── MP4_Aduio_Overload             # Standalone Tkinter utility: swap audio in MP4
│
├── core/
│   ├── __init__.py
│   └── engine.py                  # AudioEngine singleton (real-time sounddevice I/O)
│
└── ai_modules/
    ├── __init__.py
    ├── spleeter.py                # Stub: SeparationModule (raises NotImplementedError)
    ├── pitch_detection.py         # Stub: PitchDetector (raises NotImplementedError)
    └── chord_gen.py               # Stub: ChordGenerator (raises NotImplementedError)
```

### Planned structure (not yet created)

```
modules/           # Standalone CLI modules (the modular refactor)
│   ├── downloader.py
│   ├── separator.py
│   ├── transcriber.py
│   └── renderer.py
cli/               # Lean CLI entry point
│   └── main.py
gui/               # Extended GUI
│   ├── gradio_app.py
│   └── desktop_app.py
integrations/      # External service stubs (YouTube, lyrics)
utils/             # Shared helpers
requirements.txt   # Does not exist yet — must be created
```

---

## Core Pipeline (How It Works)

The full karaoke pipeline runs in this sequence:

```
YouTube URL / local file
        ↓
1. DOWNLOAD   — yt-dlp → MP4
        ↓
2. SEPARATE   — audio-separator (Kim_Vocal_2.onnx) → Vocals.wav + Playback.wav
              — optional 4-stem: htdemucs_ft.yaml → Drums, Bass, Vocals, Other
        ↓
3. TRANSCRIBE — Whisper (ivrit-ai/whisper-large-v3-turbo for Hebrew,
                         openai/whisper-large-v3 for English)
              — word-level timestamps → .ass subtitle file
        ↓
4. RENDER     — FFmpeg: overlay .ass subtitles on original video
                        swap audio to Playback.wav
              → {title}_KARAOKE.mp4
```

Output directory: `Karaoke_Output/{song_title}/`

---

## Key Files — What They Do

### `V64` (primary reference implementation)
The most complete and up-to-date working implementation. Use this as the canonical reference when building new modules.

**`BackendProcessor` class methods:**

| Method | Purpose |
|---|---|
| `get_video_info(query, logs)` | Resolves URL or search query → metadata dict |
| `download_video(video_info, logs)` | Downloads MP4 via yt-dlp |
| `separate_audio(video_path, folder, logs, mode, save_4, force)` | Separates vocals/instrumental; optionally 4 stems |
| `transcribe_audio(audio_path, folder, title, logs, lang)` | Whisper transcription → .ass file |
| `render_video(video_path, audio_path, ass_path, info, logs, use_bidi)` | FFmpeg render |
| `process_song_pipeline(query, lang, save_4, use_bidi, force)` | Full pipeline orchestrator |
| `ass_to_dataframe(ass_path)` | Parse .ass → pandas DataFrame (for UI editing) |
| `dataframe_to_ass(df, original_ass_path, output_path)` | DataFrame → .ass file |
| `update_ass_style(ass_path, font_size, color_hex)` | Edit .ass style header in-place |
| `analyze_audio(audio_path)` | BPM + musical key via librosa |
| `convert_to_wav(input, output)` | FFmpeg: any format → 16kHz mono PCM WAV |
| `_fix_hebrew_text(text)` | arabic_reshaper + python-bidi for RTL display |
| `_fmt_ass_time(seconds)` | Float seconds → `H:MM:SS.CC` ASS timestamp |
| `_sanitize_filename(name)` | Strip illegal characters for cross-platform paths |
| `cleanup_resources()` | torch.cuda.empty_cache() + gc.collect() |

**`ResourceManager` class (V64 only):**
- Semaphore limits max 2 concurrent heavy tasks (separation, transcription)
- Monitors CUDA VRAM; warns when < 4 GB free
- `get_status()` returns human-readable Hebrew status string

### `app.py`
Gradio 6.0 web UI. Imports from a `logic` module (expected to be the backend). Tabs:
- **⚡ שיר בודד** — single song pipeline
- **📚 רשימת שירים** — batch list processing
- **📝 Dashboard** — load/edit ASS subtitles in a table, re-render
- **🛠️ כלים מתקדמים** — individual tool tabs (download, separate, transcribe, analyze)

### `core/engine.py`
Singleton real-time audio engine using `sounddevice`. Currently a pass-through placeholder. **Preserve the singleton pattern** (`__new__` + `_initialized`). The callback must never do heavy work — buffer to worker threads.

### `backend.py`
~1400 lines, mostly commented-out code. Treat as a historical reference only. Do not uncomment and use — use `V64` instead.

### `V39`
Earlier version of the pipeline. Functionally similar to V64 but without `ResourceManager` or filename sanitization. Reference only.

### `MP4_Aduio_Overload`
Standalone Tkinter script. Lets the user pick an MP4 and a WAV, then replaces the audio using FFmpeg. Fully independent — no imports from the rest of the project.

---

## Models Used

| Task | Model | Notes |
|---|---|---|
| 2-stem separation | `Kim_Vocal_2.onnx` | UVR5 model, vocals + instrumental |
| 4-stem separation | `htdemucs_ft.yaml` | Demucs, drums + bass + vocals + other |
| Hebrew transcription | `ivrit-ai/whisper-large-v3-turbo` | Hugging Face pipeline |
| English transcription | `openai/whisper-large-v3` | Hugging Face pipeline |

Models are cached in `Karaoke_Output/uvr_models/` (UVR) and Hugging Face cache (Whisper).

---

## Hebrew / BIDI Text Handling

All UI labels are in Hebrew. Text rendering for subtitles uses:

```python
import arabic_reshaper
from bidi.algorithm import get_display

def _fix_hebrew_text(text):
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)
```

- Apply to ASS subtitle dialogue lines before writing to file
- A `use_bidi` boolean flag controls whether BIDI fix is applied at render time
- The `_fix_hebrew_text` method also reverses word order for RTL display

---

## Subtitle Format: ASS

The pipeline outputs `.ass` (Advanced SubStation Alpha) files.

**ASS file structure:**
```
[Script Info]
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Style: Karaoke,Arial,80,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,...

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.23,0:00:04.56,Karaoke,,0,0,0,,some words here
```

**ASS timestamp format:** `H:MM:SS.CC` (centiseconds, not milliseconds)

Use `_fmt_ass_time(seconds: float) -> str` to convert.

**Grouping words into lines:** Whisper returns word-level chunks. Lines are grouped at punctuation (`.`, `?`, `!`, `,`) or every 6 words, whichever comes first.

---

## Hardware Detection

```python
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

Always pass `device=DEVICE` and `torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32` to Hugging Face pipelines.

Force CuDNN DLL path on Windows (see `force_cudnn_path()` in V64).

---

## Dependencies (Not Yet in requirements.txt)

```
# Core processing
yt-dlp
audio-separator          # UVR5 / Demucs via Python
torch                    # GPU inference
transformers             # Whisper via HuggingFace
librosa                  # BPM + key detection
soundfile                # Audio file I/O
imageio-ffmpeg           # Bundled FFmpeg binary

# UI
gradio>=4.0

# Text / subtitles
arabic-reshaper
python-bidi
pandas

# Real-time audio (core/engine.py)
sounddevice
numpy

# Utilities
scipy
```

External requirement: **FFmpeg** must be available (either system PATH or via `imageio_ffmpeg.get_ffmpeg_exe()`).

---

## Development Workflows

### Running the current app
```bash
python app.py        # Starts Gradio web UI at http://localhost:7860
python V64           # Runs V64 standalone (also launches Gradio)
python V39           # Runs V39 standalone
```

### Running individual tools (planned, not yet implemented)
```bash
python -m modules.downloader <url> [--format wav|mp4] [--output-dir DIR]
python -m modules.separator <audio_file> [--mode 2|4] [--output-dir DIR]
python -m modules.transcriber <audio_file> [--lang he|en] [--format srt|txt|ass]
python -m modules.renderer <video> <audio> <subtitles.ass> [--output DIR]
python cli/main.py pipeline <url>     # Full pipeline
python cli/main.py batch <list.txt>   # Batch from file
```

### Audio device debugging
```python
from core.engine import AudioEngine
devices = AudioEngine.instance().list_devices()
```

### Headless/CI testing
```bash
QT_QPA_PLATFORM=offscreen python -m pytest
```
Mock `sounddevice` in unit tests to avoid hardware dependency.

---

## Conventions to Follow

### Code style
- Typed Python with type hints on all function signatures
- Module-level docstrings on every new file
- Small, single-responsibility functions
- Hebrew comments are acceptable (existing codebase uses them)

### Singleton pattern (core/engine.py)
```python
class AudioEngine:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        # ... init code ...
        self._initialized = True
```
**Never break this pattern** when extending the engine.

### Backend method pattern
All backend methods should:
1. Accept a `current_logs: list` parameter and append timestamped messages via `self.log(msg, current_logs)`
2. Check for cached output and skip processing if valid output already exists (unless `force=True`)
3. Use `try/except` and return `None` on failure
4. Call `self.cleanup_resources()` after heavy GPU operations

### ASS file writing
Always open with `encoding="utf-8-sig"` (BOM required for FFmpeg compatibility):
```python
with open(output_path, "w", encoding="utf-8-sig") as f:
    f.write(content)
```

### FFmpeg invocation
Use the bundled binary:
```python
import imageio_ffmpeg
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
cmd = [ffmpeg_exe, '-y', '-i', input_path, ...]
subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
```

### File paths
- Working directory for all output: `Karaoke_Output/` (created automatically)
- Per-song subfolder: `Karaoke_Output/{sanitized_title}/`
- Use `_sanitize_filename()` on all user-supplied or API-returned titles
- Use `os.path.join()` everywhere — no string concatenation for paths

### Logging
```python
def log(self, msg, log_list=None):
    ts = time.strftime("%H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted)
    if log_list is not None:
        log_list.append(formatted)
    return formatted
```
All user-facing status uses emoji prefixes: `✅ 📥 ⏳ 🚀 ❌ ⚠️ 🎬`

---

## Planned Features (Not Yet Implemented)

These are features the project owner has specified as requirements:

1. **Modular CLI** — each of the 4 pipeline stages runnable as a standalone script with `python -m modules.<name>`
2. **Real-time transcription progress** — stream word chunks to the terminal/UI as Whisper processes them
3. **SRT export** — in addition to ASS, export `.srt` subtitle files
4. **Word timing editor** — visual table to adjust per-word timestamps before rendering
5. **Lyrics parody editor** — side-by-side table: original transcription | user-written alternative lyrics
6. **Song analysis tab** — BPM, musical key (already implemented in `analyze_audio()`; needs UI integration)
7. **Desktop app** — launch Gradio in a thread and open a browser, or embed in a PyQt6 WebView
8. **Server/client mode** — FastAPI backend + separate web frontend; Gradio UI connects to a remote server
9. **Batch queue UI** — manage a list of songs with per-item status
10. **Lecture transcription mode** — transcription without karaoke rendering
11. **`requirements.txt`** — does not exist yet; must be created before any new contributor can run the project

---

## What NOT to Do

- **Do not use `backend.py`** as a reference — it is outdated commented-out code
- **Do not rename `V39` / `V64`** — they are working standalone scripts; treat them as read-only references
- **Do not do heavy work inside `AudioEngine._callback()`** — buffer frames and process on worker threads
- **Do not hardcode FFmpeg path** — always use `imageio_ffmpeg.get_ffmpeg_exe()`
- **Do not skip `_sanitize_filename()`** — song titles from YouTube contain characters that break file paths
- **Do not write ASS files without `utf-8-sig` encoding** — FFmpeg requires the BOM to render Hebrew correctly
