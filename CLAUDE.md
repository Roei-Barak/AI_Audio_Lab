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
├── requirements.txt               # All Python dependencies
├── .github/
│   └── copilot-instructions.md    # Legacy AI agent guidance (superseded by this file)
├── .gitignore
│
├── app.py                         # Original Gradio web UI (V71) — kept for reference
├── backend.py                     # Large backend file — mostly commented-out legacy code
├── V39                            # Working karaoke pipeline v39 (read-only reference)
├── V64                            # Working karaoke pipeline v64 (read-only reference)
├── MP4_Aduio_Overload             # Standalone Tkinter utility: swap audio in MP4
│
├── core/
│   ├── __init__.py
│   ├── engine.py                  # AudioEngine singleton (real-time sounddevice I/O)
│   ├── config.py                  # Shared constants: WORK_DIR, DEVICE, MODELS, ASS_HEADER
│   └── backend.py                 # BackendProcessor + ResourceManager (canonical engine)
│
├── modules/                       # Standalone CLI modules — each runnable independently
│   ├── __init__.py
│   ├── downloader.py              # python -m modules.downloader <url>
│   ├── separator.py               # python -m modules.separator  <file> --mode 2|4
│   ├── transcriber.py             # python -m modules.transcriber <file> --format ass,srt,txt
│   └── renderer.py                # python -m modules.renderer   <video> <audio> <subs>
│
├── cli/
│   ├── __init__.py
│   └── main.py                    # python cli/main.py pipeline|download|separate|transcribe|render|analyze|lecture|batch
│
├── gui/
│   ├── __init__.py
│   ├── gradio_app.py              # Extended 8-tab Gradio UI (web + server mode)
│   └── desktop_app.py             # Desktop launcher (browser or PyQt6 WebView)
│
├── api/
│   ├── __init__.py
│   └── server.py                  # FastAPI REST + SSE backend for server/client mode
│
└── ai_modules/
    ├── __init__.py
    ├── spleeter.py                # Stub: SeparationModule (raises NotImplementedError)
    ├── pitch_detection.py         # Stub: PitchDetector (raises NotImplementedError)
    └── chord_gen.py               # Stub: ChordGenerator (raises NotImplementedError)
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

### `core/backend.py` (canonical engine — use this)
Extracted and cleaned-up version of V64's `BackendProcessor`. All modules, CLI, and GUI import from here.

### `core/config.py`
All shared constants: `WORK_DIR`, `DEVICE`, `TORCH_DTYPE`, `MODELS`, `ASS_HEADER`, `SUBTITLE_PRESETS`, `SUBTITLE_POSITIONS`, `cleanup_gpu()`.

### `modules/downloader.py`
Public API: `download(url, output_dir, fmt, logs)` → path.
CLI: `python -m modules.downloader`.

### `modules/separator.py`
Public API: `separate(audio_path, output_dir, mode, force, logs)` → `(vocals, playback)`.
CLI: `python -m modules.separator`.

### `modules/transcriber.py`
Public API: `transcribe(audio_path, output_dir, lang, output_formats, title, force, progress_callback, logs)` → `{fmt: path}`.
Supports real-time streaming via `progress_callback(idx, total, text)`.
CLI: `python -m modules.transcriber` with live progress bar to stderr.

### `modules/renderer.py`
Public API: `render(video, audio, subtitles, output_dir, output_name, use_bidi, font_size, color_hex, position, force, logs)` → path.
CLI: `python -m modules.renderer` with `--preset` for named colour themes.

### `cli/main.py`
Thin orchestrator. Sub-commands: `pipeline`, `download`, `separate`, `transcribe`, `render`, `analyze`, `lecture`, `batch`.

### `gui/gradio_app.py`
8-tab extended Gradio UI:
1. ⚡ תהליך אוטומטי — full pipeline
2. 🛠️ כלים בנפרד — individual tools
3. 📚 עיבוד רשימה — batch with status table
4. 📝 עורך כתוביות — ASS timing/text editor
5. 🎭 עורך פרודיה — original ↔ alternative lyrics
6. 🎤 תמלול הרצאה — lecture transcription
7. 🎼 ניתוח שיר — BPM + key
8. ⚙️ הגדרות — settings + CLI reference

### `gui/desktop_app.py`
Launches Gradio in a background thread and opens the browser. Optional `--webview` mode embeds the UI in a PyQt6 WebView window.

### `api/server.py`
FastAPI REST backend. Endpoints: `/health`, `/info`, `/download`, `/separate`, `/transcribe`, `/transcribe/stream` (SSE), `/render`, `/pipeline/stream` (SSE), `/analyze`, `/files/{filename}`. Interactive docs at `/docs`.

### `V64` (read-only reference)
The original monolithic implementation before the modular refactor. Do not modify.

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

### Running the modular system
```bash
# Full Gradio web UI (recommended)
python gui/gradio_app.py                     # http://localhost:7860
python gui/gradio_app.py --server            # expose on 0.0.0.0 (LAN/server)
python gui/gradio_app.py --port 8080 --share # public Gradio link

# Desktop app (opens browser automatically)
python gui/desktop_app.py
python gui/desktop_app.py --webview          # PyQt6 embedded window

# FastAPI server/client backend
python api/server.py                         # http://0.0.0.0:8000
uvicorn api.server:app --reload              # dev mode with hot-reload
# API docs: http://localhost:8000/docs
```

### CLI — individual modules
```bash
python -m modules.downloader <url> [--format wav|mp4] [--output-dir DIR]
python -m modules.separator  <file> [--mode 2|4] [--output-dir DIR] [--force]
python -m modules.transcriber <file> [--lang he|en|auto] [--format ass,srt,txt] [--no-progress]
python -m modules.renderer   <video> <audio> <subs.ass> [--color #FFD700] [--bidi] [--preset "זהב קריוקי"]
```

### CLI — orchestrator
```bash
python cli/main.py pipeline  <url>           # Full pipeline
python cli/main.py download  <url>
python cli/main.py separate  <file> --mode 4
python cli/main.py transcribe <file> --lang en --format ass,srt,txt
python cli/main.py render    <video> <audio> <subs>
python cli/main.py analyze   <file>          # BPM + key
python cli/main.py lecture   <url|file>      # transcription only (no karaoke render)
python cli/main.py batch     songs.txt       # one URL/query per line
```

### Legacy (reference only)
```bash
python app.py        # Original V71 Gradio UI
python V64           # V64 standalone
python V39           # V39 standalone
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

## Feature Status

| Feature | Status |
|---|---|
| Modular CLI (4 standalone modules) | ✅ Implemented |
| Real-time transcription progress (streaming) | ✅ Implemented |
| SRT export | ✅ Implemented |
| TXT export | ✅ Implemented |
| Subtitle colour presets | ✅ Implemented |
| Subtitle position (top/center/bottom) | ✅ Implemented |
| Word timing editor (table) | ✅ Implemented |
| Lyrics parody editor (original ↔ alternative) | ✅ Implemented |
| Song analysis (BPM + key) | ✅ Implemented |
| Desktop app (browser + PyQt6 WebView) | ✅ Implemented |
| Server/client mode (FastAPI + SSE) | ✅ Implemented |
| Batch queue UI with per-song status | ✅ Implemented |
| Lecture transcription mode | ✅ Implemented |
| `requirements.txt` | ✅ Implemented |
| Auto language detection (`--lang auto`) | ✅ Implemented |
| `--info-only` (metadata without download) | ✅ Implemented |
| `--dry-run` / verbose/quiet modes | 🔲 Planned |
| Lyrics-from-web fetch (genius.com, etc.) | 🔲 Planned |
| Speaker diarization (identify singers) | 🔲 Planned |
| Noise reduction pre-processing | 🔲 Planned |
| Export project as ZIP | 🔲 Planned |
| Processing history / cache log | 🔲 Planned |

---

## What NOT to Do

- **Do not use `backend.py`** as a reference — it is outdated commented-out code
- **Do not rename `V39` / `V64`** — they are working standalone scripts; treat them as read-only references
- **Do not do heavy work inside `AudioEngine._callback()`** — buffer frames and process on worker threads
- **Do not hardcode FFmpeg path** — always use `imageio_ffmpeg.get_ffmpeg_exe()`
- **Do not skip `_sanitize_filename()`** — song titles from YouTube contain characters that break file paths
- **Do not write ASS files without `utf-8-sig` encoding** — FFmpeg requires the BOM to render Hebrew correctly
