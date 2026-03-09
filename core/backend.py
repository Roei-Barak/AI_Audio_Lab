"""
core/backend.py — Shared BackendProcessor and ResourceManager.

This module is the canonical backend engine, extracted from V64.
All pipeline modules (downloader, separator, transcriber, renderer) and
both the CLI and GUI import from here.

Usage:
    from core.backend import BackendProcessor, ResourceManager
    bp = BackendProcessor()
    manager = ResourceManager()
"""

import gc
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import imageio_ffmpeg
import numpy as np
import pandas as pd

from core.config import (
    ASS_HEADER,
    DEVICE,
    MODELS,
    SUBTITLE_POSITIONS,
    TORCH_DTYPE,
    UVR_MODEL_DIR,
    WORK_DIR,
    cleanup_gpu,
)

# Silence noisy library loggers
logging.getLogger("audio_separator").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Optional heavy imports (graceful degradation when not installed)
# ---------------------------------------------------------------------------
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _BIDI_AVAILABLE = True
except ImportError:
    _BIDI_AVAILABLE = False

try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    _SF_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import yt_dlp
    _YTDLP_AVAILABLE = True
except ImportError:
    _YTDLP_AVAILABLE = False

try:
    from audio_separator.separator import Separator
    _SEP_AVAILABLE = True
except ImportError:
    _SEP_AVAILABLE = False

try:
    from transformers import pipeline as hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False


# ---------------------------------------------------------------------------
# ResourceManager — concurrency and VRAM guard
# ---------------------------------------------------------------------------

class ResourceManager:
    """Limits concurrent heavy tasks and monitors CUDA VRAM."""

    def __init__(self, max_concurrent_heavy: int = 2) -> None:
        self.semaphore = threading.Semaphore(max_concurrent_heavy)
        self._active = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def start_task(self) -> None:
        with self._lock:
            self._active += 1

    def end_task(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)

    @property
    def active_tasks(self) -> int:
        with self._lock:
            return self._active

    # ------------------------------------------------------------------
    def get_status(self) -> str:
        active = self.active_tasks
        mem = ""
        if DEVICE == "cuda" and _TORCH_AVAILABLE:
            try:
                free, total = torch.cuda.mem_get_info()
                free_gb = free / 1024 ** 3
                mem = f" (VRAM: {free_gb:.1f}GB פנוי)"
                if free_gb < 4:
                    return f"🔴 עומס גבוה{mem}"
            except Exception:
                pass
        if active == 0:
            return f"🟢 מוכן{mem}"
        if active == 1:
            return f"🟡 עובד (משימה 1){mem}"
        return f"🟠 עמוס ({active} משימות){mem}"


# Module-level singleton resource manager
manager = ResourceManager(max_concurrent_heavy=2)


# ---------------------------------------------------------------------------
# BackendProcessor — all pipeline operations
# ---------------------------------------------------------------------------

class BackendProcessor:
    """
    Orchestrates the full karaoke pipeline:
        download → separate → transcribe → render

    Every public method:
    - Accepts `current_logs: list` and appends timestamped messages.
    - Returns None on failure (never raises).
    - Caches results by checking output file existence (skip unless force=True).
    - Calls cleanup_gpu() after GPU-heavy operations.
    """

    def __init__(self) -> None:
        self.ffmpeg_exe: str = imageio_ffmpeg.get_ffmpeg_exe()

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def log(self, msg: str, log_list: Optional[list] = None) -> str:
        ts = time.strftime("%H:%M:%S")
        formatted = f"[{ts}] {msg}"
        print(formatted)
        if log_list is not None:
            log_list.append(formatted)
        return formatted

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fix_hebrew_text(self, text: str) -> str:
        if not _BIDI_AVAILABLE or not text:
            return text
        try:
            reshaped = arabic_reshaper.reshape(str(text))
            return get_display(reshaped)
        except Exception:
            return text

    def _sanitize_filename(self, name: str) -> str:
        name = re.sub(r'[\\/*?:"<>|]', "", name)
        name = name.replace("'", "").replace('"', "").strip().rstrip(".")
        return name or f"Video_{int(time.time())}"

    def _fmt_ass_time(self, seconds: float) -> str:
        try:
            seconds = float(seconds)
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            cs = int((seconds - int(seconds)) * 100)
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
        except Exception:
            return "0:00:00.00"

    def _srt_time(self, seconds: float) -> str:
        """Float seconds → SRT timestamp HH:MM:SS,mmm"""
        try:
            seconds = float(seconds)
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int((seconds - int(seconds)) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        except Exception:
            return "00:00:00,000"

    # ------------------------------------------------------------------
    # FFmpeg helpers
    # ------------------------------------------------------------------

    def convert_to_wav(self, input_path: str, output_path: str) -> bool:
        """Convert any audio/video to 16 kHz mono PCM WAV."""
        try:
            cmd = [
                self.ffmpeg_exe, "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                output_path,
            ]
            subprocess.run(
                cmd, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=300,
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 1. Download
    # ------------------------------------------------------------------

    def get_video_info(self, query: str, current_logs: list) -> Optional[dict]:
        """Resolve a YouTube URL or search query → metadata dict."""
        search = query if query.startswith("http") else f"ytsearch1:{query}"
        self.log(f"🔎 {'מנתח' if query.startswith('http') else 'מחפש'}: {query!r}…", current_logs)

        if not _YTDLP_AVAILABLE:
            self.log("❌ yt-dlp לא מותקן", current_logs)
            return None

        opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(search, download=False)
                if "entries" in info:
                    entries = [e for e in (info.get("entries") or []) if e]
                    if not entries:
                        self.log("❌ לא נמצאו תוצאות.", current_logs)
                        return None
                    info = entries[0]

                title = self._sanitize_filename(info.get("title") or "")
                vid_id = info.get("id", "")
                url = info.get("webpage_url") or f"https://www.youtube.com/watch?v={vid_id}"
                if "googlevideo" in url:
                    url = f"https://www.youtube.com/watch?v={vid_id}"

                folder = os.path.join(WORK_DIR, title)
                self.log(f"✅ זוהה: {title}", current_logs)
                return {"title": title, "url": url, "id": vid_id, "folder": folder}
        except Exception as e:
            self.log(f"⚠️ שגיאה: {e}", current_logs)
            return None

    def download_video(self, video_info: dict, current_logs: list) -> Optional[str]:
        """Download MP4 via yt-dlp. Returns local path or None."""
        folder = video_info["folder"]
        title = video_info["title"]
        url = video_info["url"]
        os.makedirs(folder, exist_ok=True)

        target = os.path.join(folder, f"{title}.mp4")
        if os.path.exists(target):
            self.log("✅ קובץ מקור קיים.", current_logs)
            return target

        self.log(f"📥 מוריד: {url}", current_logs)
        tmpl = os.path.join(folder, f"tmp_{uuid.uuid4().hex[:6]}.%(ext)s")
        opts = {
            "outtmpl": tmpl,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "ffmpeg_location": self.ffmpeg_exe,
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            found = next(
                (os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("tmp_")),
                None,
            )
            if found:
                if os.path.exists(target):
                    os.remove(target)
                shutil.move(found, target)
                self.log(f"✅ הורד: {os.path.basename(target)}", current_logs)
                return target
            return None
        except Exception as e:
            self.log(f"❌ שגיאה בהורדה: {e}", current_logs)
            return None

    # ------------------------------------------------------------------
    # 2. Separate
    # ------------------------------------------------------------------

    def separate_audio(
        self,
        video_path: str,
        output_folder: str,
        current_logs: list,
        mode: str = "2_stems",
        save_4: bool = False,
        force: bool = False,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Separate vocals and instrumental.

        Returns (vocals_path, playback_path) or (None, None) on failure.
        Pass save_4=True to also run 4-stem Demucs separation.
        """
        if not _SEP_AVAILABLE:
            self.log("❌ audio-separator לא מותקן", current_logs)
            return None, None

        title = Path(video_path).stem
        vocals_out = os.path.join(output_folder, f"{title}_Vocals.wav")
        playback_out = os.path.join(output_folder, f"{title}_Playback.wav")

        if not force and os.path.exists(vocals_out) and os.path.exists(playback_out):
            self.log("✅ קבצי הפרדה קיימים.", current_logs)
            return vocals_out, playback_out

        if force:
            for p in (vocals_out, playback_out):
                if os.path.exists(p):
                    os.remove(p)

        self.log("⏳ ממתין למשאבים…", current_logs)
        with manager.semaphore:
            manager.start_task()
            try:
                self.log("🚀 מתחיל הפרדה…", current_logs)
                cleanup_gpu()

                tmp_dir = os.path.join(output_folder, f"sep_{uuid.uuid4().hex[:6]}")
                os.makedirs(tmp_dir, exist_ok=True)

                input_wav = os.path.join(tmp_dir, "input.wav")
                self.convert_to_wav(video_path, input_wav)

                sep = Separator(
                    log_level=logging.ERROR,
                    output_dir=tmp_dir,
                    model_file_dir=UVR_MODEL_DIR,
                )
                sep.load_model(MODELS["separation_2stem"])
                files = sep.separate(input_wav)

                for f in files:
                    src = os.path.join(tmp_dir, f)
                    lower = f.lower()
                    if "vocal" in lower and "no" not in lower and "inst" not in lower:
                        shutil.move(src, vocals_out)
                    elif "instrumental" in lower or "no_vocal" in lower:
                        shutil.move(src, playback_out)

                del sep
                cleanup_gpu()

                # Optional 4-stem pass
                if save_4 and os.path.exists(input_wav):
                    self.log("🎚️ מפריד ל-4 ערוצים (Demucs)…", current_logs)
                    sep4 = Separator(log_level=logging.ERROR, output_dir=tmp_dir)
                    sep4.load_model(MODELS["separation_4stem"])
                    files4 = sep4.separate(input_wav)
                    stem_map = {
                        "drums": "Drums", "bass": "Bass",
                        "vocals": "Vocals_Demucs", "other": "Other",
                    }
                    for f in files4:
                        src = os.path.join(tmp_dir, f)
                        key = next((k for k in stem_map if k in f.lower()), None)
                        dest = os.path.join(
                            output_folder,
                            f"{title}_{stem_map.get(key, f)}.wav",
                        )
                        shutil.move(src, dest)
                    del sep4
                    cleanup_gpu()

                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir, ignore_errors=True)

                self.log("✅ הפרדה הסתיימה.", current_logs)
                return vocals_out, playback_out

            except Exception as e:
                self.log(f"❌ שגיאה בהפרדה: {e}", current_logs)
                return None, None
            finally:
                manager.end_task()

    # ------------------------------------------------------------------
    # 3. Transcribe
    # ------------------------------------------------------------------

    def transcribe_audio(
        self,
        audio_path: str,
        output_folder: str,
        title: str,
        current_logs: list,
        lang: str = "he",
        output_formats: list[str] | None = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Optional[str]:
        """
        Transcribe audio with Whisper.

        Outputs ASS subtitle file (always) + optionally SRT / TXT.
        progress_callback(segment_idx, total_segments, partial_text) is called
        after each ~60-second chunk is processed.

        Returns path to the .ass file, or None on failure.
        """
        if not _TRANSFORMERS_AVAILABLE:
            self.log("❌ transformers לא מותקן", current_logs)
            return None

        if output_formats is None:
            output_formats = ["ass"]

        ass_path = os.path.join(output_folder, f"{title}.ass")
        if os.path.exists(ass_path) and "ass" in output_formats:
            self.log("✅ תמלול קיים.", current_logs)
            return ass_path

        self.log("⏳ ממתין למשאבים לתמלול…", current_logs)
        with manager.semaphore:
            manager.start_task()
            try:
                self.log(f"📝 מתחיל תמלול ({lang.upper()})…", current_logs)
                cleanup_gpu()

                model_id = MODELS["whisper_he"] if lang == "he" else MODELS["whisper_en"]
                pipe = hf_pipeline(
                    "automatic-speech-recognition",
                    model=model_id,
                    device=DEVICE,
                    torch_dtype=TORCH_DTYPE,
                    chunk_length_s=30,
                    return_timestamps="word",
                )

                # Convert to clean 16 kHz WAV
                tmp_wav = os.path.join(output_folder, f"tmp_tr_{uuid.uuid4().hex[:4]}.wav")
                self.convert_to_wav(audio_path, tmp_wav)

                all_chunks: list[dict] = []

                if _SF_AVAILABLE and progress_callback:
                    # Segment-by-segment with progress reporting
                    audio_data, sr = sf.read(tmp_wav)
                    if audio_data.ndim > 1:
                        audio_data = audio_data.mean(axis=1)

                    seg_samples = int(60 * sr)
                    total_segs = max(1, -(-len(audio_data) // seg_samples))  # ceil div

                    for idx, start in enumerate(range(0, len(audio_data), seg_samples)):
                        chunk = audio_data[start: start + seg_samples]
                        if chunk.size == 0:
                            continue
                        res = pipe(
                            {"array": chunk.astype(np.float32), "sampling_rate": sr},
                            generate_kwargs={"language": "hebrew" if lang == "he" else "english"},
                        )
                        offset = start / sr
                        for c in res.get("chunks") or []:
                            ts = c.get("timestamp", (None, None))
                            c["timestamp"] = (
                                (ts[0] + offset if ts[0] is not None else None),
                                (ts[1] + offset if ts[1] is not None else None),
                            )
                            all_chunks.append(c)

                        progress_callback(idx + 1, total_segs, res.get("text", "").strip())
                        self.log(
                            f"⏱️ קטע {idx + 1}/{total_segs}: {res.get('text', '').strip()[:60]}",
                            current_logs,
                        )
                else:
                    # Single-pass (no streaming)
                    result = pipe(
                        tmp_wav,
                        generate_kwargs={"language": "hebrew" if lang == "he" else "english"},
                    )
                    all_chunks = result.get("chunks") or []

                # Write requested output formats
                os.makedirs(output_folder, exist_ok=True)
                if "ass" in output_formats:
                    self._write_ass(all_chunks, ass_path)
                if "srt" in output_formats:
                    srt_path = os.path.join(output_folder, f"{title}.srt")
                    self._write_srt(all_chunks, srt_path)
                if "txt" in output_formats:
                    txt_path = os.path.join(output_folder, f"{title}.txt")
                    self._write_txt(all_chunks, txt_path)

                del pipe
                if os.path.exists(tmp_wav):
                    os.remove(tmp_wav)
                cleanup_gpu()
                self.log("✅ תמלול הסתיים.", current_logs)
                return ass_path

            except Exception as e:
                self.log(f"❌ שגיאה בתמלול: {e}", current_logs)
                return None
            finally:
                manager.end_task()

    # ------------------------------------------------------------------
    # Subtitle writers
    # ------------------------------------------------------------------

    def _group_chunks_into_lines(
        self, chunks: list[dict], fix_bidi: bool = False
    ) -> list[dict]:
        """Group word-level chunks into subtitle lines."""
        lines: list[dict] = []
        current: list[dict] = []

        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            ts = chunk.get("timestamp", (None, None))
            current.append({"text": text, "start": ts[0], "end": ts[1]})

            end_of_line = text.endswith((".", "?", "!", ",")) or len(current) >= 6
            if end_of_line:
                line_text = " ".join(w["text"] for w in current)
                if fix_bidi:
                    line_text = self._fix_hebrew_text(
                        " ".join(w["text"] for w in reversed(current))
                    )
                lines.append({
                    "start": current[0]["start"],
                    "end": current[-1]["end"],
                    "text": line_text,
                })
                current = []

        if current:
            line_text = " ".join(w["text"] for w in current)
            if fix_bidi:
                line_text = self._fix_hebrew_text(
                    " ".join(w["text"] for w in reversed(current))
                )
            lines.append({
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": line_text,
            })

        return lines

    def _write_ass(
        self,
        chunks: list[dict],
        output_path: str,
        fix_bidi: bool = False,
        font_size: int = 80,
        primary_color_hex: str = "#FFFFFF",
        outline_color_hex: str = "#000000",
        alignment: int = 2,
    ) -> None:
        """Write an ASS subtitle file from word-level chunks."""
        r, g, b = primary_color_hex[1:3], primary_color_hex[3:5], primary_color_hex[5:7]
        ass_primary = f"&H00{b}{g}{r}".upper()
        ro, go, bo = outline_color_hex[1:3], outline_color_hex[3:5], outline_color_hex[5:7]
        ass_outline = f"&H00{bo}{go}{ro}".upper()

        style_line = (
            f"Style: Karaoke,Arial,{font_size},{ass_primary},&H00FFFFFF,"
            f"{ass_outline},&H80000000,-1,0,0,0,100,100,0,0,1,3,0,{alignment},10,10,100,1"
        )
        header = ASS_HEADER.replace(
            "Style: Karaoke,Arial,80,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,100,1",
            style_line,
        )

        lines = self._group_chunks_into_lines(chunks, fix_bidi=fix_bidi)
        events = []
        for line in lines:
            s = self._fmt_ass_time(line["start"] or 0)
            e = self._fmt_ass_time(line["end"] or 0)
            events.append(f"Dialogue: 0,{s},{e},Karaoke,,0,0,0,,{line['text']}")

        with open(output_path, "w", encoding="utf-8-sig") as f:
            f.write(header + "\n".join(events))

    def _write_srt(self, chunks: list[dict], output_path: str) -> None:
        """Write a standard SRT file from word-level chunks."""
        lines = self._group_chunks_into_lines(chunks)
        with open(output_path, "w", encoding="utf-8") as f:
            for i, line in enumerate(lines, start=1):
                start = self._srt_time(line["start"] or 0)
                end = self._srt_time(line["end"] or 0)
                f.write(f"{i}\n{start} --> {end}\n{line['text']}\n\n")

    def _write_txt(self, chunks: list[dict], output_path: str) -> None:
        """Write plain-text transcript from word-level chunks."""
        lines = self._group_chunks_into_lines(chunks)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(line["text"] for line in lines))

    # ------------------------------------------------------------------
    # 4. Render
    # ------------------------------------------------------------------

    def render_video(
        self,
        video_path: str,
        audio_path: str,
        ass_path: str,
        video_info: dict,
        current_logs: list,
        use_bidi: bool = False,
        force: bool = False,
    ) -> Optional[str]:
        """
        Render final karaoke video via FFmpeg.

        Overlays .ass subtitles on the original video and swaps the audio
        track with the separated playback.  Returns path to output MP4 or None.
        """
        folder = video_info["folder"]
        title = video_info["title"]
        final = os.path.join(folder, f"{title}_KARAOKE.mp4")

        if os.path.exists(final) and not force:
            self.log("✅ סרטון קריוקי קיים.", current_logs)
            return final

        self.log("🎬 מרנדר סרטון קריוקי…", current_logs)
        os.makedirs(folder, exist_ok=True)

        # Optionally apply BIDI fix to a temporary copy of the ASS
        render_ass = os.path.join(folder, f"render_{uuid.uuid4().hex[:4]}.ass")
        try:
            with open(ass_path, "r", encoding="utf-8-sig") as f:
                content = f.read()
            if use_bidi:
                new_lines = []
                for line in content.splitlines():
                    if line.startswith("Dialogue:"):
                        parts = line.split(",", 9)
                        if len(parts) == 10:
                            parts[9] = self._fix_hebrew_text(parts[9])
                            line = ",".join(parts)
                    new_lines.append(line)
                content = "\n".join(new_lines)
            with open(render_ass, "w", encoding="utf-8-sig") as f:
                f.write(content)
        except Exception as e:
            self.log(f"❌ שגיאה בהכנת קובץ כתוביות: {e}", current_logs)
            return None

        tmp_out = os.path.join(folder, f"tmp_{uuid.uuid4().hex[:6]}.mp4")
        ass_name = os.path.basename(render_ass).replace("\\", "/")

        cmd = [
            self.ffmpeg_exe, "-y",
            "-i", video_path,
            "-i", audio_path,
            "-filter_complex", f"[0:v]ass='{ass_name}'[v]",
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            tmp_out,
        ]

        prev_dir = os.getcwd()
        os.chdir(folder)
        try:
            subprocess.run(
                cmd, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=600,
            )
            os.chdir(prev_dir)
            try:
                os.remove(render_ass)
            except Exception:
                pass
            if os.path.exists(final):
                os.remove(final)
            shutil.move(tmp_out, final)
            self.log(f"✅ הסתיים: {os.path.basename(final)}", current_logs)
            return final
        except Exception as e:
            os.chdir(prev_dir)
            self.log(f"❌ שגיאה ברנדור: {e}", current_logs)
            return None

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process_song_pipeline(
        self,
        query: str,
        lang: str = "he",
        save_4_stems: bool = False,
        use_bidi: bool = False,
        force: bool = False,
        output_formats: list[str] | None = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> tuple[Optional[str], str]:
        """
        Run the complete karaoke pipeline for a single song.

        Returns (final_video_path, log_string).
        """
        if output_formats is None:
            output_formats = ["ass"]

        logs: list[str] = []

        info = self.get_video_info(query, logs)
        if not info:
            return None, "\n".join(logs)

        video = self.download_video(info, logs)
        if not video:
            return None, "\n".join(logs)

        vocals, playback = self.separate_audio(
            video, info["folder"], logs, save_4=save_4_stems, force=force
        )
        if not vocals or not playback:
            return None, "\n".join(logs)

        ass = self.transcribe_audio(
            vocals, info["folder"], info["title"], logs,
            lang=lang, output_formats=output_formats,
            progress_callback=progress_callback,
        )
        if not ass:
            return None, "\n".join(logs)

        if force:
            final_path = os.path.join(info["folder"], f"{info['title']}_KARAOKE.mp4")
            if os.path.exists(final_path):
                os.remove(final_path)

        final = self.render_video(video, playback, ass, info, logs, use_bidi=use_bidi)
        return final, "\n".join(logs)

    # ------------------------------------------------------------------
    # ASS / subtitle utilities
    # ------------------------------------------------------------------

    def ass_to_dataframe(self, ass_path: str) -> pd.DataFrame:
        """Parse an ASS file into a pandas DataFrame with columns Start, End, Text."""
        if not ass_path or not os.path.exists(ass_path):
            return pd.DataFrame(columns=["Start", "End", "Text"])
        rows = []
        with open(ass_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                if line.startswith("Dialogue:"):
                    parts = line.split(",", 9)
                    if len(parts) == 10:
                        rows.append({
                            "Start": parts[1].strip(),
                            "End": parts[2].strip(),
                            "Text": parts[9].strip(),
                        })
        return pd.DataFrame(rows)

    def dataframe_to_ass(
        self,
        df: pd.DataFrame,
        original_ass_path: Optional[str],
        output_path: str,
    ) -> str:
        """Write a DataFrame (Start, End, Text columns) back to an ASS file."""
        header_lines: list[str] = []
        if original_ass_path and os.path.exists(original_ass_path):
            with open(original_ass_path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    if line.startswith("Dialogue:"):
                        break
                    header_lines.append(line)
        else:
            header_lines = [ASS_HEADER]

        with open(output_path, "w", encoding="utf-8-sig") as f:
            f.writelines(header_lines)
            for _, row in df.iterrows():
                f.write(
                    f"Dialogue: 0,{row['Start']},{row['End']}"
                    f",Karaoke,,0,0,0,,{row['Text']}\n"
                )
        return output_path

    def update_ass_style(
        self,
        ass_path: str,
        font_size: int,
        color_hex: str,
        position: int = 2,
    ) -> None:
        """Update the font size, primary colour, and alignment in an ASS Style line."""
        r, g, b = color_hex[1:3], color_hex[3:5], color_hex[5:7]
        ass_color = f"&H00{b}{g}{r}".upper()
        new_style = (
            f"Style: Karaoke,Arial,{int(font_size)},{ass_color},&H00FFFFFF,"
            f"&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,{position},10,10,100,1"
        )
        with open(ass_path, "r", encoding="utf-8-sig") as f:
            content = f.read()
        new_content = re.sub(
            r"^Style:.*Karaoke.*$", new_style, content, flags=re.MULTILINE
        )
        with open(ass_path, "w", encoding="utf-8-sig") as f:
            f.write(new_content)

    # ------------------------------------------------------------------
    # Audio analysis
    # ------------------------------------------------------------------

    def analyze_audio(self, audio_path: str) -> tuple[str, str]:
        """
        Estimate BPM and musical key using librosa.

        Returns (result_string, log_string).
        """
        if not _LIBROSA_AVAILABLE:
            return "❌ librosa לא מותקן", ""
        logs: list[str] = []
        try:
            self.log(f"🔍 מנתח: {os.path.basename(audio_path)}…", logs)
            tmp = os.path.join(os.path.dirname(audio_path), f"tmp_anl_{uuid.uuid4().hex[:4]}.wav")
            self.convert_to_wav(audio_path, tmp)
            y, sr = librosa.load(tmp, sr=None, duration=120)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = round(float(tempo[0]) if np.ndim(tempo) > 0 else float(tempo))
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            key = note_names[int(np.argmax(np.mean(chroma, axis=1)))]
            if os.path.exists(tmp):
                os.remove(tmp)
            result = f"BPM: {bpm} | מפתח: {key}"
            self.log(f"✅ {result}", logs)
            return result, "\n".join(logs)
        except Exception as e:
            return f"❌ שגיאה: {e}", "\n".join(logs)
