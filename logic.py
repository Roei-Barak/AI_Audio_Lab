import os
import shutil
import time
import uuid

from config import WORK_DIR, DEVICE, HF_TOKEN  # re-exported: logic.WORK_DIR, logic.DEVICE
from backend import BackendProcessor
from queue.manager import TaskManager
from queue.task import KaraokeTask, TaskStatus
from queue.gpu_semaphore import GpuSemaphore
from utils.ass_utils import ass_to_dataframe, dataframe_to_ass
from utils.bidi_support import apply_bidi_to_ass
from pipeline.subtitle import step_apply_style

task_manager = TaskManager()


class MissionManager:
    def get_status(self) -> str:
        return task_manager.get_status_summary()

    def get_df(self):
        return task_manager.to_dataframe()


mm = MissionManager()


class KaraokeBackend:
    """Single facade that app.py imports as `logic.backend`."""

    def __init__(self):
        self._base = BackendProcessor(log_func=print)

    # ------------------------------------------------------------------ helpers
    def log(self, msg: str, log_list=None) -> str:
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        print(entry, flush=True)
        if log_list is not None:
            log_list.append(entry)
        return entry

    def _proc(self, log_list=None):
        """Return a BackendProcessor bound to the given log list."""
        if log_list is None:
            return self._base
        return BackendProcessor(log_func=lambda m: self.log(m, log_list))

    # ------------------------------------------------------------------ info / download
    def get_video_info(self, query: str, current_logs: list):
        search_q = f"ytsearch1:{query}" if not query.startswith("http") else query
        verb = "מחפש" if not query.startswith("http") else "מנתח"
        self.log(f"🔎 {verb}: '{query}'...", current_logs)
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as ydl:
                info = ydl.extract_info(search_q, download=False)
                if "entries" in info:
                    if not info["entries"]:
                        self.log("❌ לא נמצאו תוצאות.", current_logs)
                        return None
                    info = info["entries"][0]
                raw_title = info.get("title", "Unknown")
                title = self._base._sanitize(raw_title) or f"Video_{info.get('id', 'x')}"
                real_url = info.get("webpage_url") or info.get("url") or query
                self.log(f"✅ זוהה: {title}", current_logs)
                return {"title": title, "url": real_url, "folder": os.path.join(WORK_DIR, title)}
        except Exception as e:
            self.log(f"⚠️ שגיאה בקבלת מידע: {e}", current_logs)
            return None

    def download_video(self, video_info: dict, current_logs: list):
        if not video_info:
            return None
        folder, title, url = video_info["folder"], video_info["title"], video_info["url"]
        os.makedirs(folder, exist_ok=True)
        target = os.path.join(folder, f"{title}.mp4")
        if os.path.exists(target):
            self.log("✅ קובץ קיים.", current_logs)
            return target
        result = self._proc(current_logs).download(url, folder, format_type="mp4")
        return result

    # ------------------------------------------------------------------ GPU steps
    def separate_audio(self, video_path: str, output_folder: str, current_logs: list,
                        mode: str = "2_stems", save_4: bool = False, force: bool = False):
        self.log("⏳ ממתין למשאבי GPU להפרדה...", current_logs)
        with GpuSemaphore():
            self.log("🚀 מתחיל הפרדה...", current_logs)
            stems = self._proc(current_logs).separate(video_path, output_folder, mode=mode)

        if not stems:
            return None, None

        vocals = stems[0] if len(stems) > 0 else None
        playback = stems[1] if len(stems) > 1 else None

        if save_4 and mode == "2_stems":
            self.log("🎚️ מפריד גם ל-4 ערוצים...", current_logs)
            with GpuSemaphore():
                self._proc(current_logs).separate(video_path, output_folder, mode="4_stems")

        return vocals, playback

    def transcribe_audio(self, audio_path: str, output_folder: str, title: str,
                          current_logs: list, lang: str = "he"):
        self.log("⏳ ממתין למשאבי GPU לתמלול...", current_logs)
        with GpuSemaphore():
            self.log("📝 מתמלל...", current_logs)
            ass_path = self._proc(current_logs).transcribe(audio_path, output_folder, lang, hf_token=HF_TOKEN)
        return ass_path

    # ------------------------------------------------------------------ ASS utilities
    def ass_to_dataframe(self, ass_path):
        return ass_to_dataframe(ass_path)

    def dataframe_to_ass(self, df, original_ass_path, output_path: str) -> str:
        return dataframe_to_ass(df, original_ass_path, output_path)

    def update_ass_style(self, ass_path: str, font_size, color_hex: str):
        """Modify ASS style in-place (takes file path, not content)."""
        step_apply_style(ass_path, int(font_size), color_hex)

    # ------------------------------------------------------------------ render
    def render_video(self, video_path, audio_path: str, ass_path: str,
                     video_info: dict, current_logs: list, use_bidi: bool = False) -> str:
        folder = video_info.get("folder", WORK_DIR)
        os.makedirs(folder, exist_ok=True)

        working_ass = os.path.join(folder, f"render_{uuid.uuid4().hex[:6]}.ass")
        if use_bidi:
            apply_bidi_to_ass(ass_path, working_ass)
        else:
            shutil.copy2(ass_path, working_ass)

        result = self._proc(current_logs).render_custom_karaoke(
            video_path, audio_path, working_ass, folder
        )
        try:
            os.remove(working_ass)
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------ tool wrappers (Advanced tab)
    def tool_download(self, url) -> tuple:
        logs = []
        try:
            info = self.get_video_info(url, logs)
            path = self.download_video(info, logs) if info else None
            return path, "\n".join(logs)
        except Exception as e:
            return None, f"שגיאה: {e}"

    def tool_separate(self, url, file, mode_str: str) -> tuple:
        logs = []
        try:
            target = file.name if file else None
            if not target:
                info = self.get_video_info(url, logs)
                target = self.download_video(info, logs) if info else None
            if not target:
                return None, "\n".join(logs)
            mode = "4_stems" if "4" in str(mode_str) else "2_stems"
            vocals, playback = self.separate_audio(target, WORK_DIR, logs, mode=mode)
            result = [p for p in [vocals, playback] if p]
            return result or None, "\n".join(logs)
        except Exception as e:
            return None, f"שגיאה: {e}"

    def tool_analyze(self, file) -> tuple:
        if not file:
            return "לא נבחר קובץ", ""
        path = file.name if hasattr(file, "name") else str(file)
        bpm, key = self._base.analyze_audio(path)
        if bpm:
            return f"BPM: {bpm} | Key: {key}", ""
        return "ניתוח נכשל", ""

    # ------------------------------------------------------------------ main pipeline
    def process_song_pipeline(self, query: str, lang: str, save_4_stems: bool,
                               use_bidi: bool, force: bool) -> tuple:
        task = KaraokeTask(song_name=query[:60])
        task_manager.submit(task)
        logs = task.logs

        try:
            task.status = TaskStatus.RUNNING

            task.step = "מוריד"
            info = self.get_video_info(query, logs)
            if not info:
                raise ValueError("לא נמצא שיר")
            video_path = self.download_video(info, logs)
            if not video_path:
                raise ValueError("הורדה נכשלה")
            task.progress_pct = 20

            task.step = "מפריד"
            vocals, playback = self.separate_audio(
                video_path, info["folder"], logs, save_4=save_4_stems, force=force
            )
            if not vocals or not playback:
                raise ValueError("הפרדה נכשלה")
            task.progress_pct = 50

            task.step = "מתמלל"
            ass_path = self.transcribe_audio(vocals, info["folder"], info["title"], logs, lang)
            if not ass_path:
                raise ValueError("תמלול נכשל")
            task.progress_pct = 75

            task.step = "מרנדר"
            if force:
                final_vid = os.path.join(info["folder"], f"{info['title']}_KARAOKE.mp4")
                if os.path.exists(final_vid):
                    os.remove(final_vid)

            final = self.render_video(video_path, playback, ass_path, info, logs, use_bidi=use_bidi)
            task.progress_pct = 100
            task.result_path = final
            task.status = TaskStatus.DONE

            return final, "\n".join(logs)

        except Exception as e:
            task.status = TaskStatus.ERROR
            task.error = str(e)
            self.log(f"❌ שגיאה: {e}", logs)
            return None, "\n".join(logs)


# Singleton accessible as logic.backend
backend = KaraokeBackend()


def run_karaoke_pipeline(
    source: str,
    lang: str = "he",
    save_4_stems: bool = False,
    use_bidi: bool = False,
    force: bool = False,
    font_size: int = 80,
    color_hex: str = "#00FFFF",
    on_progress=None,
) -> tuple:
    """Entry point for CLI and Desktop interfaces."""
    return backend.process_song_pipeline(source, lang, save_4_stems, use_bidi, force)
