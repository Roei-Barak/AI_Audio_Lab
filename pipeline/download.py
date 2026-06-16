import os
from pathlib import Path

from backend import BackendProcessor
from config import WORK_DIR


def step_download(query_or_url: str, output_dir: str, task_log, format_type: str = "mp4") -> dict:
    """
    Download audio/video from a URL or YouTube search query.

    Returns dict: {title, url, folder, video_path}
    Raises ValueError on failure.
    """
    proc = BackendProcessor(log_func=task_log)

    # Handle search queries vs direct URLs
    if not query_or_url.startswith("http"):
        search_url = f"ytsearch1:{query_or_url}"
        task_log(f"🔎 מחפש: '{query_or_url}'...")
    else:
        search_url = query_or_url

    # Get video info first
    try:
        import yt_dlp
        opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(search_url, download=False)
            if "entries" in info:
                if not info["entries"]:
                    raise ValueError("לא נמצאו תוצאות")
                info = info["entries"][0]
            title = proc._sanitize(info.get("title", "Unknown"))
            if not title:
                title = f"Video_{info.get('id', 'unknown')}"
            real_url = info.get("webpage_url") or info.get("url") or search_url
    except Exception as e:
        raise ValueError(f"שגיאה בקבלת מידע: {e}")

    song_folder = os.path.join(output_dir, title)
    os.makedirs(song_folder, exist_ok=True)

    # Download
    video_path = proc.download(real_url, song_folder, format_type=format_type)
    if not video_path:
        raise ValueError("הורדה נכשלה")

    task_log(f"✅ הורד: {os.path.basename(video_path)}")
    return {
        "title": title,
        "url": real_url,
        "folder": song_folder,
        "video_path": video_path,
    }
