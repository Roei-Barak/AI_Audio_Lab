"""
modules/downloader.py — Download video/audio from YouTube.

Standalone usage:
    python -m modules.downloader <url_or_query> [options]

Options:
    --format wav|mp4        Output format (default: wav)
    --output-dir DIR        Output directory (default: Karaoke_Output)
    --info-only             Print metadata without downloading
    --playlist              Allow playlist download (disabled by default)
    --quality best|audio    Video quality preset (default: best)

Examples:
    python -m modules.downloader "https://youtu.be/dQw4w9WgXcQ"
    python -m modules.downloader "rickroll" --format mp4 --output-dir /tmp/songs
    python -m modules.downloader "https://youtu.be/abc" --info-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from core.backend import BackendProcessor
from core.config import WORK_DIR


# ---------------------------------------------------------------------------
# Public API (importable by CLI, GUI, API)
# ---------------------------------------------------------------------------

def download(
    url: str,
    output_dir: str = WORK_DIR,
    fmt: str = "wav",
    logs: Optional[list] = None,
) -> Optional[str]:
    """
    Download a YouTube video and optionally convert to WAV.

    Args:
        url:        YouTube URL or search query string.
        output_dir: Directory to save the file.
        fmt:        "wav" (default) or "mp4".
        logs:       Optional list; log messages are appended to it.

    Returns:
        Absolute path to the downloaded file, or None on failure.
    """
    if logs is None:
        logs = []

    bp = BackendProcessor()
    info = bp.get_video_info(url, logs)
    if not info:
        return None

    # Override folder to the requested output_dir
    info["folder"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    mp4_path = bp.download_video(info, logs)
    if not mp4_path:
        return None

    if fmt == "wav":
        wav_path = str(Path(mp4_path).with_suffix(".wav"))
        bp.log(f"🔄 ממיר ל-WAV…", logs)
        ok = bp.convert_to_wav(mp4_path, wav_path)
        if ok and os.path.exists(wav_path):
            bp.log(f"✅ WAV נשמר: {wav_path}", logs)
            return wav_path
        bp.log("⚠️ ממיר ל-WAV נכשל — מחזיר MP4.", logs)

    return mp4_path


def get_info(url: str, logs: Optional[list] = None) -> Optional[dict]:
    """
    Return metadata dict for a YouTube URL/query without downloading.

    Keys: title, url, id, folder.
    """
    if logs is None:
        logs = []
    return BackendProcessor().get_video_info(url, logs)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m modules.downloader",
        description="📥 הורד וידאו/אודיו מיוטיוב",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("url", help="קישור יוטיוב או ביטוי חיפוש")
    p.add_argument(
        "--format", choices=["wav", "mp4"], default="wav",
        metavar="FORMAT", dest="fmt",
        help="פורמט פלט: wav (ברירת מחדל) | mp4",
    )
    p.add_argument(
        "--output-dir", default=WORK_DIR, metavar="DIR",
        help=f"תיקיית פלט (ברירת מחדל: {WORK_DIR})",
    )
    p.add_argument(
        "--info-only", action="store_true",
        help="הצג מטא-דאטה בלי להוריד",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    logs: list[str] = []

    if args.info_only:
        info = get_info(args.url, logs)
        for msg in logs:
            print(msg)
        if info:
            print(f"\n📹 כותרת : {info['title']}")
            print(f"🔗 URL    : {info['url']}")
            print(f"🆔 ID     : {info['id']}")
        else:
            print("❌ לא נמצא מידע.", file=sys.stderr)
            sys.exit(1)
        return

    result = download(args.url, args.output_dir, args.fmt, logs)
    for msg in logs:
        print(msg)

    if result:
        print(f"\n✅ קובץ נשמר: {result}")
    else:
        print("❌ הורדה נכשלה.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
