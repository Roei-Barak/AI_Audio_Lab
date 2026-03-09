"""
modules/transcriber.py — Transcribe audio to subtitles with real-time progress.

Standalone usage:
    python -m modules.transcriber <audio_file> [options]

Options:
    --lang he|en|auto       Language (default: he).  auto = Whisper auto-detect.
    --format ass|srt|txt    Output format(s), comma-separated (default: ass,srt)
    --output-dir DIR        Output directory (default: same as input)
    --title NAME            Base filename for outputs (default: input stem)
    --no-progress           Suppress real-time chunk output

Progress is printed to stderr so it can be piped or redirected independently
from the final result paths printed to stdout.

Examples:
    python -m modules.transcriber vocals.wav
    python -m modules.transcriber lecture.mp3 --lang en --format srt,txt
    python -m modules.transcriber song.wav --lang he --format ass,srt,txt --output-dir ./out
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Optional

from core.backend import BackendProcessor
from core.config import WORK_DIR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe(
    audio_path: str,
    output_dir: Optional[str] = None,
    lang: str = "he",
    output_formats: Optional[list[str]] = None,
    title: Optional[str] = None,
    force: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    logs: Optional[list] = None,
) -> dict[str, Optional[str]]:
    """
    Transcribe an audio file using Whisper and write subtitle/text output.

    Args:
        audio_path:         Path to audio or video file.
        output_dir:         Directory for output files.
        lang:               "he" | "en" | "auto" (auto uses Whisper detection).
        output_formats:     List of formats: "ass", "srt", "txt".
                            Defaults to ["ass", "srt"].
        title:              Base filename (without extension).
        force:              Re-transcribe even if outputs already exist.
        progress_callback:  Called after each ~60-second chunk with
                            (chunk_index, total_chunks, chunk_text).
        logs:               Optional list for log messages.

    Returns:
        Dict mapping format → output path (or None if failed).
        Example: {"ass": "/path/song.ass", "srt": "/path/song.srt", "txt": None}
    """
    if logs is None:
        logs = []
    if output_formats is None:
        output_formats = ["ass", "srt"]

    out_dir = output_dir or str(Path(audio_path).parent)
    os.makedirs(out_dir, exist_ok=True)

    stem = title or Path(audio_path).stem

    # Resolve "auto" → let Whisper detect; pass as "auto" (pipeline handles it)
    effective_lang = lang if lang != "auto" else "he"  # fallback; Whisper detects internally

    bp = BackendProcessor()

    # Delete existing outputs if force
    if force:
        for fmt in output_formats:
            p = os.path.join(out_dir, f"{stem}.{fmt}")
            if os.path.exists(p):
                os.remove(p)

    ass_path = bp.transcribe_audio(
        audio_path,
        out_dir,
        stem,
        logs,
        lang=effective_lang,
        output_formats=output_formats,
        progress_callback=progress_callback,
    )

    results: dict[str, Optional[str]] = {}
    for fmt in output_formats:
        p = os.path.join(out_dir, f"{stem}.{fmt}")
        results[fmt] = p if os.path.exists(p) else None

    return results


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _print_progress(idx: int, total: int, text: str) -> None:
    """Print real-time chunk progress to stderr."""
    bar_len = 30
    filled = int(bar_len * idx / max(total, 1))
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = int(100 * idx / max(total, 1))
    preview = text[:70].replace("\n", " ")
    print(f"\r[{bar}] {pct:3d}%  {preview}", end="", file=sys.stderr, flush=True)
    if idx >= total:
        print(file=sys.stderr)  # newline at end


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m modules.transcriber",
        description="🗣️ תמלל קובץ אודיו עם חיווי בזמן אמת",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("audio", help="נתיב לקובץ אודיו/וידאו")
    p.add_argument(
        "--lang", choices=["he", "en", "auto"], default="he",
        help="שפה: he (ברירת מחדל) | en | auto",
    )
    p.add_argument(
        "--format", default="ass,srt", metavar="FORMATS", dest="fmt",
        help="פורמטים מופרדים בפסיק: ass,srt,txt (ברירת מחדל: ass,srt)",
    )
    p.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help="תיקיית פלט (ברירת מחדל: תיקיית הקובץ)",
    )
    p.add_argument(
        "--title", default=None, metavar="NAME",
        help="שם בסיס לקבצי הפלט (ברירת מחדל: שם קובץ הקלט)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="תמלל מחדש גם אם קבצי הפלט קיימים",
    )
    p.add_argument(
        "--no-progress", action="store_true",
        help="הסתר חיווי התקדמות",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if not os.path.exists(args.audio):
        print(f"❌ הקובץ לא נמצא: {args.audio}", file=sys.stderr)
        sys.exit(1)

    formats = [f.strip().lower() for f in args.fmt.split(",") if f.strip()]
    valid_formats = {"ass", "srt", "txt"}
    unknown = set(formats) - valid_formats
    if unknown:
        print(f"❌ פורמטים לא מוכרים: {', '.join(unknown)}", file=sys.stderr)
        sys.exit(1)

    progress_cb = None if args.no_progress else _print_progress
    logs: list[str] = []

    print(f"📝 מתמלל: {args.audio}", file=sys.stderr)
    print(f"   שפה: {args.lang} | פורמטים: {', '.join(formats)}", file=sys.stderr)
    print(file=sys.stderr)

    results = transcribe(
        args.audio,
        output_dir=args.output_dir,
        lang=args.lang,
        output_formats=formats,
        title=args.title,
        force=args.force,
        progress_callback=progress_cb,
        logs=logs,
    )

    # Print logs to stderr
    for msg in logs:
        print(msg, file=sys.stderr)

    # Print output paths to stdout
    success = False
    print()
    for fmt, path in results.items():
        if path:
            print(f"✅ {fmt.upper():4s}: {path}")
            success = True
        else:
            print(f"❌ {fmt.upper():4s}: נכשל", file=sys.stderr)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
