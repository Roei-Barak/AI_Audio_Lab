"""
modules/renderer.py — Render a karaoke video with subtitles and alternate audio.

Standalone usage:
    python -m modules.renderer <video> <audio> <subtitles.ass> [options]

Options:
    --output-dir DIR        Output directory (default: same as video file)
    --output-name NAME      Output filename without extension
    --bidi                  Apply Hebrew BIDI text fix to subtitles
    --font-size N           Override subtitle font size (default: keep original)
    --color HEX             Override subtitle primary colour, e.g. #FFD700
    --position top|center|bottom  Subtitle position (default: bottom)
    --preset NAME           Apply a named colour preset (see --list-presets)
    --list-presets          Print available colour presets and exit
    --force                 Overwrite existing output file

Examples:
    python -m modules.renderer video.mp4 playback.wav subtitles.ass
    python -m modules.renderer v.mp4 audio.wav subs.ass --color #FFD700 --position bottom
    python -m modules.renderer v.mp4 a.wav s.ass --preset "זהב קריוקי" --bidi
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from core.backend import BackendProcessor
from core.config import SUBTITLE_POSITIONS, SUBTITLE_PRESETS, WORK_DIR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render(
    video_path: str,
    audio_path: str,
    subtitles_path: str,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    use_bidi: bool = False,
    font_size: Optional[int] = None,
    color_hex: Optional[str] = None,
    position: str = "bottom",
    force: bool = False,
    logs: Optional[list] = None,
) -> Optional[str]:
    """
    Render a karaoke MP4 by combining video, audio, and ASS subtitles.

    Args:
        video_path:     Source video file.
        audio_path:     Replacement audio (WAV/MP3/etc.).
        subtitles_path: ASS subtitle file.
        output_dir:     Directory for the output file.
        output_name:    Stem for the output filename (no extension).
        use_bidi:       Apply Hebrew BIDI text correction to subtitle dialogue.
        font_size:      Override ASS style font size.
        color_hex:      Override ASS style primary colour (e.g. "#FFD700").
        position:       "top" | "center" | "bottom".
        force:          Overwrite existing output.
        logs:           Optional list for log messages.

    Returns:
        Absolute path to the rendered MP4, or None on failure.
    """
    if logs is None:
        logs = []

    out_dir = output_dir or str(Path(video_path).parent)
    os.makedirs(out_dir, exist_ok=True)

    stem = output_name or (Path(video_path).stem + "_KARAOKE")
    video_info = {"folder": out_dir, "title": stem}

    bp = BackendProcessor()

    # Apply style overrides to a temporary copy of the ASS
    working_ass = subtitles_path
    if font_size or color_hex or position != "bottom":
        import shutil, uuid as _uuid
        tmp_ass = os.path.join(out_dir, f"render_style_{_uuid.uuid4().hex[:6]}.ass")
        shutil.copy2(subtitles_path, tmp_ass)
        align = SUBTITLE_POSITIONS.get(
            {"top": "למעלה", "center": "מרכז", "bottom": "למטה"}.get(position, "למטה"),
            2,
        )
        bp.update_ass_style(
            tmp_ass,
            font_size=font_size or 80,
            color_hex=color_hex or "#FFFFFF",
            position=align,
        )
        working_ass = tmp_ass

    result = bp.render_video(
        video_path,
        audio_path,
        working_ass,
        video_info,
        logs,
        use_bidi=use_bidi,
        force=force,
    )

    # Clean up temp ASS if we created one
    if working_ass != subtitles_path and os.path.exists(working_ass):
        try:
            os.remove(working_ass)
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    preset_names = ", ".join(f'"{n}"' for n in SUBTITLE_PRESETS)
    p = argparse.ArgumentParser(
        prog="python -m modules.renderer",
        description="🎬 רנדר סרטון קריוקי עם אודיו וכתוביות",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("video", help="נתיב לסרטון המקור")
    p.add_argument("audio", help="נתיב לקובץ האודיו (פלייבק)")
    p.add_argument("subtitles", help="נתיב לקובץ כתוביות ASS")
    p.add_argument("--output-dir", default=None, metavar="DIR", help="תיקיית פלט")
    p.add_argument("--output-name", default=None, metavar="NAME", help="שם קובץ פלט (ללא סיומת)")
    p.add_argument("--bidi", action="store_true", help="תיקון BIDI לעברית")
    p.add_argument("--font-size", type=int, default=None, metavar="N", help="גודל גופן")
    p.add_argument("--color", default=None, metavar="HEX", help="צבע כתוביות, למשל #FFD700")
    p.add_argument(
        "--position", choices=["top", "center", "bottom"], default="bottom",
        help="מיקום כתוביות (ברירת מחדל: bottom)",
    )
    p.add_argument(
        "--preset", default=None, metavar="NAME",
        help=f"פריסט צבע מוכן: {preset_names}",
    )
    p.add_argument("--list-presets", action="store_true", help="הצג פריסטים זמינים ויצא")
    p.add_argument("--force", action="store_true", help="דרוס קובץ פלט קיים")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.list_presets:
        print("פריסטים זמינים:")
        for name, (primary, outline) in SUBTITLE_PRESETS.items():
            print(f"  {name!r:20s}  primary={primary}  outline={outline}")
        return

    # Validate input files
    for label, path in [("וידאו", args.video), ("אודיו", args.audio), ("כתוביות", args.subtitles)]:
        if not os.path.exists(path):
            print(f"❌ קובץ {label} לא נמצא: {path}", file=sys.stderr)
            sys.exit(1)

    # Resolve preset
    color_hex = args.color
    if args.preset:
        if args.preset not in SUBTITLE_PRESETS:
            print(f"❌ פריסט לא מוכר: {args.preset!r}", file=sys.stderr)
            sys.exit(1)
        color_hex = SUBTITLE_PRESETS[args.preset][0]

    logs: list[str] = []
    result = render(
        args.video,
        args.audio,
        args.subtitles,
        output_dir=args.output_dir,
        output_name=args.output_name,
        use_bidi=args.bidi,
        font_size=args.font_size,
        color_hex=color_hex,
        position=args.position,
        force=args.force,
        logs=logs,
    )

    for msg in logs:
        print(msg)

    if result:
        print(f"\n✅ סרטון קריוקי: {result}")
    else:
        print("❌ רנדור נכשל.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
