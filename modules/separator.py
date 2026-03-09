"""
modules/separator.py — Separate audio into vocal and instrumental stems.

Standalone usage:
    python -m modules.separator <audio_file> [options]

Options:
    --mode 2|4              Number of stems (default: 2)
                            2 → Vocals.wav + Playback.wav  (Kim_Vocal_2.onnx)
                            4 → Drums + Bass + Vocals + Other  (htdemucs_ft.yaml)
    --output-dir DIR        Output directory (default: same as input file)
    --force                 Re-process even if output files already exist

Examples:
    python -m modules.separator song.wav
    python -m modules.separator song.mp4 --mode 4 --output-dir ./stems
    python -m modules.separator song.wav --force
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
# Public API
# ---------------------------------------------------------------------------

def separate(
    audio_path: str,
    output_dir: Optional[str] = None,
    mode: int = 2,
    force: bool = False,
    logs: Optional[list] = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    Separate an audio/video file into stems.

    Args:
        audio_path: Path to the input audio or video file.
        output_dir: Directory for output WAV files.
                    Defaults to the parent directory of audio_path.
        mode:       2 → vocals + instrumental, 4 → also runs Demucs 4-stem.
        force:      Re-process even if outputs already exist.
        logs:       Optional list for log messages.

    Returns:
        (vocals_path, playback_path) — either may be None on failure.
    """
    if logs is None:
        logs = []

    out_dir = output_dir or str(Path(audio_path).parent)
    os.makedirs(out_dir, exist_ok=True)

    bp = BackendProcessor()
    return bp.separate_audio(
        audio_path,
        out_dir,
        logs,
        save_4=(mode == 4),
        force=force,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m modules.separator",
        description="🎵 הפרד קובץ אודיו לערוצי ווקאל ופלייבק",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("audio", help="נתיב לקובץ אודיו/וידאו")
    p.add_argument(
        "--mode", type=int, choices=[2, 4], default=2,
        help="2 ערוצים (ברירת מחדל) | 4 ערוצים (Demucs)",
    )
    p.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help="תיקיית פלט (ברירת מחדל: תיקיית הקובץ)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="עבד מחדש גם אם קבצי הפלט קיימים",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if not os.path.exists(args.audio):
        print(f"❌ הקובץ לא נמצא: {args.audio}", file=sys.stderr)
        sys.exit(1)

    logs: list[str] = []
    vocals, playback = separate(
        args.audio,
        output_dir=args.output_dir,
        mode=args.mode,
        force=args.force,
        logs=logs,
    )

    for msg in logs:
        print(msg)

    if vocals and playback:
        print(f"\n✅ ווקאל   : {vocals}")
        print(f"✅ פלייבק  : {playback}")
    else:
        print("❌ הפרדה נכשלה.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
