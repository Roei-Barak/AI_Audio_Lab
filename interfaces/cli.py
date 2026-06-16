"""CLI logic — called by main_cli.py."""
import argparse
import sys

from logic import run_karaoke_pipeline
from config import ASS_DEFAULT_FONT_SIZE, ASS_DEFAULT_COLOR


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="karaoke",
        description="AI_Audio_Lab — Karaoke CLI\nURL → download → separate → transcribe → karaoke video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url",  metavar="URL",  help="YouTube URL or search query")
    src.add_argument("--file", metavar="FILE", help="Local audio/video file path")

    p.add_argument("--lang",      default="he",    choices=["he", "en"], help="שפת התמלול (ברירת מחדל: he)")
    p.add_argument("--stems",     default="2",     choices=["2", "4"],   help="מספר ערוצי הפרדה (ברירת מחדל: 2)")
    p.add_argument("--bidi",      action="store_true",                    help="תיקון טקסט עברית BIDI")
    p.add_argument("--force",     action="store_true",                    help="עיבוד מחדש גם אם קיים")
    p.add_argument("--font-size", type=int, default=ASS_DEFAULT_FONT_SIZE, metavar="N", help="גודל פונט (ברירת מחדל: 80)")
    p.add_argument("--color",     default=ASS_DEFAULT_COLOR,              metavar="#RRGGBB", help="צבע כתוביות")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    source = args.url or args.file

    def on_progress(step: str, pct: int):
        print(f"  [{pct:3d}%] {step}", flush=True)

    print(f"\n🎤 AI_Audio_Lab — מתחיל עיבוד: {source}\n")

    video, logs = run_karaoke_pipeline(
        source=source,
        lang=args.lang,
        save_4_stems=(args.stems == "4"),
        use_bidi=args.bidi,
        force=args.force,
        font_size=args.font_size,
        color_hex=args.color,
        on_progress=on_progress,
    )

    print(f"\n{'='*60}")
    if video:
        print(f"✅ הושלם בהצלחה!")
        print(f"📁 פלט: {video}")
        return 0
    else:
        print("❌ העיבוד נכשל. בדוק את הלוגים.")
        return 1
