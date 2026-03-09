"""
cli/main.py — Lean CLI entry point for the AI_Audio_Lab karaoke pipeline.

Sub-commands:
    pipeline   <url>                    Run the full pipeline (download→separate→transcribe→render)
    download   <url>                    Download only
    separate   <file>                   Audio separation only
    transcribe <file>                   Transcription only
    render     <video> <audio> <subs>   Rendering only
    analyze    <file>                   BPM + musical key analysis
    lecture    <url|file>               Transcription without karaoke rendering
    batch      <list.txt>               Process multiple songs from a text file

Usage:
    python cli/main.py pipeline "https://youtu.be/..."
    python cli/main.py pipeline "Bohemian Rhapsody Queen" --lang en --4stems
    python cli/main.py download  "https://youtu.be/..." --format mp4
    python cli/main.py separate  song.wav --mode 4
    python cli/main.py transcribe vocals.wav --lang he --format ass,srt,txt
    python cli/main.py render    video.mp4 playback.wav subs.ass --bidi
    python cli/main.py analyze   song.wav
    python cli/main.py lecture   "https://youtu.be/..." --lang en --format srt,txt
    python cli/main.py batch     songs.txt --lang he
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure repo root is importable when run as  python cli/main.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.backend import BackendProcessor
from core.config import WORK_DIR
from modules.downloader import download
from modules.separator import separate
from modules.transcriber import transcribe, _print_progress
from modules.renderer import render


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _logs_printer(logs: list[str], prev_len: int = 0) -> int:
    """Print any new log entries; return new log length."""
    for msg in logs[prev_len:]:
        print(msg)
    return len(logs)


def _progress_cb(idx: int, total: int, text: str) -> None:
    _print_progress(idx, total, text)


# ---------------------------------------------------------------------------
# Sub-command: pipeline
# ---------------------------------------------------------------------------

def cmd_pipeline(args: argparse.Namespace) -> None:
    bp = BackendProcessor()
    logs: list[str] = []

    fmt_list = [f.strip() for f in args.format.split(",") if f.strip()]

    print(f"🚀 מתחיל pipeline: {args.url!r}")
    final, log_str = bp.process_song_pipeline(
        args.url,
        lang=args.lang,
        save_4_stems=getattr(args, "stems4", False),
        use_bidi=args.bidi,
        force=args.force,
        output_formats=fmt_list,
        progress_callback=_progress_cb,
    )

    print(log_str)
    if final:
        print(f"\n🎉 סרטון קריוקי מוכן: {final}")
    else:
        print("❌ Pipeline נכשל.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Sub-command: download
# ---------------------------------------------------------------------------

def cmd_download(args: argparse.Namespace) -> None:
    logs: list[str] = []
    result = download(args.url, output_dir=args.output_dir, fmt=args.fmt, logs=logs)
    for msg in logs:
        print(msg)
    if result:
        print(f"\n✅ הורד: {result}")
    else:
        print("❌ הורדה נכשלה.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Sub-command: separate
# ---------------------------------------------------------------------------

def cmd_separate(args: argparse.Namespace) -> None:
    logs: list[str] = []
    vocals, playback = separate(
        args.audio, output_dir=args.output_dir,
        mode=args.mode, force=args.force, logs=logs,
    )
    for msg in logs:
        print(msg)
    if vocals and playback:
        print(f"\n✅ ווקאל  : {vocals}")
        print(f"✅ פלייבק : {playback}")
    else:
        print("❌ הפרדה נכשלה.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Sub-command: transcribe
# ---------------------------------------------------------------------------

def cmd_transcribe(args: argparse.Namespace) -> None:
    fmt_list = [f.strip() for f in args.fmt.split(",") if f.strip()]
    logs: list[str] = []
    results = transcribe(
        args.audio,
        output_dir=args.output_dir,
        lang=args.lang,
        output_formats=fmt_list,
        title=args.title,
        force=args.force,
        progress_callback=_progress_cb,
        logs=logs,
    )
    for msg in logs:
        print(msg, file=sys.stderr)
    print()
    ok = False
    for fmt, path in results.items():
        if path:
            print(f"✅ {fmt.upper():4s}: {path}")
            ok = True
        else:
            print(f"❌ {fmt.upper():4s}: נכשל")
    if not ok:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Sub-command: render
# ---------------------------------------------------------------------------

def cmd_render(args: argparse.Namespace) -> None:
    logs: list[str] = []
    result = render(
        args.video, args.audio, args.subtitles,
        output_dir=args.output_dir,
        output_name=args.output_name,
        use_bidi=args.bidi,
        font_size=args.font_size,
        color_hex=args.color,
        position=args.position,
        force=args.force,
        logs=logs,
    )
    for msg in logs:
        print(msg)
    if result:
        print(f"\n✅ סרטון: {result}")
    else:
        print("❌ רנדור נכשל.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Sub-command: analyze
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> None:
    bp = BackendProcessor()
    result, logs = bp.analyze_audio(args.audio)
    print(logs)
    print(f"\n🎼 {result}")


# ---------------------------------------------------------------------------
# Sub-command: lecture
# ---------------------------------------------------------------------------

def cmd_lecture(args: argparse.Namespace) -> None:
    """Transcribe a lecture/talk — no karaoke rendering, just text output."""
    bp = BackendProcessor()
    logs: list[str] = []

    # If the input is a URL, download first
    source = args.source
    if source.startswith("http"):
        print("📥 מוריד…")
        source = download(source, output_dir=args.output_dir or WORK_DIR, fmt="wav", logs=logs)
        if not source:
            for msg in logs:
                print(msg)
            print("❌ הורדה נכשלה.", file=sys.stderr)
            sys.exit(1)

    fmt_list = [f.strip() for f in args.fmt.split(",") if f.strip()]
    results = transcribe(
        source,
        output_dir=args.output_dir,
        lang=args.lang,
        output_formats=fmt_list,
        progress_callback=_progress_cb,
        logs=logs,
    )
    for msg in logs:
        print(msg, file=sys.stderr)
    print()
    for fmt, path in results.items():
        if path:
            print(f"✅ {fmt.upper():4s}: {path}")
        else:
            print(f"❌ {fmt.upper():4s}: נכשל")


# ---------------------------------------------------------------------------
# Sub-command: batch
# ---------------------------------------------------------------------------

def cmd_batch(args: argparse.Namespace) -> None:
    """Process a list of URLs/queries from a text file, one per line."""
    if not os.path.exists(args.list_file):
        print(f"❌ הקובץ לא נמצא: {args.list_file}", file=sys.stderr)
        sys.exit(1)

    with open(args.list_file, encoding="utf-8") as f:
        songs = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    if not songs:
        print("⚠️ הקובץ ריק.", file=sys.stderr)
        return

    print(f"📚 עיבוד {len(songs)} שירים…\n")
    bp = BackendProcessor()
    fmt_list = [f.strip() for f in args.format.split(",") if f.strip()]

    results: list[tuple[str, bool, str]] = []
    for i, song in enumerate(songs, start=1):
        print(f"{'─'*60}")
        print(f"[{i}/{len(songs)}] {song}")
        print(f"{'─'*60}")

        start = time.time()
        final, log_str = bp.process_song_pipeline(
            song,
            lang=args.lang,
            save_4_stems=getattr(args, "stems4", False),
            use_bidi=args.bidi,
            force=args.force,
            output_formats=fmt_list,
            progress_callback=_progress_cb,
        )
        elapsed = time.time() - start
        ok = bool(final)
        results.append((song, ok, f"{elapsed:.0f}s"))
        status = "✅" if ok else "❌"
        print(f"{status} {song} — {'הסתיים' if ok else 'נכשל'} ({elapsed:.0f}s)\n")

    # Summary
    print(f"\n{'═'*60}")
    print("סיכום:")
    for song, ok, dur in results:
        icon = "✅" if ok else "❌"
        print(f"  {icon} {song[:55]:<55} {dur:>6}")
    succeeded = sum(1 for _, ok, _ in results if ok)
    print(f"\n{succeeded}/{len(songs)} הצליחו.")


# ---------------------------------------------------------------------------
# Argument parser builder
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="python cli/main.py",
        description="🎤 AI_Audio_Lab — Karaoke Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = root.add_subparsers(dest="command", required=True)

    # ── pipeline ──
    p_pipe = sub.add_parser("pipeline", help="הרץ את כל התהליך מקצה לקצה")
    p_pipe.add_argument("url", help="קישור יוטיוב או חיפוש")
    p_pipe.add_argument("--lang", choices=["he", "en", "auto"], default="he")
    p_pipe.add_argument("--format", default="ass,srt", metavar="FORMATS",
                        help="פורמטי תמלול (ברירת מחדל: ass,srt)")
    p_pipe.add_argument("--4stems", dest="stems4", action="store_true",
                        help="שמור גם 4 ערוצים (Demucs)")
    p_pipe.add_argument("--bidi", action="store_true", help="תיקון BIDI לעברית")
    p_pipe.add_argument("--force", action="store_true", help="עבד מחדש")
    p_pipe.set_defaults(func=cmd_pipeline)

    # ── download ──
    p_dl = sub.add_parser("download", help="הורד בלבד")
    p_dl.add_argument("url", help="קישור יוטיוב או חיפוש")
    p_dl.add_argument("--format", choices=["wav", "mp4"], default="wav",
                      dest="fmt", metavar="FORMAT")
    p_dl.add_argument("--output-dir", default=WORK_DIR, metavar="DIR")
    p_dl.set_defaults(func=cmd_download)

    # ── separate ──
    p_sep = sub.add_parser("separate", help="הפרד אודיו לערוצים")
    p_sep.add_argument("audio", help="נתיב לקובץ אודיו/וידאו")
    p_sep.add_argument("--mode", type=int, choices=[2, 4], default=2)
    p_sep.add_argument("--output-dir", default=None, metavar="DIR")
    p_sep.add_argument("--force", action="store_true")
    p_sep.set_defaults(func=cmd_separate)

    # ── transcribe ──
    p_tr = sub.add_parser("transcribe", help="תמלל קובץ אודיו")
    p_tr.add_argument("audio", help="נתיב לקובץ אודיו/וידאו")
    p_tr.add_argument("--lang", choices=["he", "en", "auto"], default="he")
    p_tr.add_argument("--format", default="ass,srt", metavar="FORMATS", dest="fmt")
    p_tr.add_argument("--output-dir", default=None, metavar="DIR")
    p_tr.add_argument("--title", default=None, metavar="NAME")
    p_tr.add_argument("--force", action="store_true")
    p_tr.set_defaults(func=cmd_transcribe)

    # ── render ──
    p_ren = sub.add_parser("render", help="רנדר סרטון קריוקי")
    p_ren.add_argument("video", help="נתיב לסרטון מקור")
    p_ren.add_argument("audio", help="נתיב לאודיו פלייבק")
    p_ren.add_argument("subtitles", help="נתיב לקובץ ASS")
    p_ren.add_argument("--output-dir", default=None, metavar="DIR")
    p_ren.add_argument("--output-name", default=None, metavar="NAME")
    p_ren.add_argument("--bidi", action="store_true")
    p_ren.add_argument("--font-size", type=int, default=None, metavar="N")
    p_ren.add_argument("--color", default=None, metavar="HEX")
    p_ren.add_argument("--position", choices=["top", "center", "bottom"], default="bottom")
    p_ren.add_argument("--force", action="store_true")
    p_ren.set_defaults(func=cmd_render)

    # ── analyze ──
    p_an = sub.add_parser("analyze", help="נתח BPM ומפתח מוסיקלי")
    p_an.add_argument("audio", help="נתיב לקובץ אודיו")
    p_an.set_defaults(func=cmd_analyze)

    # ── lecture ──
    p_lec = sub.add_parser("lecture", help="תמלול הרצאה (ללא רנדור)")
    p_lec.add_argument("source", help="קישור יוטיוב או נתיב לקובץ")
    p_lec.add_argument("--lang", choices=["he", "en", "auto"], default="he")
    p_lec.add_argument("--format", default="srt,txt", metavar="FORMATS", dest="fmt")
    p_lec.add_argument("--output-dir", default=None, metavar="DIR")
    p_lec.set_defaults(func=cmd_lecture)

    # ── batch ──
    p_bat = sub.add_parser("batch", help="עבד רשימת שירים מקובץ טקסט")
    p_bat.add_argument("list_file", metavar="LIST_FILE",
                       help="קובץ טקסט — URL/חיפוש אחד בכל שורה")
    p_bat.add_argument("--lang", choices=["he", "en", "auto"], default="he")
    p_bat.add_argument("--format", default="ass,srt", metavar="FORMATS")
    p_bat.add_argument("--4stems", dest="stems4", action="store_true")
    p_bat.add_argument("--bidi", action="store_true")
    p_bat.add_argument("--force", action="store_true")
    p_bat.set_defaults(func=cmd_batch)

    return root


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
