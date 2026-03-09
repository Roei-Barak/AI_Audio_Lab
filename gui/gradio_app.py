"""
gui/gradio_app.py — Extended Gradio web UI for AI_Audio_Lab.

Tabs
────
  ⚡ תהליך אוטומטי     Full pipeline (download → separate → transcribe → render)
  🛠️ כלים בנפרד        Individual tools: download / separate / transcribe / render
  📚 עיבוד רשימה       Batch processing with per-song status table
  📝 עורך כתוביות      Load ASS, edit timing/text in a table, re-render
  🎭 עורך פרודיה       Original lyrics ↔ alternative lyrics side-by-side
  🎤 תמלול הרצאה       Transcribe a talk/lecture — no karaoke rendering
  🎼 ניתוח שיר         BPM + musical key analysis
  ⚙️ הגדרות            Output directory, server URL, default language, GPU info

Run:
    python gui/gradio_app.py                    # local UI
    python gui/gradio_app.py --server           # expose on 0.0.0.0:7860
    python gui/gradio_app.py --port 8080
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Generator

import gradio as gr
import pandas as pd

# Ensure repo root is on path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.backend import BackendProcessor, manager
from core.config import (
    DEVICE,
    SUBTITLE_POSITIONS,
    SUBTITLE_PRESETS,
    WORK_DIR,
)
from modules.downloader import download, get_info
from modules.separator import separate
from modules.transcriber import transcribe
from modules.renderer import render

bp = BackendProcessor()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LANG_CHOICES = [("עברית 🇮🇱", "he"), ("אנגלית 🇺🇸", "en"), ("זיהוי אוטומטי", "auto")]
FORMAT_CHOICES = ["ass", "srt", "txt"]
PRESET_NAMES = list(SUBTITLE_PRESETS.keys())
POSITION_NAMES = ["למטה", "מרכז", "למעלה"]


def _status() -> str:
    return manager.get_status()


def _device_info() -> str:
    if DEVICE == "cuda":
        try:
            import torch
            name = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            return f"🖥️ GPU: {name} | VRAM: {free/1024**3:.1f}/{total/1024**3:.1f} GB"
        except Exception:
            return "🖥️ GPU: CUDA"
    return "🖥️ CPU"


def _fmt_logs(logs: list[str]) -> str:
    return "\n".join(logs)


def _progress_cb_factory(log_box_state: list[str]):
    """Return a progress callback that appends to a shared log list."""
    def cb(idx: int, total: int, text: str) -> None:
        pct = int(100 * idx / max(total, 1))
        log_box_state.append(f"[{pct:3d}%] {text[:100]}")
    return cb


# ---------------------------------------------------------------------------
# Tab 1 — Full pipeline
# ---------------------------------------------------------------------------

def ui_pipeline(
    url: str,
    lang: str,
    output_formats: list[str],
    save_4: bool,
    bidi: bool,
    force: bool,
) -> Generator:
    logs: list[str] = []
    progress_text: list[str] = []

    def cb(idx, total, text):
        pct = int(100 * idx / max(total, 1))
        progress_text.append(f"[{pct:3d}%] {text[:100]}")

    # Yield early status
    yield None, "⏳ מתחיל…"

    final, log_str = bp.process_song_pipeline(
        url,
        lang=lang,
        save_4_stems=save_4,
        use_bidi=bidi,
        force=force,
        output_formats=output_formats if output_formats else ["ass"],
        progress_callback=cb,
    )

    full_log = log_str
    if progress_text:
        full_log += "\n\n── תמלול ──\n" + "\n".join(progress_text)

    yield final, full_log


# ---------------------------------------------------------------------------
# Tab 2 — Individual tools
# ---------------------------------------------------------------------------

def ui_tool_download(url: str, fmt: str, out_dir: str):
    logs: list[str] = []
    result = download(url, output_dir=out_dir or WORK_DIR, fmt=fmt, logs=logs)
    return result, _fmt_logs(logs)


def ui_tool_separate(file_obj, url: str, mode: int, out_dir: str, force: bool):
    logs: list[str] = []
    audio = file_obj.name if file_obj else None
    if not audio and url:
        audio = download(url, output_dir=out_dir or WORK_DIR, fmt="wav", logs=logs)
    if not audio:
        return None, None, _fmt_logs(logs)
    vocals, playback = separate(audio, output_dir=out_dir or None, mode=mode, force=force, logs=logs)
    return vocals, playback, _fmt_logs(logs)


def ui_tool_transcribe(
    file_obj,
    url: str,
    lang: str,
    output_formats: list[str],
    out_dir: str,
    force: bool,
) -> Generator:
    logs: list[str] = []
    progress: list[str] = []

    yield pd.DataFrame(), "", "⏳ מכין…"

    audio = file_obj.name if file_obj else None
    if not audio and url:
        audio = download(url, output_dir=out_dir or WORK_DIR, fmt="wav", logs=logs)

    if not audio:
        yield pd.DataFrame(), _fmt_logs(logs), "❌ לא נמצא קובץ"
        return

    def cb(idx, total, text):
        pct = int(100 * idx / max(total, 1))
        progress.append(f"[{pct:3d}%] {text[:100]}")
        # Yield intermediate state
    results = transcribe(
        audio,
        output_dir=out_dir or None,
        lang=lang,
        output_formats=output_formats if output_formats else ["ass"],
        force=force,
        progress_callback=cb,
        logs=logs,
    )

    ass_path = results.get("ass")
    df = bp.ass_to_dataframe(ass_path) if ass_path else pd.DataFrame()
    full_log = _fmt_logs(logs)
    if progress:
        full_log += "\n\n── תמלול ──\n" + "\n".join(progress)

    yield df, full_log, "✅ הסתיים" if ass_path else "❌ נכשל"


def ui_tool_render(
    video_obj,
    audio_obj,
    ass_obj,
    bidi: bool,
    font_size: int,
    color: str,
    position: str,
    force: bool,
):
    logs: list[str] = []
    if not (video_obj and audio_obj and ass_obj):
        return None, "❌ חסרים קבצים"
    pos_map = {"למטה": "bottom", "מרכז": "center", "למעלה": "top"}
    result = render(
        video_obj.name, audio_obj.name, ass_obj.name,
        use_bidi=bidi,
        font_size=font_size,
        color_hex=color,
        position=pos_map.get(position, "bottom"),
        force=force,
        logs=logs,
    )
    return result, _fmt_logs(logs)


# ---------------------------------------------------------------------------
# Tab 3 — Batch processing
# ---------------------------------------------------------------------------

def ui_batch(
    text: str,
    lang: str,
    output_formats: list[str],
    save_4: bool,
    bidi: bool,
    force: bool,
) -> Generator:
    songs = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if not songs:
        yield pd.DataFrame(columns=["#", "שיר", "סטטוס", "זמן"]), "⚠️ הרשימה ריקה"
        return

    rows = [{"#": i + 1, "שיר": s, "סטטוס": "⏳ ממתין", "זמן": ""} for i, s in enumerate(songs)]
    df = pd.DataFrame(rows)
    yield df, f"📚 {len(songs)} שירים בתור…"

    fmt_list = output_formats if output_formats else ["ass"]
    log_all: list[str] = []

    for i, song in enumerate(songs):
        rows[i]["סטטוס"] = "🔄 בעיבוד"
        df = pd.DataFrame(rows)
        yield df, "\n".join(log_all) + f"\n\n── [{i+1}/{len(songs)}] {song} ──"

        t0 = time.time()
        final, log_str = bp.process_song_pipeline(
            song, lang=lang, save_4_stems=save_4,
            use_bidi=bidi, force=force, output_formats=fmt_list,
        )
        elapsed = f"{time.time() - t0:.0f}s"
        log_all.append(log_str)

        rows[i]["סטטוס"] = "✅ הסתיים" if final else "❌ נכשל"
        rows[i]["זמן"] = elapsed
        df = pd.DataFrame(rows)
        yield df, "\n".join(log_all)


# ---------------------------------------------------------------------------
# Tab 4 — Subtitle timing editor
# ---------------------------------------------------------------------------

def ui_editor_load(ass_file):
    if not ass_file:
        return pd.DataFrame(columns=["Start", "End", "Text"]), "⚠️ לא נבחר קובץ"
    df = bp.ass_to_dataframe(ass_file.name)
    return df, f"✅ נטען: {len(df)} שורות"


def ui_editor_render(
    df: pd.DataFrame,
    ass_file,
    video_file,
    audio_file,
    font_size: int,
    color: str,
    position: str,
    bidi: bool,
):
    logs: list[str] = []
    if df is None or df.empty:
        return None, "❌ אין נתונים בטבלה"

    import tempfile, uuid as _uuid
    tmp_ass = os.path.join(WORK_DIR, f"edited_{_uuid.uuid4().hex[:6]}.ass")
    orig = ass_file.name if ass_file else None
    bp.dataframe_to_ass(df, orig, tmp_ass)

    align = SUBTITLE_POSITIONS.get(position, 2)
    bp.update_ass_style(tmp_ass, font_size=font_size, color_hex=color, position=align)

    info = {"folder": WORK_DIR, "title": f"Edited_{int(time.time())}"}
    v = video_file.name if video_file else None
    a = audio_file.name if audio_file else None
    if not v or not a:
        return None, "❌ חסר קובץ וידאו או אודיו"

    result = bp.render_video(v, a, tmp_ass, info, logs, use_bidi=bidi, force=True)
    return result, _fmt_logs(logs)


# ---------------------------------------------------------------------------
# Tab 5 — Parody editor
# ---------------------------------------------------------------------------

def ui_parody_load(ass_file):
    if not ass_file:
        return pd.DataFrame(columns=["Start", "End", "מקור", "חלופי"]), "⚠️ לא נבחר קובץ"
    df = bp.ass_to_dataframe(ass_file.name)
    parody_df = pd.DataFrame({
        "Start": df.get("Start", pd.Series(dtype=str)),
        "End": df.get("End", pd.Series(dtype=str)),
        "מקור": df.get("Text", pd.Series(dtype=str)),
        "חלופי": [""] * len(df),
    })
    return parody_df, f"✅ נטען: {len(df)} שורות — ערוך את העמודה 'חלופי'"


def ui_parody_export(df: pd.DataFrame, ass_file, use_alt: bool):
    """Export parody as a new ASS file using the 'חלופי' column when filled."""
    if df is None or df.empty:
        return None, "❌ אין נתונים"

    import uuid as _uuid
    out_path = os.path.join(WORK_DIR, f"parody_{_uuid.uuid4().hex[:6]}.ass")
    export_df = pd.DataFrame({
        "Start": df["Start"],
        "End": df["End"],
        "Text": df["חלופי"].where(use_alt & df["חלופי"].astype(bool), df["מקור"]),
    })
    orig = ass_file.name if ass_file else None
    bp.dataframe_to_ass(export_df, orig, out_path)
    return out_path, f"✅ יוצא: {out_path}"


def ui_parody_render(
    df: pd.DataFrame,
    ass_file,
    video_file,
    audio_file,
    use_alt: bool,
    bidi: bool,
):
    logs: list[str] = []
    if df is None or df.empty:
        return None, "❌ אין נתונים"

    import uuid as _uuid
    tmp_ass = os.path.join(WORK_DIR, f"parody_render_{_uuid.uuid4().hex[:6]}.ass")
    export_df = pd.DataFrame({
        "Start": df["Start"],
        "End": df["End"],
        "Text": df["חלופי"].where(use_alt & df["חלופי"].astype(bool), df["מקור"]),
    })
    orig = ass_file.name if ass_file else None
    bp.dataframe_to_ass(export_df, orig, tmp_ass)

    info = {"folder": WORK_DIR, "title": f"Parody_{int(time.time())}"}
    v = video_file.name if video_file else None
    a = audio_file.name if audio_file else None
    if not v or not a:
        return None, "❌ חסר קובץ וידאו או אודיו"

    result = bp.render_video(v, a, tmp_ass, info, logs, use_bidi=bidi, force=True)
    return result, _fmt_logs(logs)


# ---------------------------------------------------------------------------
# Tab 6 — Lecture transcription
# ---------------------------------------------------------------------------

def ui_lecture(
    url: str,
    file_obj,
    lang: str,
    output_formats: list[str],
) -> Generator:
    logs: list[str] = []
    progress: list[str] = []
    yield "", "⏳ מכין…"

    audio = file_obj.name if file_obj else None
    if not audio and url:
        audio = download(url, output_dir=WORK_DIR, fmt="wav", logs=logs)

    if not audio:
        yield "", "❌ לא נמצא קובץ"
        return

    def cb(idx, total, text):
        pct = int(100 * idx / max(total, 1))
        progress.append(f"[{pct:3d}%] {text[:100]}")

    fmts = output_formats if output_formats else ["srt", "txt"]
    results = transcribe(
        audio, lang=lang, output_formats=fmts,
        progress_callback=cb, logs=logs,
    )

    # Show transcript content
    txt_path = results.get("txt") or results.get("srt")
    content = ""
    if txt_path and os.path.exists(txt_path):
        with open(txt_path, encoding="utf-8") as f:
            content = f.read()

    full_log = _fmt_logs(logs)
    if progress:
        full_log += "\n\n── תמלול ──\n" + "\n".join(progress)

    paths = "\n".join(f"✅ {k.upper()}: {v}" for k, v in results.items() if v)
    yield content, full_log + "\n\n" + paths


# ---------------------------------------------------------------------------
# Tab 7 — Song analysis
# ---------------------------------------------------------------------------

def ui_analyze(file_obj, url: str):
    logs: list[str] = []
    audio = file_obj.name if file_obj else None
    if not audio and url:
        audio = download(url, output_dir=WORK_DIR, fmt="wav", logs=logs)
    if not audio:
        return "❌ לא נמצא קובץ", _fmt_logs(logs)
    result, anlogs = bp.analyze_audio(audio)
    return result, _fmt_logs(logs) + "\n" + anlogs


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    )

    with gr.Blocks(
        title="🎤 Karaoke Studio Pro",
        theme=theme,
        css="""
            .tab-nav button { font-size: 15px !important; padding: 10px 20px !important; }
            .log-box textarea { font-family: monospace; font-size: 12px; }
            footer { display: none !important; }
        """,
    ) as app:

        # ── Header ──
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("# 🎤 Karaoke Studio Pro\n*מערכת קריוקי מודולרית — AI_Audio_Lab*")
            with gr.Column(scale=1):
                status_lbl = gr.Label(value="טוען…", label="מצב מערכת")
                device_md = gr.Markdown(_device_info())

        timer = gr.Timer(3)
        timer.tick(_status, outputs=status_lbl)

        # ══════════════════════════════════════════════════════
        with gr.Tabs():

            # ── Tab 1: Full pipeline ──────────────────────────
            with gr.Tab("⚡ תהליך אוטומטי"):
                with gr.Row():
                    with gr.Column(scale=1):
                        p_url = gr.Textbox(label="🔗 קישור יוטיוב או חיפוש", placeholder="https://youtu.be/… או שם שיר")
                        p_lang = gr.Dropdown(
                            choices=[v for _, v in LANG_CHOICES],
                            value="he", label="שפה",
                            info="עברית / אנגלית / זיהוי אוטומטי"
                        )
                        p_formats = gr.CheckboxGroup(
                            FORMAT_CHOICES, value=["ass", "srt"],
                            label="פורמטי כתוביות"
                        )
                        with gr.Row():
                            p_4stems = gr.Checkbox(label="שמור 4 ערוצים (Demucs)", value=False)
                            p_bidi = gr.Checkbox(label="תיקון עברית BIDI", value=False)
                            p_force = gr.Checkbox(label="עבד מחדש", value=False)
                        p_btn = gr.Button("🚀 התחל", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        p_video = gr.Video(label="🎬 סרטון קריוקי")
                        p_log = gr.TextArea(label="📋 לוג", lines=14, elem_classes=["log-box"])

                p_btn.click(
                    ui_pipeline,
                    inputs=[p_url, p_lang, p_formats, p_4stems, p_bidi, p_force],
                    outputs=[p_video, p_log],
                )

            # ── Tab 2: Individual tools ───────────────────────
            with gr.Tab("🛠️ כלים בנפרד"):
                with gr.Tabs():

                    with gr.Tab("⬇️ הורדה"):
                        with gr.Row():
                            with gr.Column():
                                dl_url = gr.Textbox(label="קישור או חיפוש")
                                dl_fmt = gr.Radio(["wav", "mp4"], value="wav", label="פורמט")
                                dl_outdir = gr.Textbox(label="תיקיית פלט", value=WORK_DIR)
                                dl_btn = gr.Button("📥 הורד", variant="primary")
                            with gr.Column():
                                dl_out = gr.File(label="קובץ שהורד")
                                dl_log = gr.TextArea(label="לוג", lines=8, elem_classes=["log-box"])
                        dl_btn.click(ui_tool_download, [dl_url, dl_fmt, dl_outdir], [dl_out, dl_log])

                    with gr.Tab("🎵 הפרדה"):
                        with gr.Row():
                            with gr.Column():
                                sp_url = gr.Textbox(label="קישור (אם אין קובץ)")
                                sp_file = gr.File(label="קובץ אודיו/וידאו")
                                sp_mode = gr.Radio([2, 4], value=2, label="מספר ערוצים")
                                sp_outdir = gr.Textbox(label="תיקיית פלט", placeholder="ברירת מחדל: תיקיית הקובץ")
                                sp_force = gr.Checkbox(label="עבד מחדש", value=False)
                                sp_btn = gr.Button("🎵 הפרד", variant="primary")
                            with gr.Column():
                                sp_vocals = gr.File(label="ווקאל")
                                sp_playback = gr.File(label="פלייבק")
                                sp_log = gr.TextArea(label="לוג", lines=8, elem_classes=["log-box"])
                        sp_btn.click(
                            ui_tool_separate,
                            [sp_file, sp_url, sp_mode, sp_outdir, sp_force],
                            [sp_vocals, sp_playback, sp_log],
                        )

                    with gr.Tab("🗣️ תמלול"):
                        with gr.Row():
                            with gr.Column():
                                tr_url = gr.Textbox(label="קישור (אם אין קובץ)")
                                tr_file = gr.File(label="קובץ אודיו/וידאו")
                                tr_lang = gr.Dropdown(["he", "en", "auto"], value="he", label="שפה")
                                tr_fmts = gr.CheckboxGroup(FORMAT_CHOICES, value=["ass", "srt"], label="פורמטים")
                                tr_outdir = gr.Textbox(label="תיקיית פלט", placeholder="ברירת מחדל: תיקיית הקובץ")
                                tr_force = gr.Checkbox(label="עבד מחדש", value=False)
                                tr_btn = gr.Button("📝 תמלל", variant="primary")
                            with gr.Column():
                                tr_table = gr.Dataframe(
                                    headers=["Start", "End", "Text"],
                                    datatype=["str", "str", "str"],
                                    col_count=(3, "fixed"),
                                    label="תוצאת תמלול",
                                    interactive=False,
                                )
                                tr_log = gr.TextArea(label="לוג", lines=6, elem_classes=["log-box"])
                                tr_status = gr.Label(label="סטטוס")
                        tr_btn.click(
                            ui_tool_transcribe,
                            [tr_file, tr_url, tr_lang, tr_fmts, tr_outdir, tr_force],
                            [tr_table, tr_log, tr_status],
                        )

                    with gr.Tab("🎬 רנדור"):
                        with gr.Row():
                            with gr.Column():
                                rn_video = gr.File(label="קובץ וידאו")
                                rn_audio = gr.File(label="קובץ אודיו (פלייבק)")
                                rn_ass = gr.File(label="קובץ כתוביות ASS")
                                rn_bidi = gr.Checkbox(label="תיקון BIDI עברית", value=False)
                                rn_size = gr.Slider(20, 150, 80, step=1, label="גודל גופן")
                                rn_color = gr.ColorPicker(value="#FFFFFF", label="צבע כתוביות")
                                rn_pos = gr.Radio(POSITION_NAMES, value="למטה", label="מיקום")
                                rn_force = gr.Checkbox(label="דרוס קיים", value=False)
                                rn_btn = gr.Button("🎬 רנדר", variant="primary")
                            with gr.Column():
                                rn_out = gr.Video(label="תוצאה")
                                rn_log = gr.TextArea(label="לוג", lines=8, elem_classes=["log-box"])
                        rn_btn.click(
                            ui_tool_render,
                            [rn_video, rn_audio, rn_ass, rn_bidi, rn_size, rn_color, rn_pos, rn_force],
                            [rn_out, rn_log],
                        )

            # ── Tab 3: Batch ──────────────────────────────────
            with gr.Tab("📚 עיבוד רשימה"):
                gr.Markdown(
                    "הזן שיר אחד בכל שורה — קישור יוטיוב או שם שיר לחיפוש.\n"
                    "שורות המתחילות ב-`#` מתעלמות מהן."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        bt_text = gr.TextArea(
                            label="רשימת שירים", lines=10,
                            placeholder="https://youtu.be/…\nBohemian Rhapsody Queen\n# הערה\n…"
                        )
                        bt_lang = gr.Dropdown(["he", "en", "auto"], value="he", label="שפה")
                        bt_fmts = gr.CheckboxGroup(FORMAT_CHOICES, value=["ass", "srt"], label="פורמטים")
                        with gr.Row():
                            bt_4stems = gr.Checkbox(label="4 ערוצים", value=False)
                            bt_bidi = gr.Checkbox(label="BIDI", value=False)
                            bt_force = gr.Checkbox(label="עבד מחדש", value=False)
                        bt_btn = gr.Button("▶️ התחל עיבוד", variant="primary")
                    with gr.Column(scale=2):
                        bt_table = gr.Dataframe(
                            headers=["#", "שיר", "סטטוס", "זמן"],
                            datatype=["number", "str", "str", "str"],
                            col_count=(4, "fixed"),
                            label="תור עיבוד",
                        )
                        bt_log = gr.TextArea(label="לוג", lines=10, elem_classes=["log-box"])
                bt_btn.click(
                    ui_batch,
                    [bt_text, bt_lang, bt_fmts, bt_4stems, bt_bidi, bt_force],
                    [bt_table, bt_log],
                )

            # ── Tab 4: Subtitle editor ────────────────────────
            with gr.Tab("📝 עורך כתוביות"):
                gr.Markdown(
                    "טען קובץ ASS, ערוך תזמון וטקסט בטבלה, ואז רנדר מחדש."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        ed_ass = gr.File(label="קובץ ASS")
                        ed_load = gr.Button("📂 טען")
                        gr.Markdown("---")
                        ed_video = gr.File(label="וידאו מקור")
                        ed_audio = gr.File(label="אודיו פלייבק")
                        ed_size = gr.Slider(20, 150, 80, step=1, label="גודל גופן")
                        ed_color = gr.ColorPicker(value="#FFFFFF", label="צבע")
                        ed_pos = gr.Radio(POSITION_NAMES, value="למטה", label="מיקום")
                        ed_bidi = gr.Checkbox(label="תיקון BIDI", value=False)
                        ed_render = gr.Button("🎬 רנדר עם עריכות", variant="primary")
                    with gr.Column(scale=2):
                        ed_table = gr.Dataframe(
                            headers=["Start", "End", "Text"],
                            datatype=["str", "str", "str"],
                            col_count=(3, "fixed"),
                            label="כתוביות — ניתן לעריכה ישירה",
                            interactive=True,
                        )
                        ed_status = gr.Label(label="סטטוס")
                        ed_out = gr.Video(label="תוצאה")
                        ed_log = gr.TextArea(label="לוג", lines=6, elem_classes=["log-box"])

                ed_load.click(ui_editor_load, [ed_ass], [ed_table, ed_status])
                ed_render.click(
                    ui_editor_render,
                    [ed_table, ed_ass, ed_video, ed_audio, ed_size, ed_color, ed_pos, ed_bidi],
                    [ed_out, ed_log],
                )

            # ── Tab 5: Parody editor ──────────────────────────
            with gr.Tab("🎭 עורך פרודיה"):
                gr.Markdown(
                    "### עורך פרודיה\n"
                    "טען קובץ ASS כדי לראות את המילים המקוריות.\n"
                    "ערוך את העמודה **חלופי** — המילים שתכתוב שם יחליפו את המקוריות."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        par_ass = gr.File(label="קובץ ASS מקורי")
                        par_load = gr.Button("📂 טען")
                        gr.Markdown("---")
                        par_use_alt = gr.Checkbox(
                            label="השתמש במילים החלופיות בייצוא/רנדור",
                            value=True,
                        )
                        par_export = gr.Button("💾 ייצא ASS חלופי")
                        gr.Markdown("---")
                        par_video = gr.File(label="וידאו מקור")
                        par_audio = gr.File(label="אודיו פלייבק")
                        par_bidi = gr.Checkbox(label="תיקון BIDI", value=False)
                        par_render = gr.Button("🎬 רנדר פרודיה", variant="primary")
                    with gr.Column(scale=2):
                        par_table = gr.Dataframe(
                            headers=["Start", "End", "מקור", "חלופי"],
                            datatype=["str", "str", "str", "str"],
                            col_count=(4, "fixed"),
                            label="מילים מקוריות ← ← ← כתוב כאן את המילים החלופיות → → →",
                            interactive=True,
                        )
                        par_status = gr.Label(label="סטטוס")
                        par_out_file = gr.File(label="קובץ ASS שיוצא")
                        par_out_video = gr.Video(label="סרטון פרודיה")
                        par_log = gr.TextArea(label="לוג", lines=5, elem_classes=["log-box"])

                par_load.click(ui_parody_load, [par_ass], [par_table, par_status])
                par_export.click(
                    ui_parody_export, [par_table, par_ass, par_use_alt], [par_out_file, par_status]
                )
                par_render.click(
                    ui_parody_render,
                    [par_table, par_ass, par_video, par_audio, par_use_alt, par_bidi],
                    [par_out_video, par_log],
                )

            # ── Tab 6: Lecture transcription ──────────────────
            with gr.Tab("🎤 תמלול הרצאה"):
                gr.Markdown(
                    "תמלל הרצאה, פודקאסט או שיעור — **ללא רנדור קריוקי**.\n"
                    "הפלט הוא קובץ טקסט ו/או SRT."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        lc_url = gr.Textbox(label="קישור יוטיוב (אופציונלי)")
                        lc_file = gr.File(label="קובץ אודיו (אופציונלי)")
                        lc_lang = gr.Dropdown(["he", "en", "auto"], value="he", label="שפה")
                        lc_fmts = gr.CheckboxGroup(["srt", "txt", "ass"], value=["srt", "txt"], label="פורמטים")
                        lc_btn = gr.Button("🎤 תמלל", variant="primary")
                    with gr.Column(scale=2):
                        lc_text = gr.TextArea(label="תמלול", lines=20, elem_classes=["log-box"])
                        lc_log = gr.TextArea(label="לוג", lines=6, elem_classes=["log-box"])

                lc_btn.click(
                    ui_lecture,
                    [lc_url, lc_file, lc_lang, lc_fmts],
                    [lc_text, lc_log],
                )

            # ── Tab 7: Song analysis ──────────────────────────
            with gr.Tab("🎼 ניתוח שיר"):
                gr.Markdown("נתח קובץ אודיו לזיהוי **BPM** (קצב) ו**מפתח מוסיקלי**.")
                with gr.Row():
                    with gr.Column(scale=1):
                        an_url = gr.Textbox(label="קישור יוטיוב (אופציונלי)")
                        an_file = gr.File(label="קובץ אודיו (אופציונלי)")
                        an_btn = gr.Button("🔍 נתח", variant="primary")
                    with gr.Column(scale=2):
                        an_result = gr.Label(label="תוצאת ניתוח")
                        an_log = gr.TextArea(label="לוג", lines=6, elem_classes=["log-box"])
                an_btn.click(ui_analyze, [an_file, an_url], [an_result, an_log])

            # ── Tab 8: Settings ───────────────────────────────
            with gr.Tab("⚙️ הגדרות"):
                gr.Markdown("### הגדרות מערכת")
                with gr.Row():
                    with gr.Column():
                        cfg_device = gr.Label(label="חומרה", value=_device_info())
                        cfg_workdir = gr.Textbox(label="תיקיית עבודה", value=WORK_DIR, interactive=False)
                        gr.Markdown("---")
                        gr.Markdown("#### מצב שרת/לקוח")
                        gr.Markdown(
                            "להרצת המערכת כשרת: `python gui/gradio_app.py --server`\n\n"
                            "להתחברות ל-API חיצוני: הגדר את כתובת השרת ב-`api/server.py`."
                        )
                        cfg_api_url = gr.Textbox(
                            label="כתובת API חיצוני (אופציונלי)",
                            placeholder="http://myserver.com:8000",
                            interactive=True,
                        )
                        gr.Markdown("---")
                        gr.Markdown(
                            "#### פקודות CLI\n"
                            "```bash\n"
                            "python cli/main.py pipeline <url>\n"
                            "python cli/main.py batch songs.txt\n"
                            "python cli/main.py lecture <url> --lang en\n"
                            "python -m modules.downloader <url>\n"
                            "python -m modules.separator  audio.wav --mode 4\n"
                            "python -m modules.transcriber audio.wav --format ass,srt,txt\n"
                            "python -m modules.renderer  video.mp4 audio.wav subs.ass\n"
                            "```"
                        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Karaoke Studio Pro — Gradio UI")
    parser.add_argument("--server", action="store_true", help="הפעל כשרת (0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="פורט (ברירת מחדל: 7860)")
    parser.add_argument("--share", action="store_true", help="צור קישור ציבורי (Gradio Share)")
    parser.add_argument("--no-browser", action="store_true", help="אל תפתח דפדפן")
    args = parser.parse_args()

    app = build_app()
    app.queue(default_concurrency_limit=10).launch(
        server_name="0.0.0.0" if args.server else "127.0.0.1",
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
