import gradio as gr
import logic
import os
import time
import threading

def status_updater():
    return logic.mm.get_status()

def mission_control_updater():
    return logic.mm.get_df()

def get_progress():
    with logic.task_manager.lock:
        if logic.task_manager.tasks:
            first_task = next(iter(logic.task_manager.tasks.values()))
            return f"{first_task.status} {first_task.song_name[:40]}"
    return "🟢 מוכן"

def ui_process_single(url, lang, save_4_stems, use_bidi, force):
    video_path, logs = logic.backend.process_song_pipeline(url, lang, save_4_stems, use_bidi, force)
    return video_path, logs

def ui_process_batch(text_input, lang, save_4_stems, use_bidi, force):
    full_logs = []
    songs = [line.strip() for line in text_input.split('\n') if line.strip()]
    for i, song in enumerate(songs):
        full_logs.append(f"--- {i+1}/{len(songs)}: {song} ---")
        res, logs = logic.backend.process_song_pipeline(song, lang, save_4_stems, use_bidi, force)
        full_logs.append(logs)
        time.sleep(0.5)
    return "\n".join(full_logs)

def ui_render_dashboard(df, original_ass_file, video_file, audio_file, size, color, use_bidi):
    logs = []
    logic.backend.log("💾 מעבד עריכות...", logs)
    temp_ass = os.path.join(logic.WORK_DIR, f"edited_{int(time.time())}.ass")
    orig = original_ass_file.name if original_ass_file else None
    logic.backend.dataframe_to_ass(df, orig, temp_ass)
    logic.backend.update_ass_style(temp_ass, size, color)
    info = {'folder': logic.WORK_DIR, 'title': f"Edited_Karaoke_{int(time.time())}"}
    v_path = video_file.name if video_file else None
    a_path = audio_file.name if audio_file else None
    final = logic.backend.render_video(v_path, a_path, temp_ass, info, logs, use_bidi)
    return final, "\n".join(logs)

def ui_transcribe_with_table(url, file, lang):
    logs = []
    ass_path, content = None, ""
    try:
        target = file.name if file else logic.backend.download_video(logic.backend.get_video_info(url, logs), logs)
        if not target: 
            return None, None, "\n".join(logs)
        
        vocals, _ = logic.backend.separate_audio(target, logic.WORK_DIR, logs)
        if not vocals: 
            return None, None, "\n".join(logs)
        
        ass_path = logic.backend.transcribe_audio(vocals, logic.WORK_DIR, "Transcribed", logs, lang)
        if ass_path:
            df = logic.backend.ass_to_dataframe(ass_path)
            with open(ass_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
            return ass_path, df, "\n".join(logs)
    except Exception as e:
        logs.append(f"❌ שגיאה: {e}")
    
    return None, None, "\n".join(logs)

def ui_save_transcription(df, ass_file):
    if df is None or df.empty:
        return "❌ אין נתונים לשמור", ""
    
    logs = []
    try:
        output_ass = os.path.join(logic.WORK_DIR, f"edited_transcription_{int(time.time())}.ass")
        logic.backend.dataframe_to_ass(df, None, output_ass)
        logic.backend.log("✅ תמלול שמור בהצלחה", logs)
        with open(output_ass, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        return output_ass, "\n".join(logs)
    except Exception as e:
        return f"❌ שגיאה: {e}", ""

# --- בניית הממשק (Gradio 6.0 Compatible) ---
with gr.Blocks(title="Karaoke Studio V71") as app:
    gr.Markdown(f"# 🎤 Karaoke Studio Pro V71 ({logic.DEVICE.upper()})")
    
    # --- Progress Bar & Status ---
    with gr.Row():
        with gr.Column(scale=3):
            progress_label = gr.Label(value="🟢 מוכן", label="📊 התקדמות")
        with gr.Column(scale=1):
            status_label = gr.Label(value="בודק מערכת...", label="System Status")
    
    # --- Mission Control Sidebar ---
    with gr.Row():
        with gr.Column(scale=3):
            tabs_placeholder = None
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Mission Control")
            mission_table = gr.Dataframe(
                headers=["ID", "שיר", "סטטוס", "התקדמות", "VRAM", "זמן (s)"],
                datatype=["str", "str", "str", "str", "str", "str"],
                interactive=False,
                label="תור משימות"
            )
    
    # --- Update Timers ---
    timer_status = gr.Timer(2)
    timer_status.tick(status_updater, outputs=status_label)
    timer_mission = gr.Timer(1)
    timer_mission.tick(mission_control_updater, outputs=mission_table)
    timer_progress = gr.Timer(1)
    timer_progress.tick(get_progress, outputs=progress_label)

    with gr.Tabs():
        with gr.Tab("⚡ שיר בודד"):
            with gr.Row():
                with gr.Column():
                    s_url = gr.Textbox(label="חיפוש או לינק")
                    s_lang = gr.Dropdown(["he", "en"], value="he", label="שפה")
                    s_4stems = gr.Checkbox(label="שמור גם 4 ערוצים", value=False)
                    s_bidi = gr.Checkbox(label="הפוך עברית", value=False)
                    s_force = gr.Checkbox(label="Force Reprocess", value=False)
                    s_btn = gr.Button("התחל", variant="primary")
                with gr.Column():
                    s_vid = gr.Video(label="תוצאה")
                    s_log = gr.TextArea(label="לוגים", lines=10)
            s_btn.click(ui_process_single, [s_url, s_lang, s_4stems, s_bidi, s_force], [s_vid, s_log])

        with gr.Tab("📚 רשימת שירים"):
            b_text = gr.TextArea(label="רשימה (כל שיר בשורה)")
            with gr.Row():
                b_lang = gr.Dropdown(["he", "en"], value="he", label="שפה")
                b_4stems = gr.Checkbox(label="שמור גם 4 ערוצים", value=False)
                b_bidi = gr.Checkbox(label="הפוך עברית", value=False)
                b_force = gr.Checkbox(label="Force Reprocess", value=False)
            b_btn = gr.Button("התחל רשימה", variant="primary")
            b_log = gr.TextArea(label="לוגים", lines=20)
            b_btn.click(ui_process_batch, [b_text, b_lang, b_4stems, b_bidi, b_force], [b_log])

        with gr.Tab("📝 Dashboard"):
            with gr.Row():
                with gr.Column(scale=1):
                    d_ass = gr.File(label="ASS"); d_load = gr.Button("טען")
                    d_vid = gr.File(label="Video"); d_aud = gr.File(label="Playback")
                    d_size = gr.Slider(20, 150, 80, label="Size"); d_color = gr.ColorPicker("#00FFFF", label="Color")
                    d_bidi = gr.Checkbox(label="הפוך עברית", value=False)
                    d_btn = gr.Button("צור")
                with gr.Column(scale=2):
                    table = gr.Dataframe(headers=["Start", "End", "Text"], datatype=["str", "str", "str"], column_count=3, interactive=True)
                    d_res = gr.Video(); d_log = gr.TextArea()
            d_load.click(logic.backend.ass_to_dataframe, d_ass, table)
            d_btn.click(ui_render_dashboard, [table, d_ass, d_vid, d_aud, d_size, d_color, d_bidi], [d_res, d_log])

        with gr.Tab("🛠️ כלים מתקדמים"):
            with gr.Tabs():
                with gr.Tab("⬇️ הורדה"):
                    dl_u = gr.Textbox(label="URL"); dl_b = gr.Button("הורד")
                    dl_o = gr.File(label="קובץ"); dl_l = gr.TextArea(label="לוגים")
                    dl_b.click(logic.backend.tool_download, dl_u, [dl_o, dl_l])
                
                with gr.Tab("🎵 הפרדה"):
                    sp_u = gr.Textbox(label="URL"); sp_f = gr.File(label="או קובץ")
                    sp_m = gr.Radio(["2 ערוצים", "4 ערוצים"], value="2 ערוצים"); sp_b = gr.Button("הפרד")
                    sp_o = gr.Files(label="קבצים"); sp_l = gr.TextArea(label="לוגים")
                    sp_b.click(logic.backend.tool_separate, [sp_u, sp_f, sp_m], [sp_o, sp_l])

                with gr.Tab("🗣️ תמלול"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            tr_u = gr.Textbox(label="URL")
                            tr_f = gr.File(label="או קובץ")
                            tr_lang = gr.Dropdown(["he", "en"], value="he", label="שפה")
                            tr_b = gr.Button("תמלל", variant="primary")
                            tr_save = gr.Button("שמור עריכות", variant="secondary")
                        with gr.Column(scale=2):
                            tr_table = gr.Dataframe(
                                headers=["Start", "End", "Text"],
                                datatype=["str", "str", "str"],
                                column_count=3,
                                interactive=True,
                                label="טבלת תמלול"
                            )
                            tr_o = gr.File(label="ASS")
                            tr_l = gr.TextArea(label="לוגים")
                    
                    tr_b.click(ui_transcribe_with_table, [tr_u, tr_f, tr_lang], [tr_o, tr_table, tr_l])
                    tr_save.click(ui_save_transcription, [tr_table, tr_o], [tr_o, tr_l])

                with gr.Tab("🔍 ניתוח"):
                    an_f = gr.File(label="קובץ"); an_b = gr.Button("נתח")
                    an_res = gr.Label(label="תוצאה"); an_l = gr.TextArea(label="לוגים")
                    an_b.click(logic.backend.tool_analyze, an_f, [an_res, an_l])

# הרצה
if __name__ == "__main__":
    app.queue(default_concurrency_limit=10).launch(
        inbrowser=True, 
        theme=gr.themes.Soft(),
        allowed_paths=[logic.WORK_DIR]
    )