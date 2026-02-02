
import gradio as gr
import os
import time
from backend import BackendProcessor

# --- הגדרות ---
logs = []
def log_collector(msg):
    timestamp = time.strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)
    logs.append(formatted_msg)

backend = BackendProcessor(log_collector)
work_dir = os.path.abspath(os.path.join(os.getcwd(), "Web_Output"))
os.makedirs(work_dir, exist_ok=True)

def get_logs_text():
    return "\n".join(logs)

def get_audio_source(url, file):
    if url:
        log_collector(f"מתחיל הורדה: {url}")
        return backend.download(url, work_dir)
    elif file:
        log_collector(f"משתמש בקובץ מקומי: {file}")
        return file
    return None

def get_video_source(url, file):
    if url:
        log_collector(f"מוריד וידאו מיוטיוב: {url}")
        return backend.download(url, work_dir, format_type='mp4')
    elif file:
        return file
    return None

def load_ass_content(file):
    if not file: return ""
    try:
        # שימוש ב-utf-8-sig כדי לקרוא נכון
        with open(file.name, 'r', encoding='utf-8-sig') as f: return f.read()
    except:
        try:
            with open(file.name, 'r', encoding='utf-8') as f: return f.read()
        except Exception as e: return str(e)

# --- לוגיקה ---

def run_full_karaoke(url, file, lang, separation_mode, custom_path):
    global logs
    logs = []
    
    audio_path = get_audio_source(url, file)
    if not audio_path: return None, "", "", "", get_logs_text()

    sep_mode_val = "2_stems" if separation_mode == "כן (UVR5)" else "none"
    
    result_files = backend.separate(audio_path, work_dir, sep_mode_val)
    if not result_files: return None, "", "", "", get_logs_text()
    
    if len(result_files) >= 2:
        vocals, playback = result_files[0], result_files[1]
    else:
        vocals = playback = result_files[0] 

    ass_path = backend.transcribe(vocals, os.path.dirname(vocals), lang)
    if not ass_path: return None, "", "", "", get_logs_text()

    video_path = backend.render_custom_karaoke(None, playback, ass_path, os.path.dirname(vocals))
    
    # Load ASS content for editing
    ass_content = ""
    if ass_path and os.path.exists(ass_path):
        with open(ass_path, 'r', encoding='utf-8-sig') as f:
            ass_content = f.read()
    
    if custom_path and video_path:
        backend.copy_to_custom_path(video_path, custom_path)

    return video_path, ass_content, playback, ass_path, get_logs_text()

def run_render_editor(video_url, video_file, audio_file, ass_content, font_size, font_color, custom_path):
    global logs
    logs = []
    
    vid_path = get_video_source(video_url, video_file)
    aud_path = audio_file
    
    if not aud_path: return None, "❌ שגיאה: חובה להעלות קובץ אודיו (פלייבק)."
    if not ass_content: return None, "❌ שגיאה: תוכן הכתוביות ריק."

    edit_dir = os.path.join(work_dir, "Editor_Render")
    os.makedirs(edit_dir, exist_ok=True)
    
    # עדכון סגנון (גודל/צבע)
    styled_content = backend.update_ass_style(ass_content, font_size, font_color)
    
    temp_ass_path = os.path.join(edit_dir, "edited_subs.ass")
    
    # שמירה עם utf-8-sig (התיקון לעברית בעורך)
    with open(temp_ass_path, "w", encoding="utf-8-sig") as f:
        f.write(styled_content)
        
    final_video = backend.render_custom_karaoke(vid_path, aud_path, temp_ass_path, edit_dir)
    
    if custom_path and final_video:
        backend.copy_to_custom_path(final_video, custom_path)

    return final_video, get_logs_text()

def run_downloader(url, fmt, custom_path):
    global logs
    logs = []
    path = backend.download(url, work_dir, fmt)
    if custom_path and path: backend.copy_to_custom_path(path, custom_path)
    return path, get_logs_text()

def run_separator(url, file, mode_selection, custom_path):
    global logs
    logs = []
    audio_path = get_audio_source(url, file)
    if not audio_path: return None, get_logs_text()
    
    backend_mode = "4_stems" if "4 ערוצים" in mode_selection else "2_stems"
    output_files = backend.separate(audio_path, work_dir, backend_mode)
    
    if custom_path and output_files:
        for f in output_files: backend.copy_to_custom_path(f, custom_path)

    return output_files, get_logs_text()

def run_transcriber(url, file, lang, custom_path):
    global logs
    logs = []
    audio_path = get_audio_source(url, file)
    if not audio_path: return None, get_logs_text()
    
    trans_dir = os.path.join(work_dir, "Transcription_Only")
    os.makedirs(trans_dir, exist_ok=True)
    
    ass_path = backend.transcribe(audio_path, trans_dir, lang)
    
    content = ""
    if ass_path and os.path.exists(ass_path):
        with open(ass_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        
        if custom_path:
            backend.copy_to_custom_path(ass_path, custom_path)
            
    return ass_path, content, get_logs_text()

def run_analysis(file):
    global logs
    logs = []
    if not file: return "שגיאה: לא נבחר קובץ", ""
    bpm, key = backend.analyze_audio(file)
    res = f"BPM: {bpm} | סולם: {key}" if bpm else "הניתוח נכשל"
    return res, get_logs_text()

def run_playlist_processor(url, lang, custom_path):
    """Download playlist, transcribe all videos to individual and combined TXT files"""
    global logs
    logs = []
    
    log_collector(f"התחלת הורדה של רשימת השמעה: {url}")
    
    # Download all videos from playlist
    audio_files = backend.download(url, work_dir, format_type='wav', is_playlist=True)
    if not audio_files:
        return None, None, "❌ שגיאה בהורדת רשימת ההשמעה", get_logs_text()
    
    log_collector(f"הורדו {len(audio_files)} קבצים בהצלחה")
    
    # Transcribe batch
    playlist_dir = os.path.join(work_dir, "Playlist_Transcriptions")
    os.makedirs(playlist_dir, exist_ok=True)
    
    individual_files, combined_file = backend.transcribe_batch(audio_files, playlist_dir, lang)
    
    if custom_path and combined_file:
        backend.copy_to_custom_path(combined_file, custom_path)
        for f in individual_files:
            backend.copy_to_custom_path(f, custom_path)
    
    return individual_files, combined_file, get_logs_text()

def render_karaoke_after_edit(ass_content, playback_audio, font_size, font_color, custom_path):
    """Render final karaoke video after editing subtitles"""
    global logs
    
    if not ass_content:
        return None, "❌ שגיאה: תוכן הכתוביות ריק."
    
    if not playback_audio:
        return None, "❌ שגיאה: לא נמצא קובץ פלייבק."
    
    edit_dir = os.path.join(work_dir, "AutoEdit_Render")
    os.makedirs(edit_dir, exist_ok=True)
    
    # Apply style updates
    styled_content = backend.update_ass_style(ass_content, font_size, font_color)
    
    temp_ass_path = os.path.join(edit_dir, "edited_auto_subs.ass")
    with open(temp_ass_path, "w", encoding="utf-8-sig") as f:
        f.write(styled_content)
    
    final_video = backend.render_custom_karaoke(None, playback_audio, temp_ass_path, edit_dir)
    
    if custom_path and final_video:
        backend.copy_to_custom_path(final_video, custom_path)
    
    return final_video, get_logs_text()

# --- ממשק משתמש (Gradio) ---

with gr.Blocks(title="Karaoke Studio Pro") as app:
    gr.Markdown("# 🎤 Karaoke Studio Pro V16")
    
    with gr.Row():
        path_input = gr.Textbox(label="אופציונלי: נתיב שמירה אוטומטי (הדבק כתובת תיקייה)", placeholder="C:\\Users\\Name\\Desktop\\Karaoke_Output", text_align="right")

    with gr.Tabs():
        
        # --- טאב 1: עורך ---
        with gr.TabItem("🎬 עורך ומיקסר (Editor)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. מקורות")
                    e_vid_url = gr.Textbox(label="קישור לוידאו רקע (YouTube)", text_align="right")
                    e_vid_file = gr.Video(label="או קובץ וידאו (MP4)")
                    e_aud_file = gr.Audio(type="filepath", label="קובץ אודיו/פלייבק (חובה)")
                    
                    gr.Markdown("### 2. כתוביות")
                    e_ass_file = gr.File(label="טען קובץ כתוביות (.ASS)", file_types=[".ass"])
                    e_ass_editor = gr.TextArea(label="עורך טקסט (ערוך את המילים כאן)", lines=10, text_align="right")
                    e_ass_file.upload(load_ass_content, e_ass_file, e_ass_editor)
                    
                    gr.Markdown("### 3. עיצוב")
                    with gr.Row():
                        e_size = gr.Slider(20, 150, 80, label="גודל גופן")
                        e_color = gr.ColorPicker("#00FFFF", label="צבע גופן")

                    e_btn = gr.Button("🔥 צור סרטון סופי", variant="primary")

                with gr.Column(scale=1):
                    e_video_out = gr.Video(label="תוצאה סופית")
                    e_log = gr.TextArea(label="לוגים", lines=10, text_align="right")

            e_btn.click(
                fn=run_render_editor,
                inputs=[e_vid_url, e_vid_file, e_aud_file, e_ass_editor, e_size, e_color, path_input],
                outputs=[e_video_out, e_log]
            )

        # --- טאב 2: אוטומטי מלא ---
        with gr.TabItem("⚡ יצירה אוטומטית"):
            with gr.Row():
                with gr.Column(scale=1):
                    k_url = gr.Textbox(label="קישור ליוטיוב", text_align="right")
                    k_file = gr.Audio(label="או קובץ מקומי", type="filepath")
                    with gr.Row():
                        k_lang = gr.Dropdown(["he", "en"], value="he", label="שפה")
                        k_sep = gr.Radio(["כן (UVR5)", "לא (מקור)"], value="כן (UVR5)", label="הפרדת שירה?")
                    k_btn = gr.Button("התחל תהליך", variant="primary")
                    
                    gr.Markdown("### 📝 עריכת כתוביות (אחרי יצירה ראשונית)")
                    k_ass_editor = gr.TextArea(label="עריכה טקסט", lines=6, text_align="right")
                    
                    gr.Markdown("### 🎨 עיצוב")
                    with gr.Row():
                        k_size = gr.Slider(20, 150, 80, label="גודל גופן")
                        k_color = gr.ColorPicker("#00FFFF", label="צבע גופן")
                    k_edit_btn = gr.Button("📝 עדכן סגנון ורנדר מחדש", variant="secondary")
                    
                with gr.Column(scale=1):
                    k_video = gr.Video(label="תוצאה סופית")
                    k_log = gr.TextArea(label="לוגים", text_align="right", lines=10)
            
            # Hidden components to store intermediate data
            k_hidden_playback = gr.Textbox(visible=False)
            k_hidden_ass_path = gr.Textbox(visible=False)
            
            k_btn.click(
                fn=run_full_karaoke,
                inputs=[k_url, k_file, k_lang, k_sep, path_input],
                outputs=[k_video, k_ass_editor, k_hidden_playback, k_hidden_ass_path, k_log]
            )
            
            k_edit_btn.click(
                fn=render_karaoke_after_edit,
                inputs=[k_ass_editor, k_hidden_playback, k_size, k_color, path_input],
                outputs=[k_video, k_log]
            )

        # --- טאב 3: תמלול ---
        with gr.TabItem("📝 תמלול בלבד"):
            with gr.Row():
                with gr.Column():
                    t_url = gr.Textbox(label="קישור ליוטיוב", text_align="right")
                    t_file = gr.Audio(type="filepath", label="קובץ מקומי")
                    t_lang = gr.Dropdown(["he", "en"], value="he", label="שפה")
                    t_btn = gr.Button("תמלל")
                with gr.Column():
                    t_out = gr.File(label="קובץ ASS להורדה")
                    t_preview = gr.TextArea(label="תצוגה מקדימה", text_align="right")
                    t_log = gr.TextArea(label="לוגים", text_align="right")
            t_btn.click(run_transcriber, [t_url, t_file, t_lang, path_input], [t_out, t_preview, t_log])

        # --- טאב 4: הפרדה ---
        with gr.TabItem("🎵 הפרדת ערוצים"):
            with gr.Row():
                with gr.Column():
                    s_url = gr.Textbox(label="קישור ליוטיוב", text_align="right")
                    s_file = gr.Audio(type="filepath", label="קובץ מקומי")
                    s_mode = gr.Radio(["2 ערוצים", "4 ערוצים"], value="2 ערוצים", label="סוג הפרדה")
                    s_btn = gr.Button("הפרד")
                with gr.Column():
                    s_out = gr.Files(label="קבצים שהופרדו")
                    s_log = gr.TextArea(label="לוגים", text_align="right")
            s_btn.click(run_separator, [s_url, s_file, s_mode, path_input], [s_out, s_log])

        # --- טאב 5: הורדה ---
        with gr.TabItem("⬇️ הורדה"):
            d_url = gr.Textbox(label="קישור ליוטיוב", text_align="right")
            d_fmt = gr.Radio(["wav", "mp4"], value="wav", label="פורמט")
            d_btn = gr.Button("הורד")
            d_out = gr.File(label="קובץ")
            d_log = gr.TextArea(label="לוגים", text_align="right")
            d_btn.click(run_downloader, [d_url, d_fmt, path_input], [d_out, d_log])

        # --- טאב 6: ניתוח ---
        with gr.TabItem("🔍 ניתוח מוזיקלי"):
            a_file = gr.Audio(type="filepath")
            a_btn = gr.Button("נתח BPM וסולם")
            a_res = gr.Label()
            a_btn.click(run_analysis, [a_file], [a_res, gr.TextArea()])

        # --- טאב 7: רשימת השמעה ---
        with gr.TabItem("🎼 רשימת השמעה"):
            gr.Markdown("### הורדה וטמלול של רשימת השמעה שלמה מיוטיוב")
            with gr.Row():
                with gr.Column():
                    pl_url = gr.Textbox(label="קישור לרשימת השמעה (YouTube)", text_align="right")
                    pl_lang = gr.Dropdown(["he", "en"], value="he", label="שפה")
                    pl_btn = gr.Button("עבד רשימה", variant="primary")
                    
                with gr.Column():
                    pl_log = gr.TextArea(label="לוגים", text_align="right", lines=10)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### קבצים בודדים")
                    pl_individual = gr.Files(label="קובצי TXT (קובץ לכל וידאו)")
                
                with gr.Column():
                    gr.Markdown("### קובץ משולב")
                    pl_combined = gr.File(label="Combined_Transcriptions.txt")
            
            pl_btn.click(
                fn=run_playlist_processor,
                inputs=[pl_url, pl_lang, path_input],
                outputs=[pl_individual, pl_combined, pl_log]
            )

if __name__ == "__main__":
    app.queue().launch(inbrowser=True, theme=gr.themes.Soft(), allowed_paths=[work_dir, os.getcwd()])