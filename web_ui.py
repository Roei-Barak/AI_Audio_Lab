# # # import gradio as gr
# # # import os
# # # import time
# # # from backend import BackendProcessor

# # # logs = []
# # # def log_collector(msg):
# # #     timestamp = time.strftime("%H:%M:%S")
# # #     formatted_msg = f"[{timestamp}] {msg}"
# # #     print(formatted_msg)
# # #     logs.append(formatted_msg)

# # # backend = BackendProcessor(log_collector)

# # # # קביעת תיקיית העבודה בצורה אבסולוטית כדי למנוע בלבול בנתיבים
# # # work_dir = os.path.abspath(os.path.join(os.getcwd(), "Web_Output"))
# # # os.makedirs(work_dir, exist_ok=True)

# # # def get_logs_text():
# # #     return "\n".join(logs)

# # # def get_audio_source(url, file):
# # #     if url:
# # #         log_collector(f"Starting download: {url}")
# # #         return backend.download(url, work_dir)
# # #     elif file:
# # #         log_collector(f"Using local file: {file}")
# # #         return file
# # #     return None

# # # def get_video_source(url, file):
# # #     """ הורדת וידאו (MP4) מיוטיוב או קובץ מקומי """
# # #     if url:
# # #         log_collector(f"Downloading Video from: {url}")
# # #         return backend.download(url, work_dir, format_type='mp4')
# # #     elif file:
# # #         return file
# # #     return None

# # # # --- Logic Functions ---

# # # def run_full_karaoke(url, file, lang, separation_mode):
# # #     global logs
# # #     logs = []
    
# # #     audio_path = get_audio_source(url, file)
# # #     if not audio_path: return None, get_logs_text()

# # #     sep_mode_val = "2_stems" if separation_mode == "Yes (UVR5)" else "none"
    
# # #     result_files = backend.separate(audio_path, work_dir, sep_mode_val)
# # #     if not result_files: return None, get_logs_text()
    
# # #     if len(result_files) >= 2:
# # #         vocals, playback = result_files[0], result_files[1]
# # #     else:
# # #         vocals = playback = result_files[0] 

# # #     ass_path = backend.transcribe(vocals, os.path.dirname(vocals), lang)
# # #     if not ass_path: return None, get_logs_text()

# # #     video_path = backend.render_custom_karaoke(None, playback, ass_path, os.path.dirname(vocals))
    
# # #     return video_path, get_logs_text()

# # # def run_downloader(url, fmt):
# # #     global logs
# # #     logs = []
# # #     path = backend.download(url, work_dir, fmt)
# # #     return path, get_logs_text()

# # # def run_separator(url, file, mode_selection):
# # #     global logs
# # #     logs = []
# # #     audio_path = get_audio_source(url, file)
# # #     if not audio_path: return None, get_logs_text()
    
# # #     backend_mode = "4_stems" if "4 Stems" in mode_selection else "2_stems"
# # #     output_files = backend.separate(audio_path, work_dir, backend_mode)
# # #     return output_files, get_logs_text()

# # # def run_transcriber(url, file, lang):
# # #     global logs
# # #     logs = []
# # #     audio_path = get_audio_source(url, file)
# # #     if not audio_path: return None, get_logs_text()
    
# # #     trans_dir = os.path.join(work_dir, "Transcription_Only")
# # #     os.makedirs(trans_dir, exist_ok=True)
    
# # #     ass_path = backend.transcribe(audio_path, trans_dir, lang)
    
# # #     content = ""
# # #     if ass_path and os.path.exists(ass_path):
# # #         with open(ass_path, 'r', encoding='utf-8') as f:
# # #             content = f.read()
            
# # #     return ass_path, content, get_logs_text()

# # # def run_analysis(file):
# # #     global logs
# # #     logs = []
# # #     if not file: return "Error: No file", ""
# # #     bpm, key = backend.analyze_audio(file)
# # #     res = f"BPM: {bpm} | Key: {key}" if bpm else "Analysis Failed"
# # #     return res, get_logs_text()

# # # # --- Editor Logic ---

# # # def load_ass_content(file):
# # #     if not file: return ""
# # #     try:
# # #         with open(file.name, 'r', encoding='utf-8') as f:
# # #             return f.read()
# # #     except Exception as e:
# # #         return f"Error reading file: {e}"

# # # def run_render_editor(video_url, video_file, audio_file, ass_content, font_size, font_color):
# # #     global logs
# # #     logs = []
    
# # #     # Inputs check
# # #     vid_path = get_video_source(video_url, video_file)
# # #     aud_path = audio_file
    
# # #     if not aud_path:
# # #         return None, "❌ Error: Audio file is required."
# # #     if not ass_content:
# # #         return None, "❌ Error: Subtitles content is empty."

# # #     edit_dir = os.path.join(work_dir, "Editor_Render")
# # #     os.makedirs(edit_dir, exist_ok=True)
    
# # #     # Save edited subtitles
# # #     # Create temp ASS file with new styles
# # #     # Note: We need to use backend's update_ass_style but it is an instance method.
# # #     # For simplicity, we assume the ass_content is editable text. 
# # #     # If you want to use the backend logic for styling, we can call it here if added to backend, 
# # #     # or just write the file as is if user edited the style manually.
    
# # #     # Let's save the content as is (user edits raw text)
# # #     temp_ass_path = os.path.join(edit_dir, "edited_subs.ass")
# # #     with open(temp_ass_path, "w", encoding="utf-8") as f:
# # #         f.write(ass_content)
        
# # #     log_collector("Subtitles saved.")

# # #     # Render
# # #     final_video = backend.render_custom_karaoke(vid_path, aud_path, temp_ass_path, edit_dir)
    
# # #     return final_video, get_logs_text()


# # # # --- UI Setup ---

# # # with gr.Blocks(title="Karaoke Studio Pro", theme=gr.themes.Soft()) as app:
# # #     gr.Markdown("# Karaoke Studio Pro V14.1 (Fixed)")
    
# # #     with gr.Tabs():
        
# # #         # --- TAB 1: Editor & Mixer ---
# # #         with gr.TabItem("🎬 Karaoke Editor & Mixer"):
# # #             gr.Markdown("### Create custom karaoke videos")
            
# # #             with gr.Row():
# # #                 with gr.Column(scale=1):
# # #                     gr.Markdown("#### 1. Sources")
# # #                     e_vid_url = gr.Textbox(label="Background Video URL (YouTube)")
# # #                     e_vid_file = gr.Video(label="Or Video File (MP4)")
# # #                     e_aud_file = gr.Audio(type="filepath", label="Playback Audio (WAV/MP3) - Required")
                    
# # #                     gr.Markdown("#### 2. Subtitles")
# # #                     e_ass_file = gr.File(label="Load .ASS File", file_types=[".ass"])
# # #                     e_ass_editor = gr.TextArea(label="Subtitle Editor", lines=10)
                    
# # #                     e_ass_file.upload(fn=load_ass_content, inputs=e_ass_file, outputs=e_ass_editor)
                    
# # #                     gr.Markdown("#### 3. Render")
# # #                     # (Font size/color controls are visual only if backend supports parsing them, removed for simplicity to avoid errors)
# # #                     # We pass 0, "" as placeholders if backend expects them, or remove from function if backend updated.
# # #                     # Assuming backend.render_custom_karaoke takes paths directly.
                    
# # #                     e_btn = gr.Button("🔥 Render Custom Video", variant="primary")

# # #                 with gr.Column(scale=1):
# # #                     e_video_out = gr.Video(label="Final Result")
# # #                     e_log = gr.TextArea(label="Logs", lines=10)

# # #             # Note: We pass placeholders for size/color to match the signature if needed, 
# # #             # or simplify run_render_editor to not use them if you prefer manual editing in the text area.
# # #             e_btn.click(
# # #                 fn=run_render_editor,
# # #                 inputs=[e_vid_url, e_vid_file, e_aud_file, e_ass_editor, gr.State(80), gr.State("#FFFFFF")],
# # #                 outputs=[e_video_out, e_log]
# # #             )

# # #         # --- TAB 2: Full Auto ---
# # #         with gr.TabItem("⚡ Full Auto Creator"):
# # #             with gr.Row():
# # #                 with gr.Column():
# # #                     k_url = gr.Textbox(label="YouTube URL")
# # #                     k_file = gr.Audio(label="Or Upload File", type="filepath")
# # #                     with gr.Row():
# # #                         k_lang = gr.Dropdown(["he", "en"], value="he", label="Language")
# # #                         k_sep = gr.Radio(["Yes (UVR5)", "No (Original)"], value="Yes (UVR5)", label="Separate Vocals?")
# # #                     k_btn = gr.Button("Create Video", variant="primary")
# # #                 with gr.Column():
# # #                     k_video = gr.Video(label="Result")
# # #                     k_log = gr.TextArea(label="Logs", lines=8)
# # #             k_btn.click(run_full_karaoke, [k_url, k_file, k_lang, k_sep], [k_video, k_log])

# # #         # --- TAB 3: Transcriber ---
# # #         with gr.TabItem("📝 Transcriber"):
# # #             with gr.Row():
# # #                 with gr.Column():
# # #                     t_url = gr.Textbox(label="YouTube URL")
# # #                     t_file = gr.Audio(type="filepath", label="Local File")
# # #                     t_lang = gr.Dropdown(["he", "en"], value="he", label="Language")
# # #                     t_btn = gr.Button("Transcribe")
# # #                 with gr.Column():
# # #                     t_out = gr.File(label="ASS File")
# # #                     t_preview = gr.TextArea(label="Preview", lines=5) 
# # #                     t_log = gr.TextArea(label="Logs")
            
# # #             t_btn.click(run_transcriber, [t_url, t_file, t_lang], [t_out, t_preview, t_log])

# # #         # --- TAB 4: Separator ---
# # #         with gr.TabItem("🎵 Separator"):
# # #             with gr.Row():
# # #                 with gr.Column():
# # #                     s_url = gr.Textbox(label="YouTube URL")
# # #                     s_file = gr.Audio(type="filepath", label="Local File")
# # #                     s_mode = gr.Radio(["2 Stems", "4 Stems"], value="2 Stems", label="Mode")
# # #                     s_btn = gr.Button("Separate")
# # #                 with gr.Column():
# # #                     s_out = gr.Files(label="Files")
# # #                     s_log = gr.TextArea(label="Logs")
# # #             s_btn.click(run_separator, [s_url, s_file, s_mode], [s_out, s_log])

# # #         # --- TAB 5: Downloader ---
# # #         with gr.TabItem("⬇️ Downloader"):
# # #             d_url = gr.Textbox(label="YouTube URL")
# # #             d_fmt = gr.Radio(["wav", "mp4"], value="wav", label="Format")
# # #             d_btn = gr.Button("Download")
# # #             d_out = gr.File(label="File")
# # #             d_log = gr.TextArea(label="Logs")
# # #             d_btn.click(run_downloader, [d_url, d_fmt], [d_out, d_log])

# # #         # --- TAB 6: Analysis ---
# # #         with gr.TabItem("🔍 Analysis"):
# # #             a_file = gr.Audio(type="filepath")
# # #             a_btn = gr.Button("Analyze")
# # #             a_res = gr.Label()
# # #             a_btn.click(run_analysis, [a_file], [a_res, gr.TextArea()])

# # # if __name__ == "__main__":
# # #     # === התיקון הקריטי כאן ===
# # #     # allowed_paths: מאפשר ל-Gradio לגשת לקבצים בתיקיית הפלט ובתיקיית העבודה הנוכחית
# # #     app.queue().launch(
# # #         inbrowser=True, 
# # #         allowed_paths=[work_dir, os.getcwd()]
# # #     )
# # import gradio as gr
# # import os
# # import time
# # from backend import BackendProcessor

# # logs = []
# # def log_collector(msg):
# #     timestamp = time.strftime("%H:%M:%S")
# #     formatted_msg = f"[{timestamp}] {msg}"
# #     print(formatted_msg)
# #     logs.append(formatted_msg)

# # backend = BackendProcessor(log_collector)
# # work_dir = os.path.abspath(os.path.join(os.getcwd(), "Web_Output"))
# # os.makedirs(work_dir, exist_ok=True)

# # def get_logs_text():
# #     return "\n".join(logs)

# # def get_audio_source(url, file):
# #     if url:
# #         log_collector(f"Starting download: {url}")
# #         return backend.download(url, work_dir)
# #     elif file:
# #         log_collector(f"Using local file: {file}")
# #         return file
# #     return None

# # def get_video_source(url, file):
# #     if url:
# #         log_collector(f"Downloading Video from: {url}")
# #         return backend.download(url, work_dir, format_type='mp4')
# #     elif file:
# #         return file
# #     return None

# # # --- Wrapper Functions ---

# # def run_full_karaoke(url, file, lang, separation_mode, custom_path):
# #     global logs
# #     logs = []
    
# #     audio_path = get_audio_source(url, file)
# #     if not audio_path: return None, get_logs_text()

# #     sep_mode_val = "2_stems" if separation_mode == "Yes (UVR5)" else "none"
    
# #     result_files = backend.separate(audio_path, work_dir, sep_mode_val)
# #     if not result_files: return None, get_logs_text()
    
# #     if len(result_files) >= 2:
# #         vocals, playback = result_files[0], result_files[1]
# #     else:
# #         vocals = playback = result_files[0] 

# #     ass_path = backend.transcribe(vocals, os.path.dirname(vocals), lang)
# #     if not ass_path: return None, get_logs_text()

# #     video_path = backend.render_custom_karaoke(None, playback, ass_path, os.path.dirname(vocals))
    
# #     # Save to custom path if provided
# #     if custom_path and video_path:
# #         backend.copy_to_custom_path(video_path, custom_path)

# #     return video_path, get_logs_text()

# # def run_render_editor(video_url, video_file, audio_file, ass_content, font_size, font_color, custom_path):
# #     global logs
# #     logs = []
    
# #     vid_path = get_video_source(video_url, video_file)
# #     aud_path = audio_file
    
# #     if not aud_path: return None, "❌ Error: Audio required."
# #     if not ass_content: return None, "❌ Error: Content empty."

# #     edit_dir = os.path.join(work_dir, "Editor_Render")
# #     os.makedirs(edit_dir, exist_ok=True)
    
# #     styled_content = backend.update_ass_style(ass_content, font_size, font_color)
# #     temp_ass_path = os.path.join(edit_dir, "edited_subs.ass")
# #     with open(temp_ass_path, "w", encoding="utf-8") as f:
# #         f.write(styled_content)
        
# #     final_video = backend.render_custom_karaoke(vid_path, aud_path, temp_ass_path, edit_dir)
    
# #     # Save to custom path
# #     if custom_path and final_video:
# #         backend.copy_to_custom_path(final_video, custom_path)

# #     return final_video, get_logs_text()

# # def run_downloader(url, fmt, custom_path):
# #     global logs
# #     logs = []
# #     path = backend.download(url, work_dir, fmt)
    
# #     if custom_path and path:
# #         backend.copy_to_custom_path(path, custom_path)
        
# #     return path, get_logs_text()

# # def run_separator(url, file, mode_selection, custom_path):
# #     global logs
# #     logs = []
# #     audio_path = get_audio_source(url, file)
# #     if not audio_path: return None, get_logs_text()
    
# #     backend_mode = "4_stems" if "4 Stems" in mode_selection else "2_stems"
# #     output_files = backend.separate(audio_path, work_dir, backend_mode)
    
# #     if custom_path and output_files:
# #         for f in output_files:
# #             backend.copy_to_custom_path(f, custom_path)

# #     return output_files, get_logs_text()

# # def run_transcriber(url, file, lang, custom_path):
# #     global logs
# #     logs = []
# #     audio_path = get_audio_source(url, file)
# #     if not audio_path: return None, get_logs_text()
    
# #     trans_dir = os.path.join(work_dir, "Transcription_Only")
# #     os.makedirs(trans_dir, exist_ok=True)
    
# #     ass_path = backend.transcribe(audio_path, trans_dir, lang)
    
# #     content = ""
# #     if ass_path and os.path.exists(ass_path):
# #         with open(ass_path, 'r', encoding='utf-8') as f:
# #             content = f.read()
        
# #         if custom_path:
# #             backend.copy_to_custom_path(ass_path, custom_path)
            
# #     return ass_path, content, get_logs_text()

# # def run_analysis(file):
# #     global logs
# #     logs = []
# #     if not file: return "Error: No file", ""
# #     bpm, key = backend.analyze_audio(file)
# #     res = f"BPM: {bpm} | Key: {key}" if bpm else "Analysis Failed"
# #     return res, get_logs_text()

# # def load_ass_content(file):
# #     if not file: return ""
# #     try:
# #         with open(file.name, 'r', encoding='utf-8') as f: return f.read()
# #     except Exception as e: return str(e)

# # # --- UI Setup ---

# # with gr.Blocks(title="Karaoke Studio Pro", theme=gr.themes.Soft()) as app:
# #     gr.Markdown("# Karaoke Studio Pro V15")
    
# #     # Global Settings
# #     with gr.Row():
# #         path_input = gr.Textbox(label="Optional: Auto-Save Path (Paste folder path here)", placeholder="e.g. C:\\Users\\Name\\Music\\Karaoke")

# #     with gr.Tabs():
        
# #         # --- TAB 1: Editor ---
# #         with gr.TabItem("🎬 Editor & Mixer"):
# #             with gr.Row():
# #                 with gr.Column(scale=1):
# #                     gr.Markdown("#### Sources")
# #                     e_vid_url = gr.Textbox(label="Video URL")
# #                     e_vid_file = gr.Video(label="Or Video File")
# #                     e_aud_file = gr.Audio(type="filepath", label="Playback Audio (Required)")
                    
# #                     gr.Markdown("#### Subtitles")
# #                     e_ass_file = gr.File(label="Load .ASS", file_types=[".ass"])
# #                     e_ass_editor = gr.TextArea(label="Editor", lines=10)
# #                     e_ass_file.upload(load_ass_content, e_ass_file, e_ass_editor)
                    
# #                     gr.Markdown("#### Style")
# #                     with gr.Row():
# #                         e_size = gr.Slider(20, 150, 80, label="Size")
# #                         e_color = gr.ColorPicker("#00FFFF", label="Color")

# #                     e_btn = gr.Button("Render Video", variant="primary")

# #                 with gr.Column(scale=1):
# #                     e_video_out = gr.Video(label="Result")
# #                     e_log = gr.TextArea(label="Logs", lines=10)

# #             e_btn.click(
# #                 fn=run_render_editor,
# #                 inputs=[e_vid_url, e_vid_file, e_aud_file, e_ass_editor, e_size, e_color, path_input],
# #                 outputs=[e_video_out, e_log]
# #             )

# #         # --- TAB 2: Full Auto ---
# #         with gr.TabItem("⚡ Full Auto"):
# #             with gr.Row():
# #                 with gr.Column():
# #                     k_url = gr.Textbox(label="URL")
# #                     k_file = gr.Audio(label="File", type="filepath")
# #                     with gr.Row():
# #                         k_lang = gr.Dropdown(["he", "en"], value="he", label="Lang")
# #                         k_sep = gr.Radio(["Yes (UVR5)", "No"], value="Yes (UVR5)", label="Separate?")
# #                     k_btn = gr.Button("Create", variant="primary")
# #                 with gr.Column():
# #                     k_video = gr.Video(label="Result")
# #                     k_log = gr.TextArea(label="Logs")
# #             k_btn.click(run_full_karaoke, [k_url, k_file, k_lang, k_sep, path_input], [k_video, k_log])

# #         # --- TAB 3: Transcriber ---
# #         with gr.TabItem("📝 Transcriber"):
# #             with gr.Row():
# #                 with gr.Column():
# #                     t_url = gr.Textbox(label="URL")
# #                     t_file = gr.Audio(type="filepath", label="File")
# #                     t_lang = gr.Dropdown(["he", "en"], value="he", label="Lang")
# #                     t_btn = gr.Button("Transcribe")
# #                 with gr.Column():
# #                     t_out = gr.File(label="ASS File")
# #                     t_preview = gr.TextArea(label="Preview")
# #                     t_log = gr.TextArea(label="Logs")
# #             t_btn.click(run_transcriber, [t_url, t_file, t_lang, path_input], [t_out, t_preview, t_log])

# #         # --- TAB 4: Separator ---
# #         with gr.TabItem("🎵 Separator"):
# #             with gr.Row():
# #                 with gr.Column():
# #                     s_url = gr.Textbox(label="URL")
# #                     s_file = gr.Audio(type="filepath", label="File")
# #                     s_mode = gr.Radio(["2 Stems", "4 Stems"], value="2 Stems", label="Mode")
# #                     s_btn = gr.Button("Separate")
# #                 with gr.Column():
# #                     s_out = gr.Files(label="Files")
# #                     s_log = gr.TextArea(label="Logs")
# #             s_btn.click(run_separator, [s_url, s_file, s_mode, path_input], [s_out, s_log])

# #         # --- TAB 5: Downloader ---
# #         with gr.TabItem("⬇️ Downloader"):
# #             d_url = gr.Textbox(label="URL")
# #             d_fmt = gr.Radio(["wav", "mp4"], value="wav", label="Format")
# #             d_btn = gr.Button("Download")
# #             d_out = gr.File(label="File")
# #             d_log = gr.TextArea(label="Logs")
# #             d_btn.click(run_downloader, [d_url, d_fmt, path_input], [d_out, d_log])

# #         # --- TAB 6: Analysis ---
# #         with gr.TabItem("🔍 Analysis"):
# #             a_file = gr.Audio(type="filepath")
# #             a_btn = gr.Button("Analyze")
# #             a_res = gr.Label()
# #             a_btn.click(run_analysis, [a_file], [a_res, gr.TextArea()])

# # if __name__ == "__main__":
# #     app.queue().launch(inbrowser=True, allowed_paths=[work_dir, os.getcwd()])
# import gradio as gr
# import os
# import time
# from backend import BackendProcessor

# # --- Setup ---
# logs = []
# def log_collector(msg):
#     timestamp = time.strftime("%H:%M:%S")
#     formatted_msg = f"[{timestamp}] {msg}"
#     print(formatted_msg)
#     logs.append(formatted_msg)

# backend = BackendProcessor(log_collector)
# work_dir = os.path.abspath(os.path.join(os.getcwd(), "Web_Output"))
# os.makedirs(work_dir, exist_ok=True)

# def get_logs_text():
#     return "\n".join(logs)

# def get_audio_source(url, file):
#     if url:
#         log_collector(f"Starting download: {url}")
#         return backend.download(url, work_dir)
#     elif file:
#         log_collector(f"Using local file: {file}")
#         return file
#     return None

# def get_video_source(url, file):
#     if url:
#         log_collector(f"Downloading Video from: {url}")
#         return backend.download(url, work_dir, format_type='mp4')
#     elif file:
#         return file
#     return None

# def load_ass_content(file):
#     if not file: return ""
#     try:
#         # שימוש ב-utf-8-sig כדי לקרוא נכון גם קבצים שנשמרו עם BOM
#         with open(file.name, 'r', encoding='utf-8-sig') as f: return f.read()
#     except:
#         # ניסיון שני רגיל אם הראשון נכשל
#         try:
#             with open(file.name, 'r', encoding='utf-8') as f: return f.read()
#         except Exception as e: return str(e)

# # --- Logic Functions ---

# def run_full_karaoke(url, file, lang, separation_mode, custom_path):
#     global logs
#     logs = []
    
#     audio_path = get_audio_source(url, file)
#     if not audio_path: return None, get_logs_text()

#     sep_mode_val = "2_stems" if separation_mode == "Yes (UVR5)" else "none"
    
#     result_files = backend.separate(audio_path, work_dir, sep_mode_val)
#     if not result_files: return None, get_logs_text()
    
#     if len(result_files) >= 2:
#         vocals, playback = result_files[0], result_files[1]
#     else:
#         vocals = playback = result_files[0] 

#     ass_path = backend.transcribe(vocals, os.path.dirname(vocals), lang)
#     if not ass_path: return None, get_logs_text()

#     video_path = backend.render_custom_karaoke(None, playback, ass_path, os.path.dirname(vocals))
    
#     if custom_path and video_path:
#         backend.copy_to_custom_path(video_path, custom_path)

#     return video_path, get_logs_text()

# def run_render_editor(video_url, video_file, audio_file, ass_content, font_size, font_color, custom_path):
#     global logs
#     logs = []
    
#     vid_path = get_video_source(video_url, video_file)
#     aud_path = audio_file
    
#     if not aud_path: return None, "❌ Error: Audio required."
#     if not ass_content: return None, "❌ Error: Content empty."

#     edit_dir = os.path.join(work_dir, "Editor_Render")
#     os.makedirs(edit_dir, exist_ok=True)
    
#     # עדכון סגנון (גודל/צבע)
#     styled_content = backend.update_ass_style(ass_content, font_size, font_color)
    
#     temp_ass_path = os.path.join(edit_dir, "edited_subs.ass")
    
#     # === התיקון הקריטי כאן ===
#     # שימוש ב-utf-8-sig מבטיח ש-FFmpeg יזהה את העברית ולא יהפוך אותה
#     with open(temp_ass_path, "w", encoding="utf-8-sig") as f:
#         f.write(styled_content)
        
#     final_video = backend.render_custom_karaoke(vid_path, aud_path, temp_ass_path, edit_dir)
    
#     if custom_path and final_video:
#         backend.copy_to_custom_path(final_video, custom_path)

#     return final_video, get_logs_text()

# def run_downloader(url, fmt, custom_path):
#     global logs
#     logs = []
#     path = backend.download(url, work_dir, fmt)
#     if custom_path and path: backend.copy_to_custom_path(path, custom_path)
#     return path, get_logs_text()

# def run_separator(url, file, mode_selection, custom_path):
#     global logs
#     logs = []
#     audio_path = get_audio_source(url, file)
#     if not audio_path: return None, get_logs_text()
    
#     backend_mode = "4_stems" if "4 Stems" in mode_selection else "2_stems"
#     output_files = backend.separate(audio_path, work_dir, backend_mode)
    
#     if custom_path and output_files:
#         for f in output_files: backend.copy_to_custom_path(f, custom_path)

#     return output_files, get_logs_text()

# def run_transcriber(url, file, lang, custom_path):
#     global logs
#     logs = []
#     audio_path = get_audio_source(url, file)
#     if not audio_path: return None, get_logs_text()
    
#     trans_dir = os.path.join(work_dir, "Transcription_Only")
#     os.makedirs(trans_dir, exist_ok=True)
    
#     ass_path = backend.transcribe(audio_path, trans_dir, lang)
    
#     content = ""
#     if ass_path and os.path.exists(ass_path):
#         # קריאה עם utf-8-sig כדי להציג נכון בתיבת הטקסט
#         with open(ass_path, 'r', encoding='utf-8-sig') as f:
#             content = f.read()
        
#         if custom_path:
#             backend.copy_to_custom_path(ass_path, custom_path)
            
#     return ass_path, content, get_logs_text()

# def run_analysis(file):
#     global logs
#     logs = []
#     if not file: return "Error: No file", ""
#     bpm, key = backend.analyze_audio(file)
#     res = f"BPM: {bpm} | Key: {key}" if bpm else "Analysis Failed"
#     return res, get_logs_text()

# # --- UI Setup ---

# with gr.Blocks(title="Karaoke Studio Pro", theme=gr.themes.Soft()) as app:
#     gr.Markdown("# Karaoke Studio Pro V16 (Hebrew Fix)")
    
#     with gr.Row():
#         path_input = gr.Textbox(label="Optional: Auto-Save Path", placeholder="C:\\Users\\Name\\Desktop\\Karaoke_Output")

#     with gr.Tabs():
        
#         # --- TAB 1: Editor ---
#         with gr.TabItem("🎬 Editor & Mixer"):
#             with gr.Row():
#                 with gr.Column(scale=1):
#                     gr.Markdown("#### Sources")
#                     e_vid_url = gr.Textbox(label="Video URL")
#                     e_vid_file = gr.Video(label="Or Video File")
#                     e_aud_file = gr.Audio(type="filepath", label="Playback Audio (Required)")
                    
#                     gr.Markdown("#### Subtitles")
#                     e_ass_file = gr.File(label="Load .ASS", file_types=[".ass"])
#                     e_ass_editor = gr.TextArea(label="Editor (Edit text here)", lines=10, text_align="right")
#                     e_ass_file.upload(load_ass_content, e_ass_file, e_ass_editor)
                    
#                     gr.Markdown("#### Style")
#                     with gr.Row():
#                         e_size = gr.Slider(20, 150, 80, label="Size")
#                         e_color = gr.ColorPicker("#00FFFF", label="Color")

#                     e_btn = gr.Button("Render Video", variant="primary")

#                 with gr.Column(scale=1):
#                     e_video_out = gr.Video(label="Result")
#                     e_log = gr.TextArea(label="Logs", lines=10)

#             e_btn.click(
#                 fn=run_render_editor,
#                 inputs=[e_vid_url, e_vid_file, e_aud_file, e_ass_editor, e_size, e_color, path_input],
#                 outputs=[e_video_out, e_log]
#             )

#         # --- TAB 2: Full Auto ---
#         with gr.TabItem("⚡ Full Auto"):
#             with gr.Row():
#                 with gr.Column():
#                     k_url = gr.Textbox(label="URL")
#                     k_file = gr.Audio(label="File", type="filepath")
#                     with gr.Row():
#                         k_lang = gr.Dropdown(["he", "en"], value="he", label="Lang")
#                         k_sep = gr.Radio(["Yes (UVR5)", "No"], value="Yes (UVR5)", label="Separate?")
#                     k_btn = gr.Button("Create", variant="primary")
#                 with gr.Column():
#                     k_video = gr.Video(label="Result")
#                     k_log = gr.TextArea(label="Logs")
#             k_btn.click(run_full_karaoke, [k_url, k_file, k_lang, k_sep, path_input], [k_video, k_log])

#         # --- TAB 3: Transcriber ---
#         with gr.TabItem("📝 Transcriber"):
#             with gr.Row():
#                 with gr.Column():
#                     t_url = gr.Textbox(label="URL")
#                     t_file = gr.Audio(type="filepath", label="File")
#                     t_lang = gr.Dropdown(["he", "en"], value="he", label="Lang")
#                     t_btn = gr.Button("Transcribe")
#                 with gr.Column():
#                     t_out = gr.File(label="ASS File")
#                     t_preview = gr.TextArea(label="Preview", text_align="right")
#                     t_log = gr.TextArea(label="Logs")
#             t_btn.click(run_transcriber, [t_url, t_file, t_lang, path_input], [t_out, t_preview, t_log])

#         # --- TAB 4: Separator ---
#         with gr.TabItem("🎵 Separator"):
#             with gr.Row():
#                 with gr.Column():
#                     s_url = gr.Textbox(label="URL")
#                     s_file = gr.Audio(type="filepath", label="File")
#                     s_mode = gr.Radio(["2 Stems", "4 Stems"], value="2 Stems", label="Mode")
#                     s_btn = gr.Button("Separate")
#                 with gr.Column():
#                     s_out = gr.Files(label="Files")
#                     s_log = gr.TextArea(label="Logs")
#             s_btn.click(run_separator, [s_url, s_file, s_mode, path_input], [s_out, s_log])

#         # --- TAB 5: Downloader ---
#         with gr.TabItem("⬇️ Downloader"):
#             d_url = gr.Textbox(label="URL")
#             d_fmt = gr.Radio(["wav", "mp4"], value="wav", label="Format")
#             d_btn = gr.Button("Download")
#             d_out = gr.File(label="File")
#             d_log = gr.TextArea(label="Logs")
#             d_btn.click(run_downloader, [d_url, d_fmt, path_input], [d_out, d_log])

#         # --- TAB 6: Analysis ---
#         with gr.TabItem("🔍 Analysis"):
#             a_file = gr.Audio(type="filepath")
#             a_btn = gr.Button("Analyze")
#             a_res = gr.Label()
#             a_btn.click(run_analysis, [a_file], [a_res, gr.TextArea()])

# if __name__ == "__main__":
#     app.queue().launch(inbrowser=True, allowed_paths=[work_dir, os.getcwd()])
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
    if not audio_path: return None, get_logs_text()

    sep_mode_val = "2_stems" if separation_mode == "כן (UVR5)" else "none"
    
    result_files = backend.separate(audio_path, work_dir, sep_mode_val)
    if not result_files: return None, get_logs_text()
    
    if len(result_files) >= 2:
        vocals, playback = result_files[0], result_files[1]
    else:
        vocals = playback = result_files[0] 

    ass_path = backend.transcribe(vocals, os.path.dirname(vocals), lang)
    if not ass_path: return None, get_logs_text()

    video_path = backend.render_custom_karaoke(None, playback, ass_path, os.path.dirname(vocals))
    
    if custom_path and video_path:
        backend.copy_to_custom_path(video_path, custom_path)

    return video_path, get_logs_text()

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

# --- ממשק משתמש (Gradio) ---

with gr.Blocks(title="Karaoke Studio Pro", theme=gr.themes.Soft()) as app:
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
                with gr.Column():
                    k_url = gr.Textbox(label="קישור ליוטיוב", text_align="right")
                    k_file = gr.Audio(label="או קובץ מקומי", type="filepath")
                    with gr.Row():
                        k_lang = gr.Dropdown(["he", "en"], value="he", label="שפה")
                        k_sep = gr.Radio(["כן (UVR5)", "לא (מקור)"], value="כן (UVR5)", label="הפרדת שירה?")
                    k_btn = gr.Button("התחל תהליך", variant="primary")
                with gr.Column():
                    k_video = gr.Video(label="תוצאה")
                    k_log = gr.TextArea(label="לוגים", text_align="right")
            k_btn.click(run_full_karaoke, [k_url, k_file, k_lang, k_sep, path_input], [k_video, k_log])

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

if __name__ == "__main__":
    app.queue().launch(inbrowser=True, allowed_paths=[work_dir, os.getcwd()])