import os
import re
import shutil
import subprocess
import logging
import uuid
import time
import pandas as pd
from pathlib import Path
import imageio_ffmpeg
import gradio as gr

# ==========================================
# חלק 1: הגדרות וייבוא
# ==========================================

# הגדרת תיקיית עבודה
WORK_DIR = os.path.abspath("Karaoke_Output")
os.makedirs(WORK_DIR, exist_ok=True)

try:
    import yt_dlp
    import torch
    import soundfile as sf
    from transformers import pipeline
    from audio_separator.separator import Separator
    import librosa
    
    # עברית
    import arabic_reshaper
    from bidi.algorithm import get_display
    
    # אין צורך ב-MoviePy יותר!
except ImportError as e:
    print(f"❌ שגיאה: חסרות ספריות. וודא שהתקנת את: pandas arabic-reshaper python-bidi yt-dlp audio-separator transformers torch soundfile librosa imageio-ffmpeg")

# השתקת לוגים
logging.getLogger("audio_separator").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ==========================================
# חלק 2: מנוע העיבוד (Pure FFmpeg Backend)
# ==========================================

class BackendProcessor:
    def __init__(self, log_func):
        self.log = log_func
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # --- כלי עזר ---
    def _fix_hebrew_text(self, text):
        """הופך עברית לויזואלית (עבור הצריבה בלבד)"""
        try:
            if not text: return ""
            reshaped = arabic_reshaper.reshape(str(text))
            return get_display(reshaped)
        except:
            return text

    def _fmt_ass_time(self, seconds):
        """ממיר שניות לפורמט ASS"""
        try:
            seconds = float(seconds)
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            cs = int((seconds - int(seconds)) * 100)
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
        except:
            return "0:00:00.00"

    def convert_to_wav(self, input_path, output_path):
        """ממיר ל-WAV 16kHz"""
        try:
            cmd = [self.ffmpeg_exe, '-y', '-i', input_path, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', output_path]
            startupinfo = subprocess.STARTUPINFO() if os.name == 'nt' else None
            if startupinfo: startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, startupinfo=startupinfo)
            return True
        except Exception as e:
            self.log(f"Error wav conv: {e}")
            return False

    def copy_to_custom_path(self, src, dest_folder):
        if not src or not os.path.exists(src): return None
        try:
            os.makedirs(dest_folder, exist_ok=True)
            filename = os.path.basename(src)
            dest = os.path.join(dest_folder, filename)
            shutil.copy2(src, dest)
            return dest
        except: return None

    # --- 1. הורדה ---
    def download(self, url, output_dir, format_type='wav'):
        self.log(f"📥 מוריד: {url}")
        os.makedirs(output_dir, exist_ok=True)
        opts = {
            'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
            'quiet': True, 'no_warnings': True, 
            'ffmpeg_location': self.ffmpeg_exe
        }
        if format_type == 'mp4':
            opts.update({'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', 'merge_output_format': 'mp4'})
        else:
            opts.update({'format': 'bestaudio/best', 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}]})
        
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                fname = ydl.prepare_filename(info)
                final = Path(fname).with_suffix('.wav' if format_type == 'wav' else '.mp4')
                if not final.exists(): 
                    found = list(Path(output_dir).glob(f"{Path(fname).stem}*"))
                    if found: final = found[0]
                return str(final)
        except Exception as e:
            self.log(f"❌ שגיאה בהורדה: {e}")
            return None

    # --- 2. הפרדה ---
    def separate(self, audio_path, output_dir, mode="2_stems"):
        self.log(f"🎚️ מפריד ערוצים ({mode})...")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        sep_dir = os.path.join(output_dir, f"Sep_{uuid.uuid4().hex[:4]}")
        os.makedirs(sep_dir, exist_ok=True)
        
        try:
            sep = Separator(log_level=logging.ERROR, output_dir=sep_dir, model_file_dir=os.path.join(output_dir, "uvr_models"))
            sep.load_model("Kim_Vocal_2.onnx" if mode == "2_stems" else "htdemucs_ft.yaml")
            files = sep.separate(audio_path)
            
            res = []
            for f in files:
                full_p = os.path.join(sep_dir, f)
                new_name = "Vocals.wav" if "vocal" in f.lower() else "Playback.wav"
                new_p = os.path.join(sep_dir, new_name)
                if os.path.exists(new_p): os.remove(new_p)
                os.rename(full_p, new_p)
                res.append(new_p)
            
            if len(res) < 2: return [audio_path, audio_path]
            return res 
        except Exception as e:
            self.log(f"❌ שגיאה בהפרדה: {e}")
            return [audio_path, audio_path]

    # --- 3. תמלול ---
    def transcribe(self, audio_path, output_dir, lang="he"):
        self.log(f"📝 מתמלל (Whisper)...")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        try:
            model_id = "ivrit-ai/whisper-large-v3-turbo" if lang == "he" else "openai/whisper-large-v3"
            pipe = pipeline("automatic-speech-recognition", model=model_id, device="cuda" if torch.cuda.is_available() else "cpu", chunk_length_s=30, return_timestamps="word")
            
            clean_wav = os.path.join(output_dir, "clean_transcribe.wav")
            self.convert_to_wav(audio_path, clean_wav)
            
            result = pipe(clean_wav, generate_kwargs={"language": "hebrew" if lang == "he" else "english"})
            
            ass_path = os.path.join(output_dir, "karaoke.ass")
            self._create_ass_file(result['chunks'], ass_path, fix_hebrew=False)
            
            if os.path.exists(clean_wav): os.remove(clean_wav)
            return ass_path
        except Exception as e:
            self.log(f"❌ שגיאה בתמלול: {e}")
            return None

    def _create_ass_file(self, chunks, output_path, fix_hebrew=False):
        header = """[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Karaoke,Arial,80,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,100,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"""
        
        events = []
        curr_line = []
        
        for chunk in chunks:
            text = chunk['text'].strip()
            if fix_hebrew: text = self._fix_hebrew_text(text)
            
            start = chunk['timestamp'][0]
            end = chunk['timestamp'][1]
            curr_line.append({'text': text, 'start': start, 'end': end})
            
            if text.endswith(('.', '?', '!')) or len(curr_line) > 6:
                line_start = self._fmt_ass_time(curr_line[0]['start'])
                line_end = self._fmt_ass_time(curr_line[-1]['end'])
                full_text = " ".join([w['text'] for w in curr_line])
                
                if fix_hebrew: full_text = self._fix_hebrew_text(" ".join([w['text'] for w in curr_line]))
                events.append(f"Dialogue: 0,{line_start},{line_end},Karaoke,,0,0,0,,{full_text}")
                curr_line = []

        if curr_line:
             line_start = self._fmt_ass_time(curr_line[0]['start'])
             line_end = self._fmt_ass_time(curr_line[-1]['end'])
             full_text = " ".join([w['text'] for w in curr_line])
             if fix_hebrew: full_text = self._fix_hebrew_text(full_text)
             events.append(f"Dialogue: 0,{line_start},{line_end},Karaoke,,0,0,0,,{full_text}")

        with open(output_path, "w", encoding="utf-8-sig") as f:
            f.write(header + "\n".join(events))

    # --- 4. ניתוח מוזיקלי ---
    def analyze_audio(self, audio_path):
        self.log(f"🔍 מנתח: {os.path.basename(audio_path)}...")
        try:
            temp_wav = os.path.join(os.path.dirname(audio_path), f"temp_analysis_{uuid.uuid4().hex[:4]}.wav")
            self.convert_to_wav(audio_path, temp_wav)
            
            y, sr = librosa.load(temp_wav, sr=None, duration=60)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = round(float(tempo[0])) if len(tempo) > 0 else 0
            
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            key_idx = chroma.mean(axis=1).argmax()
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key = keys[key_idx]
            
            if os.path.exists(temp_wav): os.remove(temp_wav)
            return bpm, key
        except Exception as e:
            self.log(f"Analysis error: {e}")
            return None, None

    # --- 5. Dataframe ---
    def ass_to_dataframe(self, ass_path):
        if not ass_path or not os.path.exists(ass_path): return pd.DataFrame()
        data = []
        with open(ass_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                if line.startswith("Dialogue:"):
                    parts = line.split(",", 9)
                    if len(parts) == 10:
                        data.append({"Start": parts[1], "End": parts[2], "Text": parts[9].strip()})
        return pd.DataFrame(data)

    def dataframe_to_ass(self, df, original_ass_path, output_path):
        header_lines = []
        if original_ass_path and os.path.exists(original_ass_path):
            with open(original_ass_path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    if line.startswith("Dialogue:"): break
                    header_lines.append(line)
        else:
            header_lines = ["""[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Karaoke,Arial,80,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,100,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"""]

        with open(output_path, 'w', encoding='utf-8-sig') as f:
            f.writelines(header_lines)
            for _, row in df.iterrows():
                f.write(f"Dialogue: 0,{row['Start']},{row['End']},Karaoke,,0,0,0,,{row['Text']}\n")
        
        return output_path

    def update_ass_style(self, ass_path, font_size, color_hex):
        r, g, b = color_hex[1:3], color_hex[3:5], color_hex[5:7]
        ass_color = f"&H00{b}{g}{r}".upper()
        with open(ass_path, 'r', encoding='utf-8-sig') as f: content = f.read()
        new_style = f"Style: Karaoke,Arial,{int(font_size)},{ass_color},&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,100,1"
        new_content = re.sub(r"^Style:.*Karaoke.*$", new_style, content, flags=re.MULTILINE)
        with open(ass_path, 'w', encoding='utf-8-sig') as f: f.write(new_content)

    # --- 6. רינדור סופי (Pure FFmpeg) ---
    def render(self, video_path, audio_path, ass_path, output_dir):
        """
        גרסת FFmpeg בלבד - יציבה ומהירה.
        1. הופך עברית בקובץ ASS זמני.
        2. צורב את ה-ASS על הוידאו.
        3. מחליף את האודיו בפלייבק (Mapping).
        """
        self.log("🎬 מתחיל רינדור סופי (FFmpeg)...")
        out_name = f"Final_Karaoke_{uuid.uuid4().hex[:5]}.mp4"
        out_path = os.path.join(output_dir, out_name)
        
        # 1. יצירת ASS הפוך (ויזואלי) לצריבה
        temp_ass = os.path.join(output_dir, "temp_render.ass")
        try:
            with open(ass_path, 'r', encoding='utf-8-sig') as f: content = f.read()
            lines = content.splitlines()
            new_lines = []
            for line in lines:
                if line.startswith("Dialogue:"):
                    parts = line.split(",", 9)
                    if len(parts) == 10:
                        parts[9] = self._fix_hebrew_text(parts[9]) # היפוך!
                        new_lines.append(",".join(parts))
                    else: new_lines.append(line)
                else: new_lines.append(line)
            with open(temp_ass, 'w', encoding='utf-8-sig') as f: f.write("\n".join(new_lines))
        except Exception as e:
            self.log(f"Subtitle prep error: {e}")
            return None

        # 2. FFmpeg: צריבה + החלפת אודיו בפקודה אחת
        # -i video (0)
        # -i audio (1)
        # -map 0:v (וידאו מקלט 0)
        # -map 1:a (אודיו מקלט 1 - דריסה!)
        # -vf ass (צריבת כתוביות על הוידאו)
        
        cmd = [self.ffmpeg_exe, '-y']
        
        # קלט וידאו
        if video_path and os.path.exists(video_path):
            cmd.extend(['-i', video_path])
        else:
            # אם אין וידאו, מסך שחור
            try: audio_len = librosa.get_duration(filename=audio_path)
            except: audio_len = 180
            cmd.extend(['-f', 'lavfi', '-i', f'color=c=black:s=1920x1080:r=30:d={audio_len}'])
        
        # קלט אודיו (הפלייבק)
        cmd.extend(['-i', audio_path])
        
        # הגדרות
        cwd = os.getcwd()
        os.chdir(output_dir) 
        
        # הפקודה המלאה
        cmd.extend([
            '-vf', f"ass='{os.path.basename(temp_ass)}'", # פילטר הכתוביות
            '-map', '0:v', # קח וידאו מ-Input 0
            '-map', '1:a', # קח אודיו מ-Input 1
            '-c:v', 'libx264', '-preset', 'ultrafast',
            '-c:a', 'aac', '-b:a', '192k',
            '-shortest', # סיים כשהקצר מביניהם נגמר
            out_name
        ])
        
        try:
            startupinfo = subprocess.STARTUPINFO() if os.name == 'nt' else None
            if startupinfo: startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            os.chdir(cwd)
            self.log("✅ הוידאו מוכן!")
            return out_path
        except Exception as e:
            self.log(f"FFmpeg Render Error: {e}")
            os.chdir(cwd)
            return None

# ==========================================
# חלק 3: ממשק המשתמש (App)
# ==========================================

logs = []
def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    logs.append(f"[{ts}] {msg}")

def get_logs(): return "\n".join(logs)

backend = BackendProcessor(log)

# --- לוגיקה ---

def process_auto(url, file, lang, sep_mode):
    global logs
    logs = []
    
    audio_path = None
    video_path = None
    
    if url:
        log("📥 מוריד...")
        video_path = backend.download(url, WORK_DIR, 'mp4')
        audio_path = backend.download(url, WORK_DIR, 'wav')
    elif file:
        log("📂 קובץ מקומי...")
        audio_path = file.name if hasattr(file, 'name') else file
        if str(audio_path).lower().endswith('mp4'): video_path = audio_path
        
    if not audio_path: return None, None, None, None, get_logs()

    sep = "2_stems" if "כן" in sep_mode else "none"
    files = backend.separate(audio_path, WORK_DIR, sep)
    vocals = files[0]
    playback = files[1] if len(files) > 1 else files[0]
    
    ass_path = backend.transcribe(vocals, WORK_DIR, lang)
    if not ass_path: return None, None, None, None, get_logs()
    
    final_video = backend.render(video_path, playback, ass_path, WORK_DIR)
    
    return final_video, ass_path, playback, video_path, get_logs()

def load_to_dashboard(ass_file):
    if not ass_file: return None
    return backend.ass_to_dataframe(ass_file.name)

def render_from_dashboard(df, video_file, audio_file, size, color):
    global logs
    logs = []
    log("💾 מעבד עריכות...")
    
    new_ass_path = os.path.join(WORK_DIR, f"edited_{int(time.time())}.ass")
    backend.dataframe_to_ass(df, None, new_ass_path)
    backend.update_ass_style(new_ass_path, size, color)
    
    v_path = video_file.name if video_file else None
    a_path = audio_file.name if audio_file else None
    
    if not a_path: return None, "❌ חסר קובץ פלייבק!"
    
    final = backend.render(v_path, a_path, new_ass_path, WORK_DIR)
    return final, get_logs()

def run_downloader(url, fmt):
    global logs; logs = []
    path = backend.download(url, WORK_DIR, fmt)
    return path, get_logs()

def run_separator(url, file, mode):
    global logs; logs = []
    audio_path = backend.download(url, WORK_DIR, 'wav') if url else (file.name if file else None)
    if not audio_path: return None, "חסר קובץ"
    
    mode_val = "2_stems" if "2" in mode else "4_stems"
    files = backend.separate(audio_path, WORK_DIR, mode_val)
    return files, get_logs()

def run_transcriber(url, file, lang):
    global logs; logs = []
    audio_path = backend.download(url, WORK_DIR, 'wav') if url else (file.name if file else None)
    if not audio_path: return None, "", "חסר קובץ"
    
    ass_path = backend.transcribe(audio_path, WORK_DIR, lang)
    content = ""
    if ass_path:
        with open(ass_path, 'r', encoding='utf-8-sig') as f: content = f.read()
    return ass_path, content, get_logs()

def run_analysis(file):
    global logs; logs = []
    if not file: return "חסר קובץ", ""
    bpm, key = backend.analyze_audio(file.name)
    res = f"BPM: {bpm} | Key: {key}" if bpm else "נכשל"
    return res, get_logs()

# --- UI ---

with gr.Blocks(title="Karaoke V29 (FFmpeg)", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎤 Karaoke Studio Pro V29 - Fast & Stable")
    
    with gr.Tabs():
        
        with gr.Tab("⚡ יצירה אוטומטית"):
            with gr.Row():
                with gr.Column():
                    in_url = gr.Textbox(label="YouTube URL")
                    in_file = gr.File(label="קובץ מקומי")
                    in_lang = gr.Dropdown(["he", "en"], value="he", label="שפה")
                    in_sep = gr.Radio(["כן (UVR5)", "לא"], value="כן (UVR5)", label="הפרדה")
                    btn_auto = gr.Button("🚀 התחל", variant="primary")
                with gr.Column():
                    out_video = gr.Video(label="תוצאה")
                    out_log = gr.TextArea(label="לוגים", lines=6)
                    with gr.Group():
                        gr.Markdown("### 📂 קבצים להמשך עריכה")
                        res_ass = gr.File(label="כתוביות (ASS)")
                        res_audio = gr.File(label="פלייבק (WAV)")
                        res_video_src = gr.File(label="וידאו מקור")
            btn_auto.click(process_auto, [in_url, in_file, in_lang, in_sep], [out_video, res_ass, res_audio, res_video_src, out_log])

        with gr.Tab("📝 Dashboard עריכה"):
            with gr.Row():
                with gr.Column(scale=1):
                    dash_ass = gr.File(label="1. טען כתוביות (ASS)")
                    btn_load = gr.Button("📂 טען לטבלה")
                    dash_vid = gr.File(label="2. וידאו רקע")
                    dash_aud = gr.File(label="3. פלייבק (חובה!)")
                    dash_size = gr.Slider(20, 150, 80, label="גודל גופן")
                    dash_color = gr.ColorPicker("#00FFFF", label="צבע")
                    btn_render = gr.Button("🎥 צור וידאו", variant="primary")
                with gr.Column(scale=2):
                    table = gr.Dataframe(headers=["Start", "End", "Text"], datatype=["str", "str", "str"], label="עורך כתוביות", interactive=True, wrap=True, col_count=(3, "fixed"))
                    res_edit = gr.Video(label="תוצאה")
                    log_edit = gr.TextArea(label="לוגים")
            btn_load.click(load_to_dashboard, dash_ass, table)
            btn_render.click(render_from_dashboard, [table, dash_vid, dash_aud, dash_size, dash_color], [res_edit, log_edit])

        with gr.Tab("🗣️ תמלול בלבד"):
            t_url = gr.Textbox(label="YouTube URL")
            t_file = gr.Audio(type="filepath", label="קובץ")
            t_lang = gr.Dropdown(["he", "en"], value="he", label="שפה")
            t_btn = gr.Button("תמלל")
            t_out = gr.File(label="קובץ ASS")
            t_preview = gr.TextArea(label="תצוגה", text_align="right")
            t_log = gr.TextArea(label="לוגים")
            t_btn.click(run_transcriber, [t_url, t_file, t_lang], [t_out, t_preview, t_log])

        with gr.Tab("🎵 הפרדה בלבד"):
            s_url = gr.Textbox(label="YouTube URL")
            s_file = gr.Audio(type="filepath", label="קובץ")
            s_mode = gr.Radio(["2 ערוצים", "4 ערוצים"], value="2 ערוצים", label="מצב")
            s_btn = gr.Button("הפרד")
            s_out = gr.Files(label="קבצים")
            s_log = gr.TextArea(label="לוגים")
            s_btn.click(run_separator, [s_url, s_file, s_mode], [s_out, s_log])

        with gr.Tab("⬇️ הורדה בלבד"):
            d_url = gr.Textbox(label="YouTube URL")
            d_fmt = gr.Radio(["wav", "mp4"], value="wav", label="פורמט")
            d_btn = gr.Button("הורד")
            d_out = gr.File(label="קובץ")
            d_log = gr.TextArea(label="לוגים")
            d_btn.click(run_downloader, [d_url, d_fmt], [d_out, d_log])

        with gr.Tab("🔍 ניתוח"):
            a_file = gr.Audio(type="filepath", label="קובץ")
            a_btn = gr.Button("נתח")
            a_res = gr.Label(label="תוצאה")
            a_log = gr.TextArea(label="לוגים")
            a_btn.click(run_analysis, [a_file], [a_res, a_log])

if __name__ == "__main__":
    app.queue().launch(inbrowser=True)
