import os
import re
import shutil
import subprocess
import logging
import numpy as np
from pathlib import Path
import imageio_ffmpeg
import uuid

try:
    import yt_dlp
    import torch
    import soundfile as sf
    from transformers import pipeline
    from audio_separator.separator import Separator
    import librosa
except ImportError as e:
    print(f"Backend Error: {e}")

logging.getLogger("audio_separator").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

class BackendProcessor:
    def __init__(self, log_func):
        self.log = log_func
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    def copy_to_custom_path(self, source_file, target_folder):
        if not target_folder or not os.path.exists(source_file):
            return source_file 

        try:
            os.makedirs(target_folder, exist_ok=True)
            filename = os.path.basename(source_file)
            destination = os.path.join(target_folder, filename)
            
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(destination):
                destination = os.path.join(target_folder, f"{base}_{counter}{ext}")
                counter += 1
                
            shutil.copy2(source_file, destination)
            self.log(f"Saved copy to: {destination}")
            return destination
        except Exception as e:
            self.log(f"Error saving to custom path: {e}")
            return source_file

    def convert_to_wav(self, input_path, output_path):
        safe_temp_source = None
        try:
            file_ext = os.path.splitext(input_path)[1]
            if not file_ext: file_ext = ".wav"
            
            safe_name = f"safe_source_{uuid.uuid4().hex[:8]}{file_ext}"
            safe_temp_source = os.path.join(os.path.dirname(output_path), safe_name)
            shutil.copy2(input_path, safe_temp_source)

            cmd = [
                self.ffmpeg_exe, '-y',
                '-i', safe_temp_source,
                '-ar', '16000',
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                output_path
            ]
            
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, startupinfo=startupinfo)
            return True
        except Exception as e:
            self.log(f"Error converting file: {e}")
            return False
        finally:
            if safe_temp_source and os.path.exists(safe_temp_source):
                try: os.remove(safe_temp_source)
                except: pass

    def download(self, url, output_dir, format_type='wav', is_playlist=False):
        self.log(f"Downloading from URL: {url}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        opts = {
            'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'noplaylist': not is_playlist,
            'ffmpeg_location': self.ffmpeg_exe
        }
        
        if format_type == 'mp4':
            opts.update({'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'})
        else:
            opts.update({
                'format': 'bestaudio/best',
                'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}]
            })

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if is_playlist:
                    downloaded_files = []
                    for entry in info.get('entries', []):
                        fname = ydl.prepare_filename(entry)
                        final = Path(fname).with_suffix('.wav') if format_type == 'wav' else Path(fname)
                        downloaded_files.append(str(final))
                    self.log(f"Playlist download complete: {len(downloaded_files)} files")
                    return downloaded_files
                else:
                    fname = ydl.prepare_filename(info)
                    final = Path(fname).with_suffix('.wav') if format_type == 'wav' else Path(fname)
                    self.log(f"Download complete: {final.name}")
                    return str(final)
        except Exception as e:
            self.log(f"Download error: {e}")
            return None if not is_playlist else []

    def separate(self, audio_path, output_dir, mode="2_stems"):
        if mode == "none": return [audio_path]

        model_name = "Kim_Vocal_2.onnx" if mode == "2_stems" else "htdemucs_ft.yaml"
        self.log(f"Separating stems using {model_name}...")
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        temp_input = os.path.join(output_dir, "temp_sep_input.wav")
        if not self.convert_to_wav(audio_path, temp_input): return None

        file_stem = Path(audio_path).stem
        safe_folder_name = self._sanitize(file_stem)
        final_sep_dir = os.path.join(output_dir, f"Sep_{safe_folder_name}")
        Path(final_sep_dir).mkdir(parents=True, exist_ok=True)

        try:
            # Set FFmpeg path via environment variable
            os.environ['PATH'] = os.path.dirname(self.ffmpeg_exe) + os.pathsep + os.environ.get('PATH', '')
            
            separator = Separator(
                log_level=logging.ERROR,
                output_dir=final_sep_dir,
                output_single_stem=None,
                model_file_dir=os.path.join(output_dir, "uvr_models")
            )
            separator.load_model(model_filename=model_name)
            output_files = separator.separate(temp_input)
            
            full_paths = []
            if mode == "2_stems":
                vocals_path = None; playback_path = None
                for f in output_files:
                    full_path = os.path.join(final_sep_dir, f)
                    if "(Vocals)" in f:
                        target = os.path.join(final_sep_dir, "Vocals.wav")
                        if os.path.exists(target): os.remove(target)
                        os.rename(full_path, target)
                        vocals_path = target
                    elif "(Instrumental)" in f:
                        target = os.path.join(final_sep_dir, "Playback.wav")
                        if os.path.exists(target): os.remove(target)
                        os.rename(full_path, target)
                        playback_path = target
                if vocals_path and playback_path: full_paths = [vocals_path, playback_path]
            else:
                for f in output_files: full_paths.append(os.path.join(final_sep_dir, f))

            try: os.remove(temp_input)
            except: pass
            self.log("Separation complete.")
            return full_paths
        except Exception as e:
            self.log(f"Separation error: {e}")
            return None

    def transcribe(self, audio_path, output_dir, lang, hf_token=None):
        self.log(f"Preparing transcription ({lang})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "ivrit-ai/whisper-large-v3-turbo" if lang == "he" else "openai/whisper-large-v3-turbo"

        clean_path = os.path.join(output_dir, "clean_whisper.wav")
        if not self.convert_to_wav(audio_path, clean_path): return None

        try:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            self.log(f"Loading model {model_id}...")
            pipe = pipeline(
                "automatic-speech-recognition", model=model_id, tokenizer=model_id, feature_extractor=model_id,
                torch_dtype=torch_dtype, device=device, chunk_length_s=30, batch_size=8, return_timestamps="word", token=hf_token
            )
            
            data, samplerate = sf.read(clean_path)
            total_duration = len(data) / samplerate
            self.log(f"Audio duration: {total_duration:.2f}s. Starting...")

            full_results = []
            chunk_duration = 30
            for i in range(0, int(total_duration), chunk_duration):
                start_sample = int(i * samplerate)
                end_sample = int(min((i + chunk_duration) * samplerate, len(data)))
                if end_sample - start_sample < samplerate: break 
                audio_chunk = data[start_sample:end_sample]
                gen_kwargs = {"language": "hebrew" if lang == "he" else "english", "task": "transcribe"}
                result = pipe(audio_chunk, generate_kwargs=gen_kwargs)
                if result.get('text'): self.log(f"[{self._fmt_time(i)}]: {result['text'].strip()}")
                for c in result.get('chunks', []):
                    if 'timestamp' in c:
                        ts = c['timestamp']
                        if ts[0] is not None and ts[1] is not None:
                            full_results.append({'text': c['text'], 'timestamp': (ts[0]+i, ts[1]+i)})

            ass_path = os.path.join(output_dir, "karaoke.ass")
            self._generate_ass(full_results, ass_path)
            
            try: os.remove(clean_path)
            except: pass
            self.log(f"Transcription complete.")
            return ass_path
        except Exception as e:
            self.log(f"Transcription error: {e}")
            return None
        finally:
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    def analyze_audio(self, audio_path):
        self.log(f"Analyzing: {Path(audio_path).name}...")
        temp_wav = audio_path + ".temp_analysis.wav"
        if not self.convert_to_wav(audio_path, temp_wav): return None, None
        try:
            y, sr = librosa.load(temp_wav, sr=None, duration=60)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = round(tempo) if np.isscalar(tempo) else round(tempo[0])
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            key = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][np.argmax(np.mean(chroma, axis=1))]
            self.log(f"BPM: {bpm} | Key: {key}")
            return bpm, key
        except Exception as e:
            self.log(f"Analysis error: {e}")
            return None, None
        finally:
            if os.path.exists(temp_wav): os.remove(temp_wav)

    def transcribe_batch(self, audio_files, output_dir, lang):
        """Transcribe multiple audio files and create individual + combined TXT files"""
        self.log(f"Starting batch transcription for {len(audio_files)} files...")
        
        combined_content = []
        individual_files = []
        
        for idx, audio_path in enumerate(audio_files, 1):
            self.log(f"Processing file {idx}/{len(audio_files)}: {Path(audio_path).name}")
            
            file_dir = os.path.join(output_dir, f"Transcription_{idx}")
            os.makedirs(file_dir, exist_ok=True)
            
            ass_path = self.transcribe(audio_path, file_dir, lang)
            if not ass_path:
                self.log(f"Failed to transcribe file {idx}")
                continue
            
            video_title = Path(audio_path).stem
            txt_path = os.path.join(output_dir, f"{video_title}.txt")
            self.ass_to_txt(ass_path, txt_path)
            individual_files.append(txt_path)
            
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                combined_content.append(f"{'='*60}\n{video_title}\n{'='*60}\n{content}\n")
        
        combined_path = os.path.join(output_dir, "Combined_Transcriptions.txt")
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(combined_content))
        
        self.log(f"Batch transcription complete. Files: {combined_path}")
        return individual_files, combined_path

    def ass_to_txt(self, ass_path, output_path):
        """Convert ASS subtitle file to simple TXT format with timestamps"""
        try:
            with open(ass_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
            
            lines = content.split('\n')
            txt_lines = []
            
            for line in lines:
                if line.startswith('Dialogue:'):
                    parts = line.split(',', 9)
                    if len(parts) >= 10:
                        start_time = parts[1].strip()
                        end_time = parts[2].strip()
                        text = parts[9].strip()
                        txt_lines.append(f"[{start_time} - {end_time}] {text}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(txt_lines))
            
            self.log(f"TXT file created: {output_path}")
            return output_path
        except Exception as e:
            self.log(f"Error converting ASS to TXT: {e}")
            return None

    def render_custom_karaoke(self, video_path, audio_path, ass_path, output_dir):
        self.log("Rendering custom karaoke video...")
        out_name = f"Karaoke_Edit_{uuid.uuid4().hex[:6]}.mp4"
        output_file = os.path.join(output_dir, out_name)

        safe_audio = os.path.join(output_dir, "safe_audio.wav")
        if not self.convert_to_wav(audio_path, safe_audio): return None
        
        safe_ass = os.path.join(output_dir, "subs.ass")
        shutil.copy2(ass_path, safe_ass)

        cmd = [self.ffmpeg_exe, '-y']
        if video_path and os.path.exists(video_path): cmd.extend(['-i', video_path])
        else: cmd.extend(['-f', 'lavfi', '-i', 'color=c=black:s=1920x1080:r=30'])

        cmd.extend(['-i', safe_audio])
        cmd.extend([
            '-vf', f"ass='{os.path.basename(safe_ass)}'",
            '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'ultrafast', 
            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-shortest', output_file
        ])

        cwd = os.getcwd()
        os.chdir(output_dir)
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.log(f"Video created successfully: {out_name}")
            return output_file
        except Exception as e:
            self.log(f"Video rendering error: {e}")
            return None
        finally:
            os.chdir(cwd)
            if os.path.exists(safe_audio): os.remove(safe_audio)
            if os.path.exists(safe_ass): os.remove(safe_ass)

    # === עדכון פונקציית עדכון הסגנון ===
    def update_ass_style(self, ass_content, font_size, primary_color):
        """
        מחליף את כל שורת הסגנון בקובץ כדי להבטיח שהצבע והגודל יתפסו.
        """
        # המרת צבע HEX (#RRGGBB) לפורמט ASS (&HBBGGRR)
        if primary_color.startswith('#'):
            r = primary_color[1:3]
            g = primary_color[3:5]
            b = primary_color[5:7]
            ass_color = f"&H00{b}{g}{r}".upper() # סדר BGR
        else:
            ass_color = "&H00FFFFFF" 

        # בניית שורת סגנון חדשה ונקייה
        # שים לב: הפונט Arial תומך בעברית. הוספתי Encoding 1 (ברירת מחדל).
        new_style_line = f"Style: Karaoke,Arial,{int(font_size)},{ass_color},&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,100,1"

        # החלפה חכמה: מחפש כל שורה שמתחילה ב-Style: ומחליף אותה
        new_content = re.sub(r"^Style:.*$", new_style_line, ass_content, flags=re.MULTILINE)
        
        return new_content

    def _sanitize(self, name): return re.sub(r'[^\w\-_]', '', name.replace(' ', '_'))
    def _fmt_time(self, s): return f"{int(s//60)}:{int(s%60):02}"
    def _fmt_ass(self, s): return f"{int(s//3600)}:{int((s%3600)//60):02}:{int(s%60):02}.{int((s-int(s))*100):02}"
    
    def _generate_ass(self, full_results, output_path, enable_dynamic_color=True):
        # כותרת בסיסית. הפונקציה update_ass_style תדרוס את השורה הרלוונטית בעת העריכה.
        header = """[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Karaoke,Arial,80,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,100,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"""
        events = []
        curr = []
        start_t = None
        for item in full_results:
            t1, t2 = item['timestamp']
            if start_t is None: start_t = t1
            curr.append({'w': item['text'].strip(), 's': t1, 'e': t2})
            if item['text'].strip().endswith(('.', '?', '!')) or len(curr) > 7:
                if enable_dynamic_color:
                    # Generate dynamic color karaoke (white→blue)
                    line_text = self._generate_colored_line(curr)
                else:
                    line_text = " ".join([w['w'] for w in curr])
                events.append(f"Dialogue: 0,{self._fmt_ass(start_t)},{self._fmt_ass(t2)},Karaoke,,0,0,0,,{line_text}")
                curr = []
                start_t = None
        
        if curr:
            if enable_dynamic_color:
                line_text = self._generate_colored_line(curr)
            else:
                line_text = " ".join([w['w'] for w in curr])
            events.append(f"Dialogue: 0,{self._fmt_ass(start_t)},{self._fmt_ass(curr[-1]['e'])},Karaoke,,0,0,0,,{line_text}")

        with open(output_path, "w", encoding="utf-8-sig") as f:
            f.write(header + "\n".join(events))

    def _generate_colored_line(self, word_list):
        """Generate ASS line with dynamic color changes (white→blue for each word)"""
        # White color: &H00FFFFFF, Blue color: &H0000FFFF
        colored_text = ""
        for word_obj in word_list:
            word = word_obj['w']
            s_time = word_obj['s']
            e_time = word_obj['e']
            # Start white, transition to blue mid-duration, end blue
            mid_time = s_time + (e_time - s_time) * 0.3
            colored_text += f"{{\\c&H00FFFFFF&\\t({self._fmt_ass_ms(s_time)},{self._fmt_ass_ms(mid_time)},\\c&H0000FFFF&)}}{word} "
        return colored_text.strip()

    def _fmt_ass_ms(self, s):
        """Format time for ASS \\t() tags in centiseconds"""
        return f"{int(s*100)}"
