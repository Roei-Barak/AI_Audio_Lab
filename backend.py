# # # import os
# # # import re
# # # import shutil
# # # import subprocess
# # # import logging
# # # import numpy as np
# # # from pathlib import Path
# # # import imageio_ffmpeg
# # # import uuid

# # # try:
# # #     import yt_dlp
# # #     import torch
# # #     import soundfile as sf
# # #     from transformers import pipeline
# # #     from audio_separator.separator import Separator
# # #     import librosa
# # # except ImportError as e:
# # #     print(f"Backend Error: {e}")

# # # logging.getLogger("audio_separator").setLevel(logging.ERROR)
# # # logging.getLogger("transformers").setLevel(logging.ERROR)

# # # class BackendProcessor:
# # #     def __init__(self, log_func):
# # #         self.log = log_func
# # #         self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

# # #     def convert_to_wav(self, input_path, output_path):
# # #         safe_temp_source = None
# # #         try:
# # #             file_ext = os.path.splitext(input_path)[1]
# # #             if not file_ext: file_ext = ".wav"
            
# # #             safe_name = f"safe_source_{uuid.uuid4().hex[:8]}{file_ext}"
# # #             safe_temp_source = os.path.join(os.path.dirname(output_path), safe_name)
            
# # #             shutil.copy2(input_path, safe_temp_source)

# # #             cmd = [
# # #                 self.ffmpeg_exe, '-y',
# # #                 '-i', safe_temp_source,
# # #                 '-ar', '16000',
# # #                 '-ac', '1',
# # #                 '-c:a', 'pcm_s16le',
# # #                 output_path
# # #             ]
            
# # #             startupinfo = None
# # #             if os.name == 'nt':
# # #                 startupinfo = subprocess.STARTUPINFO()
# # #                 startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

# # #             subprocess.run(
# # #                 cmd, 
# # #                 check=True, 
# # #                 stdout=subprocess.DEVNULL, 
# # #                 stderr=subprocess.PIPE,
# # #                 startupinfo=startupinfo
# # #             )
# # #             return True

# # #         except Exception as e:
# # #             self.log(f"Error converting file: {e}")
# # #             return False
# # #         finally:
# # #             if safe_temp_source and os.path.exists(safe_temp_source):
# # #                 try:
# # #                     os.remove(safe_temp_source)
# # #                 except:
# # #                     pass

# # #     def download(self, url, output_dir, format_type='wav'):
# # #         self.log(f"Downloading from URL: {url}")
# # #         Path(output_dir).mkdir(parents=True, exist_ok=True)
        
# # #         opts = {
# # #             'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
# # #             'quiet': True,
# # #             'no_warnings': True,
# # #             'noplaylist': True,
# # #             'ffmpeg_location': self.ffmpeg_exe
# # #         }
        
# # #         # If user specifically requested mp4 (video), we download video+audio
# # #         if format_type == 'mp4':
# # #             opts.update({'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'})
# # #         else:
# # #             opts.update({
# # #                 'format': 'bestaudio/best',
# # #                 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}]
# # #             })

# # #         try:
# # #             with yt_dlp.YoutubeDL(opts) as ydl:
# # #                 info = ydl.extract_info(url, download=True)
# # #                 fname = ydl.prepare_filename(info)
# # #                 # Correction for output filename
# # #                 if format_type == 'wav':
# # #                     final = Path(fname).with_suffix('.wav')
# # #                 else:
# # #                     final = Path(fname)
                    
# # #                 self.log(f"Download complete: {final.name}")
# # #                 return str(final)
# # #         except Exception as e:
# # #             self.log(f"Download error: {e}")
# # #             return None

# # #     def separate(self, audio_path, output_dir, mode="2_stems"):
# # #         if mode == "none":
# # #             return [audio_path]

# # #         model_name = "Kim_Vocal_2.onnx" if mode == "2_stems" else "htdemucs_ft.yaml"
# # #         self.log(f"Separating stems using {model_name}...")
        
# # #         if torch.cuda.is_available(): torch.cuda.empty_cache()

# # #         temp_input = os.path.join(output_dir, "temp_sep_input.wav")
# # #         if not self.convert_to_wav(audio_path, temp_input): return None

# # #         file_stem = Path(audio_path).stem
# # #         safe_folder_name = self._sanitize(file_stem)
# # #         final_sep_dir = os.path.join(output_dir, f"Sep_{safe_folder_name}")
# # #         Path(final_sep_dir).mkdir(parents=True, exist_ok=True)

# # #         try:
# # #             separator = Separator(
# # #                 log_level=logging.ERROR,
# # #                 output_dir=final_sep_dir,
# # #                 output_single_stem=None,
# # #                 model_file_dir=os.path.join(output_dir, "uvr_models")
# # #             )
            
# # #             separator.load_model(model_filename=model_name)
# # #             output_files = separator.separate(temp_input)
            
# # #             full_paths = []
            
# # #             if mode == "2_stems":
# # #                 vocals_path = None
# # #                 playback_path = None
# # #                 for f in output_files:
# # #                     full_path = os.path.join(final_sep_dir, f)
# # #                     if "(Vocals)" in f:
# # #                         target = os.path.join(final_sep_dir, "Vocals.wav")
# # #                         if os.path.exists(target): os.remove(target)
# # #                         os.rename(full_path, target)
# # #                         vocals_path = target
# # #                     elif "(Instrumental)" in f:
# # #                         target = os.path.join(final_sep_dir, "Playback.wav")
# # #                         if os.path.exists(target): os.remove(target)
# # #                         os.rename(full_path, target)
# # #                         playback_path = target
                
# # #                 if vocals_path and playback_path:
# # #                     full_paths = [vocals_path, playback_path]
            
# # #             else:
# # #                 for f in output_files:
# # #                     full_paths.append(os.path.join(final_sep_dir, f))

# # #             try: os.remove(temp_input)
# # #             except: pass
            
# # #             self.log("Separation complete.")
# # #             return full_paths

# # #         except Exception as e:
# # #             self.log(f"Separation error: {e}")
# # #             return None

# # #     def transcribe(self, audio_path, output_dir, lang, hf_token=None):
# # #         self.log(f"Preparing transcription ({lang})...")
# # #         device = "cuda" if torch.cuda.is_available() else "cpu"
# # #         torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
# # #         model_id = "ivrit-ai/whisper-large-v3-turbo" if lang == "he" else "openai/whisper-large-v3-turbo"

# # #         clean_path = os.path.join(output_dir, "clean_whisper.wav")
# # #         if not self.convert_to_wav(audio_path, clean_path): return None

# # #         try:
# # #             if torch.cuda.is_available(): torch.cuda.empty_cache()
            
# # #             self.log(f"Loading model {model_id}...")
# # #             pipe = pipeline(
# # #                 "automatic-speech-recognition",
# # #                 model=model_id,
# # #                 tokenizer=model_id,
# # #                 feature_extractor=model_id,
# # #                 torch_dtype=torch_dtype,
# # #                 device=device,
# # #                 chunk_length_s=30,
# # #                 batch_size=8,
# # #                 return_timestamps="word",
# # #                 token=hf_token
# # #             )
            
# # #             data, samplerate = sf.read(clean_path)
# # #             total_duration = len(data) / samplerate
# # #             self.log(f"Audio duration: {total_duration:.2f}s. Starting...")

# # #             full_results = []
# # #             chunk_duration = 30
            
# # #             for i in range(0, int(total_duration), chunk_duration):
# # #                 start_sample = int(i * samplerate)
# # #                 end_sample = int(min((i + chunk_duration) * samplerate, len(data)))
# # #                 if end_sample - start_sample < samplerate: break 

# # #                 audio_chunk = data[start_sample:end_sample]
# # #                 gen_kwargs = {
# # #                     "language": "hebrew" if lang == "he" else "english",
# # #                     "task": "transcribe"
# # #                 }
                
# # #                 result = pipe(audio_chunk, generate_kwargs=gen_kwargs)
# # #                 if result.get('text'):
# # #                     self.log(f"[{self._fmt_time(i)}]: {result['text'].strip()}")
                
# # #                 for c in result.get('chunks', []):
# # #                     if 'timestamp' in c:
# # #                         ts = c['timestamp']
# # #                         if ts[0] is not None and ts[1] is not None:
# # #                             full_results.append({'text': c['text'], 'timestamp': (ts[0]+i, ts[1]+i)})

# # #             ass_path = os.path.join(output_dir, "karaoke.ass")
# # #             self._generate_ass(full_results, ass_path)
            
# # #             try: os.remove(clean_path)
# # #             except: pass
            
# # #             self.log(f"Transcription complete.")
# # #             return ass_path

# # #         except Exception as e:
# # #             self.log(f"Transcription error: {e}")
# # #             return None
# # #         finally:
# # #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# # #     def analyze_audio(self, audio_path):
# # #         self.log(f"Analyzing: {Path(audio_path).name}...")
# # #         temp_wav = audio_path + ".temp_analysis.wav"
# # #         if not self.convert_to_wav(audio_path, temp_wav): return None, None
        
# # #         try:
# # #             y, sr = librosa.load(temp_wav, sr=None, duration=60)
# # #             tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
# # #             bpm = round(tempo) if np.isscalar(tempo) else round(tempo[0])
# # #             chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
# # #             key = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][np.argmax(np.mean(chroma, axis=1))]
# # #             self.log(f"BPM: {bpm} | Key: {key}")
# # #             return bpm, key
# # #         except Exception as e:
# # #             self.log(f"Analysis error: {e}")
# # #             return None, None
# # #         finally:
# # #             if os.path.exists(temp_wav): os.remove(temp_wav)

# # #     def render_video(self, audio_path, ass_path, output_dir):
# # #         # Wrapper for simple render (keeps backward compatibility)
# # #         return self.render_custom_karaoke(None, audio_path, ass_path, output_dir)

# # #     def render_custom_karaoke(self, video_path, audio_path, ass_path, output_dir):
# # #         """
# # #         Combines Video + Audio (Playback) + Subtitles into a final MP4.
# # #         If video_path is None, generates black background.
# # #         """
# # #         self.log("Rendering custom karaoke video...")
# # #         out_name = f"Karaoke_Edit_{uuid.uuid4().hex[:6]}.mp4"
# # #         output_file = os.path.join(output_dir, out_name)

# # #         # 1. Prepare Safe Audio
# # #         safe_audio = os.path.join(output_dir, "safe_audio.wav")
# # #         if not self.convert_to_wav(audio_path, safe_audio):
# # #             self.log("Failed to process audio")
# # #             return None
        
# # #         # 2. Prepare Subtitles (rename to safe ascii)
# # #         safe_ass = os.path.join(output_dir, "subs.ass")
# # #         shutil.copy2(ass_path, safe_ass)

# # #         # 3. Build FFmpeg Command
# # #         cmd = [self.ffmpeg_exe, '-y']

# # #         # Input 0: Video source
# # #         if video_path and os.path.exists(video_path):
# # #             # If video exists, use it
# # #             cmd.extend(['-i', video_path])
# # #         else:
# # #             # Generate black background
# # #             cmd.extend(['-f', 'lavfi', '-i', 'color=c=black:s=1920x1080:r=30'])

# # #         # Input 1: Audio source (The instrumental)
# # #         cmd.extend(['-i', safe_audio])

# # #         # Filter complex: 
# # #         # 1. Map video (0:v)
# # #         # 2. Burn subtitles
# # #         # 3. Map audio (1:a) -> Replace video audio
# # #         # 4. Shortest (stop when shortest input ends)
        
# # #         # Using simple filtergraph for burning subtitles on the video stream
# # #         cmd.extend([
# # #             '-vf', f"ass='{os.path.basename(safe_ass)}'",
# # #             '-map', '0:v', 
# # #             '-map', '1:a', # Use the provided audio, NOT the video's audio
# # #             '-c:v', 'libx264', 
# # #             '-preset', 'ultrafast', 
# # #             '-pix_fmt', 'yuv420p',
# # #             '-c:a', 'aac', 
# # #             '-b:a', '192k',
# # #             '-shortest',
# # #             output_file
# # #         ])

# # #         cwd = os.getcwd()
# # #         os.chdir(output_dir) # Change dir so FFmpeg finds 'subs.ass' locally
        
# # #         try:
# # #             # Run FFmpeg
# # #             startupinfo = None
# # #             if os.name == 'nt':
# # #                 startupinfo = subprocess.STARTUPINFO()
# # #                 startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

# # #             subprocess.run(
# # #                 cmd, 
# # #                 check=True, 
# # #                 stdout=subprocess.DEVNULL, 
# # #                 stderr=subprocess.PIPE,
# # #                 startupinfo=startupinfo
# # #             )
# # #             self.log(f"Video created successfully: {out_name}")
# # #             return output_file
            
# # #         except subprocess.CalledProcessError as e:
# # #             err = e.stderr.decode() if e.stderr else str(e)
# # #             self.log(f"Rendering error: {err}")
# # #             return None
# # #         except Exception as e:
# # #             self.log(f"General error: {e}")
# # #             return None
# # #         finally:
# # #             os.chdir(cwd)
# # #             # Cleanup
# # #             if os.path.exists(safe_audio): os.remove(safe_audio)
# # #             if os.path.exists(safe_ass): os.remove(safe_ass)

# # #     def update_ass_style(self, ass_content, font_size, primary_color):
# # #         """
# # #         Updates the ASS header with new style using Regex.
# # #         Color format in ASS is &HBBGGRR (Hex reversed).
# # #         """
# # #         # Convert HTML Hex (#RRGGBB) to ASS Hex (&HBBGGRR)
# # #         if primary_color.startswith('#'):
# # #             r = primary_color[1:3]
# # #             g = primary_color[3:5]
# # #             b = primary_color[5:7]
# # #             ass_color = f"&H00{b}{g}{r}"
# # #         else:
# # #             ass_color = "&H00FFFFFF" # Default white

# # #         # Replace Fontsize
# # #         ass_content = re.sub(r"Fontsize,\s*\d+", f"Fontsize,{int(font_size)}", ass_content)
        
# # #         # Replace PrimaryColour
# # #         # Matches: PrimaryColour, &Hxxxxxxxx
# # #         ass_content = re.sub(r"PrimaryColour,\s*&H[0-9A-Fa-f]+", f"PrimaryColour,{ass_color}", ass_content)

# # #         return ass_content

# # #     def _sanitize(self, name): return re.sub(r'[^\w\-_]', '', name.replace(' ', '_'))
# # #     def _fmt_time(self, s): return f"{int(s//60)}:{int(s%60):02}"
# # #     def _fmt_ass(self, s): return f"{int(s//3600)}:{int((s%3600)//60):02}:{int(s%60):02}.{int((s-int(s))*100):02}"
    
# # #     def _generate_ass(self, full_results, output_path):
# # #         header = """[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Karaoke,Arial,80,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,100,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"""
# # #         events = []
# # #         curr = []
# # #         start_t = None
# # #         for item in full_results:
# # #             t1, t2 = item['timestamp']
# # #             if start_t is None: start_t = t1
# # #             curr.append({'w': item['text'], 's': t1, 'e': t2})
# # #             if item['text'].strip().endswith(('.', '?', '!')) or len(curr) > 8:
# # #                 line = ""
# # #                 curr_t = start_t
# # #                 for w in curr:
# # #                     gap = int((w['s'] - curr_t)*100); dur = int((w['e'] - w['s'])*100)
# # #                     if gap>0: line += f"{{\k{gap}}}"
# # #                     line += f"{{\kf{dur}}}{w['w']} "
# # #                     curr_t = w['e']
# # #                 events.append(f"Dialogue: 0,{self._fmt_ass(start_t)},{self._fmt_ass(t2)},Karaoke,,0,0,0,,{line.strip()}")
# # #                 curr = []; start_t = None
# # #         with open(output_path, "w", encoding="utf-8-sig") as f:
# # #             f.write(header + "\n".join(events))
# # import os
# # import re
# # import shutil
# # import subprocess
# # import logging
# # import numpy as np
# # from pathlib import Path
# # import imageio_ffmpeg
# # import uuid

# # # --- ייבוא ספריות ---
# # try:
# #     import yt_dlp
# #     import torch
# #     import soundfile as sf
# #     from transformers import pipeline
# #     from audio_separator.separator import Separator
# #     import librosa
# #     # ייבוא הספרייה לתיקון עברית
# #     from bidi.algorithm import get_display 
# # except ImportError as e:
# #     print(f"Backend Error: {e}")
# #     print("Tip: Run 'pip install python-bidi' to fix Hebrew issues.")

# # logging.getLogger("audio_separator").setLevel(logging.ERROR)
# # logging.getLogger("transformers").setLevel(logging.ERROR)

# # class BackendProcessor:
# #     def __init__(self, log_func):
# #         self.log = log_func
# #         self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

# #     def copy_to_custom_path(self, source_file, target_folder):
# #         if not target_folder or not os.path.exists(source_file):
# #             return source_file 

# #         try:
# #             os.makedirs(target_folder, exist_ok=True)
# #             filename = os.path.basename(source_file)
# #             destination = os.path.join(target_folder, filename)
            
# #             base, ext = os.path.splitext(filename)
# #             counter = 1
# #             while os.path.exists(destination):
# #                 destination = os.path.join(target_folder, f"{base}_{counter}{ext}")
# #                 counter += 1
                
# #             shutil.copy2(source_file, destination)
# #             self.log(f"Saved copy to: {destination}")
# #             return destination
# #         except Exception as e:
# #             self.log(f"Error saving to custom path: {e}")
# #             return source_file

# #     def convert_to_wav(self, input_path, output_path):
# #         safe_temp_source = None
# #         try:
# #             file_ext = os.path.splitext(input_path)[1]
# #             if not file_ext: file_ext = ".wav"
            
# #             safe_name = f"safe_source_{uuid.uuid4().hex[:8]}{file_ext}"
# #             safe_temp_source = os.path.join(os.path.dirname(output_path), safe_name)
# #             shutil.copy2(input_path, safe_temp_source)

# #             cmd = [
# #                 self.ffmpeg_exe, '-y',
# #                 '-i', safe_temp_source,
# #                 '-ar', '16000',
# #                 '-ac', '1',
# #                 '-c:a', 'pcm_s16le',
# #                 output_path
# #             ]
            
# #             startupinfo = None
# #             if os.name == 'nt':
# #                 startupinfo = subprocess.STARTUPINFO()
# #                 startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

# #             subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, startupinfo=startupinfo)
# #             return True
# #         except Exception as e:
# #             self.log(f"Error converting file: {e}")
# #             return False
# #         finally:
# #             if safe_temp_source and os.path.exists(safe_temp_source):
# #                 try: os.remove(safe_temp_source)
# #                 except: pass

# #     def download(self, url, output_dir, format_type='wav'):
# #         self.log(f"Downloading from URL: {url}")
# #         Path(output_dir).mkdir(parents=True, exist_ok=True)
        
# #         opts = {
# #             'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
# #             'quiet': True,
# #             'no_warnings': True,
# #             'noplaylist': True,
# #             'ffmpeg_location': self.ffmpeg_exe
# #         }
        
# #         if format_type == 'mp4':
# #             opts.update({'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'})
# #         else:
# #             opts.update({
# #                 'format': 'bestaudio/best',
# #                 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}]
# #             })

# #         try:
# #             with yt_dlp.YoutubeDL(opts) as ydl:
# #                 info = ydl.extract_info(url, download=True)
# #                 fname = ydl.prepare_filename(info)
# #                 final = Path(fname).with_suffix('.wav') if format_type == 'wav' else Path(fname)
# #                 self.log(f"Download complete: {final.name}")
# #                 return str(final)
# #         except Exception as e:
# #             self.log(f"Download error: {e}")
# #             return None

# #     def separate(self, audio_path, output_dir, mode="2_stems"):
# #         if mode == "none": return [audio_path]

# #         model_name = "Kim_Vocal_2.onnx" if mode == "2_stems" else "htdemucs_ft.yaml"
# #         self.log(f"Separating stems using {model_name}...")
        
# #         if torch.cuda.is_available(): torch.cuda.empty_cache()

# #         temp_input = os.path.join(output_dir, "temp_sep_input.wav")
# #         if not self.convert_to_wav(audio_path, temp_input): return None

# #         file_stem = Path(audio_path).stem
# #         safe_folder_name = self._sanitize(file_stem)
# #         final_sep_dir = os.path.join(output_dir, f"Sep_{safe_folder_name}")
# #         Path(final_sep_dir).mkdir(parents=True, exist_ok=True)

# #         try:
# #             separator = Separator(
# #                 log_level=logging.ERROR,
# #                 output_dir=final_sep_dir,
# #                 output_single_stem=None,
# #                 model_file_dir=os.path.join(output_dir, "uvr_models")
# #             )
# #             separator.load_model(model_filename=model_name)
# #             output_files = separator.separate(temp_input)
            
# #             full_paths = []
# #             if mode == "2_stems":
# #                 vocals_path = None; playback_path = None
# #                 for f in output_files:
# #                     full_path = os.path.join(final_sep_dir, f)
# #                     if "(Vocals)" in f:
# #                         target = os.path.join(final_sep_dir, "Vocals.wav")
# #                         if os.path.exists(target): os.remove(target)
# #                         os.rename(full_path, target)
# #                         vocals_path = target
# #                     elif "(Instrumental)" in f:
# #                         target = os.path.join(final_sep_dir, "Playback.wav")
# #                         if os.path.exists(target): os.remove(target)
# #                         os.rename(full_path, target)
# #                         playback_path = target
# #                 if vocals_path and playback_path: full_paths = [vocals_path, playback_path]
# #             else:
# #                 for f in output_files: full_paths.append(os.path.join(final_sep_dir, f))

# #             try: os.remove(temp_input)
# #             except: pass
# #             self.log("Separation complete.")
# #             return full_paths
# #         except Exception as e:
# #             self.log(f"Separation error: {e}")
# #             return None

# #     def transcribe(self, audio_path, output_dir, lang, hf_token=None):
# #         self.log(f"Preparing transcription ({lang})...")
# #         device = "cuda" if torch.cuda.is_available() else "cpu"
# #         torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# #         model_id = "ivrit-ai/whisper-large-v3-turbo" if lang == "he" else "openai/whisper-large-v3-turbo"

# #         clean_path = os.path.join(output_dir, "clean_whisper.wav")
# #         if not self.convert_to_wav(audio_path, clean_path): return None

# #         try:
# #             if torch.cuda.is_available(): torch.cuda.empty_cache()
# #             self.log(f"Loading model {model_id}...")
# #             pipe = pipeline(
# #                 "automatic-speech-recognition", model=model_id, tokenizer=model_id, feature_extractor=model_id,
# #                 torch_dtype=torch_dtype, device=device, chunk_length_s=30, batch_size=8, return_timestamps="word", token=hf_token
# #             )
            
# #             data, samplerate = sf.read(clean_path)
# #             total_duration = len(data) / samplerate
# #             self.log(f"Audio duration: {total_duration:.2f}s. Starting...")

# #             full_results = []
# #             chunk_duration = 30
# #             for i in range(0, int(total_duration), chunk_duration):
# #                 start_sample = int(i * samplerate)
# #                 end_sample = int(min((i + chunk_duration) * samplerate, len(data)))
# #                 if end_sample - start_sample < samplerate: break 
# #                 audio_chunk = data[start_sample:end_sample]
# #                 gen_kwargs = {"language": "hebrew" if lang == "he" else "english", "task": "transcribe"}
# #                 result = pipe(audio_chunk, generate_kwargs=gen_kwargs)
# #                 if result.get('text'): self.log(f"[{self._fmt_time(i)}]: {result['text'].strip()}")
# #                 for c in result.get('chunks', []):
# #                     if 'timestamp' in c:
# #                         ts = c['timestamp']
# #                         if ts[0] is not None and ts[1] is not None:
# #                             full_results.append({'text': c['text'], 'timestamp': (ts[0]+i, ts[1]+i)})

# #             ass_path = os.path.join(output_dir, "karaoke.ass")
# #             # שולחים את השפה לפונקציית היצירה
# #             self._generate_ass(full_results, ass_path, is_hebrew=(lang=="he"))
            
# #             try: os.remove(clean_path)
# #             except: pass
# #             self.log(f"Transcription complete.")
# #             return ass_path
# #         except Exception as e:
# #             self.log(f"Transcription error: {e}")
# #             return None
# #         finally:
# #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# #     def analyze_audio(self, audio_path):
# #         self.log(f"Analyzing: {Path(audio_path).name}...")
# #         temp_wav = audio_path + ".temp_analysis.wav"
# #         if not self.convert_to_wav(audio_path, temp_wav): return None, None
# #         try:
# #             y, sr = librosa.load(temp_wav, sr=None, duration=60)
# #             tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
# #             bpm = round(tempo) if np.isscalar(tempo) else round(tempo[0])
# #             chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
# #             key = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][np.argmax(np.mean(chroma, axis=1))]
# #             self.log(f"BPM: {bpm} | Key: {key}")
# #             return bpm, key
# #         except Exception as e:
# #             self.log(f"Analysis error: {e}")
# #             return None, None
# #         finally:
# #             if os.path.exists(temp_wav): os.remove(temp_wav)

# #     def render_custom_karaoke(self, video_path, audio_path, ass_path, output_dir):
# #         self.log("Rendering custom karaoke video...")
# #         out_name = f"Karaoke_Edit_{uuid.uuid4().hex[:6]}.mp4"
# #         output_file = os.path.join(output_dir, out_name)

# #         safe_audio = os.path.join(output_dir, "safe_audio.wav")
# #         if not self.convert_to_wav(audio_path, safe_audio): return None
        
# #         safe_ass = os.path.join(output_dir, "subs.ass")
# #         shutil.copy2(ass_path, safe_ass)

# #         cmd = [self.ffmpeg_exe, '-y']
# #         if video_path and os.path.exists(video_path): cmd.extend(['-i', video_path])
# #         else: cmd.extend(['-f', 'lavfi', '-i', 'color=c=black:s=1920x1080:r=30'])

# #         cmd.extend(['-i', safe_audio])
# #         cmd.extend([
# #             '-vf', f"ass='{os.path.basename(safe_ass)}'",
# #             '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'ultrafast', 
# #             '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-shortest', output_file
# #         ])

# #         cwd = os.getcwd()
# #         os.chdir(output_dir)
# #         try:
# #             subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
# #             self.log(f"Video created successfully: {out_name}")
# #             return output_file
# #         except Exception as e:
# #             self.log(f"Video rendering error: {e}")
# #             return None
# #         finally:
# #             os.chdir(cwd)
# #             if os.path.exists(safe_audio): os.remove(safe_audio)
# #             if os.path.exists(safe_ass): os.remove(safe_ass)

# #     def update_ass_style(self, ass_content, font_size, primary_color):
# #         if primary_color.startswith('#'):
# #             r = primary_color[1:3]; g = primary_color[3:5]; b = primary_color[5:7]
# #             ass_color = f"&H00{b}{g}{r}"
# #         else: ass_color = "&H00FFFFFF" 
# #         ass_content = re.sub(r"Fontsize,\s*\d+", f"Fontsize,{int(font_size)}", ass_content)
# #         ass_content = re.sub(r"PrimaryColour,\s*&H[0-9A-Fa-f]+", f"PrimaryColour,{ass_color}", ass_content)
# #         return ass_content

# #     def _sanitize(self, name): return re.sub(r'[^\w\-_]', '', name.replace(' ', '_'))
# #     def _fmt_time(self, s): return f"{int(s//60)}:{int(s%60):02}"
# #     def _fmt_ass(self, s): return f"{int(s//3600)}:{int((s%3600)//60):02}:{int(s%60):02}.{int((s-int(s))*100):02}"
    
# #     def _generate_ass(self, full_results, output_path, is_hebrew=False):
# #         header = """[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Karaoke,Arial,80,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,100,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"""
# #         events = []
# #         curr = []
# #         start_t = None
        
# #         for item in full_results:
# #             t1, t2 = item['timestamp']
# #             word_text = item['text'].strip()
            
# #             # === תיקון עברית: הפעלת get_display על המילה ===
# #             if is_hebrew:
# #                 try:
# #                     word_text = get_display(word_text)
# #                 except:
# #                     pass # אם הספריה לא קיימת או נכשלה
# #             # ===============================================

# #             if start_t is None: start_t = t1
# #             curr.append({'w': word_text, 's': t1, 'e': t2})
            
# #             # זיהוי סוף משפט (תומך גם בפיסוק עברי אם יש)
# #             if item['text'].strip().endswith(('.', '?', '!', ',')) or len(curr) > 8:
# #                 line = ""
# #                 curr_t = start_t
# #                 for w in curr:
# #                     gap = int((w['s'] - curr_t)*100)
# #                     dur = int((w['e'] - w['s'])*100)
# #                     if gap > 0: line += f"{{\k{gap}}}"
# #                     line += f"{{\kf{dur}}}{w['w']} "
# #                     curr_t = w['e']
                
# #                 events.append(f"Dialogue: 0,{self._fmt_ass(start_t)},{self._fmt_ass(t2)},Karaoke,,0,0,0,,{line.strip()}")
# #                 curr = []
# #                 start_t = None
        
# #         if curr: # שאריות
# #              line = ""
# #              curr_t = start_t
# #              for w in curr:
# #                 dur = int((w['e'] - w['s'])*100)
# #                 line += f"{{\kf{dur}}}{w['w']} "
# #              events.append(f"Dialogue: 0,{self._fmt_ass(start_t)},{self._fmt_ass(curr[-1]['e'])},Karaoke,,0,0,0,,{line.strip()}")

# #         with open(output_path, "w", encoding="utf-8-sig") as f:
# #             f.write(header + "\n".join(events))
# import os
# import re
# import shutil
# import subprocess
# import logging
# import numpy as np
# from pathlib import Path
# import imageio_ffmpeg
# import uuid

# try:
#     import yt_dlp
#     import torch
#     import soundfile as sf
#     from transformers import pipeline
#     from audio_separator.separator import Separator
#     import librosa
# except ImportError as e:
#     print(f"Backend Error: {e}")

# logging.getLogger("audio_separator").setLevel(logging.ERROR)
# logging.getLogger("transformers").setLevel(logging.ERROR)

# class BackendProcessor:
#     def __init__(self, log_func):
#         self.log = log_func
#         self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

#     def copy_to_custom_path(self, source_file, target_folder):
#         if not target_folder or not os.path.exists(source_file):
#             return source_file 

#         try:
#             os.makedirs(target_folder, exist_ok=True)
#             filename = os.path.basename(source_file)
#             destination = os.path.join(target_folder, filename)
            
#             base, ext = os.path.splitext(filename)
#             counter = 1
#             while os.path.exists(destination):
#                 destination = os.path.join(target_folder, f"{base}_{counter}{ext}")
#                 counter += 1
                
#             shutil.copy2(source_file, destination)
#             self.log(f"Saved copy to: {destination}")
#             return destination
#         except Exception as e:
#             self.log(f"Error saving to custom path: {e}")
#             return source_file

#     def convert_to_wav(self, input_path, output_path):
#         safe_temp_source = None
#         try:
#             file_ext = os.path.splitext(input_path)[1]
#             if not file_ext: file_ext = ".wav"
            
#             # Create safe temporary name
#             safe_name = f"safe_source_{uuid.uuid4().hex[:8]}{file_ext}"
#             safe_temp_source = os.path.join(os.path.dirname(output_path), safe_name)
#             shutil.copy2(input_path, safe_temp_source)

#             cmd = [
#                 self.ffmpeg_exe, '-y',
#                 '-i', safe_temp_source,
#                 '-ar', '16000',
#                 '-ac', '1',
#                 '-c:a', 'pcm_s16le',
#                 output_path
#             ]
            
#             startupinfo = None
#             if os.name == 'nt':
#                 startupinfo = subprocess.STARTUPINFO()
#                 startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

#             subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, startupinfo=startupinfo)
#             return True
#         except Exception as e:
#             self.log(f"Error converting file: {e}")
#             return False
#         finally:
#             if safe_temp_source and os.path.exists(safe_temp_source):
#                 try: os.remove(safe_temp_source)
#                 except: pass

#     def download(self, url, output_dir, format_type='wav'):
#         self.log(f"Downloading from URL: {url}")
#         Path(output_dir).mkdir(parents=True, exist_ok=True)
        
#         opts = {
#             'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
#             'quiet': True,
#             'no_warnings': True,
#             'noplaylist': True,
#             'ffmpeg_location': self.ffmpeg_exe
#         }
        
#         if format_type == 'mp4':
#             opts.update({'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'})
#         else:
#             opts.update({
#                 'format': 'bestaudio/best',
#                 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}]
#             })

#         try:
#             with yt_dlp.YoutubeDL(opts) as ydl:
#                 info = ydl.extract_info(url, download=True)
#                 fname = ydl.prepare_filename(info)
#                 final = Path(fname).with_suffix('.wav') if format_type == 'wav' else Path(fname)
#                 self.log(f"Download complete: {final.name}")
#                 return str(final)
#         except Exception as e:
#             self.log(f"Download error: {e}")
#             return None

#     def separate(self, audio_path, output_dir, mode="2_stems"):
#         if mode == "none": return [audio_path]

#         model_name = "Kim_Vocal_2.onnx" if mode == "2_stems" else "htdemucs_ft.yaml"
#         self.log(f"Separating stems using {model_name}...")
        
#         if torch.cuda.is_available(): torch.cuda.empty_cache()

#         temp_input = os.path.join(output_dir, "temp_sep_input.wav")
#         if not self.convert_to_wav(audio_path, temp_input): return None

#         file_stem = Path(audio_path).stem
#         safe_folder_name = self._sanitize(file_stem)
#         final_sep_dir = os.path.join(output_dir, f"Sep_{safe_folder_name}")
#         Path(final_sep_dir).mkdir(parents=True, exist_ok=True)

#         try:
#             separator = Separator(
#                 log_level=logging.ERROR,
#                 output_dir=final_sep_dir,
#                 output_single_stem=None,
#                 model_file_dir=os.path.join(output_dir, "uvr_models")
#             )
#             separator.load_model(model_filename=model_name)
#             output_files = separator.separate(temp_input)
            
#             full_paths = []
#             if mode == "2_stems":
#                 vocals_path = None; playback_path = None
#                 for f in output_files:
#                     full_path = os.path.join(final_sep_dir, f)
#                     if "(Vocals)" in f:
#                         target = os.path.join(final_sep_dir, "Vocals.wav")
#                         if os.path.exists(target): os.remove(target)
#                         os.rename(full_path, target)
#                         vocals_path = target
#                     elif "(Instrumental)" in f:
#                         target = os.path.join(final_sep_dir, "Playback.wav")
#                         if os.path.exists(target): os.remove(target)
#                         os.rename(full_path, target)
#                         playback_path = target
#                 if vocals_path and playback_path: full_paths = [vocals_path, playback_path]
#             else:
#                 for f in output_files: full_paths.append(os.path.join(final_sep_dir, f))

#             try: os.remove(temp_input)
#             except: pass
#             self.log("Separation complete.")
#             return full_paths
#         except Exception as e:
#             self.log(f"Separation error: {e}")
#             return None

#     def transcribe(self, audio_path, output_dir, lang, hf_token=None):
#         self.log(f"Preparing transcription ({lang})...")
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
#         model_id = "ivrit-ai/whisper-large-v3-turbo" if lang == "he" else "openai/whisper-large-v3-turbo"

#         clean_path = os.path.join(output_dir, "clean_whisper.wav")
#         if not self.convert_to_wav(audio_path, clean_path): return None

#         try:
#             if torch.cuda.is_available(): torch.cuda.empty_cache()
#             self.log(f"Loading model {model_id}...")
#             pipe = pipeline(
#                 "automatic-speech-recognition", model=model_id, tokenizer=model_id, feature_extractor=model_id,
#                 torch_dtype=torch_dtype, device=device, chunk_length_s=30, batch_size=8, return_timestamps="word", token=hf_token
#             )
            
#             data, samplerate = sf.read(clean_path)
#             total_duration = len(data) / samplerate
#             self.log(f"Audio duration: {total_duration:.2f}s. Starting...")

#             full_results = []
#             chunk_duration = 30
#             for i in range(0, int(total_duration), chunk_duration):
#                 start_sample = int(i * samplerate)
#                 end_sample = int(min((i + chunk_duration) * samplerate, len(data)))
#                 if end_sample - start_sample < samplerate: break 
#                 audio_chunk = data[start_sample:end_sample]
#                 gen_kwargs = {"language": "hebrew" if lang == "he" else "english", "task": "transcribe"}
#                 result = pipe(audio_chunk, generate_kwargs=gen_kwargs)
#                 if result.get('text'): self.log(f"[{self._fmt_time(i)}]: {result['text'].strip()}")
#                 for c in result.get('chunks', []):
#                     if 'timestamp' in c:
#                         ts = c['timestamp']
#                         if ts[0] is not None and ts[1] is not None:
#                             full_results.append({'text': c['text'], 'timestamp': (ts[0]+i, ts[1]+i)})

#             ass_path = os.path.join(output_dir, "karaoke.ass")
            
#             # === כאן השינוי: ביטלתי את הפרמטר שמבקש תיקון עברית ===
#             self._generate_ass(full_results, ass_path)
            
#             try: os.remove(clean_path)
#             except: pass
#             self.log(f"Transcription complete.")
#             return ass_path
#         except Exception as e:
#             self.log(f"Transcription error: {e}")
#             return None
#         finally:
#             if torch.cuda.is_available(): torch.cuda.empty_cache()

#     def analyze_audio(self, audio_path):
#         self.log(f"Analyzing: {Path(audio_path).name}...")
#         temp_wav = audio_path + ".temp_analysis.wav"
#         if not self.convert_to_wav(audio_path, temp_wav): return None, None
#         try:
#             y, sr = librosa.load(temp_wav, sr=None, duration=60)
#             tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
#             bpm = round(tempo) if np.isscalar(tempo) else round(tempo[0])
#             chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
#             key = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][np.argmax(np.mean(chroma, axis=1))]
#             self.log(f"BPM: {bpm} | Key: {key}")
#             return bpm, key
#         except Exception as e:
#             self.log(f"Analysis error: {e}")
#             return None, None
#         finally:
#             if os.path.exists(temp_wav): os.remove(temp_wav)

#     def render_custom_karaoke(self, video_path, audio_path, ass_path, output_dir):
#         self.log("Rendering custom karaoke video...")
#         out_name = f"Karaoke_Edit_{uuid.uuid4().hex[:6]}.mp4"
#         output_file = os.path.join(output_dir, out_name)

#         safe_audio = os.path.join(output_dir, "safe_audio.wav")
#         if not self.convert_to_wav(audio_path, safe_audio): return None
        
#         safe_ass = os.path.join(output_dir, "subs.ass")
#         shutil.copy2(ass_path, safe_ass)

#         cmd = [self.ffmpeg_exe, '-y']
#         if video_path and os.path.exists(video_path): cmd.extend(['-i', video_path])
#         else: cmd.extend(['-f', 'lavfi', '-i', 'color=c=black:s=1920x1080:r=30'])

#         cmd.extend(['-i', safe_audio])
#         cmd.extend([
#             '-vf', f"ass='{os.path.basename(safe_ass)}'",
#             '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'ultrafast', 
#             '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-shortest', output_file
#         ])

#         cwd = os.getcwd()
#         os.chdir(output_dir)
#         try:
#             subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
#             self.log(f"Video created successfully: {out_name}")
#             return output_file
#         except Exception as e:
#             self.log(f"Video rendering error: {e}")
#             return None
#         finally:
#             os.chdir(cwd)
#             if os.path.exists(safe_audio): os.remove(safe_audio)
#             if os.path.exists(safe_ass): os.remove(safe_ass)

#     def update_ass_style(self, ass_content, font_size, primary_color):
#         if primary_color.startswith('#'):
#             r = primary_color[1:3]; g = primary_color[3:5]; b = primary_color[5:7]
#             ass_color = f"&H00{b}{g}{r}"
#         else: ass_color = "&H00FFFFFF" 
#         ass_content = re.sub(r"Fontsize,\s*\d+", f"Fontsize,{int(font_size)}", ass_content)
#         ass_content = re.sub(r"PrimaryColour,\s*&H[0-9A-Fa-f]+", f"PrimaryColour,{ass_color}", ass_content)
#         return ass_content

#     def _sanitize(self, name): return re.sub(r'[^\w\-_]', '', name.replace(' ', '_'))
#     def _fmt_time(self, s): return f"{int(s//60)}:{int(s%60):02}"
#     def _fmt_ass(self, s): return f"{int(s//3600)}:{int((s%3600)//60):02}:{int(s%60):02}.{int((s-int(s))*100):02}"
    
#     def _generate_ass(self, full_results, output_path):
#         # כותרת ASS נקייה
#         header = """[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Karaoke,Arial,80,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,100,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"""
#         events = []
#         curr = []
#         start_t = None
        
#         for item in full_results:
#             t1, t2 = item['timestamp']
            
#             # ניקוי רווחים מיותרים מהמילה
#             word_text = item['text'].strip()
            
#             if start_t is None: start_t = t1
#             curr.append({'w': word_text, 's': t1, 'e': t2})
            
#             # חיתוך משפט (לפי סימני פיסוק או אורך)
#             if item['text'].strip().endswith(('.', '?', '!', ',')) or len(curr) > 7:
#                 # הרכבת משפט רגיל (ללא קריוקי מיוחד) עבור עברית תקינה
#                 # אנחנו פשוט מחברים את המילים עם רווחים
#                 full_sentence = " ".join([w['w'] for w in curr])
                
#                 # כתיבת השורה לקובץ (ללא תגי \k שעלולים לשבש)
#                 events.append(f"Dialogue: 0,{self._fmt_ass(start_t)},{self._fmt_ass(t2)},Karaoke,,0,0,0,,{full_sentence}")
                
#                 curr = []
#                 start_t = None
        
#         # שאריות
#         if curr:
#             full_sentence = " ".join([w['w'] for w in curr])
#             events.append(f"Dialogue: 0,{self._fmt_ass(start_t)},{self._fmt_ass(curr[-1]['e'])},Karaoke,,0,0,0,,{full_sentence}")

#         with open(output_path, "w", encoding="utf-8-sig") as f:
#             f.write(header + "\n".join(events))
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

    def download(self, url, output_dir, format_type='wav'):
        self.log(f"Downloading from URL: {url}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        opts = {
            'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
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
                fname = ydl.prepare_filename(info)
                final = Path(fname).with_suffix('.wav') if format_type == 'wav' else Path(fname)
                self.log(f"Download complete: {final.name}")
                return str(final)
        except Exception as e:
            self.log(f"Download error: {e}")
            return None

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
    
    def _generate_ass(self, full_results, output_path):
        # כותרת בסיסית. הפונקציה update_ass_style תדרוס את השורה הרלוונטית בעת העריכה.
        header = """[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Karaoke,Arial,80,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,100,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"""
        events = []
        curr = []
        start_t = None
        for item in full_results:
            t1, t2 = item['timestamp']
            if start_t is None: start_t = t1
            curr.append({'w': item['text'].strip(), 's': t1, 'e': t2})
            if item['text'].strip().endswith(('.', '?', '!')) or len(curr) > 7:
                # ללא קריוקי צבעוני (למניעת בעיות עברית)
                full_sentence = " ".join([w['w'] for w in curr])
                events.append(f"Dialogue: 0,{self._fmt_ass(start_t)},{self._fmt_ass(t2)},Karaoke,,0,0,0,,{full_sentence}")
                curr = []
                start_t = None
        
        if curr:
            full_sentence = " ".join([w['w'] for w in curr])
            events.append(f"Dialogue: 0,{self._fmt_ass(start_t)},{self._fmt_ass(curr[-1]['e'])},Karaoke,,0,0,0,,{full_sentence}")

        with open(output_path, "w", encoding="utf-8-sig") as f:
            f.write(header + "\n".join(events))