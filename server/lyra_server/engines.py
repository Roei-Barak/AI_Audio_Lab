"""Processing engines.

`ai`   - real models, same stack as the existing Gradio app (logic.py):
         yt-dlp, audio-separator (Kim_Vocal_2 for 2 stems, htdemucs_ft for 4), Whisper via transformers
         (ivrit-ai/whisper-large-v3-turbo for Hebrew) with word timestamps. Imports are lazy so the server starts
         without them.
`mock` - fast local stand-ins for development and tests: phase-cancellation "instrumental", evenly spaced
         placeholder words. Rendering with ffmpeg is real in both engines.
Every heavy step is a subprocess or a model call that can only be stopped between steps; cancellation is
checked between stages.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Callable

from .config import Config

Log = Callable[[str], None]


class SourceError(Exception):
    """The source cannot be imported (private, DRM, unsupported). Shown to the user as-is."""


class StageError(Exception):
    """A processing crash: completed stages are kept and the job can resume."""


def run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise StageError((p.stderr or p.stdout or "").strip().splitlines()[-1:] or ["ffmpeg failed"])
    return p


def probe(cfg: Config, path: Path) -> dict:
    p = subprocess.run(
        [cfg.ffprobe, "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height", "-of", "json", str(path)],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise SourceError("הקובץ לא נקרא. ייתכן שהוא פגום או בפורמט לא נתמך.")
    info = json.loads(p.stdout or "{}")
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    if not has_audio:
        raise SourceError("בקובץ אין ערוץ אודיו.")
    return {
        "duration": float(info.get("format", {}).get("duration") or 0),
        "hasVideo": v is not None,
        "width": v and v.get("width"),
        "height": v and v.get("height"),
    }


# ---------------- import ----------------

PROTECTED = re.compile(r"private|members|drm|sign in|age|copyright|unavailable", re.I)


def youtube_info(url: str) -> dict:
    try:
        import yt_dlp  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise SourceError("yt-dlp לא מותקן בשרת.") from e
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        msg = str(e)
        if PROTECTED.search(msg):
            raise SourceError("לא ניתן לייבא את המקור: הסרטון פרטי, מוגבל או מוגן. Lyra לא עוקפת הגנות. אפשר להעלות קובץ מקומי שמותר לך להשתמש בו.") from e
        raise SourceError("לא ניתן לקרוא את הקישור: " + msg[-160:]) from e
    return {"title": info.get("title") or "YouTube", "artist": info.get("uploader") or "", "duration": float(info.get("duration") or 0), "id": info.get("id")}


def youtube_download(cfg: Config, url: str, dest_dir: Path, log: Log) -> Path:
    import yt_dlp  # type: ignore

    out = dest_dir / "source.%(ext)s"
    opts = {
        "outtmpl": str(out),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "ffmpeg_location": cfg.ffmpeg,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
    }
    log("downloading " + url)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:
        if PROTECTED.search(str(e)):
            raise SourceError("המקור מוגן או פרטי. Lyra לא עוקפת הגנות.") from e
        raise StageError("download failed: " + str(e)[-200:]) from e
    found = sorted(dest_dir.glob("source.*"))
    if not found:
        raise StageError("download produced no file")
    return found[0]


def extract_wav(cfg: Config, src: Path, dest: Path, rate: int = 44100, mono: bool = False) -> Path:
    """Decodes to PCM WAV with the source timeline untouched (no trimming, no resync)."""
    cmd = [cfg.ffmpeg, "-y", "-v", "error", "-i", str(src), "-vn", "-ar", str(rate), "-ac", "1" if mono else "2", "-c:a", "pcm_s16le", str(dest)]
    run(cmd)
    return dest


# ---------------- separation ----------------

def separate(cfg: Config, wav: Path, stems_dir: Path, stems: int, profile: str, log: Log) -> dict[str, Path]:
    stems_dir.mkdir(parents=True, exist_ok=True)
    if cfg.engine == "mock":
        log("mock separation (phase cancellation)")
        voc = stems_dir / "vocals.wav"
        inst = stems_dir / "instrumental.wav"
        # center channel ~ vocals; side channels ~ accompaniment. A real but crude karaoke trick.
        run([cfg.ffmpeg, "-y", "-v", "error", "-i", str(wav), "-af", "pan=stereo|c0=0.5*c0+0.5*c1|c1=0.5*c0+0.5*c1", str(voc)])
        run([cfg.ffmpeg, "-y", "-v", "error", "-i", str(wav), "-af", "pan=stereo|c0=c0-c1|c1=c1-c0", str(inst)])
        out = {"vocals": voc, "instrumental": inst}
        if stems == 4:
            for k, f in (("drums", "lowpass=f=150"), ("bass", "bandpass=f=100"), ("other", "highpass=f=2000")):
                p = stems_dir / f"{k}.wav"
                run([cfg.ffmpeg, "-y", "-v", "error", "-i", str(inst), "-af", f, str(p)])
                out[k] = p
        return out

    try:
        from audio_separator.separator import Separator  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise StageError("audio-separator לא מותקן בשרת (pip install audio-separator).") from e
    import logging

    models = cfg.data_dir / "uvr_models"
    tmp = stems_dir / "_tmp"
    tmp.mkdir(exist_ok=True)
    out: dict[str, Path] = {}
    try:
        model = "Kim_Vocal_2.onnx" if profile == "איכותי" else "UVR-MDX-NET-Inst_HQ_3.onnx"
        log("separation model " + model)
        sep = Separator(log_level=logging.ERROR, output_dir=str(tmp), model_file_dir=str(models))
        sep.load_model(model)
        for f in sep.separate(str(wav)):
            src = tmp / Path(f).name
            low = src.name.lower()
            if "vocal" in low and "no" not in low and "inst" not in low:
                out["vocals"] = src.rename(stems_dir / "vocals.wav")
            elif "instrumental" in low or "no_vocals" in low or "inst" in low:
                out["instrumental"] = src.rename(stems_dir / "instrumental.wav")
        del sep
        if stems == 4:
            log("4-stem separation htdemucs_ft")
            sep4 = Separator(log_level=logging.ERROR, output_dir=str(tmp), model_file_dir=str(models))
            sep4.load_model("htdemucs_ft.yaml")
            for f in sep4.separate(str(wav)):
                src = tmp / Path(f).name
                for k in ("drums", "bass", "other"):
                    if k in src.name.lower():
                        out[k] = src.rename(stems_dir / f"{k}.wav")
            del sep4
    except StageError:
        raise
    except Exception as e:
        raise StageError("separation crashed: " + str(e)[-200:]) from e
    finally:
        _free_gpu()
        for f in tmp.glob("*"):
            f.unlink(missing_ok=True)
        tmp.rmdir()
    if "vocals" not in out or "instrumental" not in out:
        raise StageError("separation did not produce vocals + instrumental")
    return out


def _free_gpu() -> None:
    try:
        import gc

        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass


# ---------------- transcription ----------------

def transcribe(cfg: Config, vocals: Path, language: str, profile: str, duration: float, log: Log) -> list[dict]:
    """Returns word chunks [{w, s, e, conf}] with timings only where the model produced them."""
    if cfg.engine == "mock":
        log("mock transcription (placeholder words)")
        words = "זוהי תמלול לדוגמה מהמנוע המקומי כדי לבדוק את הזרימה מקצה לקצה".split()
        n = max(1, int((duration - 4) // 0.6))
        out, t = [], 2.0
        for i in range(min(n, 120)):
            w = words[i % len(words)]
            out.append({"w": w, "s": round(t, 2), "e": round(t + 0.45, 2), "conf": "med"})
            t += 0.6 if (i + 1) % 7 else 1.4
        return out

    try:
        import torch  # type: ignore
        from transformers import pipeline  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise StageError("transformers/torch לא מותקנים בשרת.") from e
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if language == "he":
        model_id = "ivrit-ai/whisper-large-v3-turbo"
    else:
        model_id = "openai/whisper-large-v3" if profile == "איכותי" else "openai/whisper-large-v3-turbo"
    log(f"transcription {model_id} on {device}")
    wav16 = vocals.with_name("vocals_16k.wav")
    extract_wav(cfg, vocals, wav16, rate=16000, mono=True)
    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            chunk_length_s=30,
            return_timestamps="word",
        )
        kw = {} if language == "auto" else {"generate_kwargs": {"language": "hebrew" if language == "he" else "english"}}
        result = pipe(str(wav16), **kw)
        del pipe
    except Exception as e:
        raise StageError("transcription crashed: " + str(e)[-200:]) from e
    finally:
        _free_gpu()
        wav16.unlink(missing_ok=True)
    out = []
    for c in result.get("chunks", []):
        text = (c.get("text") or "").strip()
        s, e = c.get("timestamp") or (None, None)
        if not text or s is None:
            continue
        for part in text.split():
            out.append({"w": part, "s": round(float(s), 2), "e": round(float(e if e is not None else s + 0.3), 2), "conf": "high"})
    return out


def words_to_lines(words: list[dict], max_words: int = 7, gap: float = 0.9) -> list[dict]:
    """Groups words into lines on punctuation, pauses or length."""
    lines: list[list[dict]] = []
    cur: list[dict] = []
    for i, w in enumerate(words):
        if cur and (w["s"] - cur[-1]["e"] > gap):
            lines.append(cur)
            cur = []
        cur.append(w)
        if w["w"].endswith((".", "?", "!")) or len(cur) >= max_words:
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    out = []
    for i, ws in enumerate(lines):
        out.append({
            "id": f"l{i:04d}",
            "start": ws[0]["s"],
            "end": ws[-1]["e"],
            "speaker": "1",
            "manual": False,
            "words": [{"id": f"w{i:04d}_{j:02d}", "w": w["w"], "s": w["s"], "e": w["e"], "conf": w.get("conf", "high"), "manual": False} for j, w in enumerate(ws)],
        })
    return out


# ---------------- render ----------------

ASS_COLORS = {"accent": "&H00D98491", "soft": "&H00FAF0F5", "light": "&H00FEF5F3", "deep": "&H00B05A72"}  # BGR of the UI palette


def _ass_t(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def build_ass(project: dict, width: int, height: int) -> str:
    """ASS karaoke with \\kf word sweeps when word timing exists (line-only timing otherwise)."""
    st = project.get("style", {})
    size = int(st.get("size", 44) * height / 1080 * (1.2 if st.get("template") == "bold" else 1) * (1.6 if st.get("aspect", "16:9") != "16:9" else 1))
    align = {"top": 8, "middle": 5, "bottom": 2}.get(st.get("position", "bottom"), 2)
    hi = ASS_COLORS.get(st.get("color", "accent"), ASS_COLORS["accent"])
    base = "&H00A0A0A0" if st.get("template") == "minimal" else "&H00FFFFFF"
    outline = 3 if st.get("outline", True) else 0
    shadow = 2 if st.get("shadow", True) else 0
    margin_v = int(height * (0.09 if align == 2 else 0.14))
    head = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\n"
        f"PlayResX: {width}\nPlayResY: {height}\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Karaoke,Noto Sans Hebrew,{size},{hi},{base},&H00101010,&H90000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},{align},60,60,{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    word_level = project.get("timing") == "word"
    ev = []
    for line in project.get("lines", []):
        ws = line.get("words", [])
        if not ws:
            continue
        a, b = ws[0]["s"], ws[-1]["e"]
        start = max(0.0, a - 0.4)
        if word_level:
            parts, t = [], start
            for w in ws:
                lead = max(0.0, w["s"] - t)
                if lead > 0.01:
                    parts.append(f"{{\\k{int(round(lead * 100))}}}")
                parts.append(f"{{\\kf{max(1, int(round((w['e'] - w['s']) * 100)))}}}{_esc(w['w'])} ")
                t = w["e"]
            text = "".join(parts).rstrip()
        else:
            text = " ".join(_esc(w["w"]) for w in ws)
        ev.append(f"Dialogue: 0,{_ass_t(start)},{_ass_t(b + 0.3)},Karaoke,,0,0,0,,{text}")
    return head + "\n".join(ev) + "\n"


def _esc(s: str) -> str:
    return s.replace("{", "(").replace("}", ")").replace("\\", "/")


RES = {"1080p": {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}, "720p": {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (720, 720)}}


def render(cfg: Config, project: dict, source: Path, instrumental: Path, out: Path, has_video: bool, duration: float, log: Log,
           on_pct: Callable[[float], None] | None = None, should_stop: Callable[[], bool] | None = None) -> Path:
    """Original video (muted) + instrumental + burned-in karaoke subtitles -> MP4."""
    st = project.get("style", {})
    w, h = RES.get(project.get("export", {}).get("res", "1080p"), RES["1080p"]).get(st.get("aspect", "16:9"), (1920, 1080))
    ass = out.parent / "karaoke.ass"
    ass.write_text(build_ass(project, w, h), encoding="utf-8")
    use_video = has_video and st.get("background", "video") == "video"
    bg = "0x161826" if st.get("background") == "solid" else "0x2a2552"
    inputs: list[str]
    if use_video:
        inputs = ["-i", str(source)]
        vf = f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,ass={_ff_path(ass)}[v]"
    else:
        inputs = ["-f", "lavfi", "-t", f"{duration:.3f}", "-i", f"color=c={bg}:s={w}x{h}:r=30"]
        vf = f"[0:v]ass={_ff_path(ass)}[v]"
    cmd = [cfg.ffmpeg, "-y", "-v", "error", *inputs, "-i", str(instrumental), "-filter_complex", vf, "-map", "[v]", "-map", "1:a",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)]
    log("render " + out.name)
    if on_pct and should_stop:
        run_progress(cmd, duration, on_pct, should_stop)
    else:
        run(cmd)
    return out


def _ff_path(p: Path) -> str:
    # escape for the ffmpeg filter graph
    return "'" + str(p).replace("\\", "/").replace(":", r"\:").replace("'", r"\'") + "'"


# ---------------- pasted lyrics ----------------

def _norm(w: str) -> str:
    return re.sub(r"[^\w֐-׿]", "", w).lower()


def align_lyrics(text: str, asr_words: list[dict], duration: float, vocal_spans: list[tuple[float, float]] | None = None) -> tuple[list[dict], str]:
    """Maps user-pasted lyrics onto timings.

    With transcription words: sequence-align the pasted words to the ASR words (difflib) and copy their
    timings; unmatched words are interpolated between their matched neighbours ("word" timing, conf med/low).
    Without them: each line is spread evenly over the song ("line" timing only: no invented word timing).
    """
    import difflib

    lines = [ln.split() for ln in text.splitlines() if ln.strip()]
    flat = [(li, w) for li, ws in enumerate(lines) for w in ws]
    if not flat:
        return [], "line"
    if not asr_words:
        span0, span1 = (vocal_spans[0][0], vocal_spans[-1][1]) if vocal_spans else (0.5, max(1.0, duration - 0.5))
        step = (span1 - span0) / len(lines)
        out = []
        for i, ws in enumerate(lines):
            s, e = span0 + i * step, span0 + (i + 1) * step - 0.2
            d = (e - s) / len(ws)
            out.append([{"w": w, "s": round(s + j * d, 2), "e": round(s + (j + 1) * d - 0.05, 2), "conf": "low"} for j, w in enumerate(ws)])
        return _pack(out), "line"
    a = [_norm(w) for _, w in flat]
    b = [_norm(w["w"]) for w in asr_words]
    times: list[tuple[float, float, str] | None] = [None] * len(flat)
    for blk in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            aw = asr_words[blk.b + k]
            times[blk.a + k] = (aw["s"], aw["e"], aw.get("conf", "high"))
    # interpolate gaps between matched anchors
    i = 0
    while i < len(times):
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < len(times) and times[j] is None:
            j += 1
        left = times[i - 1][1] if i > 0 else (asr_words[0]["s"] if asr_words else 0.0)
        right = times[j][0] if j < len(times) else max(left + 0.5 * (j - i), asr_words[-1]["e"])
        d = max(0.12, (right - left) / (j - i))
        for k in range(i, j):
            s = left + (k - i) * d
            times[k] = (round(s, 2), round(s + d * 0.85, 2), "low")
        i = j
    out: list[list[dict]] = [[] for _ in lines]
    for (li, w), t in zip(flat, times):
        out[li].append({"w": w, "s": t[0], "e": t[1], "conf": "med" if t[2] != "low" else "low"})
    return _pack(out), "word"


def _pack(groups: list[list[dict]]) -> list[dict]:
    res = []
    for i, ws in enumerate(groups):
        if not ws:
            continue
        res.append({"id": f"l{i:04d}", "start": ws[0]["s"], "end": ws[-1]["e"], "speaker": "1", "manual": True,
                    "words": [{"id": f"w{i:04d}_{j:02d}", "w": w["w"], "s": w["s"], "e": w["e"], "conf": w["conf"], "manual": False} for j, w in enumerate(ws)]})
    return res


# ---------------- cancellable ffmpeg with real progress ----------------

def run_progress(cmd: list[str], duration: float, on_pct: Callable[[float], None], should_stop: Callable[[], bool]) -> None:
    """Runs ffmpeg with -progress, reporting real percent and killing the process on cancel."""
    import time as _t

    full = cmd[:1] + ["-progress", "pipe:1", "-nostats"] + cmd[1:]
    p = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    last = 0.0
    assert p.stdout
    for line in p.stdout:
        if should_stop():
            p.kill()
            p.wait()
            raise InterruptedError()
        if line.startswith("out_time_us=") and duration > 0:
            try:
                us = int(line.split("=", 1)[1])
            except ValueError:
                continue
            pct = max(0.0, min(99.0, us / 1e6 / duration * 100))
            if pct - last >= 1 or _t.time() % 2 < 0.05:
                on_pct(pct)
                last = pct
    p.wait()
    if p.returncode != 0:
        err = (p.stderr.read() if p.stderr else "").strip().splitlines()[-1:] or ["ffmpeg failed"]
        raise StageError(err[0])
