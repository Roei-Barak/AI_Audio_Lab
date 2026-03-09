"""
api/server.py — FastAPI backend for server/client mode.

Exposes the karaoke pipeline as a REST API with real-time progress via
Server-Sent Events (SSE).  A web frontend or a remote Gradio client can
talk to this server instead of running the heavy models locally.

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
    # or
    python api/server.py

Endpoints:
    GET  /health                         Health check + system info
    POST /info                           Get YouTube video metadata
    POST /download                       Download video/audio
    POST /separate                       Separate audio stems
    POST /transcribe                     Transcribe audio (SSE stream)
    POST /render                         Render karaoke video
    POST /pipeline                       Full pipeline (SSE stream)
    POST /analyze                        BPM + key analysis
    GET  /status                         Resource manager status
    GET  /files/{filename}               Serve output files
    DELETE /files/{filename}             Delete an output file

All POST requests accept JSON bodies.  File paths returned are relative to
WORK_DIR; use the /files/{filename} endpoint to download them.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.backend import BackendProcessor, manager
from core.config import DEVICE, WORK_DIR
from modules.downloader import download, get_info
from modules.separator import separate
from modules.transcriber import transcribe
from modules.renderer import render

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="🎤 Karaoke Studio Pro — API",
    description=__doc__,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

bp = BackendProcessor()

# Serve output files
os.makedirs(WORK_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=WORK_DIR), name="files")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class InfoRequest(BaseModel):
    url: str

class DownloadRequest(BaseModel):
    url: str
    fmt: str = "wav"            # "wav" | "mp4"
    output_dir: Optional[str] = None

class SeparateRequest(BaseModel):
    audio_path: str             # path relative to WORK_DIR or absolute
    mode: int = 2               # 2 | 4
    output_dir: Optional[str] = None
    force: bool = False

class TranscribeRequest(BaseModel):
    audio_path: str
    lang: str = "he"            # "he" | "en" | "auto"
    output_formats: list[str] = ["ass", "srt"]
    title: Optional[str] = None
    output_dir: Optional[str] = None
    force: bool = False

class RenderRequest(BaseModel):
    video_path: str
    audio_path: str
    subtitles_path: str
    output_dir: Optional[str] = None
    output_name: Optional[str] = None
    use_bidi: bool = False
    font_size: Optional[int] = None
    color_hex: Optional[str] = None
    position: str = "bottom"    # "top" | "center" | "bottom"
    force: bool = False

class PipelineRequest(BaseModel):
    url: str
    lang: str = "he"
    output_formats: list[str] = ["ass", "srt"]
    save_4_stems: bool = False
    use_bidi: bool = False
    force: bool = False

class AnalyzeRequest(BaseModel):
    audio_path: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(p: str) -> str:
    """Resolve a path that may be relative to WORK_DIR."""
    if os.path.isabs(p):
        return p
    candidate = os.path.join(WORK_DIR, p)
    if os.path.exists(candidate):
        return candidate
    return p


def _rel(path: Optional[str]) -> Optional[str]:
    """Return a path relative to WORK_DIR for client consumption."""
    if path is None:
        return None
    try:
        return os.path.relpath(path, WORK_DIR)
    except ValueError:
        return path


async def _sse_stream(generator) -> AsyncGenerator[str, None]:
    """Wrap a synchronous generator into SSE data events."""
    loop = asyncio.get_event_loop()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        it = iter(generator)
        while True:
            try:
                item = await loop.run_in_executor(pool, next, it)
                yield f"data: {json.dumps(item)}\n\n"
            except StopIteration:
                yield "data: {\"done\": true}\n\n"
                break
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check — returns system info."""
    info: dict = {
        "status": "ok",
        "device": DEVICE,
        "work_dir": WORK_DIR,
        "resource_status": manager.get_status(),
    }
    if DEVICE == "cuda":
        try:
            import torch
            free, total = torch.cuda.mem_get_info()
            info["vram_free_gb"] = round(free / 1024 ** 3, 2)
            info["vram_total_gb"] = round(total / 1024 ** 3, 2)
            info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return info


@app.get("/status")
def status():
    return {"status": manager.get_status(), "active_tasks": manager.active_tasks}


@app.post("/info")
def video_info(req: InfoRequest):
    """Return YouTube video metadata without downloading."""
    logs: list[str] = []
    info = get_info(req.url, logs)
    if not info:
        raise HTTPException(status_code=404, detail="לא נמצא מידע לכתובת זו")
    return {"info": info, "logs": logs}


@app.post("/download")
def download_endpoint(req: DownloadRequest):
    """Download a YouTube video/audio."""
    logs: list[str] = []
    out_dir = req.output_dir or WORK_DIR
    result = download(req.url, output_dir=out_dir, fmt=req.fmt, logs=logs)
    if not result:
        raise HTTPException(status_code=500, detail="\n".join(logs))
    return {"path": result, "relative": _rel(result), "logs": logs}


@app.post("/separate")
def separate_endpoint(req: SeparateRequest):
    """Separate audio into stems."""
    audio = _resolve_path(req.audio_path)
    if not os.path.exists(audio):
        raise HTTPException(status_code=404, detail=f"קובץ לא נמצא: {audio}")
    logs: list[str] = []
    vocals, playback = separate(
        audio,
        output_dir=req.output_dir,
        mode=req.mode,
        force=req.force,
        logs=logs,
    )
    if not vocals:
        raise HTTPException(status_code=500, detail="\n".join(logs))
    return {
        "vocals": vocals, "vocals_rel": _rel(vocals),
        "playback": playback, "playback_rel": _rel(playback),
        "logs": logs,
    }


@app.post("/transcribe")
def transcribe_endpoint(req: TranscribeRequest):
    """
    Transcribe audio and return output file paths.

    For streaming progress use /transcribe/stream.
    """
    audio = _resolve_path(req.audio_path)
    if not os.path.exists(audio):
        raise HTTPException(status_code=404, detail=f"קובץ לא נמצא: {audio}")
    logs: list[str] = []
    results = transcribe(
        audio,
        output_dir=req.output_dir,
        lang=req.lang,
        output_formats=req.output_formats,
        title=req.title,
        force=req.force,
        logs=logs,
    )
    return {
        "files": {k: v for k, v in results.items()},
        "relative": {k: _rel(v) for k, v in results.items()},
        "logs": logs,
    }


@app.post("/transcribe/stream")
def transcribe_stream(req: TranscribeRequest):
    """Transcribe with real-time SSE progress stream."""
    audio = _resolve_path(req.audio_path)
    if not os.path.exists(audio):
        raise HTTPException(status_code=404, detail=f"קובץ לא נמצא: {audio}")

    def _gen():
        logs: list[str] = []
        events: list[dict] = []

        def cb(idx: int, total: int, text: str) -> None:
            events.append({"type": "progress", "idx": idx, "total": total, "text": text})

        results = transcribe(
            audio,
            output_dir=req.output_dir,
            lang=req.lang,
            output_formats=req.output_formats,
            title=req.title,
            force=req.force,
            progress_callback=cb,
            logs=logs,
        )
        # Yield all buffered progress events
        for ev in events:
            yield ev
        yield {
            "type": "done",
            "files": {k: v for k, v in results.items()},
            "logs": logs,
        }

    return StreamingResponse(
        _sse_stream(_gen()),
        media_type="text/event-stream",
    )


@app.post("/render")
def render_endpoint(req: RenderRequest):
    """Render a karaoke video."""
    for label, p in [
        ("וידאו", req.video_path),
        ("אודיו", req.audio_path),
        ("כתוביות", req.subtitles_path),
    ]:
        resolved = _resolve_path(p)
        if not os.path.exists(resolved):
            raise HTTPException(status_code=404, detail=f"קובץ {label} לא נמצא: {p}")

    logs: list[str] = []
    result = render(
        _resolve_path(req.video_path),
        _resolve_path(req.audio_path),
        _resolve_path(req.subtitles_path),
        output_dir=req.output_dir,
        output_name=req.output_name,
        use_bidi=req.use_bidi,
        font_size=req.font_size,
        color_hex=req.color_hex,
        position=req.position,
        force=req.force,
        logs=logs,
    )
    if not result:
        raise HTTPException(status_code=500, detail="\n".join(logs))
    return {"path": result, "relative": _rel(result), "logs": logs}


@app.post("/pipeline/stream")
def pipeline_stream(req: PipelineRequest):
    """
    Run the full pipeline with SSE progress stream.

    Stream format: {"type": "log"|"progress"|"done", ...}
    """
    def _gen():
        logs_buf: list[str] = []
        progress_buf: list[dict] = []

        def cb(idx: int, total: int, text: str) -> None:
            progress_buf.append({"idx": idx, "total": total, "text": text})

        final, log_str = bp.process_song_pipeline(
            req.url,
            lang=req.lang,
            save_4_stems=req.save_4_stems,
            use_bidi=req.use_bidi,
            force=req.force,
            output_formats=req.output_formats,
            progress_callback=cb,
        )
        for ev in progress_buf:
            yield {"type": "progress", **ev}
        for line in log_str.splitlines():
            yield {"type": "log", "text": line}
        yield {
            "type": "done",
            "success": bool(final),
            "path": final,
            "relative": _rel(final),
        }

    return StreamingResponse(
        _sse_stream(_gen()),
        media_type="text/event-stream",
    )


@app.post("/analyze")
def analyze_endpoint(req: AnalyzeRequest):
    """Detect BPM and musical key for an audio file."""
    audio = _resolve_path(req.audio_path)
    if not os.path.exists(audio):
        raise HTTPException(status_code=404, detail=f"קובץ לא נמצא: {audio}")
    result, logs = bp.analyze_audio(audio)
    return {"result": result, "logs": logs}


@app.delete("/files/{filename:path}")
def delete_file(filename: str):
    """Delete an output file (safety: must be inside WORK_DIR)."""
    target = os.path.realpath(os.path.join(WORK_DIR, filename))
    if not target.startswith(os.path.realpath(WORK_DIR)):
        raise HTTPException(status_code=403, detail="גישה אסורה")
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail="לא נמצא")
    os.remove(target)
    return {"deleted": filename}


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Karaoke Studio Pro — API Server")
    parser.add_argument("--host", default="0.0.0.0", help="כתובת האזנה (ברירת מחדל: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="פורט (ברירת מחדל: 8000)")
    parser.add_argument("--reload", action="store_true", help="Hot-reload בפיתוח")
    a = parser.parse_args()

    print(f"🚀 API Server — http://{a.host}:{a.port}")
    print(f"📖 Docs — http://{a.host}:{a.port}/docs")
    uvicorn.run("api.server:app", host=a.host, port=a.port, reload=a.reload)
