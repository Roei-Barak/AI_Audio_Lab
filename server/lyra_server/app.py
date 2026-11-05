"""Lyra Studio processing server (FastAPI).

Run:  uvicorn lyra_server.app:app --host 0.0.0.0 --port 8443   (from the server/ folder)
Env:  LYRA_ENGINE=ai|mock   LYRA_DATA_DIR=...   LYRA_TOKEN=... (optional shared access token)
Serves the built frontend (../app/dist) at / when present, so one process is the whole product.
"""
from __future__ import annotations

import os

import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, engines
from .config import Config, load
from .pipeline import JobQueue
from .storage import ConflictError, Storage, new_id

FILES = {"source": None, "vocals": "stems/vocals.wav", "instrumental": "stems/instrumental.wav", "drums": "stems/drums.wav", "bass": "stems/bass.wav", "other": "stems/other.wav", "karaoke": "render/karaoke.mp4"}


class SourceIn(BaseModel):
    url: str


class ProjectIn(BaseModel):
    project: dict
    source: dict | None = None


class SaveIn(BaseModel):
    baseRev: int | None = None
    project: dict
    force: bool = False


class JobIn(BaseModel):
    projectId: str
    kind: str = Field("process", pattern="^(process|render)$")
    profile: str = "מהיר"
    stems: int = Field(2, ge=2, le=4)
    language: str = "auto"
    simulateCrash: bool = False


class RetryIn(BaseModel):
    fast: bool = False


class MoveIn(BaseModel):
    delta: int


def create_app(cfg: Config | None = None, autostart: bool = True) -> FastAPI:
    cfg = cfg or load()
    st = Storage(cfg.data_dir)
    q = JobQueue(cfg, st, autostart=autostart)
    app = FastAPI(title="Lyra Studio server", version=__version__)
    app.state.cfg, app.state.storage, app.state.queue = cfg, st, q
    app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("LYRA_CORS", "http://localhost:5173,http://localhost:4173").split(","), allow_methods=["*"], allow_headers=["*"])

    token = os.environ.get("LYRA_TOKEN")

    def auth(authorization: str | None = Header(None)) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(401, "נדרש אסימון גישה")

    def project_or_404(pid: str) -> dict:
        try:
            return st.get_project(pid)
        except KeyError:
            raise HTTPException(404, "הפרויקט לא נמצא")

    @app.get("/api/health")
    def health():
        gpu = {"name": None, "vramGB": None, "device": "cpu"}
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                gpu = {"name": props.name, "vramGB": round(props.total_memory / 1024**3, 1), "device": "cuda"}
        except Exception:
            pass
        return {"ok": True, "version": __version__, "engine": cfg.engine, "host": cfg.host_label, "gpu": gpu, "gpuSlots": 1, "gpuBusy": q.gpu_busy(), "time": time.time()}

    # ---- sources ----
    @app.post("/api/sources/youtube", dependencies=[Depends(auth)])
    def youtube_info(body: SourceIn):
        if cfg.engine == "mock":
            return {"title": "YouTube (mock)", "artist": "", "duration": 66.0, "id": "mock"}
        try:
            info = engines.youtube_info(body.url)
        except engines.SourceError as e:
            raise HTTPException(422, str(e))
        if info["duration"] > cfg.max_duration_s:
            raise HTTPException(422, "הסרטון ארוך משלוש שעות.")
        return info

    # ---- projects ----
    @app.get("/api/projects", dependencies=[Depends(auth)])
    def list_projects():
        return st.list_projects()

    @app.post("/api/projects", dependencies=[Depends(auth)])
    def create_project(body: ProjectIn):
        doc = dict(body.project)
        doc["id"] = new_id("prj_")
        if body.source:
            doc["source"] = body.source
        doc["lines"] = doc.get("lines") if doc.get("source", {}).get("kind") == "demo" else []
        doc["processed"] = False
        return st.create_project(doc)

    @app.post("/api/projects/{pid}/upload", dependencies=[Depends(auth)])
    async def upload(pid: str, file: UploadFile = File(...)):
        project_or_404(pid)
        ext = Path(file.filename or "media").suffix.lower()[:6] or ".bin"
        d = st.project_dir(pid)
        for old in d.glob("source.*"):
            old.unlink()
        dest = d / f"source{ext}"
        size = 0
        with dest.open("wb") as f:
            while chunk := await file.read(8 * 1024 * 1024):
                size += len(chunk)
                if size > cfg.max_file_bytes:
                    f.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, "הקובץ גדול מ־20GB.")
                f.write(chunk)
        try:
            info = engines.probe(cfg, dest)
        except engines.SourceError as e:
            dest.unlink(missing_ok=True)
            raise HTTPException(422, str(e))
        if info["duration"] > cfg.max_duration_s:
            dest.unlink(missing_ok=True)
            raise HTTPException(422, "הקובץ ארוך משלוש שעות.")
        return st.patch_project(pid, lambda doc: doc["source"].update(kind="file", name=file.filename, size=size, duration=info["duration"], hasVideo=info["hasVideo"]))

    @app.get("/api/projects/{pid}", dependencies=[Depends(auth)])
    def get_project(pid: str):
        return project_or_404(pid)

    @app.put("/api/projects/{pid}", dependencies=[Depends(auth)])
    def save_project(pid: str, body: SaveIn):
        project_or_404(pid)
        try:
            return st.save_project(pid, body.project, None if body.force else body.baseRev)
        except ConflictError as e:
            return JSONResponse(status_code=409, content={"detail": "conflict", "current": e.current})

    @app.delete("/api/projects/{pid}", dependencies=[Depends(auth)])
    def delete_project(pid: str):
        if any(j["projectId"] == pid and j["status"] == "running" for j in q.list()):
            raise HTTPException(409, "יש עבודה שרצה על הפרויקט. בטל אותה קודם.")
        st.delete_project(pid)
        return {"ok": True}

    @app.get("/api/projects/{pid}/files/{name}", dependencies=[Depends(auth)])
    def get_file(pid: str, name: str, download: bool = False):
        p = project_or_404(pid)
        if name not in FILES:
            raise HTTPException(404, "קובץ לא מוכר")
        d = st.project_dir(pid)
        path = sorted(d.glob("source.*"))[0] if name == "source" and list(d.glob("source.*")) else d / (FILES[name] or "-")
        if not path.exists():
            raise HTTPException(404, "הקובץ עדיין לא נוצר")
        base = "".join(c for c in (p.get("title") or "lyra") if c.isalnum() or c in " -_").strip().replace(" ", "_") or "lyra"
        fname = f"{base}_{name}{path.suffix}"
        return FileResponse(path, filename=fname if download else None, content_disposition_type="attachment" if download else "inline")

    # ---- jobs ----
    @app.get("/api/jobs", dependencies=[Depends(auth)])
    def jobs():
        return q.list()

    @app.post("/api/jobs", dependencies=[Depends(auth)])
    def submit(body: JobIn):
        project_or_404(body.projectId)
        if body.kind == "render" and not (st.project_dir(body.projectId) / "stems" / "instrumental.wav").exists():
            raise HTTPException(409, "צריך להשלים את העיבוד לפני רינדור.")
        try:
            return q.submit(body.projectId, body.kind, body.profile, body.stems, body.language, body.simulateCrash)
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.get("/api/jobs/{jid}", dependencies=[Depends(auth)])
    def job(jid: str):
        try:
            return q.get(jid)
        except KeyError:
            raise HTTPException(404, "העבודה לא נמצאה")

    @app.post("/api/jobs/{jid}/cancel", dependencies=[Depends(auth)])
    def cancel(jid: str):
        return q.cancel(jid)

    @app.post("/api/jobs/{jid}/retry", dependencies=[Depends(auth)])
    def retry(jid: str, body: RetryIn):
        try:
            return q.retry(jid, body.fast)
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/jobs/{jid}/move", dependencies=[Depends(auth)])
    def move(jid: str, body: MoveIn):
        q.move(jid, body.delta)
        return q.list()

    @app.delete("/api/jobs/{jid}", dependencies=[Depends(auth)])
    def remove(jid: str):
        try:
            q.remove(jid)
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    @app.exception_handler(KeyError)
    async def not_found(_: Request, exc: KeyError):
        return JSONResponse(status_code=404, content={"detail": "לא נמצא"})

    dist = Path(os.environ.get("LYRA_WEB_DIST", Path(__file__).resolve().parents[2] / "app" / "dist"))
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    return app


app = create_app() if os.environ.get("LYRA_NO_AUTOAPP") != "1" else None  # type: ignore[assignment]

