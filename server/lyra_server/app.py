"""Lyra Studio processing server (FastAPI).

Run:  uvicorn lyra_server.app:app --host 0.0.0.0 --port 8443   (from the server/ folder)
Env:  LYRA_ENGINE=ai|mock   LYRA_DATA_DIR=...   LYRA_TOKEN=... (optional shared access token)
Serves the built frontend (../app/dist) at / when present, so one process is the whole product.
"""
from __future__ import annotations

import json
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
from .auth import COOKIE, SESSION_DAYS, Accounts, AuthError

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


class SetupIn(BaseModel):
    name: str
    user: str
    password: str


class LoginIn(BaseModel):
    user: str
    password: str


class AcceptIn(BaseModel):
    token: str
    user: str
    password: str


class ResetIn(BaseModel):
    token: str
    password: str


class RoleIn(BaseModel):
    role: str


class InviteIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    role: str = "editor"
    days: int = 7


class VersionIn(BaseModel):
    label: str
    rev: int | None = None
    lines: list
    style: dict = {}


class LyricsIn(BaseModel):
    text: str = Field(max_length=200_000)


class OutputIn(BaseModel):
    projectId: str
    kind: str = Field(pattern="^(srt|txt|json|lrc|ass|mp4|vocals|instrumental|drums|bass|other)$")


def create_app(cfg: Config | None = None, autostart: bool = True) -> FastAPI:
    cfg = cfg or load()
    st = Storage(cfg.data_dir)
    q = JobQueue(cfg, st, autostart=autostart)
    app = FastAPI(title="Lyra Studio server", version=__version__)
    app.state.cfg, app.state.storage, app.state.queue = cfg, st, q
    app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("LYRA_CORS", "http://localhost:5173,http://localhost:4173").split(","), allow_methods=["*"], allow_headers=["*"])

    token = os.environ.get("LYRA_TOKEN")
    acc = Accounts(cfg.data_dir)
    app.state.accounts = acc
    RANK = {"viewer": 0, "editor": 1, "admin": 2}
    SERVICE = {"id": "service", "name": "service", "user": "service", "role": "admin", "createdAt": 0}

    def current(request: Request, authorization: str | None = Header(None)) -> dict | None:
        if not cfg.auth:
            return SERVICE
        bearer = authorization[7:] if authorization and authorization.startswith("Bearer ") else None
        if token and bearer == token:
            return SERVICE
        return acc.user_for(request.cookies.get(COOKIE) or bearer)

    def need(role: str):
        def dep(u: dict | None = Depends(current)) -> dict:
            if not u:
                raise HTTPException(401, "נדרשת התחברות")
            if RANK[u["role"]] < RANK[role]:
                raise HTTPException(403, "אין לך הרשאה לפעולה הזו")
            return u

        return dep

    auth, write, admin = need("viewer"), need("editor"), need("admin")

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
    @app.post("/api/sources/youtube", dependencies=[Depends(write)])
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

    @app.post("/api/projects", dependencies=[Depends(write)])
    def create_project(body: ProjectIn):
        doc = dict(body.project)
        doc["id"] = new_id("prj_")
        if body.source:
            doc["source"] = body.source
        doc["lines"] = doc.get("lines") if doc.get("source", {}).get("kind") == "demo" else []
        doc["processed"] = False
        return st.create_project(doc)

    @app.post("/api/projects/{pid}/upload", dependencies=[Depends(write)])
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

    @app.put("/api/projects/{pid}", dependencies=[Depends(write)])
    def save_project(pid: str, body: SaveIn):
        project_or_404(pid)
        try:
            return st.save_project(pid, body.project, None if body.force else body.baseRev)
        except ConflictError as e:
            return JSONResponse(status_code=409, content={"detail": "conflict", "current": e.current})

    @app.delete("/api/projects/{pid}", dependencies=[Depends(write)])
    def delete_project(pid: str):
        if any(j["projectId"] == pid and j["status"] == "running" for j in q.list()):
            raise HTTPException(409, "יש עבודה שרצה על הפרויקט. בטל אותה קודם.")
        st.delete_project(pid)
        return {"ok": True}

    @app.get("/api/projects/{pid}/files/{name}", dependencies=[Depends(auth)])
    def get_file(pid: str, name: str, download: bool = False):
        p = project_or_404(pid)
        if name == "ass":
            from fastapi.responses import Response

            body = engines.build_ass(p, *engines.RES.get(p.get("export", {}).get("res", "1080p"), engines.RES["1080p"]).get(p.get("style", {}).get("aspect", "16:9"), (1920, 1080)))
            base = "".join(c for c in (p.get("title") or "lyra") if c.isalnum() or c in " -_").strip().replace(" ", "_") or "lyra"
            return Response("\ufeff" + body, media_type="text/x-ssa; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="{base}.ass"' if download else "inline"})
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

    @app.post("/api/jobs", dependencies=[Depends(write)])
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

    @app.post("/api/jobs/{jid}/cancel", dependencies=[Depends(write)])
    def cancel(jid: str):
        return q.cancel(jid)

    @app.post("/api/jobs/{jid}/retry", dependencies=[Depends(write)])
    def retry(jid: str, body: RetryIn):
        try:
            return q.retry(jid, body.fast)
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/jobs/{jid}/move", dependencies=[Depends(write)])
    def move(jid: str, body: MoveIn):
        q.move(jid, body.delta)
        return q.list()

    @app.delete("/api/jobs/{jid}", dependencies=[Depends(write)])
    def remove(jid: str):
        try:
            q.remove(jid)
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}


    # ---- accounts (private: no public signup) ----
    def _with_cookie(user: dict, tok: str, secure: bool) -> JSONResponse:
        r = JSONResponse({"user": user})
        r.set_cookie(COOKIE, tok, max_age=SESSION_DAYS * 86400, httponly=True, samesite="lax", secure=secure, path="/")
        return r

    def _run(fn):
        try:
            return fn()
        except AuthError as e:
            raise HTTPException(e.status, str(e))

    @app.get("/api/auth/status")
    def auth_status(u: dict | None = Depends(current)):
        return {"authRequired": cfg.auth, "needsSetup": cfg.auth and acc.needs_setup(), "user": u}

    @app.post("/api/auth/setup")
    def auth_setup(body: SetupIn, request: Request):
        user, tok = _run(lambda: acc.setup(body.name, body.user, body.password))
        return _with_cookie(user, tok, request.url.scheme == "https")

    @app.post("/api/auth/login")
    def auth_login(body: LoginIn, request: Request):
        user, tok = _run(lambda: acc.login(body.user, body.password))
        return _with_cookie(user, tok, request.url.scheme == "https")

    @app.post("/api/auth/logout")
    def auth_logout(request: Request):
        acc.logout(request.cookies.get(COOKIE))
        r = JSONResponse({"ok": True})
        r.delete_cookie(COOKIE, path="/")
        return r

    @app.post("/api/auth/accept")
    def auth_accept(body: AcceptIn, request: Request):
        tok_in = body.token.rstrip("/").split("/join/")[-1].split("t=")[-1]
        user, tok = _run(lambda: acc.accept(tok_in, body.user, body.password))
        return _with_cookie(user, tok, request.url.scheme == "https")

    @app.post("/api/auth/reset")
    def auth_reset(body: ResetIn, request: Request):
        tok_in = body.token.rstrip("/").split("/reset/")[-1].split("t=")[-1]
        user, tok = _run(lambda: acc.reset(tok_in, body.password))
        return _with_cookie(user, tok, request.url.scheme == "https")

    @app.get("/api/users", dependencies=[Depends(auth)])
    def users():
        return acc.users()

    @app.patch("/api/users/{uid}")
    def set_role(uid: str, body: RoleIn, u: dict = Depends(admin)):
        return _run(lambda: acc.set_role(u, uid, body.role))

    @app.delete("/api/users/{uid}")
    def remove_user(uid: str, u: dict = Depends(admin)):
        _run(lambda: acc.remove(u, uid))
        return {"ok": True}

    @app.post("/api/users/{uid}/reset", dependencies=[Depends(admin)])
    def reset_link(uid: str):
        return {"token": _run(lambda: acc.reset_link(uid))}

    @app.get("/api/invites", dependencies=[Depends(admin)])
    def invites():
        return acc.invites()

    @app.post("/api/invites", dependencies=[Depends(admin)])
    def create_invite(body: InviteIn):
        inv, tok = _run(lambda: acc.invite(body.name, body.role, body.days))
        return {**inv, "token": tok}

    @app.delete("/api/invites/{iid}", dependencies=[Depends(admin)])
    def revoke_invite(iid: str):
        acc.revoke(iid)
        return {"ok": True}

    # ---- settings ----
    settings_file = cfg.data_dir / "settings.json"

    @app.get("/api/settings", dependencies=[Depends(auth)])
    def get_settings():
        return json.loads(settings_file.read_text(encoding="utf-8")) if settings_file.exists() else {}

    @app.put("/api/settings", dependencies=[Depends(admin)])
    def put_settings(body: dict):
        allowed = {"defaultProfile", "defaultStems", "language", "autosave", "model", "separator", "outputDir"}
        cur = get_settings()
        cur.update({k: v for k, v in body.items() if k in allowed})
        settings_file.write_text(json.dumps(cur, ensure_ascii=False), encoding="utf-8")
        return cur

    # ---- versions (named checkpoints, kept on the server) ----
    @app.get("/api/projects/{pid}/versions", dependencies=[Depends(auth)])
    def versions(pid: str):
        project_or_404(pid)
        f = st.project_dir(pid) / "versions.json"
        items = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
        return [{k: v for k, v in x.items() if k not in ("lines", "style")} | {"lineTexts": [" ".join(w["w"] for w in ln.get("words", [])) for ln in x["lines"]]} for x in items]

    @app.get("/api/projects/{pid}/versions/{vid}", dependencies=[Depends(auth)])
    def version(pid: str, vid: str):
        f = st.project_dir(pid) / "versions.json"
        items = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
        v = next((x for x in items if x["id"] == vid), None)
        if not v:
            raise HTTPException(404, "הגרסה לא נמצאה")
        return v

    @app.post("/api/projects/{pid}/versions")
    def add_version(pid: str, body: VersionIn, u: dict = Depends(write)):
        project_or_404(pid)
        f = st.project_dir(pid) / "versions.json"
        items = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
        v = {"id": new_id("v_"), "at": time.time() * 1000, "label": body.label[:80], "by": u["name"], "rev": body.rev, "lines": body.lines, "style": body.style}
        items = [v, *items][:50]
        f.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        return {k: v[k] for k in ("id", "at", "label", "by", "rev")}

    @app.delete("/api/projects/{pid}/versions", dependencies=[Depends(admin)])
    def clear_versions(pid: str):
        (st.project_dir(pid) / "versions.json").unlink(missing_ok=True)
        return {"ok": True}

    # ---- pasted lyrics -> aligned to the transcription (or spread evenly) ----
    @app.post("/api/projects/{pid}/lyrics", dependencies=[Depends(write)])
    def paste_lyrics(pid: str, body: LyricsIn):
        p = project_or_404(pid)
        wf = st.project_dir(pid) / "words.json"
        words = json.loads(wf.read_text(encoding="utf-8")) if wf.exists() else []
        lines, timing = engines.align_lyrics(body.text, words, float(p.get("source", {}).get("duration") or 0))
        if not lines:
            raise HTTPException(422, "לא הודבקו מילים.")
        return {"lines": lines, "timing": timing, "aligned": bool(words)}

    # ---- outputs ----
    out_file = cfg.data_dir / "outputs.json"

    @app.get("/api/outputs", dependencies=[Depends(auth)])
    def outputs():
        return json.loads(out_file.read_text(encoding="utf-8")) if out_file.exists() else []

    @app.post("/api/outputs")
    def add_output(body: OutputIn, u: dict = Depends(auth)):
        p = project_or_404(body.projectId)
        items = outputs()
        items.insert(0, {"id": new_id("o_"), "at": time.time() * 1000, "kind": body.kind, "projectId": body.projectId, "title": p.get("title"), "lines": len(p.get("lines", [])), "by": u["name"], "simulated": False})
        out_file.write_text(json.dumps(items[:200], ensure_ascii=False), encoding="utf-8")
        return items[0]

    @app.exception_handler(KeyError)
    async def not_found(_: Request, exc: KeyError):
        return JSONResponse(status_code=404, content={"detail": "לא נמצא"})

    dist = Path(os.environ.get("LYRA_WEB_DIST", Path(__file__).resolve().parents[2] / "app" / "dist"))
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    return app


app = create_app() if os.environ.get("LYRA_NO_AUTOAPP") != "1" else None  # type: ignore[assignment]

