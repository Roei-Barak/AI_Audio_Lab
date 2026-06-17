"""KaraokeStudio FastAPI server.

All endpoints under /api/.  React SPA served at / (catch-all).

Auth:
  KARAOKE_AUTH_MODE=required  → JWT required on all /api/* except /api/auth/login
  KARAOKE_AUTH_MODE=none      → all endpoints open (Standalone)

CLI:
  python api/server.py create-user <username> <password> [--admin]
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api.auth import get_current_user, hash_password, router as auth_router
from api.db import User, create_user as db_create_user, get_user_by_username, init_db

# ── Config ─────────────────────────────────────────────────────────────────────

WORK_DIR     = os.environ.get("KARAOKE_OUTPUT_DIR", "Karaoke_Output")
CORS_ORIGINS = [o.strip() for o in os.environ.get("KARAOKE_CORS_ORIGINS", "*").split(",")]
_SPA_DIR     = Path(__file__).parent.parent / "web" / "dist"

app = FastAPI(title="KaraokeStudio API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


# ── Auth router ────────────────────────────────────────────────────────────────

app.include_router(auth_router, prefix="/api/auth")

# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ── SSE pipeline ───────────────────────────────────────────────────────────────

def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def run_pipeline(
    url: str, lang: str, output_formats: list[str],
    save_4_stems: bool, use_bidi: bool, force: bool,
) -> AsyncGenerator[str, None]:
    try:
        from modules.pipeline import KaraokePipeline  # type: ignore
        pipeline = KaraokePipeline(WORK_DIR)
        async for event in pipeline.run_async(
            url=url, lang=lang, output_formats=output_formats,
            save_4_stems=save_4_stems, use_bidi=use_bidi, force=force,
        ):
            yield sse(event)
    except ImportError:
        yield sse({"type": "log", "text": "Pipeline modules not installed"})
        yield sse({"type": "done", "success": False})
    except Exception as exc:
        yield sse({"type": "log", "text": f"Error: {exc}"})
        yield sse({"type": "done", "success": False})


class PipelineRequest(BaseModel):
    url: str
    lang: str = "he"
    output_formats: list[str] = ["ass", "srt"]
    save_4_stems: bool = False
    use_bidi: bool = False
    force: bool = False


@app.post("/api/pipeline/stream")
async def pipeline_stream(
    body: PipelineRequest,
    _user: Optional[User] = Depends(get_current_user),
):
    return StreamingResponse(
        run_pipeline(
            body.url, body.lang, body.output_formats,
            body.save_4_stems, body.use_bidi, body.force,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── File serving ───────────────────────────────────────────────────────────────

Path(WORK_DIR).mkdir(parents=True, exist_ok=True)
app.mount("/api/files", StaticFiles(directory=WORK_DIR), name="files")

# ── SPA (must be last) ────────────────────────────────────────────────────────

if _SPA_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_SPA_DIR), html=True), name="spa")


# ── CLI: create-user ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "create-user":
        username = sys.argv[2]
        password = sys.argv[3]
        role     = "admin" if "--admin" in sys.argv else "user"
        init_db()
        if get_user_by_username(username):
            print(f"User '{username}' already exists")
            sys.exit(1)
        u = db_create_user(username, hash_password(password), role)
        print(f"Created {u.role} user '{u.username}' (id={u.id})")
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
