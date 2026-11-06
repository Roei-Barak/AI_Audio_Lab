"""On-disk project and job storage.

Each project lives in <data>/projects/<id>/ with:
  project.json   - the shared project document (same shape as the frontend store)
  source.*       - imported media
  stems/*.wav    - separation output
  words.json     - raw transcription words (checkpoint)
  render/*.mp4   - karaoke renders
Jobs are persisted in <data>/jobs.json so a restarted server can resume them.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any


def new_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:10]


def _atomic_write(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


SERVER_OWNED = ("renderedRev", "renderedAt", "processed")


class ConflictError(Exception):
    def __init__(self, current: dict):
        super().__init__("revision conflict")
        self.current = current


class Storage:
    def __init__(self, root: Path):
        self.root = root
        self.projects_dir = root / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_file = root / "jobs.json"
        self._lock = threading.RLock()

    # ---- projects ----
    def project_dir(self, pid: str) -> Path:
        if not pid or "/" in pid or ".." in pid:
            raise KeyError(pid)
        return self.projects_dir / pid

    def create_project(self, doc: dict) -> dict:
        with self._lock:
            pid = doc.get("id") or new_id("prj_")
            doc = {**doc, "id": pid, "rev": doc.get("rev", 1), "serverRev": 1, "updatedAt": time.time()}
            d = self.project_dir(pid)
            (d / "stems").mkdir(parents=True, exist_ok=True)
            (d / "render").mkdir(exist_ok=True)
            _atomic_write(d / "project.json", doc)
            return doc

    def get_project(self, pid: str) -> dict:
        f = self.project_dir(pid) / "project.json"
        if not f.exists():
            raise KeyError(pid)
        return json.loads(f.read_text(encoding="utf-8"))

    def list_projects(self) -> list[dict]:
        out = []
        for d in sorted(self.projects_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            f = d / "project.json"
            if f.exists():
                p = json.loads(f.read_text(encoding="utf-8"))
                out.append({k: p.get(k) for k in ("id", "title", "artist", "createdAt", "updatedAt", "serverRev")} | {"lines": len(p.get("lines", []))})
        return out

    def save_project(self, pid: str, doc: dict, base_rev: int | None) -> dict:
        """Optimistic concurrency: base_rev must match the stored serverRev, otherwise ConflictError."""
        with self._lock:
            cur = self.get_project(pid)
            if base_rev is not None and base_rev != cur.get("serverRev"):
                raise ConflictError(cur)
            owned = {k: cur[k] for k in SERVER_OWNED if k in cur}  # fields only the pipeline writes
            doc = {**doc, **owned, "id": pid, "serverRev": cur.get("serverRev", 1) + 1, "updatedAt": time.time()}
            _atomic_write(self.project_dir(pid) / "project.json", doc)
            return doc

    def patch_project(self, pid: str, fn, bump: bool = True) -> dict:
        """Server-side update (e.g. pipeline writing the transcription). bump=False for bookkeeping fields."""
        with self._lock:
            cur = self.get_project(pid)
            fn(cur)
            if bump:
                cur["serverRev"] = cur.get("serverRev", 1) + 1
            cur["updatedAt"] = time.time()
            _atomic_write(self.project_dir(pid) / "project.json", cur)
            return cur

    def delete_project(self, pid: str) -> None:
        import shutil

        with self._lock:
            d = self.project_dir(pid)
            if d.exists():
                shutil.rmtree(d)

    # ---- jobs ----
    def load_jobs(self) -> list[dict]:
        if not self.jobs_file.exists():
            return []
        try:
            return json.loads(self.jobs_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def save_jobs(self, jobs: list[dict]) -> None:
        with self._lock:
            _atomic_write(self.jobs_file, jobs)
