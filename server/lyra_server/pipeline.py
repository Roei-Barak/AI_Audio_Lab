"""Job queue: one heavy GPU job at a time (V1 default), checkpointed stages, cancel and resume."""
from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path

from . import engines
from .config import Config
from .storage import Storage, new_id

STAGES = ["import", "separate", "transcribe", "align", "render"]
ACTIVE = ("queued", "running")


class Cancelled(Exception):
    pass


# Rough seconds of work per second of audio, used only to show an *estimated* progress bar for model stages
# (the models do not report progress). Marked estimate=True so the UI can say so.
EST = {("separate", "cuda"): 0.12, ("separate", "cpu"): 1.2, ("transcribe", "cuda"): 0.08, ("transcribe", "cpu"): 0.9}


class Estimator:
    def __init__(self, q: "JobQueue", job: dict, stage: str, duration: float, profile: str):
        dev = "cuda" if _cuda() else "cpu"
        self.expected = max(3.0, duration * EST.get((stage, dev), 0.5) * (1.8 if profile == "איכותי" else 1.0))
        self.q, self.job, self.stop = q, job, threading.Event()

    def __enter__(self):
        t0 = time.time()

        def beat():
            while not self.stop.wait(1.0):
                frac = (time.time() - t0) / self.expected
                self.q._set(self.job, pct=round(95 * (1 - 2.718 ** (-1.6 * frac)), 1), estimate=True)

        threading.Thread(target=beat, daemon=True).start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.q._set(self.job, estimate=False)


def _cuda() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class JobQueue:
    def __init__(self, cfg: Config, storage: Storage, autostart: bool = True):
        self.cfg = cfg
        self.st = storage
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.jobs: list[dict] = storage.load_jobs()
        # A job that was running when the server stopped did not finish: surface it as a crash that can resume.
        for j in self.jobs:
            if j["status"] == "running":
                j.update(status="failed", error="השרת הופעל מחדש באמצע העבודה. השלבים שהושלמו נשמרו.", crash=True)
        self._persist()
        self.current: dict | None = None
        if autostart:
            threading.Thread(target=self._worker, daemon=True, name="lyra-gpu-worker").start()

    # ---- public API ----
    def list(self) -> list[dict]:
        with self.lock:
            return [self._public(j) for j in self.jobs]

    def get(self, jid: str) -> dict:
        with self.lock:
            return self._public(self._find(jid))

    def submit(self, project_id: str, kind: str = "process", profile: str = "מהיר", stems: int = 2, language: str = "auto", simulate_crash: bool = False) -> dict:
        p = self.st.get_project(project_id)
        with self.lock:
            if any(j["projectId"] == project_id and j["kind"] == kind and j["status"] in ACTIVE for j in self.jobs):
                raise ValueError("כבר יש עבודה פעילה לפרויקט הזה.")
            job = {
                "id": new_id("job_"), "projectId": project_id, "kind": kind, "name": p.get("source", {}).get("name") or p.get("title"),
                "profile": profile, "stems": stems, "language": language, "status": "queued",
                "stage": 0 if kind == "process" else 4, "pct": 0, "attempt": 1, "error": None, "crash": False,
                "log": [], "createdAt": time.time(), "updatedAt": time.time(), "startedAt": None, "endedAt": None,
                "cancelRequested": False, "simulateCrash": bool(simulate_crash and self.cfg.engine == "mock"),
            }
            self.jobs.append(job)
            self._persist()
        self.wake.set()
        return self._public(job)

    def cancel(self, jid: str) -> dict:
        with self.lock:
            j = self._find(jid)
            if j["status"] == "queued":
                j.update(status="cancelled", endedAt=time.time())
            elif j["status"] == "running":
                # ffmpeg steps (render) stop at once; a model call (separation/transcription) finishes its stage first
                j["cancelRequested"] = True
            self._persist()
            return self._public(j)

    def retry(self, jid: str, fast: bool = False) -> dict:
        with self.lock:
            j = self._find(jid)
            if j["status"] not in ("failed", "cancelled"):
                raise ValueError("אפשר לנסות שוב רק עבודה שנכשלה או בוטלה.")
            j.update(status="queued", error=None, crash=False, cancelRequested=False, attempt=j["attempt"] + 1, pct=0, endedAt=None, simulateCrash=False)
            if fast:
                j["profile"] = "מהיר"
            self._persist()
        self.wake.set()
        return self._public(j)

    def remove(self, jid: str) -> None:
        with self.lock:
            j = self._find(jid)
            if j["status"] == "running":
                raise ValueError("אי אפשר להסיר עבודה שרצה. בטל אותה קודם.")
            self.jobs.remove(j)
            self._persist()

    def move(self, jid: str, delta: int) -> None:
        with self.lock:
            q = [j for j in self.jobs if j["status"] == "queued"]
            j = self._find(jid)
            if j not in q:
                return
            i = q.index(j)
            k = max(0, min(len(q) - 1, i + delta))
            q[i], q[k] = q[k], q[i]
            rest = [x for x in self.jobs if x["status"] != "queued"]
            self.jobs = rest + q
            self._persist()

    def gpu_busy(self) -> bool:
        return self.current is not None

    # ---- worker ----
    def _worker(self) -> None:
        while True:
            job = self._next()
            if not job:
                self.wake.wait(1.0)
                self.wake.clear()
                continue
            self.current = job
            try:
                self._run(job)
            finally:
                self.current = None

    def _next(self) -> dict | None:
        with self.lock:
            for j in self.jobs:
                if j["status"] == "queued":
                    j.update(status="running", startedAt=j.get("startedAt") or time.time(), updatedAt=time.time())
                    self._persist()
                    return j
        return None

    def run_once(self) -> bool:
        """Runs the next queued job synchronously (tests)."""
        job = self._next()
        if not job:
            return False
        self._run(job)
        return True

    def _log(self, job: dict, msg: str) -> None:
        with self.lock:
            job["log"] = (job["log"] + [time.strftime("[%H:%M:%S] ") + msg])[-60:]
            job["updatedAt"] = time.time()
            self._persist()

    def _set(self, job: dict, **kw) -> None:
        with self.lock:
            job.update(kw, updatedAt=time.time())
            self._persist()

    def _run(self, job: dict) -> None:
        pdir = self.st.project_dir(job["projectId"])
        log = lambda m: self._log(job, m)  # noqa: E731
        try:
            stages = range(job["stage"], len(STAGES)) if job["kind"] == "process" else [4]
            for i in stages:
                if job["cancelRequested"]:
                    raise Cancelled()
                self._set(job, stage=i, pct=0)
                self._stage(job, i, pdir, log)
                self._set(job, pct=100)
            self._set(job, status="done", stage=len(STAGES), endedAt=time.time())
            log("done")
        except (Cancelled, InterruptedError):
            self._set(job, status="cancelled", endedAt=time.time())
            log("cancelled by user · checkpoint kept")
        except engines.SourceError as e:
            self._set(job, status="failed", error=str(e), crash=False, endedAt=time.time())
            log("ERROR " + str(e))
        except Exception as e:  # crash: completed stages are kept on disk
            self._set(job, status="failed", error=str(e) or e.__class__.__name__, crash=True, endedAt=time.time())
            log("CRASH " + (str(e) or traceback.format_exc(limit=1)))

    def _stage(self, job: dict, i: int, pdir: Path, log) -> None:
        cfg = self.cfg
        p = self.st.get_project(job["projectId"])
        src = p.get("source", {})
        if i == 0:  # import
            existing = sorted(pdir.glob("source.*"))
            if existing:
                log("source present · " + existing[0].name)
            elif src.get("kind") == "youtube":
                engines.youtube_download(cfg, src["url"], pdir, log)
            else:
                raise engines.SourceError("קובץ המקור חסר בשרת. העלה אותו שוב.")
            info = engines.probe(cfg, sorted(pdir.glob("source.*"))[0])
            if info["duration"] > cfg.max_duration_s:
                raise engines.SourceError("הקובץ ארוך משלוש שעות.")
            self.st.patch_project(job["projectId"], lambda d: d["source"].update(duration=info["duration"] or d["source"].get("duration"), hasVideo=info["hasVideo"]))
        elif i == 1:  # separate (GPU)
            stems_dir = pdir / "stems"
            need = ["vocals", "instrumental"] + (["drums", "bass", "other"] if job["stems"] == 4 else [])
            if all((stems_dir / f"{k}.wav").exists() for k in need):
                log("stems present")
                return
            wav = engines.extract_wav(cfg, sorted(pdir.glob("source.*"))[0], pdir / "audio.wav")
            with Estimator(self, job, "separate", float(src.get("duration") or 0), job["profile"]):
                engines.separate(cfg, wav, stems_dir, job["stems"], job["profile"], log)
            wav.unlink(missing_ok=True)
            self.st.patch_project(job["projectId"], lambda d: d.update(stems=need))
        elif i == 2:  # transcribe (GPU)
            if job.get("simulateCrash") and job["attempt"] == 1:
                raise RuntimeError("CUDA out of memory (simulated)")
            words_f = pdir / "words.json"
            if words_f.exists() and job["kind"] == "process" and not p.get("rerun"):
                log("transcription present")
                return
            with Estimator(self, job, "transcribe", float(src.get("duration") or 0), job["profile"]):
                words = engines.transcribe(cfg, pdir / "stems" / "vocals.wav", job["language"], job["profile"], float(src.get("duration") or 0), log)
            if not words:
                raise engines.SourceError("לא זוהתה שירה בקובץ. אפשר להדביק מילים ידנית ולסנכרן אותן.")
            words_f.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
        elif i == 3:  # align -> lines; never overwrite manual fixes
            words = json.loads((pdir / "words.json").read_text(encoding="utf-8"))
            lines = engines.words_to_lines(words)
            has_manual = any(ln.get("manual") or any(w.get("manual") for w in ln.get("words", [])) for ln in p.get("lines", []))

            def upd(d: dict) -> None:
                if has_manual:
                    d["suggestion"] = {"at": time.time(), "lines": lines}  # AI re-run is shown as a suggestion
                else:
                    d["lines"] = lines
                    d["timing"] = "word"
                d["processed"] = True
                d.setdefault("history", []).insert(0, {"at": int(time.time() * 1000), "label": "הצעת AI חדשה מוכנה להשוואה" if has_manual else "נוצר מתמלול AI"})

            self.st.patch_project(job["projectId"], upd)
            log(f"{len(lines)} lines · {len(words)} words")
        elif i == 4:  # render
            p = self.st.get_project(job["projectId"])
            source = sorted(pdir.glob("source.*"))[0]
            out = pdir / "render" / "karaoke.mp4"
            tmp = out.with_name("karaoke.tmp.mp4")
            last = [0.0]

            def on_pct(x: float) -> None:
                if x - last[0] >= 1:
                    last[0] = x
                    self._set(job, pct=round(x, 1), estimate=False)

            engines.render(cfg, p, source, pdir / "stems" / "instrumental.wav", tmp, bool(p["source"].get("hasVideo")), float(p["source"].get("duration") or 0), log,
                           on_pct=on_pct, should_stop=lambda: job["cancelRequested"])
            tmp.replace(out)
            # The MP4 matches the revision that was rendered; later saves make it stale.
            self.st.patch_project(job["projectId"], lambda d: d.update(renderedRev=p.get("serverRev"), renderedAt=time.time()), bump=False)

    # ---- helpers ----
    def _find(self, jid: str) -> dict:
        for j in self.jobs:
            if j["id"] == jid:
                return j
        raise KeyError(jid)

    def _persist(self) -> None:
        self.st.save_jobs(self.jobs)

    def _public(self, j: dict) -> dict:
        return {k: v for k, v in j.items() if k != "simulateCrash"} | {"stageName": STAGES[min(j["stage"], len(STAGES) - 1)]}
