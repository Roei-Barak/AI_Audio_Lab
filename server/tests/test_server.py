import json
import os
import subprocess
from pathlib import Path

import pytest

os.environ["LYRA_NO_AUTOAPP"] = "1"
from fastapi.testclient import TestClient  # noqa: E402

from lyra_server.app import create_app  # noqa: E402
from lyra_server.config import Config  # noqa: E402
from lyra_server import engines  # noqa: E402


@pytest.fixture()
def env(tmp_path: Path):
    cfg = Config(data_dir=tmp_path, engine="mock", ffmpeg="ffmpeg", ffprobe="ffprobe", auth=False)
    app = create_app(cfg, autostart=False)
    return TestClient(app), app.state.queue, tmp_path


@pytest.fixture(scope="session")
def clip(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25:duration=12",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=12", "-f", "lavfi", "-i", "sine=frequency=660:duration=12",
                    "-filter_complex", "[1:a][2:a]amerge=inputs=2[a]", "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(p)], check=True)
    return p


def new_project(c, clip):
    p = c.post("/api/projects", json={"project": {"title": "בדיקה", "artist": "x", "style": {"size": 44, "aspect": "16:9", "background": "video"}, "export": {"res": "720p"}, "history": []},
                                     "source": {"kind": "file", "name": "clip.mp4", "duration": 0, "hasVideo": True}}).json()
    with clip.open("rb") as f:
        r = c.post(f"/api/projects/{p['id']}/upload", files={"file": ("clip.mp4", f, "video/mp4")})
    assert r.status_code == 200, r.text
    assert abs(r.json()["source"]["duration"] - 12) < 0.5
    return p["id"]


def test_full_pipeline_renders_karaoke(env, clip):
    c, q, root = env
    pid = new_project(c, clip)
    job = c.post("/api/jobs", json={"projectId": pid, "stems": 2}).json()
    assert job["status"] == "queued"
    assert q.run_once()
    job = c.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "done", job
    proj = c.get(f"/api/projects/{pid}").json()
    assert proj["lines"] and proj["timing"] == "word"
    mp4 = c.get(f"/api/projects/{pid}/files/karaoke")
    assert mp4.status_code == 200 and len(mp4.content) > 10000
    out = root / "projects" / pid / "render" / "karaoke.mp4"
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout)
    assert abs(dur - 12) < 0.6  # length preserved
    assert c.get(f"/api/projects/{pid}/files/instrumental").status_code == 200
    ass = (root / "projects" / pid / "render" / "karaoke.ass").read_text(encoding="utf-8")
    assert "\\kf" in ass and "PlayResY: 720" in ass


def test_crash_then_resume_from_failed_stage(env, clip):
    c, q, root = env
    pid = new_project(c, clip)
    jid = c.post("/api/jobs", json={"projectId": pid, "simulateCrash": True}).json()["id"]
    q.run_once()
    j = c.get(f"/api/jobs/{jid}").json()
    assert j["status"] == "failed" and j["crash"] and j["stageName"] == "transcribe"
    assert (root / "projects" / pid / "stems" / "vocals.wav").exists()  # checkpoint kept
    c.post(f"/api/jobs/{jid}/retry", json={"fast": True})
    q.run_once()
    j = c.get(f"/api/jobs/{jid}").json()
    assert j["status"] == "done" and j["attempt"] == 2 and j["profile"] == "מהיר"
    assert sum("mock separation" in line for line in j["log"]) == 1  # resumed at transcription, separation not repeated


def test_cancel_queued_and_one_active_job_per_project(env, clip):
    c, q, _ = env
    pid = new_project(c, clip)
    jid = c.post("/api/jobs", json={"projectId": pid}).json()["id"]
    assert c.post("/api/jobs", json={"projectId": pid}).status_code == 409
    assert c.post(f"/api/jobs/{jid}/cancel").json()["status"] == "cancelled"
    assert not q.run_once()


def test_save_conflict_detection(env, clip):
    c, q, _ = env
    pid = new_project(c, clip)
    p = c.get(f"/api/projects/{pid}").json()
    r1 = c.put(f"/api/projects/{pid}", json={"baseRev": p["serverRev"], "project": {**p, "title": "A"}})
    assert r1.status_code == 200
    r2 = c.put(f"/api/projects/{pid}", json={"baseRev": p["serverRev"], "project": {**p, "title": "B"}})
    assert r2.status_code == 409 and r2.json()["current"]["title"] == "A"
    r3 = c.put(f"/api/projects/{pid}", json={"baseRev": p["serverRev"], "project": {**p, "title": "B"}, "force": True})
    assert r3.status_code == 200 and r3.json()["title"] == "B"


def test_rerun_with_manual_edits_becomes_suggestion(env, clip):
    c, q, _ = env
    pid = new_project(c, clip)
    c.post("/api/jobs", json={"projectId": pid})
    q.run_once()
    p = c.get(f"/api/projects/{pid}").json()
    p["lines"][0]["words"][0]["w"] = "תיקון"
    p["lines"][0]["words"][0]["manual"] = True
    p["rerun"] = True
    p = c.put(f"/api/projects/{pid}", json={"baseRev": p["serverRev"], "project": p}).json()
    (Path(q.st.project_dir(pid)) / "render" / "karaoke.mp4").unlink()
    c.post("/api/jobs", json={"projectId": pid})
    q.run_once()
    p2 = c.get(f"/api/projects/{pid}").json()
    assert p2["lines"][0]["words"][0]["w"] == "תיקון"  # manual fix kept
    assert p2["suggestion"]["lines"]


def test_rejects_file_without_audio(env, tmp_path):
    c, q, _ = env
    silent = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=1", str(silent)], check=True)
    p = c.post("/api/projects", json={"project": {"title": "x"}, "source": {"kind": "file", "name": "v.mp4"}}).json()
    with silent.open("rb") as f:
        r = c.post(f"/api/projects/{p['id']}/upload", files={"file": ("v.mp4", f, "video/mp4")})
    assert r.status_code == 422


def test_words_to_lines_and_ass_line_timing():
    words = [{"w": f"w{i}", "s": i * 0.5, "e": i * 0.5 + 0.4} for i in range(10)] + [{"w": "x.", "s": 20, "e": 20.5}]
    lines = engines.words_to_lines(words)
    assert len(lines) == 3 and lines[0]["start"] == 0
    ass = engines.build_ass({"lines": lines, "timing": "line", "style": {}}, 1920, 1080)
    assert "\\kf" not in ass and ass.count("Dialogue:") == 3


def test_restart_marks_running_job_as_crash(tmp_path):
    (tmp_path / "jobs.json").write_text(json.dumps([{"id": "j1", "projectId": "p", "kind": "process", "status": "running", "stage": 2, "pct": 0, "attempt": 1, "log": []}]))
    cfg = Config(data_dir=tmp_path, engine="mock", ffmpeg="ffmpeg", ffprobe="ffprobe", auth=False)
    app = create_app(cfg, autostart=False)
    j = app.state.queue.list()[0]
    assert j["status"] == "failed" and j["crash"]
