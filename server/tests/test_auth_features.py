import os
import threading
import time
from pathlib import Path

import pytest

os.environ["LYRA_NO_AUTOAPP"] = "1"
from fastapi.testclient import TestClient  # noqa: E402

from lyra_server import engines  # noqa: E402
from lyra_server.app import create_app  # noqa: E402
from lyra_server.config import Config  # noqa: E402


@pytest.fixture()
def app(tmp_path: Path):
    return create_app(Config(data_dir=tmp_path, engine="mock", ffmpeg="ffmpeg", ffprobe="ffprobe", auth=True), autostart=False)


def test_private_accounts_flow(app):
    c = TestClient(app)
    assert c.get("/api/projects").status_code == 401
    assert c.get("/api/auth/status").json()["needsSetup"] is True
    r = c.post("/api/auth/setup", json={"name": "דניאל", "user": "daniel", "password": "password1"})
    assert r.status_code == 200 and r.json()["user"]["role"] == "admin"
    assert c.post("/api/auth/setup", json={"name": "x", "user": "x2", "password": "password1"}).status_code == 409  # no second setup / public signup
    assert c.get("/api/projects").status_code == 200  # cookie session
    inv = c.post("/api/invites", json={"name": "אורי", "role": "viewer"}).json()
    viewer = TestClient(app)
    r = viewer.post("/api/auth/accept", json={"token": "https://lyra.home/join/" + inv["token"], "user": "uri", "password": "password2"})
    assert r.status_code == 200 and r.json()["user"]["role"] == "viewer"
    assert TestClient(app).post("/api/auth/accept", json={"token": inv["token"], "user": "uri2", "password": "password2"}).status_code == 410  # one-time
    assert viewer.get("/api/projects").status_code == 200
    assert viewer.post("/api/projects", json={"project": {"title": "x"}}).status_code == 403  # viewer cannot write
    assert viewer.get("/api/invites").status_code == 403
    uid = next(u["id"] for u in c.get("/api/users").json() if u["user"] == "uri")
    assert c.patch(f"/api/users/{uid}", json={"role": "editor"}).json()["role"] == "editor"
    assert viewer.post("/api/projects", json={"project": {"title": "x"}}).status_code == 200
    me = next(u["id"] for u in c.get("/api/users").json() if u["user"] == "daniel")
    assert c.patch(f"/api/users/{me}", json={"role": "viewer"}).status_code == 409  # last admin
    tok = c.post(f"/api/users/{uid}/reset").json()["token"]
    assert TestClient(app).post("/api/auth/reset", json={"token": tok, "password": "newpass99"}).status_code == 200
    assert viewer.get("/api/projects").status_code == 401  # old sessions revoked after reset
    assert TestClient(app).post("/api/auth/login", json={"user": "uri", "password": "password2"}).status_code == 401
    assert TestClient(app).post("/api/auth/login", json={"user": "uri", "password": "newpass99"}).status_code == 200
    c.post("/api/auth/logout")
    assert c.get("/api/projects").status_code == 401


def test_align_lyrics_uses_asr_timing_and_never_invents_word_timing():
    asr = [{"w": w, "s": i * 1.0, "e": i * 1.0 + 0.6} for i, w in enumerate("הלילה יורד לאט על העיר ואני סופר".split())]
    lines, timing = engines.align_lyrics("הלילה יורד לאט על העיר\nואני סופר את האורות", asr, 30)
    assert timing == "word" and len(lines) == 2
    assert lines[0]["words"][1]["s"] == 1.0  # copied from ASR
    assert lines[1]["words"][2]["conf"] == "low"  # interpolated, marked low confidence
    lines2, timing2 = engines.align_lyrics("שורה אחת\nשורה שתיים", [], 20)
    assert timing2 == "line" and lines2[1]["start"] > lines2[0]["end"]


def test_settings_versions_outputs_and_ass(tmp_path):
    a = create_app(Config(data_dir=tmp_path, engine="mock", ffmpeg="ffmpeg", ffprobe="ffprobe", auth=False), autostart=False)
    c = TestClient(a)
    assert c.put("/api/settings", json={"defaultProfile": "איכותי", "evil": 1}).json() == {"defaultProfile": "איכותי"}
    p = c.post("/api/projects", json={"project": {"title": "t", "timing": "word", "lines": []}, "source": {"kind": "demo"}}).json()
    lines = [{"words": [{"w": "שלום", "s": 1, "e": 1.5}, {"w": "עולם", "s": 1.6, "e": 2.2}]}]
    c.put(f"/api/projects/{p['id']}", json={"baseRev": p["serverRev"], "project": {**p, "lines": lines}})
    v = c.post(f"/api/projects/{p['id']}/versions", json={"label": "נקודה", "lines": lines, "style": {}}).json()
    assert c.get(f"/api/projects/{p['id']}/versions").json()[0]["lineTexts"] == ["שלום עולם"]
    assert c.get(f"/api/projects/{p['id']}/versions/{v['id']}").json()["lines"] == lines
    ass = c.get(f"/api/projects/{p['id']}/files/ass?download=1")
    assert ass.status_code == 200 and "\\kf" in ass.text and "attachment" in ass.headers["content-disposition"]
    c.post("/api/outputs", json={"projectId": p["id"], "kind": "srt"})
    assert c.get("/api/outputs").json()[0]["kind"] == "srt"


def test_cancel_stops_render_midway(tmp_path):
    long = tmp_path / "long.mp4"
    import subprocess

    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=120", "-f", "lavfi", "-i", "sine=duration=120",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(long)], check=True)
    a = create_app(Config(data_dir=tmp_path / "d", engine="mock", ffmpeg="ffmpeg", ffprobe="ffprobe", auth=False), autostart=False)
    c, q = TestClient(a), a.state.queue
    p = c.post("/api/projects", json={"project": {"title": "t", "export": {"res": "1080p"}}, "source": {"kind": "file", "name": "long.mp4"}}).json()
    with long.open("rb") as f:
        c.post(f"/api/projects/{p['id']}/upload", files={"file": ("long.mp4", f, "video/mp4")})
    jid = c.post("/api/jobs", json={"projectId": p["id"]}).json()["id"]
    t = threading.Thread(target=q.run_once)
    t.start()
    for _ in range(600):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["stageName"] == "render" and j["pct"] > 0:
            break
        time.sleep(0.1)
    assert j["stageName"] == "render" and 0 < j["pct"] < 100  # real ffmpeg progress
    c.post(f"/api/jobs/{jid}/cancel")
    t.join(30)
    j = c.get(f"/api/jobs/{jid}").json()
    assert j["status"] == "cancelled" and j["stageName"] == "render"
