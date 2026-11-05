# Lyra Studio server

FastAPI processing server for the Lyra Studio frontend in `../app`. It runs the karaoke pipeline on your own machine:

1. **import**: download from YouTube with yt-dlp, or take an uploaded file (up to 20GB and 3 hours). Private, restricted or DRM sources are refused with a clear message; Lyra never bypasses protections.
2. **separate**: audio-separator, 2 stems (Vocals + Instrumental) or 4 (adds drums, bass, other).
3. **transcribe**: Whisper with word timestamps (`ivrit-ai/whisper-large-v3-turbo` for Hebrew).
4. **align**: words grouped into lines. If the project already has manual fixes, the new output is stored as a **suggestion** and nothing is overwritten.
5. **render**: ffmpeg burns ASS karaoke (`\kf` word sweeps) onto the original video, mutes its audio and uses the instrumental as the soundtrack. Length and timing match the source.

Behaviour:
- One heavy GPU job runs at a time.
- Every stage is a checkpoint, so a crash or a server restart resumes from the stage that failed.
- Cancel takes effect at the end of the current stage.
- Saves use optimistic concurrency: `PUT /api/projects/{id}` returns 409 with the server copy when someone else saved first.

## Run

```bash
cd server
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                     # torch: install the CUDA build for your GPU first
(cd ../app && npm install && npm run build)         # optional: the server then serves the UI at /
uvicorn lyra_server.app:app --host 0.0.0.0 --port 8443
```

Open http://localhost:8443. The UI detects the server automatically ("שרת Lyra" badge). Without a server it runs in demo mode.

For frontend development, run `npm run dev` in `app/`; it proxies `/api` to port 8443.

| Variable | Default | |
|---|---|---|
| `LYRA_ENGINE` | `ai` | `mock` runs fast stand-ins (phase-cancellation instrumental, placeholder words) for development and tests. Rendering is real in both modes. |
| `LYRA_DATA_DIR` | `Karaoke_Output/lyra` | Projects, stems, renders, `jobs.json`. |
| `LYRA_TOKEN` | unset | When set, every API call needs `Authorization: Bearer <token>`. |
| `LYRA_CORS` | `http://localhost:5173,http://localhost:4173` | Allowed origins for a separately served UI. |
| `LYRA_FFMPEG` / `LYRA_FFPROBE` | from PATH or imageio-ffmpeg | |

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q                                            # mock engine + real ffmpeg
LYRA_ENGINE=mock LYRA_DATA_DIR=/tmp/lyra uvicorn lyra_server.app:app --port 8443 &
node ../app/e2e/server.cjs clip.mp4                  # browser end-to-end against the server
```

## API

`GET /api/health` · `POST /api/sources/youtube` · `GET|POST /api/projects` · `GET|PUT|DELETE /api/projects/{id}` ·
`POST /api/projects/{id}/upload` · `GET /api/projects/{id}/files/{source|vocals|instrumental|drums|bass|other|karaoke}?download=1` ·
`GET|POST /api/jobs` · `GET|DELETE /api/jobs/{id}` · `POST /api/jobs/{id}/cancel|retry|move`

Not yet: server-side user accounts. The UI's sign-in and invites are local to the browser; use `LYRA_TOKEN` to protect a server exposed beyond your own computer.
