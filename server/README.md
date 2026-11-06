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
- **Progress**: the render reports real progress from ffmpeg. Separation and transcription show an *estimate* (marked as such), because the models don't report progress.
- **Cancel**: stops the render immediately. Separation and transcription finish their current stage first.
- **Saves**: optimistic concurrency. `PUT /api/projects/{id}` returns 409 with the server copy when someone else saved first.
- **Pasted lyrics**: `POST /api/projects/{id}/lyrics` aligns them to the transcription's word timings (difflib), and words it can't match get estimated times marked low confidence. With no transcription, it creates line timing only; word timings are never invented.

## Accounts (private, no public signup)
- **First visit**: creates the admin (`/api/auth/setup`, only while no user exists).
- **Joining**: everyone else joins with a one-time invite link the admin copies and sends. Invites expire after 1–30 days and work once.
- **Roles**: `viewer` can watch, play and download. `editor` can also edit and run jobs. `admin` can also manage users, invites and settings. At least one admin always remains.
- **Password reset**: the admin creates a one-time reset link (valid 24h). Resetting signs out the user's other sessions.
- **Storage**: passwords are scrypt-hashed. Sessions are an HttpOnly cookie, so media playback is authenticated too. `LYRA_TOKEN` is a service token for scripts.
- **Local development**: `LYRA_AUTH=off` disables accounts.

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
| `LYRA_AUTH` | `on` | `off` disables accounts (local development only). |
| `LYRA_TOKEN` | unset | Service token for scripts: `Authorization: Bearer <token>` acts as admin. |
| `LYRA_CORS` | `http://localhost:5173,http://localhost:4173` | Allowed origins for a separately served UI. |
| `LYRA_FFMPEG` / `LYRA_FFPROBE` | from PATH or imageio-ffmpeg | |

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q                                            # mock engine + real ffmpeg (12 tests)
LYRA_ENGINE=mock LYRA_DATA_DIR=/tmp/lyra uvicorn lyra_server.app:app --port 8443 &
node ../app/e2e/server.cjs clip.mp4                  # browser end-to-end against the server
```

## API

`GET /api/health` · `POST /api/sources/youtube` · `GET|POST /api/projects` · `GET|PUT|DELETE /api/projects/{id}` ·
`POST /api/projects/{id}/upload` · `GET /api/projects/{id}/files/{source|vocals|instrumental|drums|bass|other|karaoke}?download=1` ·
`GET|POST /api/jobs` · `GET|DELETE /api/jobs/{id}` · `POST /api/jobs/{id}/cancel|retry|move` ·
`GET /api/projects/{id}/files/ass` · `POST /api/projects/{id}/lyrics` · `GET|POST|DELETE /api/projects/{id}/versions` · `GET /api/projects/{id}/versions/{vid}` ·
`GET|POST /api/outputs` · `GET|PUT /api/settings` ·
`GET /api/auth/status` · `POST /api/auth/setup|login|logout|accept|reset` · `GET /api/users` · `PATCH|DELETE /api/users/{id}` · `POST /api/users/{id}/reset` · `GET|POST /api/invites` · `DELETE /api/invites/{id}`

Use HTTPS (a reverse proxy) before exposing the server beyond your home network.
