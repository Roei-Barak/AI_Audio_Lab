@echo off
REM Lyra Studio: builds the UI (first run) and starts the processing server at http://localhost:8443
cd /d %~dp0
if not exist server\.venv (
  python -m venv server\.venv
  server\.venv\Scripts\pip install -r server\requirements.txt
)
if not exist app\dist (
  pushd app && call npm install && call npm run build && popd
)
cd server
start "" http://localhost:8443
..\server\.venv\Scripts\uvicorn lyra_server.app:app --host 0.0.0.0 --port 8443
