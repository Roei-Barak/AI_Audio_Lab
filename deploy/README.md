# KaraokeStudio — Deployment Guide

## Overview

| Component | Description |
|---|---|
| **Backend** | Python (FastAPI + uvicorn) on port 8000 |
| **Frontend** | React SPA served from `web/dist/` by FastAPI |
| **Reverse proxy** | Caddy — automatic HTTPS via Let's Encrypt |
| **Auth** | `KARAOKE_AUTH_MODE=required` (multi-user) or `none` (standalone) |

---

## 1. Linux — systemd + Caddy

### Prerequisites
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y python3.11 python3.11-venv ffmpeg git
```

### Install
```bash
git clone <repo-url> /opt/karaoke-studio
cd /opt/karaoke-studio

python3.11 -m venv venv
venv/bin/pip install -r requirements.txt

# Build web SPA
cd web && npm ci && npm run build && cd ..

# Copy service & Caddy config
sudo cp deploy/linux/karaoke-studio.service /etc/systemd/system/
sudo cp deploy/linux/Caddyfile              /etc/caddy/Caddyfile

# Set your domain
sudo sed -i 's/{$KARAOKE_DOMAIN}/karaoke.example.com/' /etc/caddy/Caddyfile
sudo sed -i 's/karaoke.example.com/karaoke.example.com/' /etc/systemd/system/karaoke-studio.service

# Create service user
sudo useradd -r -s /sbin/nologin karaoke
sudo chown -R karaoke:karaoke /opt/karaoke-studio

sudo systemctl daemon-reload
sudo systemctl enable --now karaoke-studio
sudo systemctl enable --now caddy
```

### Set JWT secret
Edit `/etc/systemd/system/karaoke-studio.service`:
```
Environment=KARAOKE_JWT_SECRET=<your-long-random-secret>
```
Generate a secret: `python3 -c "import secrets; print(secrets.token_hex(32))"`

### Create first admin
```bash
cd /opt/karaoke-studio
sudo -u karaoke venv/bin/python api/server.py create-user admin <password> --admin
```

---

## 2. Windows — NSSM service

### Prerequisites
- Python 3.11+ (from python.org)
- [NSSM](https://nssm.cc) (auto-downloaded by script if missing)
- Optional: [Caddy](https://caddyserver.com/download) for HTTPS

### Install (run as Administrator)
```powershell
cd deploy\windows

# Basic (local access only):
.\install-service.ps1 -AuthMode required

# With HTTPS subdomain:
.\install-service.ps1 -Domain karaoke.example.com -AuthMode required

# Standalone mode (no login):
.\install-service.ps1 -AuthMode none
```

The script will:
1. Copy files to `C:\KaraokeStudio`
2. Create a Python venv and install dependencies
3. Register the backend as a Windows service via NSSM
4. Optionally install Caddy as a service for HTTPS

### Create first admin
```powershell
cd C:\KaraokeStudio
.\venv\Scripts\python.exe api\server.py create-user admin <password> --admin
```

---

## 3. Docker — GPU passthrough

### Prerequisites
- Docker + Docker Compose
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU access

### Configure
```bash
cp deploy/docker/.env.example deploy/docker/.env
# Edit .env:
#   KARAOKE_DOMAIN=karaoke.example.com
#   KARAOKE_JWT_SECRET=<secret>
```

### Run
```bash
cd deploy/docker
docker compose up -d

# Create first admin
docker compose exec api python api/server.py create-user admin <password> --admin
```

The `caddy` service will automatically obtain a Let's Encrypt certificate for your domain.

---

## 4. DNS & Domain Setup

1. Point your subdomain A-record to the server's public IP:
   ```
   karaoke.example.com  A  <server-public-ip>
   ```
2. If behind a router/NAT, forward **port 80 and 443** to the server.
3. Caddy handles Let's Encrypt HTTP-01 challenge automatically — no manual cert needed.

---

## 5. WPF Client — Connecting to Your Server

On first launch, the client shows a login screen. Enter:
- **Server URL**: `https://karaoke.example.com` (or `http://localhost:8000` for LAN)
- **Username / Password**: credentials you created above

The URL and JWT token are stored encrypted (DPAPI) in `%APPDATA%\KaraokeStudio\config.json`.

---

## 6. Building the Client

### Client EXE (connects to remote server)
```powershell
cd KaraokeStudio\KaraokeStudio.WPF
dotnet publish -c Release -r win-x64 --self-contained true

cd ..\..\installer
.\build-installers.ps1   # runs ISCC to produce KaraokeStudio-Client-Setup-1.0.0.exe
```

### Standalone EXE (local backend, no login)
```powershell
# 1. Prepare Python embedded distribution
$embed = "standalone-python"
New-Item -ItemType Directory $embed -Force
# Download Python 3.11 embeddable from python.org and extract to $embed
# Then install packages:
.\standalone-python\python.exe -m pip install -r requirements.txt --target standalone-python\Lib\site-packages

# 2. Build WPF + web SPA
dotnet publish KaraokeStudio\KaraokeStudio.WPF -c Release -r win-x64 --self-contained
cd web && npm ci && npm run build && cd ..

# 3. Run ISCC
cd installer
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" KaraokeStudio-Standalone.iss
```

---

## 7. Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `KARAOKE_AUTH_MODE` | `required` | `required` = JWT auth; `none` = no auth (standalone) |
| `KARAOKE_JWT_SECRET` | `change-me-in-production-please` | Secret for signing JWT tokens |
| `KARAOKE_OUTPUT_DIR` | `Karaoke_Output` | Where processed files are stored |
| `KARAOKE_CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `KARAOKE_DOMAIN` | — | Used by Docker Compose Caddyfile |
| `KARAOKE_PORT` | `8000` | Backend listening port |
