#!/usr/bin/env bash
# KaraokeStudio — one-shot Linux server installer
# Usage: KARAOKE_DOMAIN=karaoke.example.com sudo ./install.sh
set -euo pipefail

INSTALL_DIR="/opt/karaoke-studio"
SERVICE_USER="karaoke"
REPO_URL="https://github.com/Roei-Barak/AI_Audio_Lab.git"

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info() { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
die()  { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "Run as root: sudo ./install.sh"

DOMAIN="${KARAOKE_DOMAIN:-}"
if [[ -z "$DOMAIN" ]]; then read -rp "Subdomain (e.g. karaoke.example.com): " DOMAIN; fi
[[ -n "$DOMAIN" ]] || die "Domain is required"
info "Domain: $DOMAIN"

info "Installing system packages..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip ffmpeg git curl \
    debian-keyring debian-archive-keyring apt-transport-https

if ! command -v caddy &>/dev/null; then
    info "Installing Caddy..."
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
        | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
        | tee /etc/apt/sources.list.d/caddy-stable.list
    apt-get update -qq && apt-get install -y caddy
fi
ok "Packages ready"

if ! id "$SERVICE_USER" &>/dev/null; then
    useradd --system --home "$INSTALL_DIR" --shell /bin/false "$SERVICE_USER"
fi

if [[ -d "$INSTALL_DIR/.git" ]]; then
    git -C "$INSTALL_DIR" pull
else
    git clone "$REPO_URL" "$INSTALL_DIR"
fi
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
ok "Code ready at $INSTALL_DIR"

python3.11 -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip -q
"$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt" -q
ok "Venv ready"

cp "$INSTALL_DIR/deploy/linux/karaoke-studio.service" /etc/systemd/system/
systemctl daemon-reload && systemctl enable karaoke-studio
ok "Service installed"

mkdir -p /etc/caddy
export KARAOKE_DOMAIN="$DOMAIN"
envsubst < "$INSTALL_DIR/deploy/linux/Caddyfile" > /etc/caddy/Caddyfile
systemctl enable caddy
ok "Caddy configured for $DOMAIN"

systemctl start karaoke-studio caddy
ok "Services started"

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo "Create admin user:"
echo "  sudo -u $SERVICE_USER $INSTALL_DIR/venv/bin/python -m api.server create-user admin <password> --admin"
echo "Then open: https://$DOMAIN/"
