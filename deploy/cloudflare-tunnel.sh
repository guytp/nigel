#!/usr/bin/env bash
# Install cloudflared and start a quick tunnel to the MCP server so claude.ai
# (or any web client) can reach Nigel from outside the LAN.
#
# Quick tunnels give you a random *.trycloudflare.com URL with no Cloudflare
# account needed. URL changes every restart — fine for dev, swap to a named
# tunnel when you want stability.
#
# Usage:
#   ./deploy/cloudflare-tunnel.sh              # install + enable systemd unit
#   ./deploy/cloudflare-tunnel.sh --uninstall  # remove
#
# After install:
#   sudo systemctl status picrawler-tunnel
#   sudo journalctl -u picrawler-tunnel -f | grep -E "trycloudflare|INF|ERR"
# The "trycloudflare.com" URL is your MCP endpoint.
set -euo pipefail

UNINSTALL=0
for arg in "$@"; do
  case "$arg" in
    --uninstall) UNINSTALL=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

say()  { printf "\n\033[1;36m▶ %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m⚠ %s\033[0m\n" "$*"; }

if [ "$UNINSTALL" = 1 ]; then
  say "stopping + removing picrawler-tunnel"
  sudo systemctl stop picrawler-tunnel 2>/dev/null || true
  sudo systemctl disable picrawler-tunnel 2>/dev/null || true
  sudo rm -f /etc/systemd/system/picrawler-tunnel.service
  sudo systemctl daemon-reload
  warn "cloudflared binary left installed; remove with: sudo apt-get remove cloudflared"
  exit 0
fi

# ---------------------------------------------------------- install cloudflared

if ! command -v cloudflared >/dev/null 2>&1; then
  say "installing cloudflared from Cloudflare's apt repo"
  sudo mkdir -p /usr/share/keyrings
  curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg |
    sudo tee /usr/share/keyrings/cloudflare-main.gpg > /dev/null
  CODENAME="$(lsb_release -cs 2>/dev/null || echo bookworm)"
  echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared $CODENAME main" |
    sudo tee /etc/apt/sources.list.d/cloudflared.list > /dev/null
  sudo apt-get update
  sudo apt-get install -y cloudflared
else
  say "cloudflared already installed ($(cloudflared --version 2>&1 | head -1))"
fi

# ---------------------------------------------------------- systemd unit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say "installing picrawler-tunnel.service"
sudo cp "$SCRIPT_DIR/picrawler-tunnel.service" /etc/systemd/system/picrawler-tunnel.service
sudo systemctl daemon-reload
sudo systemctl enable picrawler-tunnel
sudo systemctl restart picrawler-tunnel

# ---------------------------------------------------------- wait for URL

say "waiting for Cloudflare to assign a URL (up to 30s)..."
URL=""
for _ in $(seq 1 30); do
  URL="$(sudo journalctl -u picrawler-tunnel --since '1 min ago' --no-pager 2>/dev/null |
    grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -1 || true)"
  if [ -n "$URL" ]; then break; fi
  sleep 1
done

echo
if [ -n "$URL" ]; then
  cat <<EOF

  \033[1;32m✓ Tunnel is up\033[0m
  Public MCP endpoint:  $URL/mcp

  In claude.ai, add this as an MCP server.
  If you've set MCP_TOKEN, also add header: Authorization: Bearer <token>

  URL changes each restart (quick tunnels are ephemeral). Promote to a
  named tunnel for a stable subdomain when ready.
EOF
else
  warn "no URL surfaced yet — check: sudo journalctl -u picrawler-tunnel -f"
fi
echo
