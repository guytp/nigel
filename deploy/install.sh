#!/usr/bin/env bash
# Install (or reinstall) Nigel. Idempotent — safe to run repeatedly.
#
# Called by bootstrap.sh, or directly from a checked-out repo:
#
#   cd /home/pi/nigel
#   ./deploy/install.sh             # install/upgrade
#   ./deploy/install.sh --reset     # wipe venv first
#
# Environment:
#   REPO_DIR   — default: script's parent's parent (autodetected)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(dirname "$SCRIPT_DIR")}"
RESET=0
for arg in "$@"; do
  case "$arg" in
    --reset) RESET=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

say() { printf "\n\033[1;36m▶ %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m⚠ %s\033[0m\n" "$*"; }

cd "$REPO_DIR"
say "installing from $REPO_DIR (reset=$RESET)"

# ------------------------------------------------------------ stop services

for svc in picrawler-voice picrawler-mcp; do
  if systemctl list-unit-files 2>/dev/null | grep -q "^${svc}.service"; then
    say "stopping ${svc}.service"
    sudo systemctl stop "${svc}" 2>/dev/null || true
  fi
done

# ------------------------------------------------------------ venv

if [ "$RESET" = "1" ] && [ -d .venv ]; then
  say "removing existing venv (--reset)"
  rm -rf .venv
fi

if [ ! -d .venv ]; then
  say "creating venv (--system-site-packages so vilib/picamera2 are visible)"
  python3 -m venv .venv --system-site-packages
fi

# ------------------------------------------------------------ pip install

say "upgrading pip + installing nigel"
source .venv/bin/activate
pip install --upgrade pip wheel

# SunFounder libs (picrawler/robot_hat/vilib) are NOT installed via pip —
# they come from bootstrap.sh (sudo python3 setup.py install) and the venv
# sees them via --system-site-packages. Do not add them here.
EXTRAS="vision"
if [ -f /etc/picrawler-voice.env ]; then
  EXTRAS="${EXTRAS},voice"
  say "voice env file detected — including voice extras"
fi
pip install -e ".[${EXTRAS}]"

# ------------------------------------------------------------ systemd units

say "installing systemd units"
sudo cp deploy/picrawler-mcp.service /etc/systemd/system/picrawler-mcp.service

if [ -f /etc/picrawler-voice.env ]; then
  sudo cp deploy/picrawler-voice.service /etc/systemd/system/picrawler-voice.service
else
  # Remove any prior voice unit if the env file isn't there anymore.
  if [ -f /etc/systemd/system/picrawler-voice.service ]; then
    warn "no /etc/picrawler-voice.env — removing picrawler-voice.service"
    sudo systemctl disable picrawler-voice.service 2>/dev/null || true
    sudo rm -f /etc/systemd/system/picrawler-voice.service
  fi
fi

sudo systemctl daemon-reload
sudo systemctl enable picrawler-mcp.service
sudo systemctl restart picrawler-mcp.service

if [ -f /etc/systemd/system/picrawler-voice.service ]; then
  sudo systemctl enable picrawler-voice.service
  sudo systemctl restart picrawler-voice.service
fi

# ------------------------------------------------------------ status

IP=$(hostname -I | awk '{print $1}')
say "install complete"
echo
sudo systemctl --no-pager --lines=3 status picrawler-mcp.service || true
if [ -f /etc/systemd/system/picrawler-voice.service ]; then
  echo
  sudo systemctl --no-pager --lines=3 status picrawler-voice.service || true
fi
echo
echo "  MCP endpoint:  http://${IP}:8765/mcp"
echo "  MJPEG stream:  http://${IP}:9000/mjpg"
