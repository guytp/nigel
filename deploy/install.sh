#!/usr/bin/env bash
# One-shot installer for the PiCrawler MCP server on a Raspberry Pi.
# Assumes: Raspberry Pi OS, user "pi", already cloned into /home/pi/claude-bot.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/pi/claude-bot}"
cd "$REPO_DIR"

if [ ! -d .venv ]; then
  python3 -m venv .venv --system-site-packages
fi
# --system-site-packages so picamera2/opencv from apt are visible to the venv.

source .venv/bin/activate
pip install --upgrade pip wheel

# Always install the body (MCP + vision).
pip install -e ".[pi,vision]"

# Voice agent is optional — only install if the env file exists.
if [ -f /etc/picrawler-voice.env ]; then
  echo "Installing voice agent extras..."
  pip install -e ".[voice]"
fi

sudo cp deploy/picrawler-mcp.service /etc/systemd/system/picrawler-mcp.service
sudo systemctl daemon-reload
sudo systemctl enable picrawler-mcp.service
sudo systemctl restart picrawler-mcp.service

if [ -f /etc/picrawler-voice.env ]; then
  sudo cp deploy/picrawler-voice.service /etc/systemd/system/picrawler-voice.service
  sudo systemctl daemon-reload
  sudo systemctl enable picrawler-voice.service
  sudo systemctl restart picrawler-voice.service
fi

IP=$(hostname -I | awk '{print $1}')
echo
echo "Installed. Services:"
sudo systemctl --no-pager status picrawler-mcp.service || true
if [ -f /etc/picrawler-voice.env ]; then
  sudo systemctl --no-pager status picrawler-voice.service || true
fi
echo
echo "MCP endpoint: http://$IP:8765/mcp"
echo "MJPEG stream: http://$IP:9000/mjpg"
