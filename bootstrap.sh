#!/usr/bin/env bash
# One-shot bootstrap for Nigel — the PiCrawler MCP server + voice agent.
#
# Usage (from a fresh Raspberry Pi OS install):
#
#   curl -fsSL https://raw.githubusercontent.com/guytp/nigel/main/bootstrap.sh | bash
#
# Or, to inspect first (recommended):
#
#   curl -fsSL https://raw.githubusercontent.com/guytp/nigel/main/bootstrap.sh -o /tmp/bootstrap.sh
#   less /tmp/bootstrap.sh
#   bash /tmp/bootstrap.sh
#
# Environment overrides:
#   REPO_URL   — git URL to clone (default below)
#   REPO_DIR   — where to clone (default /home/pi/nigel)
#   REF        — branch or tag to check out (default main)
#   RESET      — "1" to wipe existing install (venv + repo) before reinstalling
#   SKIP_SF    — "1" to skip the SunFounder hardware library install
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/guytp/nigel.git}"
REPO_DIR="${REPO_DIR:-/home/pi/nigel}"
REF="${REF:-main}"
RESET="${RESET:-0}"
SKIP_SF="${SKIP_SF:-0}"

say() { printf "\n\033[1;36m▶ %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m⚠ %s\033[0m\n" "$*"; }
err() { printf "\033[1;31m✗ %s\033[0m\n" "$*" >&2; }

# ---------------------------------------------------------------- preflight

if [ "$(uname -s)" != "Linux" ]; then
  err "bootstrap.sh is for Raspberry Pi OS (Linux); you're on $(uname -s)"
  exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
  err "apt-get not found — this expects Raspberry Pi OS / Debian"
  exit 1
fi

if [ "$(id -u)" -eq 0 ]; then
  err "run as the 'pi' user, not root — the script uses sudo where needed"
  exit 1
fi

say "bootstrap → repo=$REPO_URL  dir=$REPO_DIR  ref=$REF  reset=$RESET  skip_sf=$SKIP_SF"

# ---------------------------------------------------------------- apt deps

say "installing apt prerequisites"
sudo apt-get update
# --no-install-recommends matters on Trixie: python3-opencv's Recommends
# chain pulls in libatlas-base-dev which was retired, and apt fails the
# whole batch when a Recommends has no installation candidate. Our package
# set doesn't need any of the recommended extras anyway.
sudo apt-get install -y --no-install-recommends \
  git python3 python3-pip python3-venv python3-opencv ffmpeg \
  i2c-tools portaudio19-dev curl ca-certificates

# ---------------------------------------------------------------- sunfounder libs

if [ "$SKIP_SF" != "1" ]; then
  say "checking SunFounder libraries"
  missing=0
  for mod in robot_hat vilib picrawler; do
    if ! python3 -c "import ${mod}" 2>/dev/null; then
      missing=1
      break
    fi
  done
  if [ "$missing" -eq 1 ]; then
    say "installing SunFounder libraries (robot_hat, vilib, picrawler)"
    SFDIR="${HOME}/sunfounder-src"
    mkdir -p "$SFDIR"

    if [ ! -d "$SFDIR/robot-hat" ]; then
      # robot-hat's default branch IS v2.0; no v4.0 exists.
      git clone -b v2.0 https://github.com/sunfounder/robot-hat.git "$SFDIR/robot-hat"
    fi
    (cd "$SFDIR/robot-hat" && sudo python3 setup.py install)

    if [ ! -d "$SFDIR/vilib" ]; then
      git clone -b picamera2 https://github.com/sunfounder/vilib.git "$SFDIR/vilib"
    fi
    (cd "$SFDIR/vilib" && sudo python3 install.py)

    if [ ! -d "$SFDIR/picrawler" ]; then
      # picrawler ships on `main` (v2.1.x); no v2.0 tag.
      git clone https://github.com/sunfounder/picrawler.git "$SFDIR/picrawler"
    fi
    (cd "$SFDIR/picrawler" && sudo python3 setup.py install)

    warn "SunFounder installers enabled I2C/SPI/audio — a reboot may be required"
    warn "if the MCP server logs 'hardware backend: mock' after install, reboot and retry"
  else
    say "SunFounder libraries already present — skipping"
  fi
else
  warn "skipping SunFounder install (SKIP_SF=1)"
fi

# ---------------------------------------------------------------- stop existing services

for svc in picrawler-voice picrawler-mcp; do
  if systemctl list-unit-files | grep -q "^${svc}.service"; then
    say "stopping existing ${svc}.service"
    sudo systemctl stop "${svc}" 2>/dev/null || true
  fi
done

# ---------------------------------------------------------------- clone / update repo

if [ "$RESET" = "1" ] && [ -d "$REPO_DIR" ]; then
  say "RESET=1 — removing existing install at $REPO_DIR"
  sudo rm -rf "$REPO_DIR"
fi

if [ -d "$REPO_DIR/.git" ]; then
  say "updating existing checkout at $REPO_DIR"
  git -C "$REPO_DIR" fetch --all --tags --prune
  git -C "$REPO_DIR" reset --hard "origin/${REF}"
else
  say "cloning $REPO_URL → $REPO_DIR"
  git clone --branch "$REF" "$REPO_URL" "$REPO_DIR"
fi

# ---------------------------------------------------------------- hand off to install.sh

say "running deploy/install.sh"
cd "$REPO_DIR"
chmod +x deploy/install.sh
./deploy/install.sh

IP=$(hostname -I | awk '{print $1}')
say "bootstrap complete"
cat <<EOF

  MCP endpoint:   http://${IP}:8765/mcp
  MJPEG stream:   http://${IP}:9000/mjpg

  Logs:
    sudo journalctl -u picrawler-mcp -f
    sudo journalctl -u picrawler-voice -f   # if voice is enabled

  To enable the voice agent later:
    sudo cp $REPO_DIR/deploy/picrawler-voice.env.example /etc/picrawler-voice.env
    sudoedit /etc/picrawler-voice.env       # add OPENAI_API_KEY + MCP_TOKEN
    sudo chmod 600 /etc/picrawler-voice.env
    cd $REPO_DIR && ./deploy/install.sh

  To uninstall:
    cd $REPO_DIR && ./deploy/uninstall.sh

EOF

# --------------------------------------------------------- first-boot hints
if python3 -c "import picrawler, robot_hat, vilib" 2>/dev/null; then
  say "smoke test (sanity-check the hardware)"
  echo "  Stop the service first so we can open the camera:"
  echo "    sudo systemctl stop picrawler-mcp"
  echo "    cd $REPO_DIR && .venv/bin/python -m scripts.smoke"
  echo "    sudo systemctl start picrawler-mcp"
else
  warn "SunFounder libs not importable by python3 yet — a reboot is likely required"
  warn "  sudo reboot"
  warn "  after reboot, re-run this bootstrap OR just: sudo systemctl restart picrawler-mcp"
fi
