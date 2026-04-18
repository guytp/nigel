#!/usr/bin/env bash
# Remove Nigel systemd services and venv. Does NOT touch the repo itself
# or the SunFounder hardware libraries — those are harmless to leave behind.
#
#   ./deploy/uninstall.sh          # stop + disable + remove units + venv
#   ./deploy/uninstall.sh --purge  # also remove /etc/picrawler-*.env files
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PURGE=0
for arg in "$@"; do
  case "$arg" in
    --purge) PURGE=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

say() { printf "\n\033[1;36m▶ %s\033[0m\n" "$*"; }

for svc in picrawler-voice picrawler-mcp; do
  if systemctl list-unit-files 2>/dev/null | grep -q "^${svc}.service"; then
    say "disabling ${svc}.service"
    sudo systemctl stop "${svc}" 2>/dev/null || true
    sudo systemctl disable "${svc}" 2>/dev/null || true
    sudo rm -f "/etc/systemd/system/${svc}.service"
  fi
done
sudo systemctl daemon-reload

if [ -d "$REPO_DIR/.venv" ]; then
  say "removing venv at $REPO_DIR/.venv"
  rm -rf "$REPO_DIR/.venv"
fi

if [ "$PURGE" = "1" ]; then
  for env_file in /etc/picrawler-mcp.env /etc/picrawler-voice.env; do
    if [ -f "$env_file" ]; then
      say "removing $env_file"
      sudo rm -f "$env_file"
    fi
  done
fi

say "uninstall complete"
echo "  the checkout at $REPO_DIR is untouched — rm -rf it yourself if desired"
echo "  SunFounder hardware libs remain installed"
