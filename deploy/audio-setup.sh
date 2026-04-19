#!/usr/bin/env bash
# Interactive audio setup for Nigel.
#
# Does:
#   1. Makes sure bluez + pipewire-alsa + pulseaudio-utils are installed.
#   2. Scans for discoverable Bluetooth devices; lets you pick one to pair.
#   3. Pairs + trusts + connects it.
#   4. Waits for PipeWire to surface it as a sink + source.
#   5. Shows current sinks/sources and the sounddevice view.
#   6. Writes VOICE_INPUT_DEVICE / VOICE_OUTPUT_DEVICE into
#      /etc/picrawler-voice.env (preserving existing values).
#   7. Restarts picrawler-voice.
#
# Re-runnable. If you already have your speakerphone paired, it lets you
# just reselect which sounddevice index/name to use.

set -euo pipefail

say()  { printf "\n\033[1;36m▶ %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m⚠ %s\033[0m\n" "$*"; }
bad()  { printf "\033[1;31m✗ %s\033[0m\n" "$*" >&2; }

ENV_FILE="/etc/picrawler-voice.env"

# ---------------------------------------------------------- prereqs

say "ensuring audio packages are installed"
PKGS="bluez bluez-tools pulseaudio-utils pipewire-alsa"
MISSING=()
for p in $PKGS; do
  dpkg -l "$p" 2>/dev/null | grep -q "^ii" || MISSING+=("$p")
done
if [ ${#MISSING[@]} -gt 0 ]; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends "${MISSING[@]}"
else
  echo "  all present"
fi

sudo systemctl enable --now bluetooth >/dev/null 2>&1 || true
sudo rfkill unblock bluetooth >/dev/null 2>&1 || true

# Lingering keeps pi's user session (incl. PipeWire) running after we log out,
# which is what lets the picrawler-voice system service reach the PipeWire socket.
if ! loginctl show-user pi 2>/dev/null | grep -q "Linger=yes"; then
  sudo loginctl enable-linger pi
fi

# pi user needs to be in bluetooth group to access the agent
if ! groups pi 2>/dev/null | grep -qw bluetooth; then
  sudo usermod -aG bluetooth pi
  warn "added pi to bluetooth group — a relog/reboot is required for non-root"
fi

# ---------------------------------------------------------- optional BT pair

echo
read -r -p "Pair a new Bluetooth audio device now? (y/N) " REPLY
if [[ "${REPLY:-N}" =~ ^[Yy]$ ]]; then
  say "scanning for 12 seconds — put your speakerphone in pairing mode now"
  sudo bluetoothctl --timeout 12 scan on >/dev/null 2>&1 || true
  sudo bluetoothctl scan off >/dev/null 2>&1 || true
  echo
  echo "Discovered devices (named only):"
  mapfile -t NAMED < <(sudo bluetoothctl devices | awk '{mac=$2; $1=""; $2=""; sub(/^[ \t]+/,""); print mac "\t" $0}' | awk -F'\t' '$2 !~ /^[0-9A-F][0-9A-F]-/ {print}')
  if [ ${#NAMED[@]} -eq 0 ]; then
    warn "no named devices found. Make sure it's still in pairing mode and try again."
    exit 1
  fi
  for i in "${!NAMED[@]}"; do
    printf "  [%d] %s\n" "$i" "${NAMED[$i]}"
  done
  read -r -p "Pick index: " IDX
  MAC="$(echo "${NAMED[$IDX]}" | awk '{print $1}')"
  echo "Pairing $MAC..."
  sudo bluetoothctl --timeout 20 <<EOF
agent on
default-agent
pair $MAC
trust $MAC
connect $MAC
EOF
  sleep 3
  sudo bluetoothctl info "$MAC" | grep -E "Name|Paired|Trusted|Connected" || true
fi

# ---------------------------------------------------------- reload PipeWire

say "reloading PipeWire user session so ALSA bridge picks up new devices"
systemctl --user restart pipewire pipewire-pulse wireplumber 2>/dev/null || true
sleep 3

# ---------------------------------------------------------- select devices

say "current audio devices (sounddevice / PortAudio view):"
SD_OUT="$(/home/pi/nigel/.venv/bin/python - <<'PY'
import sounddevice as sd
for i, d in enumerate(sd.query_devices()):
    kind = "IN " if d["max_input_channels"] > 0 else ("OUT" if d["max_output_channels"] > 0 else "   ")
    print(f"[{i:2d}] {kind} {d['name'][:60]:60s} sr={d['default_samplerate']:.0f}")
PY
)"
echo "$SD_OUT"

echo
echo "(Enter the index or device name string. For Bluetooth speakerphones the"
echo " input is usually 'pipewire' and the output is usually 'default'.)"
echo
read -r -p "VOICE_INPUT_DEVICE  [pipewire]: " IN_DEV
read -r -p "VOICE_OUTPUT_DEVICE [default]:  " OUT_DEV
IN_DEV="${IN_DEV:-pipewire}"
OUT_DEV="${OUT_DEV:-default}"

read -r -p "Output volume percentage (0-100, blank = don't touch): " VOL_PCT

# ---------------------------------------------------------- write env

say "writing device selection to $ENV_FILE"
[ -f "$ENV_FILE" ] || sudo touch "$ENV_FILE"
sudo chmod 600 "$ENV_FILE"

# Drop existing device + volume lines then append fresh values
TMP="$(mktemp)"
sudo grep -v -E '^(VOICE_INPUT_DEVICE|VOICE_OUTPUT_DEVICE|VOICE_HW_INPUT_RATE|VOICE_HW_OUTPUT_RATE|VOICE_SPEAKER_VOLUME_PCT)=' "$ENV_FILE" 2>/dev/null | sudo tee "$TMP" >/dev/null || true
{
  cat "$TMP"
  echo "VOICE_INPUT_DEVICE=$IN_DEV"
  echo "VOICE_OUTPUT_DEVICE=$OUT_DEV"
  echo "VOICE_HW_INPUT_RATE=48000"
  echo "VOICE_HW_OUTPUT_RATE=48000"
  if [ -n "$VOL_PCT" ]; then
    echo "VOICE_SPEAKER_VOLUME_PCT=$VOL_PCT"
  fi
} | sudo tee "$ENV_FILE" >/dev/null
sudo rm -f "$TMP"

# ---------------------------------------------------------- restart + verify

say "restarting picrawler-voice"
sudo systemctl restart picrawler-voice
sleep 8
echo
sudo systemctl --no-pager --lines=5 status picrawler-voice || true

say "done"
echo "  watch live:  sudo journalctl -u picrawler-voice -f"
