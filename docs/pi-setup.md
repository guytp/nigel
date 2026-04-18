# Pi setup — from flash to first move

End-to-end path: blank SD card → assembled PiCrawler → Claude Code controlling the robot through the MCP server. Budget about an hour if the hardware is already built.

> **Shortcut:** after flashing and SSH'ing in, you can skip most of steps 2 and 4 by running the one-liner from the main README:
> ```bash
> curl -fsSL https://raw.githubusercontent.com/guytp/nigel/main/bootstrap.sh | bash
> ```
> This doc remains as the canonical reference and for troubleshooting.

## 0. Hardware checklist

- SunFounder PiCrawler, assembled and calibrated per SunFounder's own docs
- Raspberry Pi 5 (or 4B / Zero 2W) with SD card
- USB-C power (PiCrawler power pack is fine) — servos draw amps, don't underfeed it
- Camera cable connected (CSI)
- Wi-Fi credentials on hand
- A development machine on the same LAN (your laptop, running Claude Code)

## 1. Flash the OS

1. Install Raspberry Pi Imager.
2. Flash **Raspberry Pi OS (64-bit, Bookworm)**. Lite is fine — we don't need a desktop.
3. Under "Advanced options" (gear icon):
   - set hostname: `picrawler` (gives you `picrawler.local`)
   - enable SSH with your public key
   - set user: `pi` (matches the systemd unit — change both if you want something else)
   - set Wi-Fi
   - set locale + keyboard
4. Write, boot the Pi, wait ~1 minute.

## 2. First SSH + install SunFounder base libraries

```bash
ssh pi@picrawler.local

sudo apt update && sudo apt -y full-upgrade
sudo apt -y install git python3-pip python3-venv python3-opencv libatlas-base-dev \
                    ffmpeg i2c-tools portaudio19-dev

# SunFounder's one-shot installer — brings in robot_hat, vilib, picrawler,
# enables I2C/SPI, sets up audio routing.
cd ~
git clone https://github.com/sunfounder/robot-hat.git -b v2.0
cd robot-hat && sudo python3 setup.py install && cd ~

git clone https://github.com/sunfounder/vilib.git -b picamera2
cd vilib && sudo python3 install.py && cd ~

git clone https://github.com/sunfounder/picrawler.git
cd picrawler && sudo python3 setup.py install && cd ~

sudo reboot
```

Reboot matters — I2C group perms and camera overlays only take effect after.

## 3. Validate the hardware stack in isolation

```bash
ssh pi@picrawler.local
cd picrawler/examples
sudo python3 0_calibration.py   # or whichever SunFounder calibration you used
sudo python3 1_ready.py         # should stand
sudo python3 2_move.py          # should walk forward
```

If these don't work, stop here and debug with SunFounder's docs — nothing we build on top will fix servo wiring.

Camera check: `python3 -c "from vilib import Vilib; Vilib.camera_start(); Vilib.display(local=False, web=True); input()"` then browse to `http://picrawler.local:9000/mjpg`.

## 4. Install Nigel

```bash
cd ~
git clone https://github.com/guytp/nigel.git
cd nigel

# Optional: lock in an auth token before the service goes live
sudo tee /etc/picrawler-mcp.env > /dev/null <<EOF
MCP_TOKEN=$(openssl rand -hex 32)
EOF
sudo chmod 600 /etc/picrawler-mcp.env
cat /etc/picrawler-mcp.env   # copy this for step 5

./deploy/install.sh
```

`install.sh` creates a venv (with `--system-site-packages` so the SunFounder installs above are visible), pip-installs `mcp-picrawler`, drops the systemd unit, enables and starts the service.

Verify:

```bash
systemctl status picrawler-mcp
journalctl -u picrawler-mcp -f
# should see: "hardware backend: real"
```

Open the MJPEG stream in a browser: `http://picrawler.local:9000/mjpg`. If vilib started, you see a live feed.

## 5. Connect Claude Code to it

On your dev machine, add to project `.mcp.json` (or `~/.claude.json`):

```jsonc
{
  "mcpServers": {
    "picrawler": {
      "type": "http",
      "url": "http://picrawler.local:8765/mcp",
      "headers": {
        "Authorization": "Bearer PUT_THE_TOKEN_HERE"
      }
    }
  }
}
```

Restart Claude Code. The MCP server should show up in `/mcp` with 11 tools and 2 resources.

## 6. First moves

Ask Claude to: `read distance`, then `scan`, then `move forward 1 step`, then `snapshot`. The snapshot should come back showing the world from the crawler's camera.

If you want a human cockpit view while the agent drives: keep `http://picrawler.local:9000/mjpg` open in a browser tab.

## Troubleshooting

| symptom | likely cause |
|---|---|
| `hardware backend: mock` in journal | the SunFounder libs weren't installed as root / `--system-site-packages` missed them. Re-run step 2 with `sudo`. |
| 401 Unauthorized from Claude Code | header not set or wrong token. Check `/etc/picrawler-mcp.env` vs your `.mcp.json`. |
| "camera frame not ready" on `snapshot` | vilib camera didn't start. Check `journalctl` — often a CSI cable seated wrong. |
| distance always 0 | ultrasonic pins differ on your build. Edit `RealHardware.__init__` in `hardware.py` (currently assumes `D2` / `D3`). |
| servos jitter / reset | power: USB-C power bank isn't delivering enough current. Swap to a 5V/3A supply. |

## Network hardening (before the service goes beyond your desk)

- Bind to Tailscale interface only: set `MCP_HOST` to the Tailscale IP rather than `0.0.0.0`.
- Or firewall port 8765 to LAN only (`ufw allow from 192.168.1.0/24 to any port 8765`).
- The MCP token is not a silver bullet — it's a second layer. Don't expose port 8765 to the public internet.
