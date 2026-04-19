# Nigel

A body for your LLM, shaped like a small quadruped spider — the [SunFounder PiCrawler](https://www.sunfounder.com/products/picrawler-robot-kit). Two brains can drive Nigel interchangeably:

- `mcp_picrawler` — MCP server wrapping movement, camera, tiered vision, and speech. Used by Claude Code.
- `voice_agent` — OpenAI Realtime client that connects to the MCP server locally and gives Nigel a voice. Same body, different brain.

Run one or both. They share the same tool surface.

## Install on a Pi (one-liner)

On a freshly-flashed Raspberry Pi OS (64-bit, Bookworm) with the crawler assembled:

```bash
curl -fsSL https://raw.githubusercontent.com/guytp/nigel/main/bootstrap.sh | bash
```

Inspect-first variant (recommended first time):

```bash
curl -fsSL https://raw.githubusercontent.com/guytp/nigel/main/bootstrap.sh -o /tmp/bootstrap.sh
less /tmp/bootstrap.sh
bash /tmp/bootstrap.sh
```

What it does:

1. Installs apt prerequisites.
2. Clones + builds the SunFounder libraries (`robot-hat`, `vilib`, `picrawler`) if not already present — skip with `SKIP_SF=1`.
3. Clones this repo to `/home/pi/nigel`.
4. Sets up a venv and installs the `mcp_picrawler` package.
5. Drops the systemd unit, enables + starts `picrawler-mcp.service`.
6. If `/etc/picrawler-voice.env` exists, also installs and starts `picrawler-voice.service`.

The script is idempotent — running it again updates code and restarts services. To wipe and reinstall:

```bash
RESET=1 bash /tmp/bootstrap.sh
```

To uninstall:

```bash
cd /home/pi/nigel && ./deploy/uninstall.sh           # remove services + venv
cd /home/pi/nigel && ./deploy/uninstall.sh --purge   # also remove /etc/picrawler-*.env
```

After install, connect Claude Code by adding to `~/.claude.json` or project `.mcp.json`:

```jsonc
{
  "mcpServers": {
    "picrawler": {
      "type": "http",
      "url": "http://nigel.local:8765/mcp",
      "headers": { "Authorization": "Bearer PUT_TOKEN_HERE" }
    }
  }
}
```

Full manual walkthrough, hardware calibration notes, and troubleshooting live in [`docs/pi-setup.md`](docs/pi-setup.md).

Want to connect from claude.ai or anywhere outside the LAN? [`docs/public-mcp.md`](docs/public-mcp.md) has a Cloudflare tunnel script that gives you a public HTTPS endpoint in ~30 seconds.

### Smoke test after install

Once the bootstrap finishes (reboot first if it said one was needed), run the self-test:

```bash
sudo systemctl stop picrawler-mcp            # free the camera
cd /home/pi/nigel
.venv/bin/python -m scripts.smoke
sudo systemctl start picrawler-mcp
```

It exercises every subsystem — servos, ultrasonic, camera, JPEG encode, MJPEG stream, vision toggles, TTS, YOLO — and prints pass/fail per item. If it passes, Claude Code and the voice agent will work. If a step fails, the line tells you what to fix.

## What it does

- Exposes movement, camera, sensor, and speech as MCP tools
- Every movement tool auto-returns a fresh camera frame — that's the near-realtime loop
- Piggybacks on `vilib`'s built-in MJPEG server (port 9000) so a human can watch from a browser while the agent drives
- Auto-detects hardware: runs identically on a laptop using a mock backend, then drops onto the Pi unchanged

## Tools

Movement / sensors / speech:

| tool | what it does |
| --- | --- |
| `move(direction, steps, speed)` | forward / backward / turn left / turn right; returns frame |
| `action(name, steps, speed)` | sit, stand, wave, twist, push-up, look up/down, beckon; returns frame |
| `stop()` | stabilise in a stand |
| `read_distance()` | ultrasonic cm |
| `set_vision(feature, enabled)` | toggle vilib's face/color/qr/gesture/object detection |
| `set_target_color(color)` | named target for color detection |
| `read_detections()` | latest vilib detection results |
| `speak(text)` | on-board TTS |

Tiered vision — "scan ten times before you snapshot once":

| tool | tier | latency | returns | when to use |
| --- | --- | --- | --- | --- |
| `scan(include_objects, object_conf)` | T0 + T2 | ~1–30ms warm | motion delta, pHash, YOLO objects | default "glance" — use this first |
| `caption(prompt?)` | T3 | ~1–2s | one-sentence caption (Moondream2) | scan found something, want a label |
| `snapshot()` | T4 | turn-cost | full image back to the model | need to actually see the frame |

Resources: `picrawler://state`, `picrawler://stream`.

## Local development (no Pi required)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,vision]"         # add ,caption for Moondream
mcp-picrawler                          # stdio transport, mock hardware
```

Optional extras:

- `.[vision]` — OpenCV (already in base) + `ultralytics` (YOLOv8n for `scan`'s objects tier). ~500MB with torch.
- `.[caption]` — `transformers` + `torch` for Moondream2 (T3). Heavy; skip on Pi if tight on RAM and run on a bridge machine instead.
- `.[pi]` — `picrawler`/`robot-hat`/`vilib`. Only installable on a Raspberry Pi.

Wire it into Claude Code:

```jsonc
// ~/.claude.json or project .mcp.json
{
  "mcpServers": {
    "picrawler": {
      "command": "/abs/path/to/.venv/bin/mcp-picrawler"
    }
  }
}
```

## Deploy to the Pi

```bash
# on the Pi
git clone https://github.com/guytp/nigel.git /home/pi/nigel
cd /home/pi/nigel
./deploy/install.sh
```

The service listens on `:8765` (MCP over HTTP) and `:9000` (MJPEG). Bind it behind Tailscale or a LAN firewall — there is no auth yet.

Add to your client:

```json
{
  "mcpServers": {
    "picrawler": {
      "type": "http",
      "url": "http://nigel.local:8765/mcp"
    }
  }
}
```

## Hardware

- SunFounder PiCrawler (12 servos, Robot HAT, camera, ultrasonic, speaker)
- Raspberry Pi 5/4/3B+/Zero 2W

Libraries used on the Pi: [`picrawler`](https://github.com/sunfounder/picrawler), [`robot-hat`](https://github.com/sunfounder/robot-hat), [`vilib`](https://github.com/sunfounder/vilib).

## Voice agent (OpenAI Realtime)

Lives in `src/voice_agent/`. Connects to the local MCP server over HTTP, translates its tools into OpenAI function-tool defs, opens a Realtime session, pipes mic in and TTS out. Server-side VAD does turn-taking; barge-in works out of the box.

```bash
pip install -e ".[voice]"
export OPENAI_API_KEY=sk-...
export MCP_URL=http://127.0.0.1:8765/mcp
export MCP_TOKEN=<same token used by mcp_picrawler>
picrawler-voice
```

On the Pi the voice agent runs as a second systemd service (`picrawler-voice.service`) alongside `picrawler-mcp.service`. Setup creds in `/etc/picrawler-voice.env` (see `deploy/picrawler-voice.env.example`). If the env file isn't present, `install.sh` skips voice — so you can ship body-only first and add voice later.

Instructions given to the model steer it away from `snapshot` (returns an image it can't read over voice) toward `scan` / `caption` / `read_distance`. It narrates actions as it performs them.

**Text-mode variant** (`picrawler-text`) — same Realtime session and tool-call plumbing, no audio. Useful when mic permissions aren't available (fresh macOS, SSH session, etc.). See [`docs/local-test.md`](docs/local-test.md) §3a.

## Auth

Set `MCP_TOKEN` to require `Authorization: Bearer <token>` on the HTTP transport. Stdio sessions bypass this. This is a second layer — put the server on Tailscale or a LAN firewall anyway; don't expose port 8765 to the public internet.

## Layout

```
src/mcp_picrawler/
  server.py       FastMCP server — tool + resource defs
  hardware.py     Hardware protocol, MockHardware, RealHardware (picrawler + robot_hat + vilib)
  vision.py       Tiered vision stack (T0 motion/pHash, T2 YOLO, T3 Moondream)
  auth.py         Bearer token middleware
  __main__.py     entrypoint

src/voice_agent/
  agent.py        OpenAI Realtime session loop
  audio.py        sounddevice mic + speaker @ 24kHz pcm16
  mcp_bridge.py   MCP client that translates to OpenAI function tools
  __main__.py     entrypoint

deploy/
  picrawler-mcp.service         systemd unit for body
  picrawler-voice.service       systemd unit for voice agent
  picrawler-voice.env.example   copy to /etc/picrawler-voice.env
  install.sh                    Pi installer (skips voice if env missing)
  mcp.json.example              Claude Code MCP config

docs/
  pi-setup.md                   flash-to-first-move
```
