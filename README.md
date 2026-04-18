# claude-bot — PiCrawler MCP server + voice agent

Two processes that give an LLM a body on a [SunFounder PiCrawler](https://www.sunfounder.com/products/picrawler-robot-kit):

- `mcp_picrawler` — MCP server wrapping movement, camera, vision, and speech. Drives the body. Used by Claude Code.
- `voice_agent` — OpenAI Realtime client that connects to the MCP server locally and gives the crawler a voice. Same body, different brain.

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
git clone <this repo> /home/pi/claude-bot
cd /home/pi/claude-bot
./deploy/install.sh
```

The service listens on `:8765` (MCP over HTTP) and `:9000` (MJPEG). Bind it behind Tailscale or a LAN firewall — there is no auth yet.

Add to your client:

```json
{
  "mcpServers": {
    "picrawler": {
      "type": "http",
      "url": "http://picrawler.local:8765/mcp"
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
