# Local testing — before hardware arrives

What you can prove on a Mac (or any dev machine) without the PiCrawler:

1. **Automated tests** — 68 unit + integration tests, all mock-backed, ~15s to run
2. **Claude Code ↔ mock body** — real MCP roundtrips against a fake crawler
3. **OpenAI Realtime ↔ mock body** — talk to Nigel via the laptop's mic + speaker; tool calls routed through MCP to the mock
4. **Tier C: webcam as the eye** — optional; turns the mock into something that actually sees

After these, what's left to validate on the Pi is narrow: servo wiring, vilib camera init, ultrasonic pin assignments, audio through robot_hat's speaker.

## 1. Automated tests

```bash
python3.12 -m venv .venv          # if not already
.venv/bin/pip install -e ".[dev,vision,voice]"
.venv/bin/pytest tests/ -q
```

Expected: `68 passed, 3 skipped`. Skipped are caption (T3 pulls ~2GB) and two webcam tests (gated by `PICRAWLER_TEST_WEBCAM=1`).

Want the webcam ones to run too? `PICRAWLER_TEST_WEBCAM=1 .venv/bin/pytest tests/ -q`. macOS will prompt once for camera access.

## 2. Claude Code against the mock

Add to `~/.claude.json` or project `.mcp.json`:

```jsonc
{
  "mcpServers": {
    "nigel-mock": {
      "command": "/Users/guytp/code/chippy-tcg-poc/claude-bot/.venv/bin/mcp-picrawler"
    }
  }
}
```

Restart Claude Code. Ask:

> *"read the distance. Then scan. Then move forward 1 step and take a snapshot."*

Claude should invoke `read_distance`, `scan`, `move`, `snapshot` in sequence. `read_distance` returns a random number; `scan` returns motion+phash+objects (empty on synthetic frames); `move` prints a log line and returns a synthetic frame; `snapshot` returns an image Claude can see.

If Claude complains about tool shapes or errors, that's a real bug to fix.

## 3. Talking to Nigel (OpenAI Realtime)

Two terminals. **Terminal 1** — the mock body:

```bash
cd /Users/guytp/code/chippy-tcg-poc/claude-bot
MCP_TRANSPORT=streamable-http \
MCP_HOST=127.0.0.1 \
MCP_PORT=8765 \
MCP_TOKEN=local-test \
PICRAWLER_MOCK_CAMERA=webcam \
  .venv/bin/mcp-picrawler
```

The `PICRAWLER_MOCK_CAMERA=webcam` makes `scan` and `caption` see your room through the Mac's FaceTime camera — Nigel will report actual objects he detects. Omit it if you prefer synthetic frames.

**Terminal 2** — the brain:

```bash
cd /Users/guytp/code/chippy-tcg-poc/claude-bot
export OPENAI_API_KEY=sk-...          # your key
MCP_URL=http://127.0.0.1:8765/mcp \
MCP_TOKEN=local-test \
  .venv/bin/picrawler-voice
```

Speak into the mic. "Hey Nigel, what can you see?" He'll invoke `scan` (and maybe `caption`), describe the result in his voice, and narrate as he goes. "Move forward" will invoke `move` (mock, but he'll reply as though he moved).

**Barge-in check:** while he's mid-sentence, start talking — he should cut himself off. If he doesn't, check the server-side VAD event wiring.

**Costs:** ~$0.06/min input audio, ~$0.24/min output audio on `gpt-realtime`. A 15-minute chat is under $5.

### What you're validating with this

Anything that works here is effectively proved for the Pi. Specifically:
- OpenAI Realtime event names and field shapes (I wrote these from memory)
- Tool schema translation (MCP → OpenAI function def)
- Image summarisation in tool results (voice agent doesn't need raw JPEG bytes)
- Audio I/O at 24kHz pcm16
- Bearer auth on the MCP transport
- Whole async plumbing under real load

What's still untested after this: actual servo motion, vilib-provided frames (cv2 BGR format), ultrasonic pins, the robot_hat speaker.

## 4. Tier C — webcam as Nigel's eye

Already covered above — `PICRAWLER_MOCK_CAMERA=webcam` on the MCP server. With it set:

- `scan(include_objects=True)` runs YOLOv8n on your webcam feed; likely finds "person", "keyboard", etc.
- `caption()` would run Moondream on the frame if `.[caption]` is installed and `RUN_CAPTION_TESTS=1` plumbing is hooked up (it's lazy-loaded — first call pulls ~2GB of weights).
- `snapshot()` returns a JPEG of your actual room.

This lets you stress-test the vision pipeline with varied input, not just a synthetic grey square.

## 5. When tests pass but the Pi misbehaves

Expect these discrepancies and plan for them:

| difference on Pi | symptom | fix |
|---|---|---|
| `robot_hat.Ultrasonic` pin numbers | `read_distance` always 0 | edit `RealHardware.__init__` in `hardware.py`, try `D0/D1` or `D4/D5` |
| `vilib.Vilib.img` not ready on first call | `snapshot` raises "camera frame not ready" | add a 1s sleep after `camera_start` or retry loop |
| robot_hat speaker sample rate | TTS sounds chipmunky | check `robot_hat.TTS().say()` docs — might need `lang=en-US` |
| servo calibration drift | walking curves to one side | run SunFounder's `0_calibration.py` |

## 6. CI'able subset

Everything except the webcam + caption tests runs without hardware. Plumb into a GitHub Action later:

```yaml
- run: python3.12 -m venv .venv
- run: .venv/bin/pip install -e ".[dev,vision]"
- run: .venv/bin/pytest tests/ -q
```

(Don't include `[voice]` in CI — `sounddevice` needs portaudio and audio tests skip themselves without a device anyway.)
