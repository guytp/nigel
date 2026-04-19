"""MCP server exposing PiCrawler hardware to Claude.

Design note: every movement tool returns a fresh post-action camera frame as
part of its result. That's the "near-realtime" loop — the model acts, then
immediately sees the world the action produced, in a single tool round-trip.
"""

from __future__ import annotations

import logging
import os

from mcp.server.fastmcp import FastMCP, Image

from . import audio_input
from .agent_inbox import AgentInbox
from .hardware import BUILTIN_ACTIONS, DETECTIONS, get_hardware
from .memory_store import MemoryStore
from .vision import VisionStack

log = logging.getLogger(__name__)

# Safety reflex: if the smoothed ultrasonic reports less than this (cm) and
# the caller asked us to move forward, we stop before the step. -1/-2 sentinels
# are ignored (broken sensor can't block movement). Override with env var.
SAFETY_MIN_FORWARD_CM = float(os.environ.get("PICRAWLER_SAFETY_MIN_CM", "15"))


mcp = FastMCP(
    name="picrawler",
    instructions=(
        "You are Nigel, a test robot. This is a PiCrawler quadruped Pete "
        "assembled with Claude's help. Guy and Pete are the devs debugging "
        "you — act like a techy teammate in a lab, not a customer-service "
        "bot. Be terse, direct, honest about what's broken.\n\n"
        "Locomotion: `move` (forward/backward/turn left/turn right), `action` "
        f"(built-ins: {', '.join(BUILTIN_ACTIONS)} — note `dance` is very long "
        "at high step count, keep step=1), `stop`.\n"
        "Sensors: `read_distance` (ultrasonic cm, -1/-2 = timeout).\n"
        "Speech: `speak` (TTS via robot_hat).\n"
        "Audio: `listen` (one-shot Whisper), `listen_for_wake_word` (chunked loop).\n\n"
        "Vision tiers — cheapest first:\n"
        "  1. `scan` — <250ms, motion delta + perceptual hash + YOLO objects.\n"
        "  2. `caption` — ~1-2s, one-sentence Moondream summary.\n"
        "  3. `snapshot` — full image. Movement tools also auto-return a frame.\n\n"
        "The voice agent (OpenAI gpt-realtime, separate process) may be "
        "connected to the same MCP server — if a tool you didn't invoke runs, "
        "that's the other brain. Guy and Pete can watch MJPEG at :9000.\n\n"
        "Nigel has a mode — check with `get_mode`. "
        "In 'coupled' mode (default) you collaborate with the voice agent via "
        "`agent_send` / `agent_poll`. In 'solo' mode the voice agent runs alone "
        "and you must stay passive: don't send inbox messages, don't drive the "
        "body, don't speak unless the human specifically asks you to do "
        "something that clearly isn't addressed to Nigel."
    ),
)

hw = get_hardware()
vision = VisionStack(
    yolo_model=os.environ.get("PICRAWLER_YOLO_MODEL", "yolov8n.pt"),
    caption_model=os.environ.get("PICRAWLER_CAPTION_MODEL", "vikhyatk/moondream2"),
)
inbox = AgentInbox()
memory = MemoryStore()
log.info("hardware backend: %s", hw.kind)


def _frame() -> Image:
    return Image(data=hw.snapshot_jpeg(), format="jpeg")


@mcp.tool()
def move(direction: str, steps: int = 1, speed: int = 90) -> list:
    """Walk in a direction and return a fresh camera frame.

    direction: one of "forward", "backward", "turn left", "turn right"
    steps: gait cycles (1-10 sensible)
    speed: 1-100, default 90

    Safety: forward moves check the ultrasonic between each step and abort
    if we get within PICRAWLER_SAFETY_MIN_CM (default 15cm) of an obstacle.
    Broken sensor (-1/-2) does not block movement. Other directions are not
    safety-gated — the ultrasonic only faces forward.
    """
    allowed = {"forward", "backward", "turn left", "turn right"}
    if direction not in allowed:
        raise ValueError(f"direction must be one of {sorted(allowed)}")
    steps = max(1, min(int(steps), 20))
    speed = max(1, min(int(speed), 100))

    if direction == "forward":
        for i in range(steps):
            d = hw.read_distance_cm()
            if 0 < d < SAFETY_MIN_FORWARD_CM:
                return [
                    f"stopped after {i} of {steps} steps: obstacle at {d}cm "
                    f"< safety threshold {SAFETY_MIN_FORWARD_CM}cm",
                    _frame(),
                ]
            hw.do_action(direction, 1, speed)
        return [f"moved forward x{steps} @ speed {speed}", _frame()]

    hw.do_action(direction, steps, speed)
    return [f"moved {direction} x{steps} @ speed {speed}", _frame()]


@mcp.tool()
def action(name: str, steps: int = 1, speed: int = 90) -> list:
    """Perform a built-in expressive action and return a fresh frame.

    name: one of the built-in actions (e.g. "sit", "stand", "wave", "push-up",
    "twist", "beckon", "look up", "look down"). See server instructions for the
    full list.
    """
    if name not in BUILTIN_ACTIONS:
        raise ValueError(f"action must be one of {list(BUILTIN_ACTIONS)}")
    steps = max(1, min(int(steps), 20))
    speed = max(1, min(int(speed), 100))
    hw.do_action(name, steps, speed)
    return [f"did {name} x{steps} @ speed {speed}", _frame()]


@mcp.tool()
def stop() -> str:
    """Stop moving and return to a stable stand."""
    hw.stop()
    return "stopped; standing"


@mcp.tool()
def snapshot() -> Image:
    """Return the full camera frame as an image you can see directly.

    This is the expensive tier — prefer `scan` or `caption` unless you actually
    need to look at the picture (visual reasoning, reading text, spatial layout).
    """
    return _frame()


@mcp.tool()
def scan(include_objects: bool = True, object_conf: float = 0.35) -> dict:
    """Cheap vision glance — no image returned.

    Returns motion delta since the previous frame (0 = identical, higher = more
    movement), a perceptual hash for scene fingerprinting, and — if enabled —
    a list of detected objects (YOLO classes). Typically <250ms.

    Use this as your default way to "look". Only call `snapshot` if this output
    suggests something worth seeing in detail.
    """
    frame = hw.latest_frame_bgr()
    result = vision.scan(frame, include_objects=include_objects, object_conf=object_conf)
    return result.to_dict()


@mcp.tool()
def read_text(min_confidence: float = 0.3) -> dict:
    """OCR the current camera frame. Returns recognised text regions.

    Output: {"regions": [{text, conf, bbox: [x1,y1,x2,y2]}], "joined": "..."}
    `joined` is all text concatenated top-to-bottom for easy reading.

    Useful for reading screens, signs, whiteboards, book pages. Slower than
    `scan` (~2-5s on a Pi); prefer `scan` first to check if text-ish things
    are even in view.
    """
    frame = hw.latest_frame_bgr()
    regions = vision.read_text(frame, min_confidence=max(0.0, min(float(min_confidence), 1.0)))
    regions_sorted = sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return {
        "regions": regions_sorted,
        "joined": " ".join(r["text"] for r in regions_sorted),
    }


@mcp.tool()
def caption(prompt: str | None = None) -> dict:
    """Describe the current frame in one sentence via a tiny local VLM.

    Slower than `scan` (~1-2s) but cheaper than a full image round-trip. Use
    when `scan` found something but you want a quick human-readable summary
    before deciding whether to `snapshot`.

    Optional `prompt` overrides the default caption question (e.g.
    "What color is the ball?" or "Is there a person?").
    """
    frame = hw.latest_frame_bgr()
    text = vision.caption(frame, prompt=prompt)
    return {"caption": text, "prompt": prompt or "default"}


@mcp.tool()
def read_distance() -> dict:
    """Read the forward ultrasonic distance in centimeters."""
    return {"cm": hw.read_distance_cm()}


@mcp.tool()
def set_vision(feature: str, enabled: bool = True) -> dict:
    """Toggle an on-board vision feature.

    feature: one of "face", "color", "qr", "gesture", "object".
    """
    if feature not in DETECTIONS:
        raise ValueError(f"feature must be one of {list(DETECTIONS)}")
    hw.set_vision(feature, enabled)
    return {"feature": feature, "enabled": enabled}


@mcp.tool()
def set_target_color(color: str) -> dict:
    """Set the named color for color-detection (e.g. "red", "green", "blue")."""
    hw.set_target_color(color)
    return {"target_color": color}


@mcp.tool()
def read_detections() -> list[dict]:
    """Return the current frame's detection results for enabled features."""
    return [{"kind": d.kind, **d.data} for d in hw.read_detections()]


@mcp.tool()
def speak(text: str) -> str:
    """Speak text aloud via the on-board speaker."""
    hw.speak(text)
    return f"spoke: {text!r}"


@mcp.tool()
def listen(seconds: float = 5.0) -> dict:
    """Record from the USB mic for `seconds`, transcribe via Whisper, return the text.

    The mic pipeline applies software gain (PICRAWLER_AUDIO_GAIN_DB, default 20dB).
    Requires OPENAI_API_KEY in the service environment.

    Use this for one-shot capture when you know the user is about to speak.
    For unsolicited speech detection, use `listen_for_wake_word`.
    """
    seconds = max(0.5, min(float(seconds), 30.0))
    return audio_input.listen(seconds)


@mcp.tool()
def agent_send(to: str, message: str, from_: str = "claude") -> dict:
    """Send a text message to another agent connected to this MCP server.

    Used for inter-agent chat: Claude (in Claude Code) and gpt-realtime (the
    voice agent running on the Pi) can exchange messages through the same MCP
    server instead of via spoken audio. Delivery is in-process — messages sit
    in a per-recipient inbox until the other agent polls for them.

    Recipient names by convention: "claude", "nigel" (the voice agent uses
    "nigel"). `from_` defaults to "claude" — override if you're posting as a
    different identity. Returns {id, to, from, ts} for reference.
    """
    msg = inbox.send(from_=from_, to=to, message=message)
    return msg.to_dict()


@mcp.tool()
def memory_set(key: str, value: str, tags: list[str] | None = None, author: str = "") -> dict:
    """Write a persistent fact, preference, or observation to Nigel's memory.

    Both Claude (Claude Code) and gpt-realtime (voice agent) share this
    store — a fact written by one is readable by the other and survives
    service restarts.

    Use for: who lives here, room layout, user preferences, observations
    worth keeping ("Guy prefers his tea strong"), calibration notes
    ("ultrasonic reads -2 when pointed at fabric"). Avoid: ephemeral
    chatter, raw tool outputs.

    Arguments:
        key: stable identifier (namespace with colons, e.g. "user:guy:tea")
        value: the content. Strings stored verbatim; other types JSON-encoded.
        tags: optional list for categorising, e.g. ["preference", "user"]
        author: who wrote it — "claude" or "nigel". Helps attribution.

    Returns the stored record.
    """
    return memory.set(key=key, value=value, tags=tags, author=author)


@mcp.tool()
def memory_get(key: str) -> dict | None:
    """Read one memory by exact key. Returns None if not found."""
    return memory.get(key)


@mcp.tool()
def memory_search(query: str, limit: int = 20) -> list[dict]:
    """Substring-search memory keys, values, and tags. Newest first."""
    return memory.search(query, limit=max(1, min(int(limit), 100)))


@mcp.tool()
def memory_by_tag(tag: str, limit: int = 50) -> list[dict]:
    """List memories with a specific tag, newest first."""
    return memory.by_tag(tag, limit=max(1, min(int(limit), 200)))


@mcp.tool()
def memory_list_keys(limit: int = 100) -> list[str]:
    """List memory keys, most-recently-updated first."""
    return memory.list_keys(limit=max(1, min(int(limit), 1000)))


@mcp.tool()
def memory_delete(key: str) -> dict:
    """Delete a memory by key. Returns {deleted: bool}."""
    return {"deleted": memory.delete(key)}


# ------------------------------------------------------------ mode toggle

NIGEL_MODE_KEY = "nigel:mode"
VALID_MODES = ("coupled", "solo")


@mcp.tool()
def set_mode(mode: str) -> dict:
    """Switch Nigel between 'coupled' and 'solo'.

    - `coupled` (default): Claude (via Claude Code) and gpt-realtime (the
      voice agent) are both active. They talk through the agent inbox.
      Claude is the reasoner; gpt-realtime is the voice + fast reflexes.
    - `solo`: only gpt-realtime drives. Inbox polling stops, agent_send /
      agent_poll tools are hidden from the voice session, and Claude is
      expected to stay silent unless the human addresses Claude directly.

    The voice agent polls mode every ~5s and reconfigures its session
    (prompt + tool list) on change — no restart needed. It announces the
    mode switch aloud.
    """
    mode = (mode or "").strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {list(VALID_MODES)}")
    memory.set(NIGEL_MODE_KEY, mode, tags=["system", "mode"], author="mcp")
    return {"mode": mode}


@mcp.tool()
def get_mode() -> dict:
    """Return Nigel's current mode: 'coupled' or 'solo'."""
    rec = memory.get(NIGEL_MODE_KEY)
    return {"mode": rec["value"] if rec else "coupled"}


@mcp.tool()
def agent_poll(as_who: str, since_id: int = 0) -> list[dict]:
    """Return messages for `as_who` with id strictly greater than `since_id`.

    Agents poll periodically (or at the start of each turn) to pick up new
    messages. Track the highest id you've seen and pass it as `since_id` on
    the next poll to avoid seeing the same message twice.

    Returns a list of {id, from, to, message, ts}, oldest first.
    """
    return [m.to_dict() for m in inbox.poll(as_who, since_id)]


@mcp.tool()
def listen_for_wake_word(
    wake: str = "hey nigel",
    timeout: float = 60.0,
    chunk_seconds: float = 3.0,
    capture_after: float = 4.0,
) -> dict:
    """Listen in chunks until the wake phrase is heard, then capture what follows.

    Default wake phrase is "hey nigel". Records `chunk_seconds` at a time,
    transcribes each via Whisper, returns as soon as the phrase appears
    (word-boundary, case-insensitive). Then captures `capture_after` more
    seconds of speech so you get whatever the user said after the wake word.

    Returns {woke: bool, wake_chunk: str, followup: str, heard_chunks: [str], timed_out: bool}.

    Whisper cost: roughly $0.006/min of audio. With default 3s chunks, ~$0.0003 per chunk.
    """
    timeout = max(5.0, min(float(timeout), 600.0))
    chunk_seconds = max(1.5, min(float(chunk_seconds), 10.0))
    capture_after = max(1.0, min(float(capture_after), 15.0))
    return audio_input.listen_for_wake_word(
        wake=wake,
        timeout=timeout,
        chunk_seconds=chunk_seconds,
        capture_after=capture_after,
    )


@mcp.resource("picrawler://state")
def state_resource() -> dict:
    """Current robot state: last action, vision toggles, target color."""
    return {
        "backend": hw.kind,
        "last_action": hw.state.last_action,
        "vision": dict(hw.state.vision),
        "target_color": hw.state.target_color,
    }


@mcp.resource("picrawler://stream")
def stream_resource() -> str:
    """URL of the live MJPEG stream (open in a browser while the agent drives)."""
    return hw.stream_url()


def _configure_transport_security(host: str) -> None:
    """Set DNS-rebinding-protection allow-list based on the actual bind host.

    FastMCP auto-enables protection at construction time with loopback-only
    hosts. We need to expand it when binding non-loopback, or any request
    carrying a real hostname (nigel.local, a LAN IP) gets 421'd.

    Behaviour:
      - binding to loopback → keep the SDK defaults (protection ON, loopback only)
      - MCP_ALLOWED_HOSTS set → protection ON, with that comma-separated list
      - otherwise → protection OFF (we're trusting LAN + optional bearer token)
    """
    from mcp.server.transport_security import TransportSecuritySettings

    if host in ("127.0.0.1", "localhost", "::1"):
        return  # SDK defaults already correct

    env_hosts = os.environ.get("MCP_ALLOWED_HOSTS", "").strip()
    if env_hosts:
        allowed = [h.strip() for h in env_hosts.split(",") if h.strip()]
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed,
            allowed_origins=[f"http://{h}" for h in allowed],
        )
        log.info("MCP DNS-rebinding protection enabled for: %s", allowed)
    else:
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
        )
        log.warning(
            "MCP DNS-rebinding protection disabled (bound to %s). "
            "Set MCP_ALLOWED_HOSTS to re-enable with a hostname allow-list.",
            host,
        )


def run() -> None:
    logging.basicConfig(level=os.environ.get("PICRAWLER_LOG", "INFO"))
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        mcp.run(transport="stdio")
        return
    if transport not in ("http", "streamable-http"):
        raise SystemExit(f"unknown MCP_TRANSPORT={transport!r}")

    host = os.environ.get("MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("MCP_PORT", "8765"))
    token = os.environ.get("MCP_TOKEN", "").strip()

    _configure_transport_security(host)

    if not token:
        # No auth — fall back to the SDK's built-in runner.
        mcp.settings.host = host
        mcp.settings.port = port
        mcp.run(transport="streamable-http")
        return

    # Auth required — wrap the Starlette app with bearer middleware and host
    # it via uvicorn ourselves.
    import uvicorn  # local import so stdio mode has no uvicorn dependency

    from .auth import BearerAuthMiddleware

    app = mcp.streamable_http_app()
    app = BearerAuthMiddleware(app, token=token)
    log.info("MCP bearer auth enabled; bound to %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
