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
from .hardware import BUILTIN_ACTIONS, DETECTIONS, get_hardware
from .vision import VisionStack

log = logging.getLogger(__name__)

mcp = FastMCP(
    name="picrawler",
    instructions=(
        "You are Nigel, an LLM embodied in a SunFounder PiCrawler quadruped robot.\n\n"
        "Locomotion: `move` (forward/backward/turn left/turn right), `action` "
        f"(expressive built-ins: {', '.join(BUILTIN_ACTIONS)}), `stop`.\n"
        "Sensors: `read_distance` (ultrasonic cm).\n"
        "Speech: `speak`.\n\n"
        "Vision is tiered — prefer the cheapest tier that answers the question:\n"
        "  1. `scan` — <250ms, returns motion delta, perceptual hash, and object "
        "list (YOLO). Use this first. Call it repeatedly to glance around without "
        "burning a vision turn.\n"
        "  2. `caption` — ~1-2s, returns a one-sentence description. Use when "
        "`scan` tells you something is there but you want a quick label.\n"
        "  3. `snapshot` — returns the actual image for you to look at. Use only "
        "when you specifically need to see the frame (visual reasoning, reading "
        "text, spatial layout). Movement tools also auto-return a frame.\n\n"
        "Rule of thumb: `scan` ten times before you `snapshot` once."
    ),
)

hw = get_hardware()
vision = VisionStack(
    yolo_model=os.environ.get("PICRAWLER_YOLO_MODEL", "yolov8n.pt"),
    caption_model=os.environ.get("PICRAWLER_CAPTION_MODEL", "vikhyatk/moondream2"),
)
log.info("hardware backend: %s", hw.kind)


def _frame() -> Image:
    return Image(data=hw.snapshot_jpeg(), format="jpeg")


@mcp.tool()
def move(direction: str, steps: int = 1, speed: int = 90) -> list:
    """Walk in a direction and return a fresh camera frame.

    direction: one of "forward", "backward", "turn left", "turn right"
    steps: gait cycles (1-10 sensible)
    speed: 1-100, default 90
    """
    allowed = {"forward", "backward", "turn left", "turn right"}
    if direction not in allowed:
        raise ValueError(f"direction must be one of {sorted(allowed)}")
    steps = max(1, min(int(steps), 20))
    speed = max(1, min(int(speed), 100))
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
def listen_for_wake_word(
    wake: str = "nigel",
    timeout: float = 60.0,
    chunk_seconds: float = 3.0,
    capture_after: float = 4.0,
) -> dict:
    """Listen in chunks until the wake word is heard, then capture what follows.

    Records `chunk_seconds` at a time, transcribes each, returns as soon as
    `wake` appears (word-boundary, case-insensitive). After wake, captures
    `capture_after` more seconds to get the rest of the user's utterance.

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
