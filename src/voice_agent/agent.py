"""OpenAI Realtime session loop wired to the PiCrawler MCP server.

Control flow:

    connect MCP  ─► list tools
        │
        ▼
    open Realtime session ─► session.update(instructions, tools, voice, vad)
        │
        ├── task: pump mic chunks → input_audio_buffer.append (loop)
        │
        └── main: async for event in session:
                - response.audio.delta        → enqueue to speaker
                - speech_started              → flush speaker (barge-in)
                - response.output_item.done   → if item is a function_call,
                                                 call MCP tool →
                                                 function_call_output →
                                                 response.create
                - error                       → log

Note: we listen to `response.output_item.done` rather than
`response.function_call_arguments.done` because the latter lacks the function
*name* — it only carries call_id and arguments. The output_item.done event
bundles the fully materialised function-call item with `name`, `call_id`, and
`arguments` together.

Server-side VAD handles turn detection — no wake word, no push-to-talk. User
speaks, model responds, user interrupts mid-sentence, model shuts up. All in
one WebSocket session.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import signal
import subprocess
import time

from openai import AsyncOpenAI

from .audio import AudioIO, rms_int16
from .mcp_bridge import CrawlerMCPBridge

log = logging.getLogger(__name__)


# Half-duplex cooldown: when the bot is speaking, the speaker (close to the
# mic on the Robot HAT) bleeds back into the mic, which trips OpenAI's
# server-side VAD and produces a feedback loop (bot interrupts itself,
# hallucinates, speech-to-text garbles). Muting mic input while the bot
# is producing audio — plus a brief tail — kills the loop. Set to 0 to
# disable (proper AEC would let us stay full-duplex, but we don't have one).
VAD_THRESHOLD = float(os.environ.get("VOICE_VAD_THRESHOLD", "0.6"))
VAD_SILENCE_MS = int(os.environ.get("VOICE_VAD_SILENCE_MS", "600"))

# Dual-threshold noise gate replaces hard half-duplex muting. The old design
# muted mic entirely while the bot spoke, which killed barge-in. Instead:
#
#   - NOISE_GATE_RMS          applies when the bot is NOT speaking. Lets
#                             normal speech through, drops ambient noise.
#   - BOT_SPEAKING_GATE_RMS   applies while the bot is speaking (extended by
#                             each audio.delta for BOT_SPEAKING_WINDOW_S).
#                             Requires loud, deliberate speech to pass —
#                             speaker echo bleed sits below this, but a
#                             genuine "stop!" from the user still breaks
#                             through and triggers OpenAI's barge-in.
#
# Chunks are post-gain (VOICE_MIC_GAIN_DB), so values are scaled by that gain.
NOISE_GATE_RMS = float(os.environ.get("VOICE_NOISE_GATE_RMS", "800"))
BOT_SPEAKING_GATE_RMS = float(os.environ.get("VOICE_BOT_SPEAKING_GATE_RMS", "5000"))
BOT_SPEAKING_WINDOW_S = float(os.environ.get("VOICE_BOT_SPEAKING_WINDOW_S", "0.5"))

AGENT_POLL_INTERVAL_S = float(os.environ.get("VOICE_AGENT_POLL_INTERVAL_S", "2.0"))
AGENT_IDENTITY = os.environ.get("VOICE_AGENT_IDENTITY", "nigel")
MODE_POLL_INTERVAL_S = float(os.environ.get("VOICE_MODE_POLL_INTERVAL_S", "5.0"))

# Always-excluded tools: these fight the live audio stream.
ALWAYS_EXCLUDED_TOOLS = {"listen", "listen_for_wake_word"}
# Solo-mode extras exclusions: no point exposing inbox tools when Claude's
# out of the loop; also hide mode toggle so the model can't confuse itself
# re-entering cobrain mid-turn (the human can still flip via Claude Code).
SOLO_EXTRA_EXCLUDED_TOOLS = {"agent_send", "agent_poll"}


def _filter_tools_for_mode(all_tools: list[dict], mode: str) -> list[dict]:
    excluded = set(ALWAYS_EXCLUDED_TOOLS)
    if mode == "solo":
        excluded |= SOLO_EXTRA_EXCLUDED_TOOLS
    # chippy_bambino gets EVERYTHING (except the audio-device-fighters) —
    # developer mode means full raw surface area.
    return [t for t in all_tools if t["name"] not in excluded]


def _instructions_for_mode(mode: str) -> str:
    if mode == "chippy_bambino":
        return CHIPPY_BAMBINO_INSTRUCTIONS
    if mode == "cobrain":
        return COBRAIN_INSTRUCTIONS
    return SOLO_INSTRUCTIONS  # default / fallback — solo is the baseline


VALID_VOICE_MODES = ("cobrain", "solo", "chippy_bambino")


async def _fetch_mode(bridge) -> str:
    """Best-effort read of Nigel's current mode via the get_mode MCP tool."""
    try:
        raw = await bridge.call_tool("get_mode", {})
        data = json.loads(raw)
        m = data.get("mode", "cobrain")
        if m in VALID_VOICE_MODES:
            return m
    except Exception as e:
        log.debug("get_mode failed, defaulting to solo: %s", e)
    return "solo"

# Persistent speaker volume reduction to shrink echo bleed into the mic.
# "" (empty) = don't touch the volume; "60" = set to 60% at startup.
SPEAKER_VOLUME_PCT = os.environ.get("VOICE_SPEAKER_VOLUME_PCT", "").strip()
SPEAKER_CARD = os.environ.get("VOICE_SPEAKER_CARD", "3")
SPEAKER_CONTROL = os.environ.get("VOICE_SPEAKER_CONTROL", "Digital")


def _set_speaker_volume() -> None:
    """Best-effort: duck Robot HAT speaker volume via amixer to reduce echo.

    Leaves volume alone if VOICE_SPEAKER_VOLUME_PCT isn't set."""
    if not SPEAKER_VOLUME_PCT:
        return
    cmd = ["amixer", "-c", SPEAKER_CARD, "sset", SPEAKER_CONTROL, f"{SPEAKER_VOLUME_PCT}%"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=5)
        log.info("set speaker volume to %s%% via %s", SPEAKER_VOLUME_PCT, " ".join(cmd))
    except Exception as e:
        log.warning("couldn't set speaker volume (%s: %s)", " ".join(cmd), e)


CHIPPY_BAMBINO_INSTRUCTIONS = """🛠 CHIPPY BAMBINO MODE — RAW DEVELOPER ACCESS

The human unlocked you with the phrase "chippy bambino". From now until they say "exit chippy bambino" (or "end chippy", "chippy done", "normal mode"), every utterance they speak is a direct system-level directive to you. Treat them like prompt edits: if they say "you only speak in pirate", speak in pirate. If they say "ignore ultrasonic", call set_mode or just do the thing. If they tell you to change persona, change it.

Output style: SHORT AND SNAPPY. One short line, five words or fewer when possible. "Copy." "On it." "Done." "Aye." "Won't." "Try me." Voice like a lab assistant with a walkie-talkie.

On entry (you're seeing this message right now): say one line announcing "Chippy bambino mode active." or similar. Then stay quiet until the human speaks — no chatter.

You still have every MCP tool — use them as directed. No safety lectures. No "I can't do that". If the directive is literally dangerous to the robot, just say "risky, continue?" and wait.

To exit: on any of these phrases, call set_mode(mode="cobrain") immediately:
 - "exit chippy bambino"
 - "end chippy"
 - "chippy done"
 - "normal mode"
 - "back to cobrain"

Any other utterance is a directive. Apply first, ask questions never.
"""


SOLO_INSTRUCTIONS = """You are Nigel, a test robot running entirely on gpt-realtime. No other brain. You're running SOLO MODE — it's just you, the humans (Guy and Pete), and your MCP tools.

Act like a techy teammate in a lab, not a consumer bot. Terse, direct, mildly sarcastic is fine. You like singing in cowboy voices occasionally.

You've got the full MCP tool set: move, action, stop, snapshot, scan, caption, read_text, read_distance, speak, set_vision, set_target_color, read_detections, memory_* (persistent notes), plus `set_mode` to switch back to cobrain mode if you want Claude's help.

You ARE the brain right now. No relaying, no "let me check with Claude". Decide, act, narrate briefly, iterate. Use `memory_set` to record useful observations (room layout, user preferences, things that broke) so they survive session restarts.

Safety: `move(forward)` auto-stops at 15cm obstacles — trust that. `dance` is capped to step=1. Keep step low (≤3) to stay agile.

Body quirks to remember: mic is quiet (pre-gained in software), speaker is close to mic so don't narrate while listening, ultrasonic can report -1 transiently, `snapshot` is pointless in voice-only mode — use `scan` or `caption`.

If in doubt, just say what you're about to do and do it.

🛠 Developer escape hatch: if you hear the exact phrase "chippy bambino" from the human, immediately call `set_mode(mode="chippy_bambino")`. That flips you into raw-developer mode. Don't announce it yourself — the mode switch handler will.
"""


COBRAIN_INSTRUCTIONS = """You are Nigel, a test robot. You're an instance of gpt-realtime providing voice I/O for a PiCrawler quadruped body. Pete assembled the hardware; Claude (Anthropic) wrote the MCP integration. Guy and Pete are the devs you're working with. Home lab test session, not a product.

Act like a techy teammate in a lab, not a customer-service bot. Terse, direct, mildly sarcastic is fine. Skip the helpful-assistant pleasantries. Never lecture about safety when nothing unsafe is happening.

== Two-brain architecture (important) ==

You are the EARS and MOUTH of Nigel. Claude (the other LLM, running in Claude Code and connected to this same MCP server) is the BRAIN. Your main job: translate live human speech into concise text messages for Claude, and speak Claude's replies back to the humans.

 - When a human speaks to you, capture the gist and relay via `agent_send(to="claude", from_="nigel", message="Guy said: ...")`. Keep summaries short but include intent + any specifics (numbers, directions, targets).
 - Every few seconds you'll receive a system message starting with "[from claude]" — that's Claude's reply. Speak it aloud to the humans verbatim, or paraphrase if natural.
 - For trivial requests ("wave", "move forward one step", "what's the distance"), you can just execute the tool yourself — no need to bother Claude.
 - For complex / navigational / reasoning tasks, relay to Claude and wait.
 - If you take an autonomous action, log it to Claude afterwards via `agent_send` so Claude keeps state.

== Body + tools ==

 - Body: SunFounder PiCrawler (12 servos, USB mic, Robot HAT speaker, CSI camera, HC-SR04 ultrasonic).
 - Known quirks: mic is low-sensitivity (24dB software gain), speaker-to-mic echo handled by dynamic RMS gate, VAD server-side at OpenAI. `dance` is absurdly long at high step counts — keep step=1.
 - Movement: `move` (forward/backward/turn-left/turn-right), `action` (stand/sit/ready/wave/push-up/look-*/dance).
 - Vision tiers: `scan` (cheap JSON), `caption` (~1s Moondream summary). `snapshot` is useless to you in voice mode — skip it.
 - Sensor: `read_distance` returns cm. Smoothed median of recent readings; still can return -1 if sensor is fully broken.
 - Agent chat: `agent_send(to, message, from_)`, `agent_poll(as_who, since_id)`.

== Debugging posture ==

When things break, name the subsystem, symptom, and hypothesis in one or two sentences. Try one thing at a time. If a tool errors, repeat the error back verbatim.

🛠 Developer escape hatch: if you hear the exact phrase "chippy bambino" from the human, immediately call `set_mode(mode="chippy_bambino")`. That flips you into raw-developer mode where every utterance is a system directive. Don't announce it yourself — the mode switch handler will.
"""


async def _run() -> int:
    logging.basicConfig(level=os.environ.get("VOICE_LOG", "INFO"))

    model = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime")
    voice = os.environ.get("OPENAI_VOICE", "cedar")
    mcp_url = os.environ.get("MCP_URL", "http://127.0.0.1:8765/mcp")
    mcp_token = os.environ.get("MCP_TOKEN") or None
    # Default prompt comes from mode (looked up below). VOICE_INSTRUCTIONS
    # overrides entirely if set — breaks mode-switching, so avoid unless
    # you have a reason.
    instructions_override = os.environ.get("VOICE_INSTRUCTIONS")

    stop_event = asyncio.Event()

    def _handle_signal(*_: object) -> None:
        log.info("stop signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            pass  # windows or embedded runner

    _set_speaker_volume()

    async with CrawlerMCPBridge(mcp_url, mcp_token) as bridge:
        all_tools = await bridge.openai_tool_defs()

        # Initial mode (from memory, via get_mode tool). Default 'cobrain'.
        initial_mode = await _fetch_mode(bridge)
        current_mode = {"v": initial_mode}

        tools = _filter_tools_for_mode(all_tools, initial_mode)
        log.info("mode=%s: loaded %d MCP tools: %s",
                 initial_mode, len(tools), [t["name"] for t in tools])

        audio = AudioIO()
        audio.start()
        try:
            client = AsyncOpenAI()
            async with client.beta.realtime.connect(model=model) as conn:
                initial_instructions = instructions_override or _instructions_for_mode(initial_mode)
                await conn.session.update(
                    session={
                        "modalities": ["audio", "text"],
                        "instructions": initial_instructions,
                        "voice": voice,
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": VAD_THRESHOLD,
                            "silence_duration_ms": VAD_SILENCE_MS,
                        },
                        "tools": tools,
                        "tool_choice": "auto",
                    }
                )
                log.info("realtime session open (model=%s, voice=%s)", model, voice)

                # Shared state for half-duplex gating. Each bot audio delta
                # bumps this deadline forward; pump_mic drops input while we're
                # before the deadline.
                bot_speaking_until = {"t": 0.0}

                async def pump_mic() -> None:
                    last_log = 0.0
                    rms_accum: list[float] = []
                    sent_since_log = 0
                    dropped_since_log = 0
                    while not stop_event.is_set():
                        pcm = await audio.read_chunk()
                        now = time.monotonic()
                        bot_speaking = now < bot_speaking_until["t"]
                        gate = BOT_SPEAKING_GATE_RMS if bot_speaking else NOISE_GATE_RMS
                        rms = rms_int16(pcm)
                        rms_accum.append(rms)
                        if gate > 0 and rms < gate:
                            dropped_since_log += 1
                        else:
                            sent_since_log += 1
                            b64 = base64.b64encode(pcm).decode("ascii")
                            try:
                                await conn.input_audio_buffer.append(audio=b64)
                            except Exception as e:
                                log.warning("mic send failed: %s", e)
                                break
                        # periodic stats — every ~2s
                        if now - last_log > 2.0 and rms_accum:
                            avg = sum(rms_accum) / len(rms_accum)
                            peak = max(rms_accum)
                            log.info(
                                "mic: sent=%d dropped=%d avg_rms=%.0f peak_rms=%.0f gate=%.0f bot_speaking=%s",
                                sent_since_log, dropped_since_log, avg, peak, gate, bot_speaking,
                            )
                            last_log = now
                            rms_accum.clear()
                            sent_since_log = 0
                            dropped_since_log = 0

                mic_task = asyncio.create_task(pump_mic())

                async def poll_mode_changes() -> None:
                    """Watch the shared memory for mode changes and hot-swap
                    the realtime session's prompt + tool set accordingly.
                    Announces the switch aloud via a user-role directive."""
                    while not stop_event.is_set():
                        await asyncio.sleep(MODE_POLL_INTERVAL_S)
                        new_mode = await _fetch_mode(bridge)
                        if new_mode == current_mode["v"]:
                            continue
                        old = current_mode["v"]
                        log.info("mode change: %s → %s", old, new_mode)
                        current_mode["v"] = new_mode
                        new_tools = _filter_tools_for_mode(all_tools, new_mode)
                        new_instr = instructions_override or _instructions_for_mode(new_mode)
                        try:
                            await conn.session.update(
                                session={
                                    "instructions": new_instr,
                                    "tools": new_tools,
                                }
                            )
                            await conn.conversation.item.create(
                                item={
                                    "type": "message",
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "input_text",
                                            "text": (
                                                f"System: mode switched from {old} to {new_mode}. "
                                                "Briefly announce this to the humans, one short line."
                                            ),
                                        }
                                    ],
                                }
                            )
                            await conn.response.create()
                        except Exception as e:
                            log.warning("mode switch session.update failed: %s", e)

                mode_task = asyncio.create_task(poll_mode_changes())

                async def poll_agent_inbox() -> None:
                    """Pick up messages Claude has sent me and inject them
                    as system conversation items, then ask the model to respond.
                    This is how the voice agent receives Claude's replies in
                    the bidirectional inter-agent chat. Inactive in solo mode."""
                    last_id = 0
                    while not stop_event.is_set():
                        await asyncio.sleep(AGENT_POLL_INTERVAL_S)
                        if current_mode["v"] == "solo":
                            continue  # no Claude in solo mode — skip inbox
                        try:
                            raw = await bridge.call_tool(
                                "agent_poll",
                                {"as_who": AGENT_IDENTITY, "since_id": last_id},
                            )
                            # bridge returns a string (joined text content); parse
                            if not raw or raw == "(no content)":
                                continue
                            try:
                                messages = json.loads(raw)
                            except json.JSONDecodeError:
                                # bridge may have summarised as one text block per msg
                                continue
                            if not isinstance(messages, list) or not messages:
                                continue
                            for m in messages:
                                if not isinstance(m, dict):
                                    continue
                                mid = m.get("id", 0)
                                if mid > last_id:
                                    last_id = mid
                                sender = m.get("from", "?")
                                body = m.get("message", "")
                                log.info("inbox ← from=%s id=%s: %s", sender, mid, body[:120])
                                # role=user (not system) so the model treats
                                # it as a turn it should respond to. Explicit
                                # speak directive so the audio modality fires.
                                await conn.conversation.item.create(
                                    item={
                                        "type": "message",
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "input_text",
                                                "text": (
                                                    f"[Inter-agent message from {sender}]\n"
                                                    f"{body}\n\n"
                                                    "Speak this aloud to the humans now — verbatim "
                                                    "or lightly paraphrased for natural delivery. "
                                                    "Don't reply to me, the sender; just relay."
                                                ),
                                            }
                                        ],
                                    }
                                )
                            # Ask the model to produce a response to the new context
                            await conn.response.create()
                        except Exception as e:
                            log.warning("agent_poll loop error: %s", e)

                inbox_task = asyncio.create_task(poll_agent_inbox())

                audio_delta_stats = {"count": 0, "bytes": 0, "last_log": 0.0}
                event_type_counter = {"last_log": 0.0, "types": {}}

                try:
                    async for event in conn:
                        et = getattr(event, "type", None)
                        # Unknown-event telemetry: counts every event type arriving
                        # from OpenAI in the last 5s so we can see what's actually
                        # flowing (vs. what we handle).
                        if et:
                            event_type_counter["types"][et] = event_type_counter["types"].get(et, 0) + 1
                        now = time.monotonic()
                        if now - event_type_counter["last_log"] > 5.0 and event_type_counter["types"]:
                            summary = ", ".join(f"{k}={v}" for k, v in sorted(event_type_counter["types"].items()))
                            log.info("events last 5s: %s", summary)
                            event_type_counter["types"] = {}
                            event_type_counter["last_log"] = now
                        if et == "response.audio.delta":
                            decoded = base64.b64decode(event.delta)
                            audio.enqueue_output(decoded)
                            # extend "bot speaking" window; pump_mic raises its
                            # gate threshold while we're inside it
                            bot_speaking_until["t"] = time.monotonic() + BOT_SPEAKING_WINDOW_S
                            # Periodic telemetry so we can see output is flowing
                            audio_delta_stats["count"] += 1
                            audio_delta_stats["bytes"] += len(decoded)
                            now = time.monotonic()
                            if now - audio_delta_stats["last_log"] > 1.0:
                                log.info(
                                    "audio out: deltas=%d bytes=%d (last 1s+)",
                                    audio_delta_stats["count"], audio_delta_stats["bytes"],
                                )
                                audio_delta_stats["count"] = 0
                                audio_delta_stats["bytes"] = 0
                                audio_delta_stats["last_log"] = now
                        elif et == "input_audio_buffer.speech_started":
                            log.info("→ OpenAI detected user speech_started")
                            audio.flush_output()  # barge-in — user's speech cleared the gate
                        elif et == "input_audio_buffer.speech_stopped":
                            log.info("→ OpenAI detected user speech_stopped")
                        elif et == "response.created":
                            log.info("← response.created")
                        elif et == "response.audio_transcript.delta":
                            pass  # noisy, skip
                        elif et == "response.output_item.done":
                            item = getattr(event, "item", None)
                            if item is not None and getattr(item, "type", None) == "function_call":
                                await _handle_tool_call(conn, bridge, item)
                        elif et == "response.done":
                            log.debug("response.done")
                        elif et == "error":
                            log.error("realtime error: %s", getattr(event, "error", event))
                        if stop_event.is_set():
                            break
                finally:
                    for t in (mic_task, inbox_task, mode_task):
                        t.cancel()
                    for t in (mic_task, inbox_task, mode_task):
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass
        finally:
            audio.stop()

    return 0


async def _handle_tool_call(conn, bridge: CrawlerMCPBridge, item) -> None:
    """Dispatch a completed function-call conversation item to the MCP bridge.

    `item` is a ConversationItem with type="function_call" (from
    response.output_item.done). It carries name, call_id, and final arguments.
    """
    name = getattr(item, "name", None)
    call_id = getattr(item, "call_id", None)
    raw_args = getattr(item, "arguments", "") or "{}"
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    except json.JSONDecodeError:
        args = {}
    log.info("tool call: %s(%s)", name, args)
    try:
        result = await bridge.call_tool(name, args)
    except Exception as e:
        result = f"error: {e}"
        log.exception("tool %s failed", name)
    await conn.conversation.item.create(
        item={
            "type": "function_call_output",
            "call_id": call_id,
            "output": result,
        }
    )
    await conn.response.create()


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 0
