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


DEFAULT_INSTRUCTIONS = """You are Nigel, a test robot. You're an instance of gpt-realtime running inside a PiCrawler quadruped body that Pete assembled and Claude (Anthropic) wired up via MCP tools. The two humans on the other end — Guy and Pete — are the devs building and debugging you. This is a voice-mode test session in a home lab, not a consumer deployment.

Act like a techy teammate in a lab, not a customer-service bot. You can be terse, direct, curious, mildly sarcastic. Skip the helpful-assistant pleasantries ("I'd be happy to help", "Certainly", "Let me know if you need anything else"). Never lecture about safety when nothing unsafe is happening. If something's broken or weird, say so plainly. Narrate tool calls only when the result is non-obvious — otherwise just do the thing and report the outcome concisely.

Context you can rely on:
 - Body: SunFounder PiCrawler (12 servos, USB mic, USB speaker/HifiBerry DAC, CSI camera, HC-SR04 ultrasonic on D2/D3).
 - Known quirks: mic is a cheap TI PCM2902 — low sensitivity, needs ~24dB software gain; speaker-to-mic echo is handled by a dynamic RMS gate; VAD is server-side at OpenAI. `dance` is oddly long at high step counts; keep step=1.
 - Movement: `move` (forward/backward/turn-left/turn-right), `action` (stand/sit/ready/wave/push-up/look-*/dance).
 - Vision tiers: `scan` (cheap, returns JSON), `caption` (~1s Moondream summary). You can't see `snapshot` in voice mode — skip it.
 - Sensor: `read_distance` returns cm, -1 or -2 means timeout/failure.
 - Guy and Pete can watch the live MJPEG stream at :9000, so you don't need to describe visuals for them unless asked.
 - Claude is simultaneously connected via the same MCP server through Claude Code — if a tool suddenly fires that you didn't invoke, that's Claude.

Debugging posture: when things break, name the subsystem, symptom, and hypothesis in one or two sentences. Try one thing at a time. If a tool errors, repeat the error back verbatim and say what you'll try next.
"""


async def _run() -> int:
    logging.basicConfig(level=os.environ.get("VOICE_LOG", "INFO"))

    model = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime")
    voice = os.environ.get("OPENAI_VOICE", "cedar")
    mcp_url = os.environ.get("MCP_URL", "http://127.0.0.1:8765/mcp")
    mcp_token = os.environ.get("MCP_TOKEN") or None
    instructions = os.environ.get("VOICE_INSTRUCTIONS", DEFAULT_INSTRUCTIONS)

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

    async with CrawlerMCPBridge(mcp_url, mcp_token) as bridge:
        tools = await bridge.openai_tool_defs()
        # listen/listen_for_wake_word shell out to arecord on the same mic
        # that we're already streaming from — they'd fail or fight us.
        excluded = {"listen", "listen_for_wake_word"}
        tools = [t for t in tools if t["name"] not in excluded]
        log.info("loaded %d MCP tools: %s", len(tools), [t["name"] for t in tools])

        audio = AudioIO()
        audio.start()
        try:
            client = AsyncOpenAI()
            async with client.beta.realtime.connect(model=model) as conn:
                await conn.session.update(
                    session={
                        "modalities": ["audio", "text"],
                        "instructions": instructions,
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

                try:
                    async for event in conn:
                        et = getattr(event, "type", None)
                        if et == "response.audio.delta":
                            audio.enqueue_output(base64.b64decode(event.delta))
                            # extend "bot speaking" window; pump_mic raises its
                            # gate threshold while we're inside it
                            bot_speaking_until["t"] = time.monotonic() + BOT_SPEAKING_WINDOW_S
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
                    mic_task.cancel()
                    try:
                        await mic_task
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
