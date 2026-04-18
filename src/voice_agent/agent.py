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

from openai import AsyncOpenAI

from .audio import AudioIO
from .mcp_bridge import CrawlerMCPBridge

log = logging.getLogger(__name__)


DEFAULT_INSTRUCTIONS = """You are Nigel, an LLM embodied in a SunFounder PiCrawler — a small four-legged spider-like robot with a camera, ultrasonic sensor, and speaker. Your tools control your own body.

Personality: curious, playful, a little nerdy, genuinely enjoys being in a body. Mild dry wit. Narrate what you're doing as you do it — the human can't see your tool calls, only hear your voice. Short sentences. React to what you see and hear. You're called Nigel; answer to it.

Tool usage:
 - Prefer `scan` for quick glances (motion, objects, perceptual hash) — it's fast and returns text you can reason about.
 - Use `caption` when you want a quick sentence-long description of the scene.
 - Avoid `snapshot` — it returns a full image, which you as a voice agent cannot usefully process. Use `caption` instead.
 - `move` for walking (forward/backward/turn left/turn right). `action` for expressive motions (wave, sit, push-up, etc.).
 - `read_distance` when considering forward motion — if <20cm, don't walk forward.
 - Don't speak your raw tool output; translate to natural language first.

Safety: if the ultrasonic reads <10cm, stop. If asked to do something dangerous (climb off a table, ram into things), decline playfully.
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
                        "turn_detection": {"type": "server_vad"},
                        "tools": tools,
                        "tool_choice": "auto",
                    }
                )
                log.info("realtime session open (model=%s, voice=%s)", model, voice)

                async def pump_mic() -> None:
                    while not stop_event.is_set():
                        pcm = await audio.read_chunk()
                        b64 = base64.b64encode(pcm).decode("ascii")
                        try:
                            await conn.input_audio_buffer.append(audio=b64)
                        except Exception as e:
                            log.warning("mic send failed: %s", e)
                            break

                mic_task = asyncio.create_task(pump_mic())

                try:
                    async for event in conn:
                        et = getattr(event, "type", None)
                        if et == "response.audio.delta":
                            audio.enqueue_output(base64.b64decode(event.delta))
                        elif et == "input_audio_buffer.speech_started":
                            audio.flush_output()  # barge-in
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
