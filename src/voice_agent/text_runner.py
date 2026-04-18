"""Text-mode Realtime runner — validates the whole brain⇄body loop without audio.

Same session/event/tool plumbing as the voice agent, but:
 - `modalities: ["text"]` (no audio I/O, no portaudio, no mic permissions)
 - user input via a list of text prompts passed in, or stdin interactive loop
 - model output streamed to stdout as plain text

This exists to let us verify the OpenAI Realtime integration against the real
API while we're blocked from audio by mic-permission dialogs. Every bug that
would bite voice mode will also bite here: session config, tool-call event
shape, function_call_output semantics, turn lifecycle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from .mcp_bridge import CrawlerMCPBridge

log = logging.getLogger(__name__)


TEXT_MODE_INSTRUCTIONS = """You are Nigel, an LLM embodied in a SunFounder PiCrawler quadruped robot.

In this session you have no voice — respond in short text. Use the tools to act on the world. Narrate what you're doing. Prefer `scan` / `read_distance` / `caption` to `snapshot` since `snapshot` returns an image you can't see in a text-only session.

Be concise. One tool call per turn is normal. Report back what you learned.
"""


@dataclass
class TurnLog:
    prompt: str
    text: str = ""
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    error: str | None = None


async def run_prompts(
    prompts: list[str],
    bridge: CrawlerMCPBridge,
    model: str = "gpt-realtime",
    instructions: str = TEXT_MODE_INSTRUCTIONS,
    timeout_per_turn: float = 45.0,
) -> list[TurnLog]:
    """Run a canned list of prompts against a live Realtime session.

    Returns a TurnLog per prompt with collected text + tool invocations.
    Useful as a scripted smoke test.
    """
    tools = await bridge.openai_tool_defs()
    client = AsyncOpenAI()
    logs: list[TurnLog] = []

    async with client.beta.realtime.connect(model=model) as conn:
        await conn.session.update(
            session={
                "modalities": ["text"],
                "instructions": instructions,
                "tools": tools,
                "tool_choice": "auto",
            }
        )

        for prompt in prompts:
            turn = TurnLog(prompt=prompt)
            logs.append(turn)
            await conn.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            )
            await conn.response.create()
            try:
                await asyncio.wait_for(
                    _consume_one_turn(conn, bridge, turn),
                    timeout=timeout_per_turn,
                )
            except asyncio.TimeoutError:
                turn.error = f"turn timed out after {timeout_per_turn}s"
                break

    return logs


async def _consume_one_turn(conn, bridge: CrawlerMCPBridge, turn: TurnLog) -> None:
    """Drain events for one prompt until we see a response.done at the top level.

    A turn may contain multiple tool calls → sub-responses. We count:
      opened = number of response.created events seen
      closed = number of response.done events seen
    The turn ends when opened == closed and opened >= 1.
    """
    opened = 0
    closed = 0
    async for event in conn:
        et = getattr(event, "type", None)
        if et == "response.created":
            opened += 1
        elif et == "response.text.delta":
            turn.text += getattr(event, "delta", "") or ""
        elif et == "response.audio_transcript.delta":
            # model may emit transcript deltas even in text mode — ignore
            pass
        elif et == "response.output_item.done":
            item = getattr(event, "item", None)
            if item is not None and getattr(item, "type", None) == "function_call":
                await _dispatch_tool(conn, bridge, item, turn)
        elif et == "response.done":
            closed += 1
            if opened >= 1 and opened == closed:
                return
        elif et == "error":
            err = getattr(event, "error", event)
            turn.error = str(err)
            log.error("realtime error during turn: %s", err)
            return


async def _dispatch_tool(conn, bridge: CrawlerMCPBridge, item, turn: TurnLog) -> None:
    name = getattr(item, "name", None)
    call_id = getattr(item, "call_id", None)
    raw_args = getattr(item, "arguments", "") or "{}"
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    except json.JSONDecodeError:
        args = {}
    log.info("tool call: %s(%s)", name, args)
    turn.tool_calls.append((name or "?", args))

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


async def interactive(
    bridge: CrawlerMCPBridge,
    model: str = "gpt-realtime",
    instructions: str = TEXT_MODE_INSTRUCTIONS,
) -> None:
    """REPL: each stdin line = one user turn. Ctrl-D to exit."""
    tools = await bridge.openai_tool_defs()
    client = AsyncOpenAI()
    async with client.beta.realtime.connect(model=model) as conn:
        await conn.session.update(
            session={
                "modalities": ["text"],
                "instructions": instructions,
                "tools": tools,
                "tool_choice": "auto",
            }
        )
        print("connected — talk to Nigel (Ctrl-D to quit)", file=sys.stderr)
        loop = asyncio.get_running_loop()
        while True:
            print("\n> ", end="", flush=True)
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            prompt = line.strip()
            if not prompt:
                continue
            turn = TurnLog(prompt=prompt)
            await conn.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            )
            await conn.response.create()
            await _consume_one_turn(conn, bridge, turn)
            if turn.tool_calls:
                for name, args in turn.tool_calls:
                    print(f"  [tool: {name}({args})]", file=sys.stderr)
            print(turn.text or "(no text)")
            if turn.error:
                print(f"  ERROR: {turn.error}", file=sys.stderr)


def main() -> int:
    logging.basicConfig(level=os.environ.get("VOICE_LOG", "INFO"))
    mcp_url = os.environ.get("MCP_URL", "http://127.0.0.1:8765/mcp")
    mcp_token = os.environ.get("MCP_TOKEN") or None
    model = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime")

    async def _main() -> int:
        async with CrawlerMCPBridge(mcp_url, mcp_token) as bridge:
            await interactive(bridge, model=model)
        return 0

    try:
        return asyncio.run(_main())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
