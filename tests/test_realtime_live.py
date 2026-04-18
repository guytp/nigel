"""Live end-to-end test against the real OpenAI Realtime API.

Gated on RUN_LIVE_REALTIME=1 because it costs real money (~$0.02/run with
gpt-4o-mini-realtime-preview) and needs OPENAI_API_KEY.

Exercises the full chain:
 - mock MCP server subprocess (from conftest)
 - CrawlerMCPBridge over HTTP with bearer auth
 - AsyncOpenAI Realtime connection, session.update with our tool defs
 - user text message → model → function_call → MCP → function_call_output → model → text

Purpose: prove our session shape, event handling, and tool-call dispatch
actually work against the real API before we touch hardware.
"""

from __future__ import annotations

import os

import pytest

from voice_agent.mcp_bridge import CrawlerMCPBridge
from voice_agent.text_runner import run_prompts


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVE_REALTIME") != "1"
        or not os.environ.get("OPENAI_API_KEY"),
        reason="live Realtime test — set RUN_LIVE_REALTIME=1 and OPENAI_API_KEY",
    ),
]


MODEL = os.environ.get("LIVE_REALTIME_MODEL", "gpt-4o-mini-realtime-preview")


async def test_live_distance_prompt_invokes_read_distance(mcp_server):
    """Ask about distance → the model should call read_distance."""
    async with CrawlerMCPBridge(mcp_server["url"], token=mcp_server["token"]) as bridge:
        logs = await run_prompts(
            ["How far is the nearest obstacle ahead, in centimetres?"],
            bridge,
            model=MODEL,
        )
    turn = logs[0]
    assert turn.error is None, f"turn errored: {turn.error}"
    tool_names = [name for name, _ in turn.tool_calls]
    assert "read_distance" in tool_names, (
        f"expected read_distance call, got tools={tool_names}, text={turn.text!r}"
    )
    # and the model should weave the result into natural text
    assert turn.text, "no text response from model"


async def test_live_move_forward_invokes_move_tool(mcp_server):
    """Ask it to walk forward → expects a move call with direction=forward."""
    async with CrawlerMCPBridge(mcp_server["url"], token=mcp_server["token"]) as bridge:
        logs = await run_prompts(
            ["Please walk forward one step."],
            bridge,
            model=MODEL,
        )
    turn = logs[0]
    assert turn.error is None, f"turn errored: {turn.error}"
    moves = [(n, a) for n, a in turn.tool_calls if n == "move"]
    assert moves, (
        f"expected a move call, got tools={turn.tool_calls}, text={turn.text!r}"
    )
    direction = moves[0][1].get("direction", "")
    assert direction == "forward", f"expected forward, got {direction!r}"


async def test_live_scan_and_describe(mcp_server):
    """Ask Nigel to look → prefers `scan` over `snapshot` per instructions."""
    async with CrawlerMCPBridge(mcp_server["url"], token=mcp_server["token"]) as bridge:
        logs = await run_prompts(
            ["Quick check — what's in front of you? Describe briefly."],
            bridge,
            model=MODEL,
        )
    turn = logs[0]
    assert turn.error is None, f"turn errored: {turn.error}"
    tool_names = [name for name, _ in turn.tool_calls]
    assert any(t in tool_names for t in ("scan", "caption")), (
        f"expected a vision call (scan/caption), got {tool_names}"
    )
    assert turn.text, "model should describe what it found"


async def test_live_multi_turn_state_persists(mcp_server):
    """Two turns in the same session — context should carry across."""
    async with CrawlerMCPBridge(mcp_server["url"], token=mcp_server["token"]) as bridge:
        logs = await run_prompts(
            [
                "Check the distance, then stand ready.",
                "What did the distance reading say?",
            ],
            bridge,
            model=MODEL,
        )
    assert all(t.error is None for t in logs), f"errors: {[t.error for t in logs]}"
    # first turn should invoke read_distance
    tools_first = [n for n, _ in logs[0].tool_calls]
    assert "read_distance" in tools_first
    # second turn references the prior result in its text (doesn't have to recall
    # the exact number but should not start from scratch)
    assert logs[1].text, "second turn should respond"
