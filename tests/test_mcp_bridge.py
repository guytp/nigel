"""Tests for the voice_agent MCP bridge against a live mock MCP server.

Ensures: tool translation to OpenAI function-tool shape, round-trip calls,
and image content summarisation (voice agent gets a text note, not base64
image bytes that would blow up the session context).
"""

from __future__ import annotations

import json

import pytest

from voice_agent.mcp_bridge import CrawlerMCPBridge


pytestmark = pytest.mark.asyncio


async def test_bridge_connects_and_lists_tools(mcp_server):
    async with CrawlerMCPBridge(mcp_server["url"], token=mcp_server["token"]) as bridge:
        defs = await bridge.openai_tool_defs()
    names = {d["name"] for d in defs}
    assert {"move", "action", "stop", "snapshot", "scan", "caption"} <= names


async def test_openai_tool_def_shape(mcp_server):
    async with CrawlerMCPBridge(mcp_server["url"], token=mcp_server["token"]) as bridge:
        defs = await bridge.openai_tool_defs()
    for d in defs:
        assert d["type"] == "function"
        assert isinstance(d["name"], str) and d["name"]
        assert isinstance(d["description"], str)
        assert isinstance(d["parameters"], dict)
        assert d["parameters"].get("type") == "object"


async def test_bridge_rejects_wrong_token(mcp_server):
    """With a bad token, entering the context should surface an auth failure."""
    bridge = CrawlerMCPBridge(mcp_server["url"], token="wrong")
    with pytest.raises(BaseException):
        await bridge.__aenter__()
        try:
            await bridge.openai_tool_defs()
        finally:
            await bridge.__aexit__(None, None, None)


async def test_call_tool_read_distance_returns_text(mcp_server):
    async with CrawlerMCPBridge(mcp_server["url"], token=mcp_server["token"]) as bridge:
        text = await bridge.call_tool("read_distance", {})
    data = json.loads(text)
    assert "cm" in data
    assert isinstance(data["cm"], (int, float))


async def test_call_tool_move_summarises_image(mcp_server):
    """move returns text + image; bridge must flatten to text with a note."""
    async with CrawlerMCPBridge(mcp_server["url"], token=mcp_server["token"]) as bridge:
        result = await bridge.call_tool(
            "move", {"direction": "forward", "steps": 1, "speed": 50}
        )
    assert "moved forward" in result
    assert "[image returned" in result or "image/jpeg" in result
    # and critically, no raw base64 payload leaking into the voice prompt
    assert len(result) < 1000  # images should be noted, not embedded


async def test_call_tool_scan(mcp_server):
    async with CrawlerMCPBridge(mcp_server["url"], token=mcp_server["token"]) as bridge:
        result = await bridge.call_tool("scan", {"include_objects": False})
    # scan returns dict; may be JSON text directly
    data = json.loads(result)
    assert "motion" in data
    assert "phash" in data


async def test_call_tool_invalid_arguments_raises(mcp_server):
    async with CrawlerMCPBridge(mcp_server["url"], token=mcp_server["token"]) as bridge:
        result = await bridge.call_tool("move", {"direction": "sideways"})
    # MCP returns errors as result content (isError=True); bridge stringifies
    assert "error" in result.lower() or "direction" in result.lower()
