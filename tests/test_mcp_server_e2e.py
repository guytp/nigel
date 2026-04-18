"""End-to-end tests against a live MCP server subprocess.

Spawns `mcp-picrawler` with mock hardware + bearer auth on a random port, then
drives it through the real `mcp` Python client. Exercises every tool and
verifies the response shape. This catches transport/serialisation bugs that
unit tests would miss.
"""

from __future__ import annotations

import base64
import json

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


pytestmark = pytest.mark.asyncio


async def _session(mcp_server):
    headers = {"Authorization": f"Bearer {mcp_server['token']}"}
    # Return the context managers to the caller; it has to async-with both.
    return streamablehttp_client(mcp_server["url"], headers=headers)


async def test_server_starts_and_initializes(mcp_server):
    headers = {"Authorization": f"Bearer {mcp_server['token']}"}
    async with streamablehttp_client(mcp_server["url"], headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            result = await session.initialize()
            assert result.serverInfo.name == "picrawler"


async def test_list_tools_returns_expected_set(mcp_server):
    headers = {"Authorization": f"Bearer {mcp_server['token']}"}
    expected = {
        "move",
        "action",
        "stop",
        "snapshot",
        "scan",
        "caption",
        "read_distance",
        "set_vision",
        "set_target_color",
        "read_detections",
        "speak",
    }
    async with streamablehttp_client(mcp_server["url"], headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            names = {t.name for t in tools}
            assert names == expected
            for t in tools:
                assert t.description  # all tools have descriptions
                assert isinstance(t.inputSchema, dict)


async def test_list_resources(mcp_server):
    headers = {"Authorization": f"Bearer {mcp_server['token']}"}
    async with streamablehttp_client(mcp_server["url"], headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            resources = (await session.list_resources()).resources
            uris = {str(res.uri) for res in resources}
            assert "picrawler://state" in uris
            assert "picrawler://stream" in uris


async def test_read_distance_returns_number(mcp_server):
    headers = {"Authorization": f"Bearer {mcp_server['token']}"}
    async with streamablehttp_client(mcp_server["url"], headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool("read_distance", {})
            text = _text_of(result)
            data = json.loads(text)
            assert "cm" in data
            assert isinstance(data["cm"], (int, float))


async def test_move_returns_text_plus_image(mcp_server):
    headers = {"Authorization": f"Bearer {mcp_server['token']}"}
    async with streamablehttp_client(mcp_server["url"], headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool(
                "move", {"direction": "forward", "steps": 1, "speed": 50}
            )
            kinds = [_kind_of(c) for c in result.content]
            assert "text" in kinds
            assert "image" in kinds
            image_blob = _first_image(result)
            assert image_blob[:2] == b"\xff\xd8"  # JPEG


async def test_move_rejects_invalid_direction(mcp_server):
    headers = {"Authorization": f"Bearer {mcp_server['token']}"}
    async with streamablehttp_client(mcp_server["url"], headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool("move", {"direction": "sideways"})
            assert result.isError, "invalid direction should return an error"


async def test_scan_returns_structured_json(mcp_server):
    headers = {"Authorization": f"Bearer {mcp_server['token']}"}
    async with streamablehttp_client(mcp_server["url"], headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool("scan", {"include_objects": False})
            text = _text_of(result)
            data = json.loads(text)
            assert {"motion", "phash", "objects", "elapsed_ms", "tiers"} <= data.keys()


async def test_snapshot_returns_image(mcp_server):
    headers = {"Authorization": f"Bearer {mcp_server['token']}"}
    async with streamablehttp_client(mcp_server["url"], headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool("snapshot", {})
            blob = _first_image(result)
            assert blob[:2] == b"\xff\xd8"


async def test_action_wave(mcp_server):
    headers = {"Authorization": f"Bearer {mcp_server['token']}"}
    async with streamablehttp_client(mcp_server["url"], headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool("action", {"name": "wave", "steps": 1})
            assert not result.isError
            assert "wave" in _text_of(result).lower()


async def test_set_vision_and_read_detections(mcp_server):
    headers = {"Authorization": f"Bearer {mcp_server['token']}"}
    async with streamablehttp_client(mcp_server["url"], headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            toggle = await session.call_tool("set_vision", {"feature": "face", "enabled": True})
            assert not toggle.isError
            detections_result = await session.call_tool("read_detections", {})
            data = _structured(detections_result)
            assert isinstance(data, list)
            assert any(d.get("kind") == "face" for d in data)


async def test_speak_does_not_raise(mcp_server):
    headers = {"Authorization": f"Bearer {mcp_server['token']}"}
    async with streamablehttp_client(mcp_server["url"], headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool("speak", {"text": "hello"})
            assert not result.isError
            assert "hello" in _text_of(result)


async def test_state_resource_reflects_last_action(mcp_server):
    headers = {"Authorization": f"Bearer {mcp_server['token']}"}
    async with streamablehttp_client(mcp_server["url"], headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            await session.call_tool("action", {"name": "sit"})
            res = await session.read_resource("picrawler://state")
            text = res.contents[0].text
            data = json.loads(text)
            assert data["last_action"] == "sit"
            assert data["backend"] == "mock"


# ------------------------------------------------------------ helpers

def _kind_of(content) -> str:
    if hasattr(content, "text"):
        return "text"
    if hasattr(content, "data"):
        return "image"
    return "other"


def _text_of(result) -> str:
    for c in result.content:
        if hasattr(c, "text") and c.text is not None:
            return c.text
    raise AssertionError(f"no text in result: {result.content!r}")


def _first_image(result) -> bytes:
    for c in result.content:
        if hasattr(c, "data") and c.data:
            return base64.b64decode(c.data)
    raise AssertionError(f"no image in result: {result.content!r}")


def _structured(result):
    """Prefer structuredContent; fall back to concatenated text JSON.

    FastMCP emits `structuredContent` for typed returns (wrapping list[...] as
    {"result": [...]}). When that's absent we stitch TextContent blocks.
    """
    sc = getattr(result, "structuredContent", None)
    if sc is not None:
        if isinstance(sc, dict) and list(sc.keys()) == ["result"]:
            return sc["result"]
        return sc
    text = "".join(c.text for c in result.content if getattr(c, "text", None))
    return json.loads(text)
