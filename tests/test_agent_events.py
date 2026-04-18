"""Unit tests for agent's tool-call event handler.

Exercises `_handle_tool_call` with a fake OpenAI conn + fake MCP bridge,
asserting the call_id is echoed, result is stringified, and that the
agent issues response.create to continue the turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from voice_agent.agent import _handle_tool_call


pytestmark = pytest.mark.asyncio


class FakeBridge:
    def __init__(self, result: str = "ok"):
        self._result = result
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, args: dict) -> str:
        self.calls.append((name, args))
        return self._result


class FakeConvItem:
    def __init__(self, sink):
        self._sink = sink

    async def create(self, item: dict[str, Any]) -> None:
        self._sink.append(("conversation.item.create", item))


class FakeResponse:
    def __init__(self, sink):
        self._sink = sink

    async def create(self) -> None:
        self._sink.append(("response.create", None))


class FakeConn:
    def __init__(self):
        self.events: list = []
        self.conversation = type("C", (), {"item": FakeConvItem(self.events)})()
        self.response = FakeResponse(self.events)


@dataclass
class FakeFunctionCallItem:
    """Stand-in for a ConversationItem with type='function_call'."""
    call_id: str
    name: str
    arguments: str  # JSON-encoded
    type: str = "function_call"


async def test_handle_tool_call_routes_and_responds():
    conn = FakeConn()
    bridge = FakeBridge(result='{"cm": 42.0}')
    event = FakeFunctionCallItem(call_id="call_1", name="read_distance", arguments="{}")

    await _handle_tool_call(conn, bridge, event)

    # bridge got invoked
    assert bridge.calls == [("read_distance", {})]
    # conn got the function_call_output + response.create, in that order
    kinds = [k for k, _ in conn.events]
    assert kinds == ["conversation.item.create", "response.create"]
    item = conn.events[0][1]
    assert item["type"] == "function_call_output"
    assert item["call_id"] == "call_1"
    assert item["output"] == '{"cm": 42.0}'


async def test_handle_tool_call_parses_args_json():
    conn = FakeConn()
    bridge = FakeBridge()
    event = FakeFunctionCallItem(
        call_id="c2",
        name="move",
        arguments='{"direction": "forward", "steps": 2, "speed": 80}',
    )
    await _handle_tool_call(conn, bridge, event)
    assert bridge.calls[0][0] == "move"
    assert bridge.calls[0][1] == {"direction": "forward", "steps": 2, "speed": 80}


async def test_handle_tool_call_handles_malformed_json():
    conn = FakeConn()
    bridge = FakeBridge()
    event = FakeFunctionCallItem(call_id="c3", name="stop", arguments="not json")
    await _handle_tool_call(conn, bridge, event)
    # malformed args fall through to empty dict, tool still invoked
    assert bridge.calls[0][1] == {}


async def test_handle_tool_call_surfaces_bridge_exception_as_text():
    """If the tool raises, the output should contain the error — not crash the loop."""

    class BoomBridge:
        async def call_tool(self, name: str, args: dict) -> str:
            raise RuntimeError("boom")

    conn = FakeConn()
    event = FakeFunctionCallItem(call_id="c4", name="move", arguments="{}")
    await _handle_tool_call(conn, BoomBridge(), event)

    item = conn.events[0][1]
    assert item["type"] == "function_call_output"
    assert "boom" in item["output"].lower()
    # response.create is still called so the model can recover
    assert conn.events[1][0] == "response.create"
