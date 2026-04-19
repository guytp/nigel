"""Unit tests for voice_agent mode-switching helpers."""

from __future__ import annotations

from voice_agent.agent import (
    ALWAYS_EXCLUDED_TOOLS,
    COUPLED_INSTRUCTIONS,
    SOLO_EXTRA_EXCLUDED_TOOLS,
    SOLO_INSTRUCTIONS,
    _filter_tools_for_mode,
    _instructions_for_mode,
)


def _tool(name: str) -> dict:
    return {"type": "function", "name": name, "description": "", "parameters": {}}


ALL_TOOLS = [
    _tool(n) for n in [
        "move", "action", "scan", "snapshot", "caption", "speak",
        "read_distance", "read_text",
        "set_vision", "set_target_color", "read_detections",
        "listen", "listen_for_wake_word",
        "agent_send", "agent_poll",
        "memory_set", "memory_get", "memory_search",
        "set_mode", "get_mode",
    ]
]


def test_coupled_includes_agent_tools():
    names = {t["name"] for t in _filter_tools_for_mode(ALL_TOOLS, "coupled")}
    assert "agent_send" in names
    assert "agent_poll" in names


def test_coupled_still_excludes_listen_tools():
    names = {t["name"] for t in _filter_tools_for_mode(ALL_TOOLS, "coupled")}
    for n in ALWAYS_EXCLUDED_TOOLS:
        assert n not in names


def test_solo_hides_agent_tools_and_listen_tools():
    names = {t["name"] for t in _filter_tools_for_mode(ALL_TOOLS, "solo")}
    for n in ALWAYS_EXCLUDED_TOOLS | SOLO_EXTRA_EXCLUDED_TOOLS:
        assert n not in names


def test_solo_keeps_everything_else():
    """Solo should still expose movement, vision, memory, mode tools."""
    names = {t["name"] for t in _filter_tools_for_mode(ALL_TOOLS, "solo")}
    for expected in (
        "move", "action", "scan", "snapshot", "caption", "speak",
        "memory_set", "memory_get", "set_mode", "get_mode",
    ):
        assert expected in names


def test_instructions_differ_by_mode():
    assert _instructions_for_mode("solo") == SOLO_INSTRUCTIONS
    assert _instructions_for_mode("coupled") == COUPLED_INSTRUCTIONS
    assert _instructions_for_mode("anything_else") == COUPLED_INSTRUCTIONS  # default
    assert SOLO_INSTRUCTIONS != COUPLED_INSTRUCTIONS
