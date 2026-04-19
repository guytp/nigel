"""Unit tests for voice_agent mode-switching helpers."""

from __future__ import annotations

from voice_agent.agent import (
    ALWAYS_EXCLUDED_TOOLS,
    CHIPPY_BAMBINO_INSTRUCTIONS,
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
    assert _instructions_for_mode("chippy_bambino") == CHIPPY_BAMBINO_INSTRUCTIONS
    assert _instructions_for_mode("anything_else") == COUPLED_INSTRUCTIONS  # default
    assert SOLO_INSTRUCTIONS != COUPLED_INSTRUCTIONS
    assert CHIPPY_BAMBINO_INSTRUCTIONS not in (SOLO_INSTRUCTIONS, COUPLED_INSTRUCTIONS)


def test_coupled_and_solo_prompts_mention_chippy_trigger():
    """Both default modes must tell the model the escape-hatch phrase
    so it can recognise it and call set_mode."""
    assert "chippy bambino" in COUPLED_INSTRUCTIONS.lower()
    assert "chippy bambino" in SOLO_INSTRUCTIONS.lower()


def test_chippy_prompt_lists_exit_phrases():
    prompt = CHIPPY_BAMBINO_INSTRUCTIONS.lower()
    for exit_phrase in ("exit chippy bambino", "end chippy", "chippy done", "normal mode"):
        assert exit_phrase in prompt


def test_chippy_mode_exposes_full_tool_surface():
    """Developer mode gets everything the other modes would, at minimum."""
    coupled_names = {t["name"] for t in _filter_tools_for_mode(ALL_TOOLS, "coupled")}
    chippy_names = {t["name"] for t in _filter_tools_for_mode(ALL_TOOLS, "chippy_bambino")}
    assert coupled_names <= chippy_names
