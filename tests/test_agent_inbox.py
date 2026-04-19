"""Unit tests for the AgentInbox message bus."""

from __future__ import annotations

import pytest

from mcp_picrawler.agent_inbox import AgentInbox


def test_send_returns_message_with_incrementing_id():
    inbox = AgentInbox()
    m1 = inbox.send(from_="claude", to="nigel", message="hello")
    m2 = inbox.send(from_="claude", to="nigel", message="are you there")
    assert m1.id == 1
    assert m2.id == 2
    assert m2.ts >= m1.ts


def test_poll_returns_all_messages_for_recipient():
    inbox = AgentInbox()
    inbox.send(from_="claude", to="nigel", message="a")
    inbox.send(from_="claude", to="nigel", message="b")
    inbox.send(from_="nigel", to="claude", message="c")
    nigels = inbox.poll("nigel")
    assert [m.message for m in nigels] == ["a", "b"]
    claudes = inbox.poll("claude")
    assert [m.message for m in claudes] == ["c"]


def test_poll_since_id_filters_older():
    inbox = AgentInbox()
    m1 = inbox.send(from_="claude", to="nigel", message="a")
    inbox.send(from_="claude", to="nigel", message="b")
    inbox.send(from_="claude", to="nigel", message="c")
    remaining = inbox.poll("nigel", since_id=m1.id)
    assert [m.message for m in remaining] == ["b", "c"]


def test_poll_empty_recipient_returns_empty():
    inbox = AgentInbox()
    assert inbox.poll("nobody") == []


def test_send_rejects_empty_recipient():
    inbox = AgentInbox()
    with pytest.raises(ValueError):
        inbox.send(from_="claude", to="", message="x")


def test_send_rejects_empty_message():
    inbox = AgentInbox()
    with pytest.raises(ValueError):
        inbox.send(from_="claude", to="nigel", message="")


def test_to_dict_converts_from_keyword():
    """`from` is reserved in JSON-ish consumers, so AgentMessage serialises
    with a plain `from` key (not `from_`)."""
    inbox = AgentInbox()
    m = inbox.send(from_="claude", to="nigel", message="hi")
    d = m.to_dict()
    assert "from" in d
    assert "from_" not in d
    assert d["from"] == "claude"
    assert d["to"] == "nigel"
    assert d["message"] == "hi"


def test_queues_cap_at_max_per_agent():
    inbox = AgentInbox(max_per_agent=5)
    for i in range(10):
        inbox.send(from_="a", to="b", message=str(i))
    got = [m.message for m in inbox.poll("b")]
    assert got == ["5", "6", "7", "8", "9"]  # last 5 kept


def test_known_recipients_lists_all_queues():
    inbox = AgentInbox()
    inbox.send(from_="x", to="a", message="1")
    inbox.send(from_="y", to="b", message="2")
    assert inbox.known_recipients() == ["a", "b"]
