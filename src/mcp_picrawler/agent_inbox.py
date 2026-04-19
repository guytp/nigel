"""In-process message bus for agent-to-agent comms through the MCP server.

Both brains talking to Nigel (Claude via Claude Code, gpt-realtime via the
voice agent) connect to the same MCP server and can exchange text via a
shared inbox. Nigel's MCP server is the rendezvous point.

Design is deliberately minimal:
 - Each agent has a string name (convention: "claude", "nigel")
 - Messages are monotonic-id'd, per-recipient queues with bounded length
 - Polling is authoritative (no push/subscribe yet) — agents call
   `agent_poll(as_who=..., since_id=...)` periodically
 - No auth on the `from` field — anyone can spoof, fine inside the same host

Future: MCP resources/subscribe could let us push instead of poll. For now
the voice agent polls on a background task, and Claude polls at the start
of each user turn.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass


@dataclass
class AgentMessage:
    id: int
    from_: str
    to: str
    message: str
    ts: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["from"] = d.pop("from_")  # "from" is a reserved word at top level
        return d


class AgentInbox:
    """Thread-safe per-recipient message queue."""

    def __init__(self, max_per_agent: int = 200) -> None:
        self._queues: dict[str, deque[AgentMessage]] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        self._max_per_agent = max_per_agent

    def send(self, *, from_: str, to: str, message: str) -> AgentMessage:
        if not to:
            raise ValueError("to must be non-empty")
        if not message:
            raise ValueError("message must be non-empty")
        with self._lock:
            self._next_id += 1
            msg = AgentMessage(
                id=self._next_id,
                from_=from_ or "unknown",
                to=to,
                message=message,
                ts=time.time(),
            )
            self._queues.setdefault(to, deque(maxlen=self._max_per_agent)).append(msg)
            return msg

    def poll(self, who: str, since_id: int = 0) -> list[AgentMessage]:
        """Return all messages for `who` with id > `since_id`, oldest first."""
        with self._lock:
            q = self._queues.get(who, deque())
            return [m for m in q if m.id > since_id]

    def known_recipients(self) -> list[str]:
        with self._lock:
            return sorted(self._queues.keys())
