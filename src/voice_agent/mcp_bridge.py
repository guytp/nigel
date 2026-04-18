"""Thin MCP client that exposes the crawler's tools to the Realtime agent.

Translates MCP tool schemas into the OpenAI function-tool shape and relays
tool calls. Image content in tool results is summarised as text — the voice
agent doesn't need to see images; if Claude Code wants them it hits the MCP
server directly.
"""

from __future__ import annotations

import logging
from types import TracebackType

from mcp import ClientSession
# Using the legacy name that accepts `headers=` directly. The newer
# streamable_http_client takes a pre-built httpx.AsyncClient; revisit when we
# can justify plumbing that through.
from mcp.client.streamable_http import streamablehttp_client

log = logging.getLogger(__name__)


class CrawlerMCPBridge:
    def __init__(self, url: str, token: str | None = None) -> None:
        self._url = url
        self._token = token
        self._transport_cm = None
        self._session_cm = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "CrawlerMCPBridge":
        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        self._transport_cm = streamablehttp_client(self._url, headers=headers)
        read, write, _ = await self._transport_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        log.info("MCP bridge connected: %s", self._url)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(exc_type, exc, tb)
        if self._transport_cm is not None:
            await self._transport_cm.__aexit__(exc_type, exc, tb)
        self._session = None

    async def openai_tool_defs(self) -> list[dict]:
        assert self._session is not None
        res = await self._session.list_tools()
        defs: list[dict] = []
        for t in res.tools:
            schema = t.inputSchema or {"type": "object", "properties": {}}
            defs.append(
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": schema,
                }
            )
        return defs

    async def call_tool(self, name: str, arguments: dict) -> str:
        assert self._session is not None
        result = await self._session.call_tool(name, arguments or {})
        parts: list[str] = []
        for c in result.content:
            if hasattr(c, "text") and c.text is not None:
                parts.append(c.text)
            elif getattr(c, "type", None) == "image":
                mime = getattr(c, "mimeType", "image/*")
                data_len = len(getattr(c, "data", "") or "")
                parts.append(f"[image returned: {mime}, {data_len} bytes base64]")
            else:
                parts.append(repr(c))
        return "\n".join(parts) if parts else "(no content)"
