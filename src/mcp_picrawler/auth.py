"""Bearer-token auth for the HTTP transport.

Stdio sessions are local-process only and bypass this. For the streamable-http
transport, if MCP_TOKEN is set, requests must carry `Authorization: Bearer <token>`.

This is not a replacement for putting the server behind Tailscale / a LAN
firewall — it's a second layer so an accidentally exposed port doesn't hand
the whole robot to anyone who finds it.
"""

from __future__ import annotations

import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse


class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(self, request, call_next):
        header = request.headers.get("authorization", "")
        prefix = "Bearer "
        if not header.startswith(prefix):
            return PlainTextResponse("missing bearer token", status_code=401)
        presented = header[len(prefix):].strip()
        if not hmac.compare_digest(presented, self._token):
            return PlainTextResponse("invalid bearer token", status_code=401)
        return await call_next(request)
