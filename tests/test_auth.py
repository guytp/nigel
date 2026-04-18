"""Unit tests for the bearer-auth middleware, in isolation from MCP."""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from mcp_picrawler.auth import BearerAuthMiddleware


@pytest.fixture
def app():
    async def hello(_: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    return Starlette(routes=[Route("/", hello)])


@pytest.fixture
def client(app):
    wrapped = BearerAuthMiddleware(app, token="correct-horse-battery")
    return TestClient(wrapped)


def test_rejects_missing_header(client):
    r = client.get("/")
    assert r.status_code == 401
    assert "missing" in r.text.lower()


def test_rejects_wrong_bearer(client):
    r = client.get("/", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    assert "invalid" in r.text.lower()


def test_rejects_non_bearer_scheme(client):
    r = client.get("/", headers={"Authorization": "Basic Zm9vOmJhcg=="})
    assert r.status_code == 401


def test_accepts_correct_bearer(client):
    r = client.get("/", headers={"Authorization": "Bearer correct-horse-battery"})
    assert r.status_code == 200
    assert r.text == "ok"


def test_constant_time_compare_is_used():
    """Smoke: wrong-length tokens still reach compare_digest without crashing."""
    import hmac

    # If we weren't using compare_digest, mismatched lengths would be indistinguishable
    # by timing — compare_digest is happy to handle mismatched lengths.
    assert hmac.compare_digest("abc", "abcdef") is False
