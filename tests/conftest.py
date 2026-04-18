"""Shared fixtures.

Tests fall into three bands:

- unit (fast, in-process): hardware mock, vision, auth middleware
- integration (spawn subprocess): full MCP server over streamable-http with token
- manual E2E (requires OpenAI key, mic, speaker): not automated — see docs/local-test.md
"""

from __future__ import annotations

import asyncio
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest


REPO_ROOT = Path(__file__).parent.parent
VENV_BIN = REPO_ROOT / ".venv" / "bin"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def free_port() -> int:
    return _free_port()


@pytest.fixture
def mcp_token() -> str:
    return secrets.token_hex(16)


@pytest.fixture
def mcp_server(free_port: int, mcp_token: str):
    """Spawn the MCP server as a subprocess with mock hardware + bearer auth.

    Yields a dict with url, token, port. Terminates on teardown.
    """
    env = {
        **os.environ,
        "MCP_TRANSPORT": "streamable-http",
        "MCP_HOST": "127.0.0.1",
        "MCP_PORT": str(free_port),
        "MCP_TOKEN": mcp_token,
        "PICRAWLER_LOG": "WARNING",  # keep test output clean
    }
    proc = subprocess.Popen(
        [str(VENV_BIN / "mcp-picrawler")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )
    url = f"http://127.0.0.1:{free_port}/mcp"
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            r = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {mcp_token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                content=b'{"jsonrpc":"2.0","id":0,"method":"ping","params":{}}',
                timeout=1.0,
            )
            if r.status_code in (200, 400, 406):  # alive, any valid MCP response
                break
        except httpx.HTTPError:
            pass
        if proc.poll() is not None:
            stdout = proc.stdout.read().decode() if proc.stdout else ""
            pytest.fail(f"MCP server exited early (rc={proc.returncode}):\n{stdout}")
        time.sleep(0.1)
    else:
        proc.terminate()
        proc.wait(timeout=5)
        stdout = proc.stdout.read().decode() if proc.stdout else ""
        pytest.fail(f"MCP server didn't become ready in 15s\n{stdout}")

    yield {"url": url, "token": mcp_token, "port": free_port, "proc": proc}

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
