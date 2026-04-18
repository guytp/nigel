"""OpenAI Realtime voice agent that drives the PiCrawler through our MCP server.

Runs as a second process alongside mcp_picrawler on the Pi. Connects to the
local MCP server over HTTP, relays those tools to an OpenAI Realtime session,
and pipes mic + speaker through the user's desk.
"""

__version__ = "0.1.0"
