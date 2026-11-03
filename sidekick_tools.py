"""
Tools for the Sidekick.

This module combines:
- LangChain tools
- Custom tools
- MCP servers

The Sidekick receives all of these tools through get_all_tools().
"""

import asyncio
import os
from contextlib import AsyncExitStack

import requests
import wikipedia

from dotenv import load_dotenv

from langchain_community.tools import (
    GoogleSerperRun,
    WikipediaQueryRun,
)

from langchain_community.utilities import (
    GoogleSerperAPIWrapper,
    WikipediaAPIWrapper,
)

from langchain_core.tools import tool

from langchain_mcp_adapters.client import (
    MultiServerMCPClient,
)

from langchain_mcp_adapters.tools import (
    load_mcp_tools,
)


# ============================================================
# Environment
# ============================================================

load_dotenv(override=True)


# ============================================================
# Wikipedia
# ============================================================

wikipedia.set_user_agent(
    "agentic-track-course (https://edwarddonner.com)"
)


# ============================================================
# Ready-made tools
# ============================================================

search = GoogleSerperRun(
    api_wrapper=GoogleSerperAPIWrapper()
)

wikipedia_lookup = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper()
)


# ============================================================
# Custom push notification tool
# ============================================================

@tool
def send_push_notification(text: str) -> str:
    """Send a short push notification to the user's phone."""

    response = requests.post(
        "https://api.pushover.net/1/messages.json",

        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        },
    )

    response.raise_for_status()

    return "Notification sent"


# ============================================================
# Human help tool
# ============================================================

@tool
def request_human_help(instructions: str) -> str:
    """
    Ask the user to do something in the browser that
    the agent cannot do itself.

    Examples:
    - Login
    - CAPTCHA
    - Two-factor authentication
    """

    return (
        "The user says it is done. "
        "Continue with the task."
    )


# ============================================================
# MCP configuration
# ============================================================

def mcp_connections(sandbox: str) -> dict:
    """
    Define the MCP servers used by Sidekick.
    """

    return {

        # Browser MCP
        "playwright": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "@playwright/mcp@latest",
                "--isolated",
            ],
        },

        # Filesystem MCP
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                sandbox,
            ],
        },
    }


# ============================================================
# Persistent MCP sessions
# ============================================================

class McpSessions:
    """
    Maintain persistent MCP sessions.

    This keeps the browser and filesystem MCP servers alive
    while the Sidekick is working.
    """

    def __init__(self, connections: dict):

        self.connections = connections
        self.tools = []

        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task = None

    async def _run(self):

        client = MultiServerMCPClient(
            self.connections
        )

        async with AsyncExitStack() as stack:

            for name in self.connections:

                session = await stack.enter_async_context(
                    client.session(name)
                )

                self.tools += await load_mcp_tools(
                    session,
                    server_name=name,
                )

            # MCP servers are ready
            self._ready.set()

            # Keep sessions alive
            await self._stop.wait()

    async def start(self) -> list:
        """
        Start MCP servers and wait until they are ready.
        """

        self._task = asyncio.create_task(
            self._run()
        )

        ready = asyncio.create_task(
            self._ready.wait()
        )

        await asyncio.wait(
            [ready, self._task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        ready.cancel()

        if self._task.done():
            self._task.result()

        return self.tools

    def stop(self):
        """Stop MCP sessions."""

        self._stop.set()


# ============================================================
# Collect all tools
# ============================================================

async def get_all_tools(sandbox: str):
    """
    Start MCP servers and return all Sidekick tools.
    """

    sessions = McpSessions(
        mcp_connections(sandbox)
    )

    mcp_tools = await sessions.start()

    our_tools = [
        search,
        send_push_notification,
        wikipedia_lookup,
        request_human_help,
    ]

    return (
        our_tools + mcp_tools,
        sessions,
    )