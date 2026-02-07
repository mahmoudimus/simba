"""FastMCP server — registers orchestration MCP tools and runs the server."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from pathlib import Path

import simba.orchestration.agents
import simba.orchestration.proxy

mcp = FastMCP("Orchestration")


# --- Agent tools ---


@mcp.tool()
def dispatch_agent(agent_name: str, ticket_id: str, instructions: str) -> str:
    """Launches an async subagent. Non-blocking, returns immediately.

    Check status via agent_status_check().
    """
    return simba.orchestration.agents.dispatch_agent(
        agent_name, ticket_id, instructions
    )


@mcp.tool()
def agent_status_update(ticket_id: str, status: str, message: str = "") -> str:
    """Updates the status of an async agent task.

    Called by subagents to report progress.
    """
    return simba.orchestration.agents.agent_status_update(ticket_id, status, message)


@mcp.tool()
def agent_status_check(ticket_id: str | None = None) -> str:
    """Check the status of async agent tasks.

    Auto-detects completion via process state.
    """
    return simba.orchestration.agents.agent_status_check(ticket_id)


# --- Proxy tools ---


@mcp.tool()
def reload_server() -> str:
    """Hot-reloads the MCP server backend (requires proxy mode).

    Sends SIGHUP to the proxy process, which restarts the backend
    while maintaining the connection to Claude Code.
    """
    return simba.orchestration.proxy.reload_server()


# --- Server entry point ---


def run_server(root_dir: Path | None = None) -> None:
    """Configure and start the FastMCP server."""
    if root_dir is not None:
        resolved = root_dir.resolve()
        os.chdir(resolved)

    # Register SIGCHLD handler for zombie reaping
    simba.orchestration.agents.register_sigchld_handler()

    mcp.run()
