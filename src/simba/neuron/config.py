"""Configuration for the Neuron MCP server."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Runtime configuration for the Neuron MCP Server."""

    python_cmd: str
    souffle_cmd: str | None


CONFIG = ServerConfig(
    python_cmd=sys.executable,
    souffle_cmd=shutil.which("souffle"),
)
