"""Configuration, enums, and constants for the Neuron MCP server."""

from __future__ import annotations

import logging
import shutil
import sys
import time
from dataclasses import dataclass
from enum import IntEnum


class Status(IntEnum):
    """Agent run status (maps to status_types table)."""

    PENDING = 1
    STARTED = 2
    RUNNING = 3
    COMPLETED = 4
    FAILED = 5


class LogLevel(IntEnum):
    """Log levels (maps to log_levels table)."""

    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4


LOGGING_LEVEL_MAP = {
    logging.DEBUG: LogLevel.DEBUG,
    logging.INFO: LogLevel.INFO,
    logging.WARNING: LogLevel.WARNING,
    logging.ERROR: LogLevel.ERROR,
    logging.CRITICAL: LogLevel.ERROR,
}

STATUS_NAME_MAP = {
    "pending": Status.PENDING,
    "started": Status.STARTED,
    "running": Status.RUNNING,
    "completed": Status.COMPLETED,
    "failed": Status.FAILED,
}


@dataclass
class ServerConfig:
    """Runtime configuration for the Neuron MCP Server."""

    python_cmd: str
    souffle_cmd: str | None


CONFIG = ServerConfig(
    python_cmd=sys.executable,
    souffle_cmd=shutil.which("souffle"),
)


def utc_now() -> int:
    """Return current UTC time as Unix timestamp."""
    return int(time.time())
