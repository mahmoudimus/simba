"""Pipe-separated activity log for tracking tool usage per project.

Ported from claude-turbo-search/scripts/track-activity.sh.
"""

from __future__ import annotations

import contextlib
import datetime
from typing import TYPE_CHECKING

import simba.search.project_memory

if TYPE_CHECKING:
    import pathlib


def _get_log_path(cwd: pathlib.Path) -> pathlib.Path:
    """Return the activity log path for the repository containing *cwd*."""
    root = simba.search.project_memory.find_repo_root(cwd)
    base = root if root is not None else cwd
    return base / ".simba" / "search" / "activity.log"


def log_activity(cwd: pathlib.Path, tool_name: str, detail: str) -> None:
    """Append a timestamped activity entry and rotate if needed."""
    log_path = _get_log_path(cwd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as fh:
        fh.write(f"{timestamp}|{tool_name}|{detail}\n")
    _rotate_log(log_path)


def read_activity_log(
    cwd: pathlib.Path,
) -> list[tuple[str, str, str]]:
    """Parse the activity log into (timestamp, tool_name, detail) tuples.

    Returns an empty list when the log file does not exist.
    Malformed lines are silently skipped.
    """
    log_path = _get_log_path(cwd)
    if not log_path.exists():
        return []
    entries: list[tuple[str, str, str]] = []
    for line in log_path.read_text().splitlines():
        parts = line.split("|", maxsplit=2)
        if len(parts) == 3:
            entries.append((parts[0], parts[1], parts[2]))
    return entries


def clear_activity_log(cwd: pathlib.Path) -> None:
    """Delete the activity log file if it exists."""
    log_path = _get_log_path(cwd)
    with contextlib.suppress(FileNotFoundError):
        log_path.unlink()


def _rotate_log(log_path: pathlib.Path, max_lines: int = 200) -> None:
    """Keep only the last 100 lines when the log exceeds *max_lines*."""
    lines = log_path.read_text().splitlines(keepends=True)
    if len(lines) > max_lines:
        log_path.write_text("".join(lines[-100:]))
