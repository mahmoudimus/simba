"""Token economics dashboard and project statistics.

Gathers data from the activity log, project memory, and codebase size
to display a summary of tool usage and estimated token savings.
"""

from __future__ import annotations

import contextlib
import subprocess
from typing import TYPE_CHECKING

import simba.search.activity_tracker
import simba.search.project_memory

if TYPE_CHECKING:
    import pathlib

_TOKENS_PER_SEARCH = 50
_TOKENS_PER_FILE_READ = 1000
_BLIND_EXPLORATION_FILES = 20


def _count_activities(
    entries: list[tuple[str, str, str]],
) -> dict[str, int]:
    """Tally activity entries by tool name."""
    counts: dict[str, int] = {}
    for _ts, tool, _detail in entries:
        counts[tool] = counts.get(tool, 0) + 1
    return counts


def _codebase_size(cwd: pathlib.Path) -> tuple[int, int]:
    """Return (file_count, line_count) using ripgrep.

    Returns ``(0, 0)`` when ``rg`` is unavailable.
    """
    try:
        files_result = subprocess.run(
            ["rg", "--files"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(cwd),
        )
        stdout = files_result.stdout.strip()
        lines = stdout.splitlines() if stdout else []
        file_count = len(lines)

        count_result = subprocess.run(
            ["rg", "-c", ""],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(cwd),
        )
        line_count = 0
        for entry in count_result.stdout.strip().splitlines():
            # rg -c outputs "path:count"
            parts = entry.rsplit(":", 1)
            if len(parts) == 2:
                with contextlib.suppress(ValueError):
                    line_count += int(parts[1])
        return file_count, line_count
    except (subprocess.SubprocessError, OSError):
        return 0, 0


def run_stats(cwd: pathlib.Path) -> str:
    """Build and return the stats dashboard string."""
    project_name = cwd.name
    sections: list[str] = []

    # -- Codebase --
    file_count, line_count = _codebase_size(cwd)
    est_tokens = int(line_count * 4 / 3) if line_count else 0
    sections.append(
        f"Codebase: {project_name}\n"
        f"  Files: {file_count}  Lines: {line_count}  Est. tokens: {est_tokens:,}"
    )

    # -- Activity (this session / recent) --
    entries = simba.search.activity_tracker.read_activity_log(cwd)
    counts = _count_activities(entries)
    searches = counts.get("search", 0) + counts.get("grep", 0)
    reads = counts.get("read", 0) + counts.get("Read", 0)
    edits = counts.get("edit", 0) + counts.get("Edit", 0)
    sections.append(
        f"Activity ({len(entries)} logged events)\n"
        f"  Searches: {searches}  Reads: {reads}  Edits: {edits}"
    )

    # -- Token economics --
    with_plugin = (searches * _TOKENS_PER_SEARCH) + (reads * _TOKENS_PER_FILE_READ)
    without_plugin = _BLIND_EXPLORATION_FILES * _TOKENS_PER_FILE_READ
    if without_plugin > 0 and with_plugin < without_plugin:
        savings_pct = int((1 - with_plugin / without_plugin) * 100)
    else:
        savings_pct = 0
    sections.append(
        f"Token economics\n"
        f"  With plugin:    ~{with_plugin:,} tokens\n"
        f"  Without plugin: ~{without_plugin:,} tokens (blind exploration)\n"
        f"  Savings:        ~{savings_pct}%"
    )

    # -- Project memory --
    conn = simba.search.project_memory.get_connection(cwd)
    if conn is not None:
        try:
            pm_stats = simba.search.project_memory.get_stats(conn)
            sections.append(
                f"Project memory\n"
                f"  Sessions: {pm_stats['sessions']}  "
                f"Knowledge: {pm_stats['knowledge']}  "
                f"Facts: {pm_stats['facts']}"
            )
        finally:
            conn.close()
    else:
        sections.append("Project memory: not initialized (run `simba search init`)")

    return "\n\n".join(sections)
