"""Extract content between SIMBA:core markers from project files.

Scans CLAUDE.md, AGENTS.md, and .claude/**/*.md (excluding handoffs/notes)
for SIMBA:core markers. Used by UserPromptSubmit hook to inject essential
rules on every prompt — the compaction-safe layer.
"""

from __future__ import annotations

import json
import pathlib
import sys

import simba.markers
import simba.markers_cli


def extract_core_blocks(content: str) -> list[str]:
    """Extract all content blocks between SIMBA:core markers."""
    return simba.markers.extract_blocks(content, "core")


def main(cwd: pathlib.Path | None = None) -> str:
    """Scan project files and return concatenated CORE blocks."""
    if cwd is None:
        cwd = pathlib.Path.cwd()

    blocks: list[str] = []
    for md_file in simba.markers_cli._collect_project_files(cwd):
        try:
            content = md_file.read_text()
        except OSError:
            continue
        blocks.extend(extract_core_blocks(content))

    return "\n".join(blocks)


if __name__ == "__main__":
    hook_input = sys.stdin.read()
    cwd = None
    if hook_input:
        try:
            data = json.loads(hook_input)
            if "cwd" in data:
                cwd = pathlib.Path(data["cwd"])
        except (json.JSONDecodeError, KeyError):
            pass

    result = main(cwd=cwd)
    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": result,
        }
    }
    json.dump(output, sys.stdout)
