"""Check Claude's response for [✓ rules] signal marker.

If missing, outputs CLAUDE.md with self-audit instructions.
"""

from __future__ import annotations

import json
import pathlib
import sys

_SIGNAL = "[✓ rules]"


def check_signal(response: str) -> bool:
    """Return True if the response contains the rules signal marker."""
    return _SIGNAL in response


def main(response: str, cwd: pathlib.Path | None = None) -> str:
    """Check response for signal. Return warning + CLAUDE.md if missing."""
    if cwd is None:
        cwd = pathlib.Path.cwd()

    if check_signal(response):
        return ""

    claude_md = cwd / "CLAUDE.md"
    if not claude_md.exists():
        return ""

    claude_content = claude_md.read_text()
    lines = [
        "\u26a0\ufe0f MEMORY ALERT: Signal marker missing from your last response.",
        "",
        "Required actions:",
        "1. Re-read the project rules below",
        "2. Review your previous response for any rule violations",
        "3. Report any issues you find and correct them",
        "4. Resume with [✓ rules] on all future responses",
        "",
        "=== CLAUDE.md ===",
        claude_content,
        "=== END CLAUDE.md ===",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    response = sys.stdin.read()
    result = main(response=response)
    if result:
        print(result)
    output = {
        "hookSpecificOutput": {
            "hookEventName": "Stop",
            "additionalContext": result,
        }
    }
    json.dump(output, sys.stdout)
