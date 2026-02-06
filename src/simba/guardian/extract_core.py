"""Extract content between <!-- CORE --> tags from CLAUDE.md.

Used by UserPromptSubmit hook to inject essential rules every prompt.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

_CORE_PATTERN = re.compile(
    r"<!--\s*CORE\s*-->\n(.*?)<!--\s*/CORE\s*-->",
    re.DOTALL,
)


def extract_core_blocks(content: str) -> list[str]:
    """Extract all content blocks between <!-- CORE --> and <!-- /CORE --> tags."""
    return _CORE_PATTERN.findall(content)


def main(cwd: pathlib.Path | None = None) -> str:
    """Read CLAUDE.md and return concatenated CORE blocks."""
    if cwd is None:
        cwd = pathlib.Path.cwd()
    claude_md = cwd / "CLAUDE.md"
    if not claude_md.exists():
        return ""
    content = claude_md.read_text()
    blocks = extract_core_blocks(content)
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
