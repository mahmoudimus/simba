"""Shared SIMBA marker utilities for structured content blocks.

Markers follow the format::

    <!-- BEGIN SIMBA:{name} -->
    ...content...
    <!-- END SIMBA:{name} -->

All functions are string-based (no file I/O). Callers handle persistence.
"""

from __future__ import annotations

import re

NAMESPACE = "SIMBA"


def begin_tag(name: str) -> str:
    """Return the opening marker string for a named block."""
    return f"<!-- BEGIN {NAMESPACE}:{name} -->"


def end_tag(name: str) -> str:
    """Return the closing marker string for a named block."""
    return f"<!-- END {NAMESPACE}:{name} -->"


def _block_pattern(name: str) -> re.Pattern[str]:
    """Compile the regex for a named SIMBA block."""
    escaped = re.escape(name)
    return re.compile(
        rf"<!--\s*BEGIN\s+{NAMESPACE}:{escaped}\s*-->\n?"
        rf"(.*?)"
        rf"<!--\s*END\s+{NAMESPACE}:{escaped}\s*-->",
        re.DOTALL,
    )


def extract_blocks(content: str, name: str) -> list[str]:
    """Extract all content blocks between BEGIN/END SIMBA markers.

    Returns a list to support multiple blocks with the same name
    (e.g. multiple ``SIMBA:core`` blocks in one CLAUDE.md).
    """
    return _block_pattern(name).findall(content)


def update_blocks(content: str, updates: dict[str, str]) -> str:
    """Replace content between existing SIMBA markers for the given sections.

    Only touches sections whose names are keys in *updates*.
    Other SIMBA blocks (including ``core``) are left untouched.
    """
    for name, new_content in updates.items():
        pattern = re.compile(
            rf"(<!--\s*BEGIN\s+{NAMESPACE}:{re.escape(name)}\s*-->)"
            rf".*?"
            rf"(<!--\s*END\s+{NAMESPACE}:{re.escape(name)}\s*-->)",
            re.DOTALL,
        )
        content = pattern.sub(rf"\1\n{new_content}\2", content)
    return content


def has_marker(content: str, name: str) -> bool:
    """Return True if the content contains a BEGIN SIMBA marker for *name*."""
    return begin_tag(name) in content


def make_empty_block(name: str) -> str:
    """Return an empty marker pair ready for injection."""
    return f"{begin_tag(name)}\n{end_tag(name)}"
