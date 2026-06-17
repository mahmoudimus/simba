"""Preflight brief builder (spec 28) — the body of ``simba preflight <task>``.

Given task text, ``simba preflight`` returns (a) the intent-relevant doctrine
(project-scoped recall — spec 26), (b) the applicable TOOL_RULEs + redirect rules
for the project, and (c) a short "here's the right approach" brief. It also emits
the ``🦁☑`` ledger (spec 27) and sets the per-turn preflight flag so the
PreToolUse gate sees it.

This module holds the PURE assembly (``build_brief`` + the rule/redirect
adapters); the CLI / daemon endpoint wire the recall + lookups + flag side-effect.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

MARKER = "🦁☑"


def tool_rule_lines(memories: Sequence[dict]) -> list[str]:
    """Extract the human-readable warning line from each TOOL_RULE memory."""
    lines: list[str] = []
    for m in memories:
        content = (m.get("content") or "").strip()
        if content:
            lines.append(content)
    return lines


def redirect_lines(rules: Sequence[typing.Any]) -> list[str]:
    """One ``program -> replacement`` (or ``pattern -> rewrite``) line per rule."""
    lines: list[str] = []
    for r in rules:
        program = getattr(r, "program", "") or ""
        replacement = getattr(r, "replacement", "") or ""
        pattern = getattr(r, "pattern", "") or ""
        rewrite = getattr(r, "rewrite", "") or ""
        if program and replacement:
            lines.append(f"{program} -> {replacement}")
        elif pattern:
            lines.append(f"{pattern} -> {rewrite}")
    return lines


def build_brief(
    *,
    task: str,
    doctrine_lines: Sequence[str],
    tool_rules: Sequence[str],
    redirects: Sequence[str],
) -> str:
    """Render the preflight brief: a one-line ``🦁☑`` ledger + the surfaced detail.

    Always emits the marker + the task (so even a zero-match preflight is an
    observable, logged interaction — the grounded ``🦁☑``, not a self-claimed
    emoji). Sections are omitted when empty.
    """
    ledger = (
        f"{MARKER} preflight: {len(doctrine_lines)} doctrine · "
        f"{len(tool_rules)} rule · {len(redirects)} redirect"
    )
    parts: list[str] = [ledger, f"<preflight task={task!r}>"]
    if doctrine_lines:
        parts.append("Doctrine for this task:")
        parts.extend(f"  - {d}" for d in doctrine_lines)
    if tool_rules:
        parts.append("Applicable TOOL_RULEs:")
        parts.extend(f"  - {r}" for r in tool_rules)
    if redirects:
        parts.append("Applicable redirects:")
        parts.extend(f"  - {r}" for r in redirects)
    if not (doctrine_lines or tool_rules or redirects):
        parts.append("  (nothing matched — proceed, but consult as you go)")
    parts.append("</preflight>")
    return "\n".join(parts)
