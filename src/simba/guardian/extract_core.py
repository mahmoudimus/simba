"""Extract content between SIMBA:core markers from project files.

Scans only the intended project rule surfaces for SIMBA:core markers: top-level
CLAUDE.md / AGENTS.md and the configured .claude/rules core file. Used by
UserPromptSubmit to inject essential rules — the compaction-safe layer.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import sys

import simba.markers

_TOP_LEVEL_FILES = ("CLAUDE.md", "AGENTS.md")
_FENCE_RE = re.compile(r"^\s*```")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_SPACE_RE = re.compile(r"\s+")


@dataclasses.dataclass(frozen=True)
class CoreBlock:
    """A CORE block plus where it came from."""

    content: str
    path: pathlib.Path


def extract_core_blocks(content: str) -> list[str]:
    """Extract all content blocks between SIMBA:core markers.

    Markdown examples often show literal SIMBA markers inside fenced code blocks;
    those examples must not become live guardian context.
    """
    return simba.markers.extract_blocks(_strip_fenced_code(content), "core")


def _cfg(cwd: pathlib.Path):
    """Load guardian config using the target project root."""
    import simba.config
    import simba.guardian.config

    _ = simba.guardian.config  # ensure "guardian" section is registered
    return simba.config.load("guardian", root=cwd)


def _strip_fenced_code(content: str) -> str:
    """Drop fenced markdown code blocks before marker extraction."""
    lines: list[str] = []
    in_fence = False
    for line in content.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return "\n".join(lines)


def _candidate_files(cwd: pathlib.Path, core_filename: str) -> list[pathlib.Path]:
    """Return the narrow, ordered set of files guardian is allowed to scan."""
    candidates: list[pathlib.Path] = []
    for name in _TOP_LEVEL_FILES:
        candidates.append(cwd / name)

    configured = pathlib.Path(core_filename)
    if configured.is_absolute():
        candidates.append(configured)
    elif len(configured.parts) > 1:
        candidates.append(cwd / configured)
    else:
        candidates.append(cwd / ".claude" / "rules" / configured)

    seen: set[pathlib.Path] = set()
    unique: list[pathlib.Path] = []
    for path in candidates:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _read_core_blocks(cwd: pathlib.Path, core_filename: str) -> list[CoreBlock]:
    """Read CORE blocks from the allowed guardian source files."""
    blocks: list[CoreBlock] = []
    for md_file in _candidate_files(cwd, core_filename):
        if not md_file.is_file():
            continue
        try:
            content = md_file.read_text()
        except OSError:
            continue
        blocks.extend(
            CoreBlock(content=block, path=md_file)
            for block in extract_core_blocks(content)
        )
    return blocks


def _compact_rule(text: str, max_chars: int) -> str:
    """Return one compact rule line suitable for the CORE capsule."""
    text = _BOLD_RE.sub(r"\1", text).strip()
    text = _SPACE_RE.sub(" ", text)
    for sep in (" — ", ". ", "; "):
        head, found, _tail = text.partition(sep)
        if found and 24 <= len(head) <= max_chars:
            text = head
            break
    if len(text) > max_chars:
        text = text[: max_chars - 3].rsplit(" ", 1)[0].rstrip() + "..."
    return text


def build_capsule(
    blocks: list[CoreBlock],
    *,
    cwd: pathlib.Path,
    max_chars: int,
    rule_chars: int,
) -> str:
    """Compile full CORE blocks into a short deterministic guardian capsule."""
    if not blocks:
        return ""

    rules: list[str] = []
    signal = ""
    seen: set[str] = set()
    for block in blocks:
        for raw in block.content.splitlines():
            line = raw.strip()
            if not line:
                continue
            if "[✓ rules]" in line:
                signal = "Signal: end every response with `[✓ rules]`."
                continue
            if not line.startswith("- "):
                continue
            compacted = _compact_rule(line[2:], rule_chars)
            key = compacted.lower()
            if compacted and key not in seen:
                seen.add(key)
                rules.append(compacted)

    if not rules and not signal:
        return ""

    source_paths = sorted(
        {
            str(block.path.relative_to(cwd))
            if block.path.is_relative_to(cwd)
            else str(block.path)
            for block in blocks
        }
    )
    lines = [
        "<simba-core-capsule>",
        "Project CORE rules are active; read the source file if any detail is unclear.",
        f"Source: {', '.join(source_paths)}",
    ]
    for rule in rules:
        lines.append(f"- {rule}")
    if signal:
        lines.append(signal)
    lines.append("</simba-core-capsule>")

    capsule = "\n".join(lines)
    if len(capsule) <= max_chars:
        return capsule

    # Preserve source + signal; trim least-specific tail rules to fit the budget.
    footer = [signal] if signal else []
    footer.append("</simba-core-capsule>")
    prefix = lines[:3]
    budget = max_chars - len("\n".join(prefix + footer)) - 1
    kept: list[str] = []
    used = 0
    for rule in rules:
        rendered = f"- {rule}"
        cost = len(rendered) + 1
        if used + cost > budget:
            break
        kept.append(rendered)
        used += cost
    return "\n".join(prefix + kept + footer)


def main(cwd: pathlib.Path | None = None) -> str:
    """Scan project files and return concatenated CORE blocks."""
    if cwd is None:
        cwd = pathlib.Path.cwd()
    cfg = _cfg(cwd)
    blocks = _read_core_blocks(cwd, cfg.core_filename)
    if getattr(cfg, "core_injection_mode", "full") == "capsule":
        return build_capsule(
            blocks,
            cwd=cwd,
            max_chars=max(200, getattr(cfg, "core_capsule_max_chars", 1200)),
            rule_chars=max(40, getattr(cfg, "core_capsule_rule_chars", 140)),
        )

    return "\n".join(block.content for block in blocks)


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
