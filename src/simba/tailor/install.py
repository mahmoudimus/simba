"""Project installer for tailor-nano.

Ported from claude-tailor/src/install.js. Creates dirs, copies hooks,
registers in settings.local.json, handles CLAUDE.md merge/overwrite/skip.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import stat
import sys

# Template content for CLAUDE.md
CLAUDE_MD_TEMPLATE = """\
# tailor-nano Patterns

Automatic error memory via `.claude-tailor/memory/reflections.jsonl`.

Use `/mark <name>` to save progress checkpoints.
Use `/recall <term>` to search error history.
"""

# Source file content for the hook
HOOK_PY_CONTENT = """\
#!/usr/bin/env python3
\"\"\"tailor-nano error capture hook.\"\"\"
import sys
from simba.tailor.hook import process_hook

if __name__ == "__main__":
    process_hook(sys.stdin.read())
"""

SESSION_START_PY_CONTENT = """\
#!/usr/bin/env python3
\"\"\"tailor-nano session start hook.\"\"\"
from simba.tailor.session_start import main

print(main())
"""


@dataclasses.dataclass
class InstallOptions:
    dry_run: bool = False
    force: bool = False
    claude_md: str | None = None


def parse_args(args: list[str]) -> InstallOptions:
    """Parse command-line arguments."""
    opts = InstallOptions()
    for arg in args:
        if arg in ("--dry-run", "-n"):
            opts.dry_run = True
        elif arg in ("--force", "-f"):
            opts.force = True
        elif arg.startswith("--claude-md="):
            value = arg.split("=", 1)[1]
            if value in ("skip", "overwrite", "merge"):
                opts.claude_md = value
    return opts


def install(
    cwd: pathlib.Path | None = None,
    dry_run: bool = False,
    force: bool = False,
    claude_md_action: str | None = None,
) -> list[str]:
    """Install tailor-nano into the given directory.

    Returns a list of log messages.
    """
    if cwd is None:
        cwd = pathlib.Path.cwd()
    logs: list[str] = []

    if dry_run:
        logs.append("[DRY RUN] Would install tailor-nano...")
        mem = cwd / ".claude-tailor" / "memory"
        cmd = cwd / ".claude" / "commands"
        logs.append(f"[DRY RUN] Would create directory: {mem}")
        logs.append(f"[DRY RUN] Would create directory: {cmd}")
        return logs

    logs.append("Installing tailor-nano...")

    # 1. Create directories
    memory_dir = cwd / ".claude-tailor" / "memory"
    commands_dir = cwd / ".claude" / "commands"
    memory_dir.mkdir(parents=True, exist_ok=True)
    commands_dir.mkdir(parents=True, exist_ok=True)

    # 2. Copy hook files
    hook_path = cwd / ".claude-tailor" / "hook.py"
    session_start_path = cwd / ".claude-tailor" / "session-start.py"

    if force or not hook_path.exists():
        hook_path.write_text(HOOK_PY_CONTENT)
        hook_path.chmod(hook_path.stat().st_mode | stat.S_IEXEC)
    else:
        logs.append("hook.py already exists (use --force to overwrite)")

    if force or not session_start_path.exists():
        session_start_path.write_text(SESSION_START_PY_CONTENT)
        session_start_path.chmod(session_start_path.stat().st_mode | stat.S_IEXEC)
    else:
        logs.append("session-start.py already exists (use --force to overwrite)")

    # 3. Handle CLAUDE.md
    claude_md_path = cwd / "CLAUDE.md"
    if claude_md_path.exists():
        action = claude_md_action or "merge"
        if action == "skip":
            logs.append("Skipped CLAUDE.md (already exists)")
        elif action == "overwrite":
            claude_md_path.write_text(CLAUDE_MD_TEMPLATE)
            logs.append("Overwrote CLAUDE.md")
        else:  # merge
            existing = claude_md_path.read_text()
            if "tailor-nano" not in existing:
                claude_md_path.write_text(existing + "\n\n" + CLAUDE_MD_TEMPLATE)
                logs.append("Appended patterns to existing CLAUDE.md")
            else:
                logs.append("CLAUDE.md already contains tailor-nano patterns")
    else:
        claude_md_path.write_text(CLAUDE_MD_TEMPLATE)
        logs.append("Created CLAUDE.md")

    # 4. Register hook in settings.local.json
    settings_path = cwd / ".claude" / "settings.local.json"
    settings: dict = {"hooks": {}}
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())

    hooks = settings.setdefault("hooks", {})
    session_start_hooks = hooks.get("SessionStart", [])

    session_start_exists = any(
        any(
            "session-start" in h.get("command", "")
            or "session_start" in h.get("command", "")
            for h in entry.get("hooks", [])
        )
        for entry in session_start_hooks
    )

    if not session_start_exists:
        hooks.setdefault("SessionStart", []).append(
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": "python -m simba.tailor.session_start",
                    }
                ]
            }
        )
        settings_path.write_text(json.dumps(settings, indent=2))
        logs.append("Registered SessionStart hook in settings.local.json")
    elif not settings_path.exists():
        settings_path.write_text(json.dumps(settings, indent=2))

    # Always write settings if it doesn't exist yet
    if not settings_path.exists():
        settings_path.write_text(json.dumps(settings, indent=2))

    logs.append("Installation complete!")
    return logs


if __name__ == "__main__":
    opts = parse_args(sys.argv[1:])
    messages = install(
        dry_run=opts.dry_run, force=opts.force, claude_md_action=opts.claude_md
    )
    for msg in messages:
        print(msg)
