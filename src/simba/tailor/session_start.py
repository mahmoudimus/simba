"""SessionStart hook — inject concise session context.

Ported from claude-tailor/src/session-start.sh.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time


def gather_git_status(cwd: pathlib.Path | None = None) -> str:
    """Gather git branch and status info."""
    if cwd is None:
        cwd = pathlib.Path.cwd()

    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=cwd,
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no repo"

    # Get branch
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        branch = result.stdout.strip() or "detached"
    except (subprocess.CalledProcessError, FileNotFoundError):
        branch = "detached"

    # Check dirty
    try:
        subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=cwd,
            capture_output=True,
            check=True,
        )
        dirty = "clean"
    except subprocess.CalledProcessError:
        dirty = "dirty"
    except FileNotFoundError:
        dirty = "unknown"

    # Commits ahead
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", "@{upstream}..HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        ahead = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        ahead = "0"

    git_status = f"{branch} ({dirty})"
    if ahead != "0":
        git_status += f" \u2191{ahead}"

    return git_status


def format_time_ago(seconds: int) -> str:
    """Format seconds into a human-readable 'ago' string."""
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    elif seconds < 86400:
        return f"{seconds // 3600}h ago"
    else:
        return f"{seconds // 86400}d ago"


def gather_checkpoints(cwd: pathlib.Path | None = None) -> list[tuple[str, str]]:
    """Find recent progress checkpoint files and return (name, time_ago) pairs."""
    if cwd is None:
        cwd = pathlib.Path.cwd()

    memory_dir = cwd / ".claude-tailor" / "memory"
    if not memory_dir.is_dir():
        return []

    progress_files = sorted(
        memory_dir.glob("progress-*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:3]

    marks: list[tuple[str, str]] = []
    now = time.time()
    for f in progress_files:
        name = f.stem.removeprefix("progress-")
        mod_time = f.stat().st_mtime
        diff = int(now - mod_time)
        marks.append((name, format_time_ago(diff)))

    return marks


def gather_context(cwd: pathlib.Path | None = None) -> str:
    """Build the full context string."""
    if cwd is None:
        cwd = pathlib.Path.cwd()

    time_str = time.strftime("%H:%M %Z (%a)")
    git_status = gather_git_status(cwd=cwd)

    context = f"Time: {time_str} | Git: {git_status}"

    memory_dir = cwd / ".claude-tailor" / "memory"
    if memory_dir.is_dir():
        marks = gather_checkpoints(cwd=cwd)
        if marks:
            marks_str = " ".join(f"{name}({ago})" for name, ago in marks)
            context += f" | Marks: {marks_str}"
        else:
            context += " | Marks: none (use /mark <name> to save)"

    return context


def main(cwd: pathlib.Path | None = None) -> str:
    """Generate session start hook JSON output."""
    if cwd is None:
        cwd = pathlib.Path.cwd()

    context = gather_context(cwd=cwd)

    output = {
        "systemMessage": f"\U0001f4cd {context}",
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        },
    }
    return json.dumps(output)


if __name__ == "__main__":
    # Read hook input from stdin (may contain cwd)
    cwd = None
    try:
        input_str = sys.stdin.read()
        if input_str:
            data = json.loads(input_str)
            if "cwd" in data:
                cwd = pathlib.Path(data["cwd"])
    except (json.JSONDecodeError, KeyError):
        pass

    print(main(cwd=cwd))
