"""CLI for managing tool rules (TOOL_RULE memories).

Usage:
    simba rule add --tool TOOL --correction TEXT [--pattern PAT] [--project DIR]
    simba rule list [--tool TOOL] [--project DIR] [--all]
    simba rule remove <rule_id>
    simba rule prune [--older-than D] [--project P] [--all-projects] [--dry-run]
"""

from __future__ import annotations

import argparse
import calendar
import json
import pathlib
import re
import sys
import time

import httpx

import simba.config
import simba.db
import simba.hooks.config

_DURATION_RE = re.compile(r"^(\d+)([smhdw])$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def _parse_duration_seconds(raw: str) -> int | None:
    """Parse 30m / 48h / 14d / 2w into seconds; None if malformed."""
    m = _DURATION_RE.match(raw.strip())
    if not m:
        return None
    return int(m.group(1)) * _UNIT_SECONDS[m.group(2)]


def _age_seconds(created_at: str) -> float | None:
    """Age of an ISO-8601 ``createdAt`` in seconds, or None if unparseable."""
    if not created_at:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return time.time() - calendar.timegm(time.strptime(created_at, fmt))
        except ValueError:
            continue
    return None


def _daemon_url() -> str:
    _ = simba.hooks.config  # register hooks section
    cfg = simba.config.load("hooks")
    return f"http://{cfg.daemon_host}:{cfg.daemon_port}"


def _cmd_add(args: argparse.Namespace) -> int:
    """Store a new TOOL_RULE memory."""
    content = f"{args.tool}: {args.correction}"[:200]
    context_data = {
        "tool": args.tool,
        "pattern": args.pattern or "",
        "error_source": "",
        "correction": args.correction,
    }
    payload: dict = {
        "type": "TOOL_RULE",
        "content": content,
        "context": json.dumps(context_data),
        "tags": [args.tool],
        "confidence": 0.95,
    }
    if args.project:
        # Canonicalize to the same opaque id the learner stores and the matcher
        # recalls by, so a manually-added rule is actually reachable at runtime.
        payload["projectPath"] = simba.db.resolve_project_id(
            pathlib.Path(args.project)
        )

    try:
        resp = httpx.post(
            f"{_daemon_url()}/store", json=payload, timeout=5.0
        )
        data = resp.json()
        if data.get("status") == "stored":
            print(f"Rule stored: {data.get('id', '?')}")
            return 0
        elif data.get("status") == "duplicate":
            print(
                f"Duplicate of {data.get('existing_id', '?')} "
                f"(similarity {data.get('similarity', 0):.2f})"
            )
            return 0
        else:
            print(f"Unexpected response: {data}", file=sys.stderr)
            return 1
    except httpx.HTTPError as exc:
        print(f"Error contacting daemon: {exc}", file=sys.stderr)
        return 1


def _cmd_list(args: argparse.Namespace) -> int:
    """List existing TOOL_RULE memories."""
    params: dict = {"type": "TOOL_RULE", "limit": 50}
    try:
        resp = httpx.get(
            f"{_daemon_url()}/list", params=params, timeout=5.0
        )
        data = resp.json()
    except httpx.HTTPError as exc:
        print(f"Error contacting daemon: {exc}", file=sys.stderr)
        return 1

    memories = data.get("memories", [])
    if not memories:
        print("No tool rules found.")
        return 0

    # Default: only this project's rules (the opaque id the matcher scopes by).
    # ``--all`` shows every project; ``--project`` narrows by substring.
    current_id = None if getattr(args, "all", False) else simba.db.resolve_project_id()

    shown = 0
    for m in memories:
        mid = m.get("id", "?")
        content = m.get("content", "")
        project = m.get("projectPath", "")
        ctx_str = m.get("context", "{}")
        correction = ""
        tool = ""
        try:
            ctx = json.loads(ctx_str)
            correction = ctx.get("correction", "")
            tool = ctx.get("tool", "")
        except (json.JSONDecodeError, TypeError):
            pass

        # Apply filters
        if current_id is not None and project != current_id:
            continue
        if args.tool and tool and tool != args.tool:
            continue
        if args.project and project and args.project not in project:
            continue

        shown += 1
        print(f"  {mid}  [{tool or '?'}]  {content}")
        if correction:
            print(f"         INSTEAD: {correction}")
        if project:
            print(f"         project: {project}")

    if shown == 0:
        print("No tool rules for this project (use --all to see all projects).")
    return 0


def _cmd_prune(args: argparse.Namespace) -> int:
    """Bulk-delete TOOL_RULE memories (this project by default)."""
    max_age_seconds: int | None = None
    if args.older_than:
        max_age_seconds = _parse_duration_seconds(args.older_than)
        if max_age_seconds is None:
            print(
                f"Error: invalid --older-than '{args.older_than}' "
                "(use forms like 14d / 48h / 2w / 30m)",
                file=sys.stderr,
            )
            return 1

    try:
        resp = httpx.get(
            f"{_daemon_url()}/list",
            params={"type": "TOOL_RULE", "limit": 1000},
            timeout=10.0,
        )
        memories = resp.json().get("memories", [])
    except httpx.HTTPError as exc:
        print(f"Error contacting daemon: {exc}", file=sys.stderr)
        return 1

    current_id = None if args.all_projects else simba.db.resolve_project_id()

    targets = []
    for m in memories:
        project = m.get("projectPath", "")
        if current_id is not None and project != current_id:
            continue
        if args.project and args.project not in project:
            continue
        if max_age_seconds is not None:
            age = _age_seconds(m.get("createdAt", ""))
            if age is None or age < max_age_seconds:
                continue
        targets.append(m)

    if not targets:
        print("No matching tool rules to prune.")
        return 0

    if args.dry_run:
        print(f"Would prune {len(targets)} tool rule(s):")
        for m in targets:
            print(f"  {m.get('id', '?')}  {m.get('content', '')}")
        return 0

    deleted = 0
    for m in targets:
        mid = m.get("id", "")
        if not mid:
            continue
        try:
            r = httpx.delete(f"{_daemon_url()}/memory/{mid}", timeout=10.0)
            if r.status_code == 200:
                deleted += 1
        except httpx.HTTPError:
            pass

    print(f"Pruned {deleted} tool rule(s).")
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    """Delete a TOOL_RULE memory by ID."""
    try:
        resp = httpx.delete(
            f"{_daemon_url()}/memory/{args.rule_id}", timeout=5.0
        )
        if resp.status_code == 200:
            print(f"Removed rule {args.rule_id}")
            return 0
        else:
            print(f"Failed: {resp.text}", file=sys.stderr)
            return 1
    except httpx.HTTPError as exc:
        print(f"Error contacting daemon: {exc}", file=sys.stderr)
        return 1


def main(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="simba rule")
    sub = parser.add_subparsers(dest="subcmd")

    add_p = sub.add_parser("add", help="Add a tool rule")
    add_p.add_argument("--tool", required=True, help="Tool name (e.g. Bash)")
    add_p.add_argument(
        "--correction", required=True, help="What to do instead"
    )
    add_p.add_argument("--pattern", default="", help="Command pattern")
    add_p.add_argument("--project", default="", help="Project path scope")

    list_p = sub.add_parser("list", help="List tool rules")
    list_p.add_argument("--tool", default="", help="Filter by tool name")
    list_p.add_argument("--project", default="", help="Filter by project")
    list_p.add_argument(
        "--all", action="store_true", help="Show rules from all projects"
    )

    rm_p = sub.add_parser("remove", help="Remove a tool rule")
    rm_p.add_argument("rule_id", help="Memory ID to remove")

    prune_p = sub.add_parser("prune", help="Bulk-delete tool rules")
    prune_p.add_argument(
        "--older-than", default="", help="Only prune entries older than 14d/48h/2w"
    )
    prune_p.add_argument("--project", default="", help="Restrict by project substring")
    prune_p.add_argument(
        "--all-projects", action="store_true", help="Prune across all projects"
    )
    prune_p.add_argument(
        "--dry-run", action="store_true", help="List what would be pruned"
    )

    parsed = parser.parse_args(args)
    if parsed.subcmd == "add":
        return _cmd_add(parsed)
    elif parsed.subcmd == "list":
        return _cmd_list(parsed)
    elif parsed.subcmd == "remove":
        return _cmd_remove(parsed)
    elif parsed.subcmd == "prune":
        return _cmd_prune(parsed)
    else:
        parser.print_help()
        return 1
