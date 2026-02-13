"""CLI for managing tool rules (TOOL_RULE memories).

Usage:
    simba rule add --tool TOOL --correction TEXT [--pattern PAT] [--project DIR]
    simba rule list [--tool TOOL] [--project DIR]
    simba rule remove <rule_id>
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

import simba.config
import simba.hooks.config


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
        payload["projectPath"] = args.project

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
        if args.tool and tool and tool != args.tool:
            continue
        if args.project and project and args.project not in project:
            continue

        print(f"  {mid}  [{tool or '?'}]  {content}")
        if correction:
            print(f"         INSTEAD: {correction}")
        if project:
            print(f"         project: {project}")

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

    rm_p = sub.add_parser("remove", help="Remove a tool rule")
    rm_p.add_argument("rule_id", help="Memory ID to remove")

    parsed = parser.parse_args(args)
    if parsed.subcmd == "add":
        return _cmd_add(parsed)
    elif parsed.subcmd == "list":
        return _cmd_list(parsed)
    elif parsed.subcmd == "remove":
        return _cmd_remove(parsed)
    else:
        parser.print_help()
        return 1
