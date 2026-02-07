"""CLI for the Orchestration server.

Usage:
    simba orchestration run --root-dir .             Run MCP server
    simba orchestration proxy --root-dir .           Run via hot-reload proxy
    simba orchestration install                      Register with Claude Code
    simba orchestration status <ticket_id> <state>   Update agent status
    simba orchestration agents [--inject] [--update] Manage agent definitions
    simba orchestration sync                         Update managed sections
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import simba.db
import simba.orchestration.config
import simba.orchestration.install
import simba.orchestration.templates


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Orchestration: Agent dispatch and management for Claude"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- install ---
    install_parser = subparsers.add_parser(
        "install",
        help="Register MCP server with Claude and bootstrap agents",
    )
    install_parser.add_argument(
        "--name", default="orchestration", help="Name of the MCP server"
    )
    install_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing agent definition files",
    )
    install_parser.add_argument(
        "--proxy",
        action="store_true",
        help="Use hot-reload proxy mode",
    )

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run the MCP server (internal use)")
    run_parser.add_argument("--root-dir", type=Path, required=True, help="Project root")

    # --- proxy ---
    proxy_parser = subparsers.add_parser(
        "proxy", help="Run MCP server via hot-reload proxy"
    )
    proxy_parser.add_argument(
        "--pid-file",
        type=Path,
        default=Path(".claude/proxy.pid"),
        help="Path to PID file",
    )
    proxy_parser.add_argument(
        "--root-dir", type=Path, required=True, help="Project root"
    )

    # --- status ---
    status_parser = subparsers.add_parser(
        "status", help="Update agent status (called by subagents)"
    )
    status_parser.add_argument("ticket_id", help="Ticket ID")
    status_parser.add_argument("state", choices=["running", "completed", "failed"])
    status_parser.add_argument("--message", "-m", default="", help="Status message")

    # --- agents ---
    agents_parser = subparsers.add_parser(
        "agents", help="Manage agent definition files"
    )
    agents_parser.add_argument(
        "--inject",
        action="store_true",
        help="Inject managed section markers into agent files",
    )
    agents_parser.add_argument(
        "--update",
        action="store_true",
        help="Update managed sections in agent files",
    )
    agents_parser.add_argument(
        "--dir",
        default=".claude/agents",
        type=Path,
        help="Agent definitions directory",
    )
    agents_parser.add_argument(
        "--sections",
        nargs="+",
        default=list(simba.orchestration.templates.MANAGED_SECTIONS.keys()),
        help="Sections to inject/update",
    )

    # --- sync ---
    sync_parser = subparsers.add_parser(
        "sync",
        help="Update managed sections in CLAUDE.md and agent files",
    )
    sync_parser.add_argument(
        "--claude-md",
        default="CLAUDE.md",
        type=Path,
        help="Path to CLAUDE.md",
    )
    sync_parser.add_argument(
        "--agents-dir",
        default=".claude/agents",
        type=Path,
        help="Agent definitions directory",
    )

    args, _ = parser.parse_known_args(argv)

    if args.command == "status":
        status_id = simba.orchestration.config.STATUS_NAME_MAP.get(args.state.lower())
        if not status_id:
            print(f"Invalid status: {args.state}", file=sys.stderr)
            return 1

        with simba.db.get_db() as conn:
            if status_id in (
                simba.orchestration.config.Status.COMPLETED,
                simba.orchestration.config.Status.FAILED,
            ):
                conn.execute(
                    """UPDATE agent_runs
                       SET status_id=?, error=?, completed_at_utc=?
                       WHERE ticket_id=?""",
                    (
                        status_id,
                        args.message
                        if status_id == simba.orchestration.config.Status.FAILED
                        else None,
                        simba.orchestration.config.utc_now(),
                        args.ticket_id,
                    ),
                )
            else:
                conn.execute(
                    "UPDATE agent_runs SET status_id=? WHERE ticket_id=?",
                    (status_id, args.ticket_id),
                )
            conn.commit()
        print(f"{args.ticket_id} -> {args.state}")
        return 0

    if args.command == "agents":
        agents_dir = Path(args.dir)
        if not agents_dir.exists():
            print(f"Directory not found: {agents_dir}", file=sys.stderr)
            return 1
        if args.inject:
            print(f"Injecting markers into {agents_dir}...")
            simba.orchestration.templates.inject_markers(agents_dir, args.sections)
        if args.update or not args.inject:
            print(f"Updating managed sections in {agents_dir}...")
            for agent_file in agents_dir.glob("*.md"):
                original = agent_file.read_text()
                updated = simba.orchestration.templates.update_managed_sections(
                    original
                )
                if original != updated:
                    agent_file.write_text(updated)
                    print(f"   {agent_file.name}")
        print("Done.")
        return 0

    if args.command == "sync":
        print("Syncing managed sections...")

        claude_md = Path(args.claude_md)
        if claude_md.exists():
            original = claude_md.read_text()
            updated = simba.orchestration.templates.update_managed_sections(original)
            if original != updated:
                claude_md.write_text(updated)
                print(f"   {claude_md}")
            else:
                print(f"   {claude_md} (no changes)")
        else:
            print(f"   {claude_md} not found", file=sys.stderr)

        agents_dir = Path(args.agents_dir)
        if agents_dir.exists():
            for agent_file in agents_dir.glob("*.md"):
                original = agent_file.read_text()
                updated = simba.orchestration.templates.update_managed_sections(
                    original
                )
                if original != updated:
                    agent_file.write_text(updated)
                    print(f"   {agent_file.name}")
        else:
            print(f"   {agents_dir} not found", file=sys.stderr)

        print("Done.")
        return 0

    if args.command == "install":
        simba.orchestration.install.install_routine(args.name, args.force, args.proxy)
        return 0

    if args.command == "proxy":
        import simba.orchestration.proxy

        root_dir = args.root_dir.resolve()
        simba.orchestration.proxy.run_proxy(pid_file=args.pid_file, root_dir=root_dir)
        return 0

    # Default: run mode
    if not args.command or args.command == "run":
        import simba.orchestration.server

        root_dir = getattr(args, "root_dir", None)
        simba.orchestration.server.run_server(root_dir=root_dir)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
