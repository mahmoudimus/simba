"""CLI for the Neuron neuro-symbolic logic server.

Usage:
    simba neuron run --root-dir .                Run MCP server
    simba neuron proxy --root-dir .              Run via hot-reload proxy
    simba neuron install [--global] [--remove]   Register the MCP with Claude Code
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _install_mcp(
    root_dir: Path | None, *, user_scope: bool = False, remove: bool = False
) -> int:
    """Register (or remove) the Neuron MCP server with Claude Code.

    Shells out to the ``claude`` CLI so registration stays in sync with how
    Claude Code manages MCP servers. Default scope is the current project;
    ``--global`` registers at user scope.
    """
    claude = shutil.which("claude")
    if claude is None:
        print(
            "error: 'claude' CLI not found on PATH; cannot register the MCP server",
            file=sys.stderr,
        )
        return 1

    scope = ["--scope", "user"] if user_scope else []

    if remove:
        result = subprocess.run(
            [claude, "mcp", "remove", "neuron", *scope], check=False
        )
        if result.returncode == 0:
            print("neuron MCP removed")
        return result.returncode

    if root_dir is None:
        import simba.db

        root_dir = simba.db.find_repo_root(Path.cwd()) or Path.cwd()
    root_dir = Path(root_dir).resolve()

    simba_bin = shutil.which("simba") or sys.argv[0]
    result = subprocess.run(
        [
            claude, "mcp", "add", "neuron", *scope, "--",
            simba_bin, "neuron", "run", "--root-dir", str(root_dir),
        ],
        check=False,
    )
    if result.returncode == 0:
        print(
            f"neuron MCP registered for {root_dir} "
            "(reconnect with /mcp or restart to load rlm_* tools)"
        )
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Neuron: Neuro-Symbolic Logic Server for Claude"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_parser = subparsers.add_parser(
        "run", help="Run the MCP server (internal use)"
    )
    run_parser.add_argument(
        "--root-dir", type=Path, required=True, help="Project root"
    )

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

    # --- install ---
    install_parser = subparsers.add_parser(
        "install", help="Register the Neuron MCP server with Claude Code"
    )
    install_parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Project root to register (default: cwd repo root)",
    )
    install_parser.add_argument(
        "--global",
        dest="user_scope",
        action="store_true",
        help="Register at user scope instead of just this project",
    )
    install_parser.add_argument(
        "--remove", action="store_true", help="Unregister instead of register"
    )

    args, _ = parser.parse_known_args()

    if args.command == "install":
        return _install_mcp(
            getattr(args, "root_dir", None),
            user_scope=getattr(args, "user_scope", False),
            remove=getattr(args, "remove", False),
        )

    if args.command == "proxy":
        import simba.orchestration.proxy

        root_dir = args.root_dir.resolve()
        simba.orchestration.proxy.run_proxy(
            pid_file=args.pid_file, root_dir=root_dir
        )
        return 0

    # Default: run mode
    if not args.command or args.command == "run":
        import simba.neuron.server

        root_dir = getattr(args, "root_dir", None)
        simba.neuron.server.run_server(root_dir=root_dir)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
