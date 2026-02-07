"""CLI for the Neuron neuro-symbolic logic server.

Usage:
    simba neuron run --root-dir .             Run MCP server
    simba neuron proxy --root-dir .           Run via hot-reload proxy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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

    args, _ = parser.parse_known_args()

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
