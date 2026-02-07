"""CLI for the sync subsystem.

Usage:
    simba sync run               Run all pipelines once (index + extract)
    simba sync index [options]   Index new DB rows into LanceDB/QMD
    simba sync extract [options] Extract facts from memories
    simba sync status            Show watermarks and pipeline state
    simba sync schedule [opts]   Run pipelines on a periodic schedule
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import simba.db
from simba.sync.extractor import ExtractResult, run_extract
from simba.sync.indexer import IndexResult, run_index
from simba.sync.watermarks import get_all_watermarks

logger = logging.getLogger("simba.sync")


def _print_index_result(result: IndexResult, *, dry_run: bool = False) -> None:
    prefix = "[dry-run] " if dry_run else ""
    print(f"{prefix}Index: {result.rows_indexed} indexed, "
          f"{result.duplicates} duplicates, {result.errors} errors "
          f"({result.tables_polled} tables polled, "
          f"{result.rows_exported} exported)")


def _print_extract_result(result: ExtractResult, *, dry_run: bool = False) -> None:
    prefix = "[dry-run] " if dry_run else ""
    print(f"{prefix}Extract: {result.facts_extracted} facts from "
          f"{result.memories_processed} memories, "
          f"{result.facts_duplicate} duplicates, {result.errors} errors")
    if result.agent_dispatched:
        print("  Claude researcher agent dispatched for deeper extraction")


def _cmd_run(args: argparse.Namespace) -> int:
    """Run all pipelines once."""
    cwd = Path(args.cwd)
    daemon_url = args.daemon_url

    idx = run_index(cwd, daemon_url=daemon_url, dry_run=args.dry_run)
    _print_index_result(idx, dry_run=args.dry_run)

    ext = run_extract(
        cwd,
        daemon_url=daemon_url,
        use_claude=args.use_claude,
        dry_run=args.dry_run,
    )
    _print_extract_result(ext, dry_run=args.dry_run)

    return 1 if (idx.errors + ext.errors) > 0 else 0


def _cmd_index(args: argparse.Namespace) -> int:
    """Run the indexing pipeline once."""
    result = run_index(
        Path(args.cwd), daemon_url=args.daemon_url, dry_run=args.dry_run
    )
    _print_index_result(result, dry_run=args.dry_run)
    return 1 if result.errors > 0 else 0


def _cmd_extract(args: argparse.Namespace) -> int:
    """Run the extraction pipeline once."""
    result = run_extract(
        Path(args.cwd),
        daemon_url=args.daemon_url,
        use_claude=args.use_claude,
        dry_run=args.dry_run,
    )
    _print_extract_result(result, dry_run=args.dry_run)
    return 1 if result.errors > 0 else 0


def _cmd_status(args: argparse.Namespace) -> int:
    """Show watermarks and pipeline state."""
    cwd = Path(args.cwd)
    conn = simba.db.get_connection(cwd)
    if conn is None:
        print("Database not found. Run a simba command first to initialize it.")
        return 1

    try:
        watermarks = get_all_watermarks(conn)
    finally:
        conn.close()

    if not watermarks:
        print("No sync watermarks recorded yet.")
        print("Run 'simba sync run' to start syncing.")
        return 0

    print(f"{'Table':<20s} {'Pipeline':<10s} {'Cursor':<24s} "
          f"{'Last Run':<24s} {'Rows':>6s} {'Errors':>6s}")
    print("-" * 94)
    for wm in watermarks:
        cursor = wm["last_cursor"]
        if len(cursor) > 22:
            cursor = cursor[:22] + ".."
        last_run = wm["last_run_at"] or "-"
        if len(last_run) > 22:
            last_run = last_run[:22] + ".."
        print(
            f"{wm['table_name']:<20s} {wm['pipeline']:<10s} "
            f"{cursor:<24s} {last_run:<24s} "
            f"{wm['rows_processed']:>6d} {wm['errors']:>6d}"
        )
    return 0


def _cmd_schedule(args: argparse.Namespace) -> int:
    """Run pipelines on a periodic schedule."""
    from simba.sync.scheduler import SyncScheduler

    scheduler = SyncScheduler(
        cwd=Path(args.cwd),
        daemon_url=args.daemon_url,
        interval_seconds=args.interval,
        use_claude=args.use_claude,
    )
    print(f"Starting sync scheduler (interval: {args.interval}s)")
    print(f"  cwd: {args.cwd}")
    print(f"  daemon: {args.daemon_url}")
    print("Press Ctrl+C to stop.")
    try:
        asyncio.run(scheduler.run_forever())
    except KeyboardInterrupt:
        print("\nScheduler stopped.")
    return 0


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add args shared by all subcommands."""
    parser.add_argument(
        "--cwd",
        default=".",
        help="Working directory (default: current directory)",
    )
    parser.add_argument(
        "--daemon-url",
        default="http://localhost:8741",
        help="Memory daemon URL (default: http://localhost:8741)",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simba sync — keep SQLite, LanceDB, and QMD in sync"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run all pipelines once")
    _add_common_args(run_parser)
    run_parser.add_argument("--dry-run", action="store_true", help="Preview only")
    run_parser.add_argument(
        "--use-claude", action="store_true",
        help="Use Claude agent for deeper fact extraction",
    )

    # --- index ---
    index_parser = subparsers.add_parser("index", help="Run indexing pipeline")
    _add_common_args(index_parser)
    index_parser.add_argument("--dry-run", action="store_true", help="Preview only")

    # --- extract ---
    extract_parser = subparsers.add_parser(
        "extract", help="Run fact extraction pipeline"
    )
    _add_common_args(extract_parser)
    extract_parser.add_argument("--dry-run", action="store_true", help="Preview only")
    extract_parser.add_argument(
        "--use-claude", action="store_true",
        help="Use Claude agent for deeper fact extraction",
    )

    # --- status ---
    status_parser = subparsers.add_parser(
        "status", help="Show watermarks and pipeline state"
    )
    _add_common_args(status_parser)

    # --- schedule ---
    schedule_parser = subparsers.add_parser(
        "schedule", help="Run pipelines periodically"
    )
    _add_common_args(schedule_parser)
    schedule_parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between sync cycles (default: 300)",
    )
    schedule_parser.add_argument(
        "--use-claude", action="store_true",
        help="Use Claude agent for deeper fact extraction",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "run":
        return _cmd_run(args)
    elif args.command == "index":
        return _cmd_index(args)
    elif args.command == "extract":
        return _cmd_extract(args)
    elif args.command == "status":
        return _cmd_status(args)
    elif args.command == "schedule":
        return _cmd_schedule(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
