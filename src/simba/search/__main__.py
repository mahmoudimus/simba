"""CLI for project memory operations.

Usage:
    python -m simba.search init
    python -m simba.search index
    python -m simba.search add-session "summary" '["file1"]' '["Read"]' "topics"
    python -m simba.search add-knowledge "area" "summary" "patterns"
    python -m simba.search add-fact "fact" "category"
    python -m simba.search search "query"
    python -m simba.search context "query" [token_limit]
    python -m simba.search recent [n]
    python -m simba.search stats
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import simba.db
import simba.search.deps
import simba.search.project_memory


def _cmd_index(cwd: pathlib.Path) -> int:
    """Check dependencies, initialize project memory, and index with QMD."""
    project_name = cwd.name

    # 1. Check dependencies
    print("Checking dependencies...")
    deps = simba.search.deps.check_all()
    for name, (found, version) in deps.items():
        if found:
            status = f"  {version}"
        else:
            hint = simba.search.deps.get_install_instructions(name)
            status = f"  MISSING — {hint}"
        print(f"  {name}: {'ok' if found else 'missing'}{status}")
    print()

    # 2. Initialize SQLite project memory
    db_path = simba.db.get_db_path(cwd)
    with simba.db.get_db(cwd):
        pass
    print(f"Project memory: {db_path}")

    # 3. Index with QMD (optional — skip if not installed)
    qmd_available = deps.get("qmd", (False, ""))[0]
    if qmd_available:
        print(f"\nIndexing with QMD (collection: {project_name})...")
        try:
            subprocess.run(
                [
                    "qmd",
                    "collection",
                    "add",
                    ".",
                    "--name",
                    project_name,
                    "--mask",
                    "**/*.md",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(cwd),
            )
            desc = f"Codebase documentation for {project_name}"
            subprocess.run(
                ["qmd", "context", "add", ".", desc],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(cwd),
            )
            result = subprocess.run(
                ["qmd", "embed"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(cwd),
            )
            if result.returncode == 0:
                print("  QMD indexing complete.")
            else:
                print(f"  QMD embed returned code {result.returncode}", file=sys.stderr)
        except (subprocess.SubprocessError, OSError) as exc:
            print(f"  QMD indexing failed: {exc}", file=sys.stderr)
    else:
        print("\nQMD not installed — skipping semantic indexing.")

    # 4. Summary
    rg_available = deps.get("rg", (False, ""))[0]
    if rg_available:
        try:
            result = subprocess.run(
                ["rg", "--files"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(cwd),
            )
            stdout = result.stdout.strip()
            file_count = len(stdout.splitlines()) if stdout else 0
            print(f"\nProject: {project_name} ({file_count} files)")
        except (subprocess.SubprocessError, OSError):
            print(f"\nProject: {project_name}")
    else:
        print(f"\nProject: {project_name}")

    print("Index complete.")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1

    cmd = args[0]
    cwd = pathlib.Path.cwd()

    if cmd == "init":
        db_path = simba.db.get_db_path(cwd)
        with simba.db.get_db(cwd):
            pass
        print(f"Memory database initialized at {db_path}")
        return 0

    if cmd == "index":
        return _cmd_index(cwd)

    # For all other commands, get connection (init if needed)
    conn = simba.db.get_connection(cwd)
    if conn is None:
        # DB does not exist yet; create it via get_db context manager,
        # but we need a persistent connection for the commands below.
        db_path = simba.db.get_db_path(cwd)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        simba.db._init_schemas(conn)

    try:
        if cmd == "add-session" and len(args) >= 5:
            rowid = simba.search.project_memory.add_session(
                conn, args[1], args[2], args[3], args[4]
            )
            print(f"Session saved (id={rowid}).")

        elif cmd == "add-knowledge" and len(args) >= 4:
            rowid = simba.search.project_memory.add_knowledge(
                conn, args[1], args[2], args[3]
            )
            print(f"Knowledge saved for: {args[1]}")

        elif cmd == "add-fact" and len(args) >= 2:
            category = args[2] if len(args) >= 3 else "general"
            rowid = simba.search.project_memory.add_fact(conn, args[1], category)
            print(f"Fact saved (id={rowid}).")

        elif cmd == "search" and len(args) >= 2:
            limit = int(args[2]) if len(args) >= 3 else 10
            results = simba.search.project_memory.search_fts(conn, args[1], limit)
            print(json.dumps(results, indent=2))

        elif cmd == "context" and len(args) >= 2:
            budget = int(args[2]) if len(args) >= 3 else 500
            ctx = simba.search.project_memory.get_context(conn, args[1], budget)
            print(ctx)

        elif cmd == "recent":
            limit = int(args[1]) if len(args) >= 2 else 5
            sessions = simba.search.project_memory.get_recent_sessions(conn, limit)
            print(json.dumps(sessions, indent=2))

        elif cmd == "stats":
            stats = simba.search.project_memory.get_stats(conn)
            print(json.dumps(stats, indent=2))

        else:
            print(__doc__)
            return 1
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
