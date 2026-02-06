"""CLI for project memory operations.

Usage:
    python -m simba.search init
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
import sys

import simba.search.project_memory


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1

    cmd = args[0]
    cwd = pathlib.Path.cwd()

    if cmd == "init":
        db_path = simba.search.project_memory.get_db_path(cwd)
        conn = simba.search.project_memory.init_db(db_path)
        conn.close()
        print(f"Memory database initialized at {db_path}")
        return 0

    # For all other commands, get connection (init if needed)
    conn = simba.search.project_memory.get_connection(cwd)
    if conn is None:
        db_path = simba.search.project_memory.get_db_path(cwd)
        conn = simba.search.project_memory.init_db(db_path)

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
