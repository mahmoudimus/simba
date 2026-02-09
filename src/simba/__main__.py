"""Simba CLI — unified Claude Code plugin.

Usage:
    simba install          Register hooks in current project
    simba install --global Register hooks globally (~/.claude/settings.json)
    simba install --remove Remove hooks (add --global for global)
    simba server [opts]    Start the memory daemon
    simba search <cmd>     Project memory operations
    simba sync <cmd>       Sync SQLite, LanceDB, and QMD
    simba stats            Show token economics and project statistics
    simba neuron <cmd>     Neuro-symbolic logic server (MCP)
    simba orchestration <cmd> Agent orchestration server (MCP)
    simba config <cmd>     Unified configuration (get/set/list/show)
    simba markers <cmd>    Discover, audit, and update SIMBA markers
    simba db <subcmd>      Inspect or migrate the shared database
    simba hook <event>     Run a hook (called by Claude Code, not users)
"""

from __future__ import annotations

import json
import pathlib
import sys

_HOOK_EVENTS = {
    "SessionStart": "simba.hooks.session_start",
    "UserPromptSubmit": "simba.hooks.user_prompt_submit",
    "PreToolUse": "simba.hooks.pre_tool_use",
    "PostToolUse": "simba.hooks.post_tool_use",
    "PreCompact": "simba.hooks.pre_compact",
    "Stop": "simba.hooks.stop",
}

_HOOK_TIMEOUTS = {
    "SessionStart": 15000,
    "UserPromptSubmit": 3000,
    "PreToolUse": 3000,
    "PostToolUse": 3000,
    "PreCompact": 5000,
    "Stop": 5000,
}

_GLOBAL_SETTINGS = pathlib.Path.home() / ".claude" / "settings.json"


def _build_hooks_config() -> dict:
    """Build the hooks section for settings.json."""
    hooks: dict = {}
    for event in _HOOK_EVENTS:
        timeout = _HOOK_TIMEOUTS[event]
        hooks[event] = [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"simba hook {event}",
                        "timeout": timeout,
                    }
                ]
            }
        ]
    return hooks


def _bundled_skill_names() -> list[str]:
    """Return names of all bundled skills."""
    import importlib.resources

    skills_pkg = importlib.resources.files("simba") / "skills"
    if not skills_pkg.is_dir():
        return []
    return [
        d.name
        for d in skills_pkg.iterdir()
        if d.is_dir() and (d / "skill.md").is_file()
    ]


def _install_skills(skills_dir: pathlib.Path) -> int:
    """Copy bundled skills into *skills_dir*."""
    import importlib.resources

    skills_pkg = importlib.resources.files("simba") / "skills"
    if not skills_pkg.is_dir():
        return 0

    copied = 0
    for skill_dir in skills_pkg.iterdir():
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "skill.md"
        if not skill_md.is_file():
            continue
        dest_dir = skills_dir / skill_dir.name
        dest_file = dest_dir / "skill.md"
        if dest_file.exists():
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file.write_text(skill_md.read_text())
        print(f"  + skill: /{skill_dir.name}")
        copied += 1
    return copied


def _remove_skills(skills_dir: pathlib.Path) -> int:
    """Remove bundled skills from *skills_dir*."""
    import shutil

    removed = 0
    for name in _bundled_skill_names():
        dest_dir = skills_dir / name
        if dest_dir.is_dir():
            shutil.rmtree(dest_dir)
            print(f"  - skill: /{name}")
            removed += 1
    return removed


def _cmd_install(args: list[str]) -> int:
    """Register or remove simba hooks.

    By default writes to .claude/settings.local.json in the current
    project.  Use ``--global`` to write to ~/.claude/settings.json
    instead.
    """
    remove = "--remove" in args
    is_global = "--global" in args

    if is_global:
        settings_path = _GLOBAL_SETTINGS
    else:
        settings_path = pathlib.Path.cwd() / ".claude" / "settings.local.json"

    if not settings_path.parent.exists():
        settings_path.parent.mkdir(parents=True, exist_ok=True)

    settings: dict = {}
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())

    if is_global:
        skills_dir = pathlib.Path.home() / ".claude" / "skills"
    else:
        skills_dir = pathlib.Path.cwd() / ".claude" / "skills"

    if remove:
        if "hooks" in settings:
            del settings["hooks"]
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        print("Simba hooks removed from", settings_path)
        removed = _remove_skills(skills_dir)
        if removed:
            print(f"  {removed} skill(s) removed")
        return 0

    settings["hooks"] = _build_hooks_config()
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    scope = "global" if is_global else "project"
    print(f"Simba hooks registered ({scope}) in {settings_path}")
    print(f"  {len(_HOOK_EVENTS)} hooks: {', '.join(_HOOK_EVENTS)}")

    skill_count = _install_skills(skills_dir)
    if skill_count:
        print(f"  {skill_count} skill(s) installed")

    return 0


def _cmd_hook(args: list[str]) -> int:
    """Dispatch a hook event. Called by Claude Code, not users."""
    if not args:
        print("Usage: simba hook <event>", file=sys.stderr)
        print(f"Events: {', '.join(_HOOK_EVENTS)}", file=sys.stderr)
        return 1

    event = args[0]
    module_name = _HOOK_EVENTS.get(event)
    if module_name is None:
        print(f"Unknown hook event: {event}", file=sys.stderr)
        return 1

    import importlib

    module = importlib.import_module(module_name)

    hook_data: dict = {}
    try:
        raw = sys.stdin.read()
        if raw:
            hook_data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    print(module.main(hook_data))
    return 0


def _cmd_server(args: list[str]) -> int:
    """Start the memory daemon."""
    # Rewrite sys.argv so argparse in server.main() sees the right args
    sys.argv = ["simba server", *args]
    import simba.memory.server

    simba.memory.server.main()
    return 0


def _cmd_search(args: list[str]) -> int:
    """Project memory operations."""
    sys.argv = ["simba search", *args]
    import simba.search.__main__

    return simba.search.__main__.main()


def _cmd_stats() -> int:
    """Show token economics and project statistics."""
    import simba.stats

    print(simba.stats.run_stats(pathlib.Path.cwd()))
    return 0


def _cmd_sync(args: list[str]) -> int:
    """Sync SQLite, LanceDB, and QMD."""
    sys.argv = ["simba sync", *args]
    import simba.sync.__main__

    return simba.sync.__main__.main()


def _cmd_neuron(args: list[str]) -> int:
    """Neuro-symbolic logic server (MCP)."""
    sys.argv = ["simba neuron", *args]
    import simba.neuron.__main__

    return simba.neuron.__main__.main()


def _cmd_orchestration(args: list[str]) -> int:
    """Agent orchestration server (MCP)."""
    sys.argv = ["simba orchestration", *args]
    import simba.orchestration.__main__

    return simba.orchestration.__main__.main(args)


_DB_USAGE = """\
Usage: simba db <subcommand> [options]

Subcommands:
    stats                  Row counts for all tables
    reflections [options]  Show error reflections
    activities [options]   Show tool activity log
    facts                  Show proven facts (neuron)
    agents [options]       Show agent runs
    sessions [options]     Show project memory sessions
    migrate                Migrate data from old per-module databases

Options:
    --limit N              Max rows to display (default: 20)
    --type TYPE            Filter reflections by error type
    --status STATUS        Filter agents by status
"""


def _parse_db_opts(args: list[str]) -> dict[str, str]:
    """Parse --key value pairs from args."""
    opts: dict[str, str] = {}
    i = 0
    while i < len(args):
        if args[i].startswith("--") and i + 1 < len(args):
            opts[args[i][2:]] = args[i + 1]
            i += 2
        else:
            i += 1
    return opts


def _cmd_db(args: list[str]) -> int:
    """Inspect or migrate the shared simba.db database."""
    if not args:
        print(_DB_USAGE)
        return 1

    import simba.db

    # Ensure all schemas are registered by importing modules
    import simba.neuron.truth
    import simba.orchestration.agents
    import simba.search.activity_tracker
    import simba.search.project_memory
    import simba.tailor.hook

    _use = (
        simba.orchestration.agents,
        simba.neuron.truth,
        simba.search.activity_tracker,
        simba.search.project_memory,
        simba.tailor.hook,
    )
    del _use

    subcmd = args[0]
    opts = _parse_db_opts(args[1:])
    limit = int(opts.get("limit", "20"))
    cwd = pathlib.Path.cwd()

    if subcmd == "stats":
        return _db_stats(cwd)
    elif subcmd == "reflections":
        return _db_reflections(cwd, limit, opts.get("type"))
    elif subcmd == "activities":
        return _db_activities(cwd, limit)
    elif subcmd == "facts":
        return _db_facts(cwd, limit)
    elif subcmd == "agents":
        return _db_agents(cwd, limit, opts.get("status"))
    elif subcmd == "sessions":
        return _db_sessions(cwd, limit)
    elif subcmd == "migrate":
        return _db_migrate(cwd)
    else:
        print(f"Unknown db subcommand: {subcmd}")
        print(_DB_USAGE)
        return 1


def _db_stats(cwd: pathlib.Path) -> int:
    """Print row counts for all tables."""
    import simba.db

    conn = simba.db.get_connection(cwd)
    if conn is None:
        print("Database not found. Run a simba command first to initialize it.")
        return 1

    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()

        print(f"Database: {simba.db.get_db_path(cwd)}")
        print()
        total = 0
        for (name,) in tables:
            q = f"SELECT COUNT(*) FROM [{name}]"
            count = conn.execute(q).fetchone()[0]
            total += count
            print(f"  {name:<20s} {count:>6d} rows")
        print(f"  {'─' * 28}")
        print(f"  {'total':<20s} {total:>6d} rows")
    finally:
        conn.close()
    return 0


def _db_reflections(cwd: pathlib.Path, limit: int, error_type: str | None) -> int:
    """Print recent reflections."""
    import simba.db

    conn = simba.db.get_connection(cwd)
    if conn is None:
        print("Database not found.")
        return 1

    try:
        query = "SELECT id, ts, error_type, snippet, signature FROM reflections"
        params: list[str] = []
        if error_type:
            query += " WHERE error_type = ?"
            params.append(error_type)
        query += " ORDER BY ts DESC LIMIT ?"
        params.append(str(limit))

        rows = conn.execute(query, params).fetchall()
        if not rows:
            print("No reflections found.")
            return 0

        for row in rows:
            s = row["snippet"]
            snippet = s[:80] + "..." if len(s) > 80 else s
            print(f"[{row['ts']}] {row['error_type']} — {row['signature']}")
            if snippet:
                print(f"  {snippet}")
            print()
    finally:
        conn.close()
    return 0


def _db_activities(cwd: pathlib.Path, limit: int) -> int:
    """Print recent activities."""
    import simba.db

    conn = simba.db.get_connection(cwd)
    if conn is None:
        print("Database not found.")
        return 1

    try:
        rows = conn.execute(
            "SELECT timestamp, tool_name, detail FROM activities "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        if not rows:
            print("No activities logged.")
            return 0

        for row in rows:
            d = row["detail"]
            detail = d[:60] + "..." if len(d) > 60 else d
            print(f"[{row['timestamp']}] {row['tool_name']:<12s} {detail}")
    finally:
        conn.close()
    return 0


def _db_facts(cwd: pathlib.Path, limit: int) -> int:
    """Print proven facts."""
    import simba.db

    conn = simba.db.get_connection(cwd)
    if conn is None:
        print("Database not found.")
        return 1

    try:
        rows = conn.execute(
            "SELECT subject, predicate, object, proof FROM proven_facts LIMIT ?",
            (limit,),
        ).fetchall()
        if not rows:
            print("No proven facts recorded.")
            return 0

        for row in rows:
            p = row["proof"]
            proof = p[:40] + "..." if len(p) > 40 else p
            print(f"  {row['subject']} {row['predicate']} {row['object']}")
            print(f"    proof: {proof}")
    finally:
        conn.close()
    return 0


def _db_agents(cwd: pathlib.Path, limit: int, status: str | None) -> int:
    """Print agent runs."""
    import simba.db

    conn = simba.db.get_connection(cwd)
    if conn is None:
        print("Database not found.")
        return 1

    try:
        query = (
            "SELECT ar.ticket_id, ar.agent, ar.pid, st.name AS status, "
            "ar.created_at_utc, ar.completed_at_utc, ar.result, ar.error "
            "FROM agent_runs ar "
            "LEFT JOIN status_types st ON ar.status_id = st.id"
        )
        params: list[str] = []
        if status:
            query += " WHERE st.name = ?"
            params.append(status)
        query += " ORDER BY ar.created_at_utc DESC LIMIT ?"
        params.append(str(limit))

        rows = conn.execute(query, params).fetchall()
        if not rows:
            print("No agent runs found.")
            return 0

        for row in rows:
            elapsed = ""
            if row["completed_at_utc"] and row["created_at_utc"]:
                secs = row["completed_at_utc"] - row["created_at_utc"]
                elapsed = f" [{secs}s]"
            result_preview = ""
            if row["result"]:
                r = row["result"]
                result_preview = f"\n    Result: {r[:80]}{'...' if len(r) > 80 else ''}"
            error = ""
            if row["error"]:
                error = f"\n    Error: {row['error']}"
            print(
                f"  {row['ticket_id']} ({row['agent']}, PID {row['pid']}): "
                f"{row['status'] or 'unknown'}{elapsed}{result_preview}{error}"
            )
    finally:
        conn.close()
    return 0


def _db_sessions(cwd: pathlib.Path, limit: int) -> int:
    """Print project memory sessions."""
    import simba.db

    conn = simba.db.get_connection(cwd)
    if conn is None:
        print("Database not found.")
        return 1

    try:
        rows = conn.execute(
            "SELECT session_id, started_at, summary FROM sessions "
            "ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        if not rows:
            print("No sessions recorded.")
            return 0

        for row in rows:
            sm = row["summary"]
            summary = sm[:60] + "..." if len(sm) > 60 else sm
            print(f"[{row['started_at']}] {row['session_id']}")
            if summary:
                print(f"  {summary}")
    finally:
        conn.close()
    return 0


def _db_migrate(cwd: pathlib.Path) -> int:
    """Migrate data from old per-module databases into simba.db."""
    import sqlite3

    import simba.db

    base = simba.db.find_repo_root(cwd)
    if base is None:
        base = cwd
    simba_dir = base / ".simba"

    # Ensure the target DB exists with all schemas
    with simba.db.get_db(cwd):
        pass

    migrated: dict[str, int] = {}

    # 1. neuron/truth.db → proven_facts
    truth_db = simba_dir / "neuron" / "truth.db"
    if truth_db.exists():
        src = sqlite3.connect(str(truth_db))
        try:
            rows = src.execute(
                "SELECT subject, predicate, object, proof FROM facts"
            ).fetchall()
            if rows:
                with simba.db.get_db(cwd) as conn:
                    conn.executemany(
                        "INSERT OR IGNORE INTO proven_facts "
                        "(subject, predicate, object, proof) "
                        "VALUES (?, ?, ?, ?)",
                        rows,
                    )
                    conn.commit()
                migrated["proven_facts (from neuron/truth.db)"] = len(rows)
        except sqlite3.OperationalError:
            pass
        finally:
            src.close()

    # 2. neuron/agents.db → agent_runs, agent_logs
    agents_db = simba_dir / "neuron" / "agents.db"
    if agents_db.exists():
        src = sqlite3.connect(str(agents_db))
        try:
            runs = src.execute("SELECT * FROM agent_runs").fetchall()
            desc = src.execute("SELECT * FROM agent_runs LIMIT 0").description
            run_cols = [d[0] for d in desc]
            if runs:
                placeholders = ", ".join("?" * len(run_cols))
                cols = ", ".join(run_cols)
                with simba.db.get_db(cwd) as conn:
                    q = (
                        f"INSERT OR IGNORE INTO agent_runs "
                        f"({cols}) VALUES ({placeholders})"
                    )
                    conn.executemany(
                        q,
                        runs,
                    )
                    conn.commit()
                migrated["agent_runs (from neuron/agents.db)"] = len(runs)

            logs = src.execute("SELECT * FROM agent_logs").fetchall()
            desc = src.execute("SELECT * FROM agent_logs LIMIT 0").description
            log_cols = [d[0] for d in desc]
            if logs:
                # Skip the auto-increment id column
                non_id_cols = [c for c in log_cols if c != "id"]
                non_id_idx = [i for i, c in enumerate(log_cols) if c != "id"]
                placeholders = ", ".join("?" * len(non_id_cols))
                cols = ", ".join(non_id_cols)
                filtered_logs = [tuple(row[i] for i in non_id_idx) for row in logs]
                with simba.db.get_db(cwd) as conn:
                    conn.executemany(
                        f"INSERT INTO agent_logs ({cols}) VALUES ({placeholders})",
                        filtered_logs,
                    )
                    conn.commit()
                migrated["agent_logs (from neuron/agents.db)"] = len(logs)
        except sqlite3.OperationalError:
            pass
        finally:
            src.close()

    # 3. search/memory.db → sessions, knowledge, facts
    memory_db = simba_dir / "search" / "memory.db"
    if memory_db.exists():
        src = sqlite3.connect(str(memory_db))
        try:
            for table in ("sessions", "knowledge", "facts"):
                try:
                    rows = src.execute(f"SELECT * FROM {table}").fetchall()
                    desc = src.execute(f"SELECT * FROM {table} LIMIT 0").description
                    cols = [d[0] for d in desc]
                except sqlite3.OperationalError:
                    continue
                if rows:
                    placeholders = ", ".join("?" * len(cols))
                    col_str = ", ".join(cols)
                    with simba.db.get_db(cwd) as conn:
                        q = (
                            f"INSERT OR IGNORE INTO {table} "
                            f"({col_str}) VALUES ({placeholders})"
                        )
                        conn.executemany(
                            q,
                            rows,
                        )
                        conn.commit()
                    migrated[f"{table} (from search/memory.db)"] = len(rows)
        except sqlite3.OperationalError:
            pass
        finally:
            src.close()

    # 4. search/activity.log → activities
    activity_log = simba_dir / "search" / "activity.log"
    if activity_log.exists():
        try:
            lines = activity_log.read_text().strip().splitlines()
            count = 0
            with simba.db.get_db(cwd) as conn:
                for line in lines:
                    parts = line.split("|", 2)
                    if len(parts) >= 2:
                        ts = parts[0].strip()
                        tool = parts[1].strip()
                        detail = parts[2].strip() if len(parts) > 2 else ""
                        conn.execute(
                            "INSERT INTO activities (timestamp, tool_name, detail) "
                            "VALUES (?, ?, ?)",
                            (ts, tool, detail),
                        )
                        count += 1
                conn.commit()
            if count:
                migrated["activities (from search/activity.log)"] = count
        except OSError:
            pass

    # 5. tailor/reflections.jsonl → reflections
    reflections_jsonl = simba_dir / "tailor" / "reflections.jsonl"
    if reflections_jsonl.exists():
        try:
            lines = reflections_jsonl.read_text().strip().splitlines()
            count = 0
            with simba.db.get_db(cwd) as conn:
                for line in lines:
                    try:
                        entry = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    conn.execute(
                        "INSERT OR IGNORE INTO reflections "
                        "(id, ts, error_type, snippet, context, signature) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            entry.get("id", ""),
                            entry.get("ts", ""),
                            entry.get("error_type", ""),
                            entry.get("snippet", ""),
                            json.dumps(entry.get("context", {})),
                            entry.get("signature", ""),
                        ),
                    )
                    count += 1
                conn.commit()
            if count:
                migrated["reflections (from tailor/reflections.jsonl)"] = count
        except OSError:
            pass

    # Report
    if not migrated:
        print("No old data files found to migrate.")
        print(f"Looked in: {simba_dir}")
        return 0

    print(f"Migration complete → {simba.db.get_db_path(cwd)}")
    print()
    for source, count in migrated.items():
        print(f"  {source}: {count} rows")
    print()
    print("Old files were NOT deleted. Remove manually when satisfied:")
    print(
        f"  rm -rf {simba_dir}/neuron/ {simba_dir}/search/ "
        f"{simba_dir}/tailor/reflections.jsonl"
    )
    return 0


def _cmd_config(args: list[str]) -> int:
    """Unified configuration."""
    import simba.config_cli

    return simba.config_cli.main(args)


def _cmd_markers(args: list[str]) -> int:
    """Discover, audit, and update SIMBA markers."""
    import simba.markers_cli

    return simba.markers_cli.main(args)


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    cmd = args[0]
    rest = args[1:]

    if cmd == "install":
        sys.exit(_cmd_install(rest))
    elif cmd == "hook":
        sys.exit(_cmd_hook(rest))
    elif cmd == "server":
        sys.exit(_cmd_server(rest))
    elif cmd == "search":
        sys.exit(_cmd_search(rest))
    elif cmd == "stats":
        sys.exit(_cmd_stats())
    elif cmd == "sync":
        sys.exit(_cmd_sync(rest))
    elif cmd == "neuron":
        sys.exit(_cmd_neuron(rest))
    elif cmd == "orchestration":
        sys.exit(_cmd_orchestration(rest))
    elif cmd == "config":
        sys.exit(_cmd_config(rest))
    elif cmd == "markers":
        sys.exit(_cmd_markers(rest))
    elif cmd == "db":
        sys.exit(_cmd_db(rest))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
