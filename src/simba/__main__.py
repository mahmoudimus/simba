"""Simba CLI — unified Claude Code plugin.

Usage:
    simba install          Register hooks in current project
    simba install --global Register hooks globally (~/.claude/settings.json)
    simba install --remove Remove hooks (add --global for global)
    simba codex-install    Install bundled skills for Codex (~/.codex/skills)
    simba codex-install --remove
                           Remove bundled Codex skills
    simba codex-status     Check daemon health + pending transcript extraction
    simba codex-extract    Show extraction prompt for pending transcript
    simba codex-recall     Query semantic memory (/recall) for a text query
    simba codex-finalize   Run end-of-task signal/error checks
    simba codex-automation Print suggested Codex automation directive
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

import contextlib
import json
import os
import pathlib
import re
import sys
from typing import Any

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


def _codex_home() -> pathlib.Path:
    """Return CODEX_HOME (or ~/.codex)."""
    env_home = os.environ.get("CODEX_HOME")
    if env_home:
        return pathlib.Path(env_home).expanduser()
    return pathlib.Path.home() / ".codex"


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


def _latest_transcript_metadata() -> dict[str, Any] | None:
    """Load latest transcript metadata from ~/.claude/transcripts/latest.json."""
    latest = pathlib.Path.home() / ".claude" / "transcripts" / "latest.json"
    if not latest.exists():
        return None
    try:
        return json.loads(latest.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _extract_transcript_text(path: pathlib.Path) -> str:
    """Extract plain text from markdown or JSONL transcript."""
    if not path.exists():
        return ""
    try:
        raw = path.read_text()
    except OSError:
        return ""

    # JSONL transcript: parse message/tool fields.
    if path.suffix == ".jsonl":
        parts: list[str] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            msg = entry.get("message", {})
            if isinstance(msg, dict):
                content = msg.get("content", [])
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            txt = (
                                item.get("text")
                                or item.get("content")
                                or item.get("thinking")
                            )
                            if isinstance(txt, str) and txt.strip():
                                parts.append(txt.strip())
            for key in ("toolUseResult", "text", "content"):
                val = entry.get(key)
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
        return "\n".join(parts)

    # Markdown transcript: drop tags and use remaining text.
    return re.sub(r"<[^>]+>", " ", raw)


def _classify_learning(sentence: str) -> tuple[str, float] | None:
    """Classify a sentence into a memory type with confidence."""
    s = sentence.lower()
    if re.search(r"\b(prefer|prefers|always use|always prefer|likes?)\b", s):
        return ("PREFERENCE", 0.90)
    if re.search(r"\b(fail|fails|failed|broke|broken|error|exception)\b", s):
        return ("FAILURE", 0.88)
    if re.search(r"\b(chose|decided|selected|picked)\b", s):
        return ("DECISION", 0.90)
    if re.search(r"\b(watch out|beware|careful|avoid|don't|never)\b", s):
        return ("GOTCHA", 0.88)
    if re.search(r"\b(pattern|convention|workflow|approach)\b", s):
        return ("PATTERN", 0.85)
    if re.search(r"\b(use|run|fix|resolve|works|worked|solves?)\b", s):
        return ("WORKING_SOLUTION", 0.86)
    return None


def _extract_learnings(
    transcript_text: str,
    *,
    max_items: int = 15,
) -> list[dict[str, Any]]:
    """Extract candidate learnings from transcript text heuristically."""
    # Split into sentence-like units and normalize whitespace.
    chunks = re.split(r"[.\n]+", transcript_text)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    for raw in chunks:
        sentence = " ".join(raw.strip().split())
        if len(sentence) < 24 or len(sentence) > 220:
            continue
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)

        tagged = _classify_learning(sentence)
        if tagged is None:
            continue

        mtype, conf = tagged
        out.append(
            {
                "type": mtype,
                "content": sentence[:200],
                "context": "extracted from transcript",
                "confidence": conf,
            }
        )
        if len(out) >= max_items:
            break

    return out


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


def _bundled_codex_skill_names() -> list[str]:
    """Return names of bundled Codex skills (SKILL.md)."""
    import importlib.resources

    skills_pkg = importlib.resources.files("simba") / "codex_skills"
    if not skills_pkg.is_dir():
        return []
    return [
        d.name
        for d in skills_pkg.iterdir()
        if d.is_dir() and (d / "SKILL.md").is_file()
    ]


def _install_codex_skills(skills_dir: pathlib.Path) -> int:
    """Copy bundled Codex skills (SKILL.md + agents metadata)."""
    import importlib.resources

    skills_pkg = importlib.resources.files("simba") / "codex_skills"
    if not skills_pkg.is_dir():
        return 0

    installed = 0
    for skill_dir in skills_pkg.iterdir():
        if not skill_dir.is_dir():
            continue
        src_skill = skill_dir / "SKILL.md"
        if not src_skill.is_file():
            continue

        dest_dir = skills_dir / skill_dir.name
        dest_skill = dest_dir / "SKILL.md"
        dest_dir.mkdir(parents=True, exist_ok=True)
        if not dest_skill.exists():
            dest_skill.write_text(src_skill.read_text())
            print(f"  + codex skill: {skill_dir.name}")
            installed += 1

        src_agents = skill_dir / "agents"
        if src_agents.is_dir():
            for src_file in src_agents.rglob("*"):
                if not src_file.is_file():
                    continue
                rel_path = src_file.relative_to(skill_dir)
                dst_file = dest_dir / rel_path
                if dst_file.exists():
                    continue
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                dst_file.write_text(src_file.read_text())
    return installed


def _remove_codex_skills(skills_dir: pathlib.Path) -> int:
    """Remove bundled Codex skills from CODEX_HOME."""
    import shutil

    removed = 0
    for name in _bundled_codex_skill_names():
        dest_dir = skills_dir / name
        if dest_dir.is_dir():
            shutil.rmtree(dest_dir)
            print(f"  - codex skill: {name}")
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


def _cmd_codex_install(args: list[str]) -> int:
    """Install or remove bundled skills for Codex."""
    remove = "--remove" in args
    skills_dir = _codex_home() / "skills"

    if remove:
        removed = _remove_codex_skills(skills_dir)
        print(f"Codex skills removed from {skills_dir}")
        if removed:
            print(f"  {removed} skill(s) removed")
        return 0

    skills_dir.mkdir(parents=True, exist_ok=True)
    installed = _install_codex_skills(skills_dir)
    print(f"Codex skills installed in {skills_dir}")
    if installed:
        print(f"  {installed} skill(s) installed")
    return 0


def _cmd_codex_status(args: list[str]) -> int:
    """Check Codex-oriented Simba status: daemon + pending extraction."""
    del args
    import httpx

    import simba.hooks._memory_client

    url = simba.hooks._memory_client.daemon_url()
    print(f"[codex] daemon: {url}")

    health_ok = False
    try:
        resp = httpx.get(f"{url}/health", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            health_ok = True
            print(
                "[codex] memory: up "
                f"(count={data.get('memoryCount', 0)}, "
                f"model={data.get('embeddingModel', 'unknown')})"
            )
            # Mirror Claude SessionStart behavior: trigger one sync cycle.
            with contextlib.suppress(httpx.HTTPError, ValueError):
                httpx.post(f"{url}/sync", timeout=1.0)
                print("[codex] sync: triggered")
    except (httpx.HTTPError, ValueError):
        pass

    if not health_ok:
        print("[codex] memory: down (start with `simba server`)")

    meta = _latest_transcript_metadata()
    if not meta:
        print("[codex] extraction: no latest transcript metadata found")
        return 0

    status = meta.get("status", "unknown")
    transcript = meta.get("transcript_path", "")
    session_id = meta.get("session_id", "")
    print(f"[codex] latest transcript: {transcript or 'unknown'}")
    print(f"[codex] latest session: {session_id or 'unknown'}")
    print(f"[codex] extraction status: {status}")
    if status == "pending_extraction":
        print("[codex] next: run `simba codex-extract`")
    return 0


def _cmd_codex_extract(args: list[str]) -> int:
    """Print a ready-to-run extraction prompt and optionally mark it done."""
    import httpx

    mark_done = "--mark-done" in args
    run_mode = "--run" in args

    meta = _latest_transcript_metadata()
    if not meta:
        print("No latest transcript metadata found at ~/.claude/transcripts/latest.json")
        return 1

    transcript = meta.get("transcript_path", "")
    session_id = meta.get("session_id", "")
    project_path = meta.get("project_path", str(pathlib.Path.cwd()))
    status = meta.get("status", "")

    if not transcript:
        print("latest.json is missing transcript_path")
        return 1

    if status and status != "pending_extraction":
        print(f"Extraction status is '{status}' (not pending).")

    if run_mode:
        transcript_path = pathlib.Path(transcript)
        text = _extract_transcript_text(transcript_path)
        if not text.strip():
            print(f"No readable transcript content found in {transcript_path}")
            return 1

        learnings = _extract_learnings(text, max_items=15)
        if not learnings:
            print("No candidate learnings found heuristically.")
            print("Fallback: run `simba codex-extract` without --run for manual prompt.")
            return 1

        daemon = "http://localhost:8741"
        stored = 0
        duplicates = 0
        errors = 0

        for mem in learnings:
            payload = {
                "type": mem["type"],
                "content": mem["content"],
                "context": mem["context"],
                "confidence": mem["confidence"],
                "sessionSource": session_id,
                "projectPath": project_path,
            }
            try:
                resp = httpx.post(f"{daemon}/store", json=payload, timeout=10.0)
                resp.raise_for_status()
                body = resp.json()
                if body.get("status") == "stored":
                    stored += 1
                elif body.get("status") == "duplicate":
                    duplicates += 1
                else:
                    errors += 1
            except (httpx.HTTPError, ValueError):
                errors += 1

        print(
            f"[codex] extract run complete: candidates={len(learnings)} "
            f"stored={stored} duplicate={duplicates} errors={errors}"
        )
    else:
        print("Use this prompt with Codex (or the `memories-learn` skill):")
        print("---")
        print(
            f"Read transcript `{transcript}` and extract 5-15 high-value learnings. "
            "Store each learning to semantic memory using:"
        )
        print(
            "curl -X POST http://localhost:8741/store "
            '-H "Content-Type: application/json" '
            f"-d '{{\"type\":\"<TYPE>\",\"content\":\"<LEARNING>\","
            f"\"context\":\"<CONTEXT>\",\"confidence\":<SCORE>,"
            f"\"sessionSource\":\"{session_id}\",\"projectPath\":\"{project_path}\"}}'"
        )
        print(
            "Types: WORKING_SOLUTION, GOTCHA, PATTERN, DECISION, FAILURE, PREFERENCE."
        )
        print("---")

    if mark_done:
        meta["status"] = "extracted"
        latest = pathlib.Path.home() / ".claude" / "transcripts" / "latest.json"
        target = latest.resolve() if latest.is_symlink() else latest
        target.write_text(json.dumps(meta, indent=2))
        print(f"Updated extraction status to 'extracted' in {target}")

    return 0


def _cmd_codex_recall(args: list[str]) -> int:
    """Recall memories for a query via the memory daemon."""
    if not args:
        print("Usage: simba codex-recall <query text>", file=sys.stderr)
        return 1

    query = " ".join(args).strip()
    if not query:
        print("Usage: simba codex-recall <query text>", file=sys.stderr)
        return 1

    import simba.hooks._memory_client

    memories = simba.hooks._memory_client.recall_memories(
        query,
        project_path=str(pathlib.Path.cwd()),
    )
    if not memories:
        print("[codex] recall: no memories")
        return 0

    print(f"[codex] recall: {len(memories)} memories")
    for m in memories:
        mtype = m.get("type", "UNKNOWN")
        sim = m.get("similarity", 0.0)
        content = str(m.get("content", "")).strip()
        print(f"- [{mtype}] ({sim:.2f}) {content}")
    return 0


def _parse_opt_value(args: list[str], key: str) -> str | None:
    """Parse `--key value` from args."""
    if key not in args:
        return None
    idx = args.index(key)
    if idx + 1 >= len(args):
        return None
    return args[idx + 1]


def _cmd_codex_finalize(args: list[str]) -> int:
    """Run end-of-task checks equivalent to the Stop hook."""
    response = _parse_opt_value(args, "--response") or ""
    response_file = _parse_opt_value(args, "--response-file")
    transcript = _parse_opt_value(args, "--transcript")

    if response_file:
        try:
            response = pathlib.Path(response_file).read_text()
        except OSError as exc:
            print(f"Failed to read --response-file: {exc}", file=sys.stderr)
            return 1

    if not transcript:
        meta = _latest_transcript_metadata()
        if meta:
            transcript = meta.get("transcript_path", "")

    import simba.guardian.check_signal
    import simba.tailor.hook

    if response:
        signal_result = simba.guardian.check_signal.main(
            response=response, cwd=pathlib.Path.cwd()
        )
        if signal_result:
            print(signal_result)
        else:
            print("[codex] signal check: ok ([✓ rules] present)")
    else:
        print("[codex] signal check: skipped (no response provided)")

    if transcript:
        simba.tailor.hook.process_hook(
            json.dumps(
                {
                    "transcript_path": transcript,
                    "cwd": str(pathlib.Path.cwd()),
                }
            )
        )
        print(f"[codex] reflection capture: processed {transcript}")
    else:
        print("[codex] reflection capture: skipped (no transcript found)")

    return 0


def _cmd_codex_automation(args: list[str]) -> int:
    """Print a suggested Codex automation directive for Simba checks."""
    del args
    cwd = str(pathlib.Path.cwd())
    print(
        "::automation-update{mode=\"suggested create\" "
        "name=\"Simba Codex Health\" "
        "prompt=\"Run simba codex-status and report whether extraction is pending "
        "or memory daemon is down. If pending extraction exists, include the exact "
        "simba codex-extract command in the result.\" "
        "rrule=\"FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9;BYMINUTE=0\" "
        f"cwds=\"{cwd}\" status=\"ACTIVE\"}}"
    )
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
    elif cmd == "codex-install":
        sys.exit(_cmd_codex_install(rest))
    elif cmd == "codex-status":
        sys.exit(_cmd_codex_status(rest))
    elif cmd == "codex-extract":
        sys.exit(_cmd_codex_extract(rest))
    elif cmd == "codex-recall":
        sys.exit(_cmd_codex_recall(rest))
    elif cmd == "codex-finalize":
        sys.exit(_cmd_codex_finalize(rest))
    elif cmd == "codex-automation":
        sys.exit(_cmd_codex_automation(rest))
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
