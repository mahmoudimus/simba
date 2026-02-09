"""SessionStart hook — combined daemon health, tailor context, memory status.

Reads stdin JSON, checks memory daemon health (auto-starts if needed),
gathers tailor session context, outputs combined additionalContext.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import time

import httpx

import simba.config
import simba.db
import simba.hooks._memory_client
import simba.search.project_memory
import simba.tailor.session_start


def _hooks_cfg():
    import simba.hooks.config

    return simba.config.load("hooks")


def _check_health() -> dict | None:
    """Check daemon health. Returns health dict or None if unreachable."""
    try:
        resp = httpx.get(
            f"{simba.hooks._memory_client.daemon_url()}/health",
            timeout=_hooks_cfg().health_timeout,
        )
        if resp.status_code == 200:
            return resp.json()
    except (httpx.HTTPError, ValueError):
        pass
    return None


def _auto_start_daemon() -> bool:
    """Attempt to start the memory daemon and poll until healthy."""
    try:
        subprocess.Popen(
            ["uv", "run", "python", "-m", "simba.memory.server"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (FileNotFoundError, OSError):
        return False

    cfg = _hooks_cfg()
    for _ in range(cfg.poll_attempts):
        time.sleep(cfg.poll_interval)
        if _check_health() is not None:
            return True
    return False


def _check_pending_extraction(session_id: str) -> str:
    """Check if there's a pending transcript extraction and return instructions."""
    transcripts_dir = pathlib.Path.home() / ".claude" / "transcripts"
    latest = transcripts_dir / "latest.json"
    if not latest.exists():
        return ""

    try:
        metadata = json.loads(latest.read_text())
    except (json.JSONDecodeError, OSError):
        return ""

    if metadata.get("status") != "pending_extraction":
        return ""

    transcript_path = metadata.get("transcript_path", "")
    export_session = metadata.get("session_id", session_id)
    url = simba.hooks._memory_client.daemon_url()

    return (
        "\n<learning-extraction-required>\n"
        "Session transcript was exported before compaction and needs "
        "learning extraction.\n\n"
        f"TRANSCRIPT: {transcript_path}\n"
        f"SESSION: {export_session}\n"
        f"DAEMON: {url}\n\n"
        "You MUST dispatch a sub-agent using the Task tool with this prompt:\n\n"
        "---\n"
        f"Read the transcript at {transcript_path} and extract learnings "
        "to store in the semantic memory database.\n\n"
        "For each learning found, store it by running:\n"
        f"curl -X POST {url}/store "
        '-H "Content-Type: application/json" '
        '-d \'{"type": "<TYPE>", "content": "<LEARNING>", '
        '"context": "<CONTEXT>", "confidence": <SCORE>, '
        f'"sessionSource": "{export_session}"}}\'\n\n'
        "LEARNING TYPES:\n"
        "- WORKING_SOLUTION: Commands, code, or approaches that worked\n"
        '- GOTCHA: Traps, counterintuitive behaviors, "watch out for this"\n'
        "- PATTERN: Recurring architectural decisions or workflows\n"
        "- DECISION: Explicit design choices with reasoning\n"
        "- FAILURE: What didn't work and why\n"
        "- PREFERENCE: User's stated preferences\n\n"
        "RULES:\n"
        "- Be specific - include actual commands, paths, error messages\n"
        "- Confidence 0.95+ for explicitly confirmed, 0.85+ for strong evidence\n"
        "- Skip generic programming knowledge Claude already knows\n"
        "- Focus on user-specific infrastructure, preferences, workflows\n"
        "- Keep content under 200 characters, use context for details\n"
        "---\n"
        "</learning-extraction-required>"
    )


def main(hook_input: dict) -> str:
    """Run the SessionStart hook pipeline. Returns JSON output string."""

    cwd_str = hook_input.get("cwd")
    cwd = pathlib.Path(cwd_str) if cwd_str else pathlib.Path.cwd()
    session_id = hook_input.get("session_id", "")

    parts: list[str] = []

    # 1. Memory daemon health check + auto-start
    health = _check_health()
    if health is None:
        started = _auto_start_daemon()
        if started:
            health = _check_health()

    if health:
        model = health.get("embeddingModel", "unknown")
        count = health.get("memoryCount", 0)
        parts.append(
            f"[Semantic Memory] Active: {count} memories available (model: {model})"
        )

        # Fire-and-forget: trigger a sync cycle so new DB rows get indexed.
        import contextlib

        with contextlib.suppress(httpx.HTTPError, ValueError):
            httpx.post(
                f"{simba.hooks._memory_client.daemon_url()}/sync",
                timeout=1.0,
            )

    # 2. Tailor session context (time, git, marks)
    tailor_ctx = simba.tailor.session_start.gather_context(cwd=cwd)
    if tailor_ctx:
        parts.append(tailor_ctx)

    # 3. Project memory status
    try:
        conn = simba.db.get_connection(cwd)
        if conn is not None:
            stats = simba.search.project_memory.get_stats(conn)
            conn.close()
            sessions = stats.get("sessions", 0)
            knowledge = stats.get("knowledge", 0)
            facts = stats.get("facts", 0)
            if sessions or knowledge or facts:
                parts.append(
                    f"[Project Memory] {sessions} sessions, "
                    f"{knowledge} knowledge areas, {facts} facts"
                )
    except Exception:
        pass

    # 4. Check for pending transcript extraction
    if hook_input.get("source") == "compact" or session_id:
        extraction = _check_pending_extraction(session_id)
        if extraction:
            parts.append(extraction)

    combined = "\n\n".join(parts)
    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": combined,
        }
    }
    return json.dumps(output)
