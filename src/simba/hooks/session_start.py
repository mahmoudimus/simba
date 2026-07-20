"""SessionStart hook — combined daemon health, tailor context, memory status.

Reads stdin JSON, checks memory daemon health (auto-starts if needed),
gathers tailor session context, outputs combined additionalContext.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time

import httpx

import simba.config
import simba.db
import simba.guardian.signal_flag
import simba.hooks._memory_client
import simba.memory.config
import simba.search.project_memory
import simba.tailor.session_start
from simba.harness.core import CanonicalResult


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


def _daemon_log_path(cwd: pathlib.Path | None) -> pathlib.Path:
    """Resolve ``.simba/memory/daemon.log`` for the project.

    Reuses ``simba.db.get_db_path``'s repo-root-aware resolution (its
    ``.parent`` is ``.simba``) rather than inventing a second path
    convention --- see ``_auto_start_daemon`` for why this matters.
    """
    return simba.db.get_db_path(cwd).parent / "memory" / "daemon.log"


def _auto_start_daemon(cwd: pathlib.Path | None = None) -> bool:
    """Attempt to start the memory daemon and poll until healthy.

    stdout/stderr go to an append-mode ``.simba/memory/daemon.log`` ---
    never ``DEVNULL``. A spawned daemon that hits a startup exception, or
    later a failed ``POST /restart`` (see routes.py's
    ``_run_restart_sequence``), must leave SOME trace: discarding both
    streams is exactly how those failures went unnoticed live until sampled
    via ``lsof``. Deliberately simple append, no rotation --- log growth is
    bounded by daemon verbosity (one INFO line per request/heartbeat), not
    by anything unbounded from outside.
    """
    log_path = _daemon_log_path(cwd)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a")  # noqa: SIM115 -- closed in `finally` below
    except OSError:
        return False
    # Never mutate this (the hook's own) process's environment --- always
    # spawn the daemon against a copy. When `memory.malloc_stack_logging` is
    # True, arm the copy with `MallocStackLogging=lite` (2026-07-19: a
    # 16.7GB RSS burst had no attributable stacks because the daemon that
    # ended up serving was an unarmed hook auto-start like this one) --- the
    # env can only be set at spawn time, never injected after the fact.
    env = os.environ.copy()
    if simba.memory.config.resolve_malloc_stack_logging(cwd):
        env["MallocStackLogging"] = "lite"
    try:
        subprocess.Popen(
            ["uv", "run", "python", "-m", "simba.memory.server"],
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
            env=env,
        )
    except (FileNotFoundError, OSError):
        return False
    finally:
        # The child's dup'd fd (from Popen/fork+exec) is independent of this
        # handle --- closing it here doesn't touch the daemon's own copy.
        log_file.close()

    cfg = _hooks_cfg()
    for _ in range(cfg.poll_attempts):
        time.sleep(cfg.poll_interval)
        if _check_health() is not None:
            return True
    return False


def _curated_import_nudge(cwd: pathlib.Path | None) -> str:
    """Nudge when the curated re-import bridge (spec 33 R4) is stale.

    ``simba memory import-curated --run`` stamps ``.simba/curated-import.json``
    under the target project with the curated dir + import time. If that
    dir's MEMORY.md has since been touched, the daemon's mirror of the
    curated layer is out of date. Missing marker, missing MEMORY.md, or
    corrupt/short JSON all resolve to "" — this must never raise, since it is
    called unconditionally from ``_lifecycle_nudges``.
    """
    if cwd is None:
        return ""
    marker_path = pathlib.Path(cwd) / ".simba" / "curated-import.json"
    try:
        marker = json.loads(marker_path.read_text())
        curated_dir = marker["dir"]
        last_import_at = float(marker["last_import_at"])
        memory_md_mtime = (pathlib.Path(curated_dir) / "MEMORY.md").stat().st_mtime
    except (OSError, ValueError, KeyError, TypeError):
        return ""
    if memory_md_mtime <= last_import_at:
        return ""
    return (
        "[Lifecycle] curated memory changed since last import — run: "
        f"simba memory import-curated --dir {curated_dir} --run"
    )


def _pending_rule_candidates_line(cwd: pathlib.Path | None) -> str:
    """Pending arc-derived redirect-rule candidates (redirect/candidates.py),
    awaiting `simba rule promote` review.

    Unlike the rest of ``_lifecycle_nudges``, this is local-only -- the
    ``rule_candidate`` table lives in the project's own ``.simba/simba.db``,
    not the memory daemon's LanceDB, so it needs no HTTP round trip and stays
    correct even when the daemon (and therefore ``/digest``) is down. Fail-
    soft on any DB/filesystem trouble, same as everything else here.
    """
    if cwd is None:
        return ""
    try:
        import simba.redirect.candidates as candidates

        count = candidates.count_pending(cwd=pathlib.Path(cwd))
    except Exception:
        return ""
    if not count:
        return ""
    return f"{count} pending rule candidate(s) — `simba rule promote`"


def _lifecycle_nudges(cfg, cwd: pathlib.Path | None = None) -> str:
    """One-glance lifecycle state at session start (spec 33 Phase 5).

    A line for the latest maintenance heartbeat (so shadow-mode results are
    seen, not buried in daemon logs), one for promotion candidates awaiting
    review, and one for a stale curated re-import (spec 33 R4). Two
    sub-second local GETs plus a couple of local stat() calls; default-off ⇒
    "" and zero HTTP/filesystem; fail-soft on any daemon or filesystem
    trouble.
    """
    if not getattr(cfg, "session_start_lifecycle_nudges", False):
        return ""
    lines: list[str] = []
    base = simba.hooks._memory_client.daemon_url()

    # Preferred: one /digest call (spec 33 v2) — heartbeat + promotion inbox +
    # supersession pendings + knowledge gaps. Falls back to the two-call
    # /stats + /promotions path against an older daemon.
    digest: dict | None = None
    try:
        resp = httpx.get(f"{base}/digest", timeout=1.0)
        if resp.status_code == 200:
            digest = resp.json()
    except (httpx.HTTPError, ValueError):
        digest = None
    if digest is not None:
        heartbeat = digest.get("heartbeat") or {}
        if heartbeat:
            decay = heartbeat.get("decay") or {}
            mode = "apply" if heartbeat.get("apply") else "shadow"
            lines.append(
                f"[Lifecycle] last maintenance {heartbeat.get('at', '?')} "
                f"({mode}): decay updated={decay.get('updated', 0)} "
                f"dormant={decay.get('newly_dormant', 0)}"
            )
        inbox: list[str] = []
        promotions = int((digest.get("promotions") or {}).get("total", 0))
        if promotions:
            inbox.append(
                f"{promotions} promotion candidate(s) — `simba memory promote`"
            )
        pending = int((digest.get("supersessions") or {}).get("pending", 0))
        if pending:
            inbox.append(
                f"{pending} pending supersession(s) — `simba memory supersession`"
            )
        gap_count = int((digest.get("gaps") or {}).get("total", 0))
        if gap_count:
            inbox.append(f"{gap_count} knowledge gap(s) — `simba memory gaps`")
        rule_candidates_line = _pending_rule_candidates_line(cwd)
        if rule_candidates_line:
            inbox.append(rule_candidates_line)
        if inbox:
            lines.append("[Lifecycle] inbox: " + " | ".join(inbox))

        # Graduation-readiness nudge (spec 33 Part 8 rule R1): surface ONLY
        # when the DATA criteria are met AND maintenance_apply hasn't
        # already flipped — `heartbeat.apply` mirrors the CURRENT config
        # value (set by run_maintenance every pass), so it doubles as the
        # "already applying" check without a second daemon round-trip. Once
        # applying, this line would be stale noise (a human already turned
        # the lever); progress otherwise lives in the digest, not here.
        graduation = digest.get("graduation") or {}
        if graduation.get("ready") and not heartbeat.get("apply"):
            lines.append(
                "[Lifecycle] maintenance_apply data criteria MET "
                f"({graduation.get('signalDays', 0):.0f}d signals, "
                f"{graduation.get('usedRatio', 0) * 100:.0f}% fired rules "
                "used) — run bench guards, then: "
                "simba config set memory.maintenance_apply true"
            )
    else:
        try:
            resp = httpx.get(f"{base}/stats", timeout=1.0)
            if resp.status_code == 200:
                last = resp.json().get("lastMaintenance") or {}
                if last:
                    decay = last.get("decay") or {}
                    mode = "apply" if last.get("apply") else "shadow"
                    lines.append(
                        f"[Lifecycle] last maintenance {last.get('at', '?')} "
                        f"({mode}): decay updated={decay.get('updated', 0)} "
                        f"dormant={decay.get('newly_dormant', 0)}"
                    )
        except (httpx.HTTPError, ValueError):
            pass
        try:
            resp = httpx.get(
                f"{base}/promotions/candidates", params={"limit": 1}, timeout=1.0
            )
            if resp.status_code == 200:
                total = int(resp.json().get("total", 0))
                if total:
                    lines.append(
                        f"[Lifecycle] {total} promotion candidate(s) — "
                        "`simba memory promote` to review"
                    )
        except (httpx.HTTPError, ValueError, TypeError):
            pass
        rule_candidates_line = _pending_rule_candidates_line(cwd)
        if rule_candidates_line:
            lines.append(f"[Lifecycle] {rule_candidates_line}")

    curated_line = _curated_import_nudge(cwd)
    if curated_line:
        lines.append(curated_line)

    return "\n".join(lines)


def _check_pending_extraction(session_id: str, cwd: str = "") -> str:
    """Return extraction instructions for THIS project's pending transcript.

    Project-scoped (via ``transcripts.find_pending``) — never the global
    ``latest.json``, which is overwritten by whichever session compacted last
    across all projects, so it cross-wired sessions (a session in project A would
    be told to extract project B's transcript, mis-attributing the memories).
    """
    import simba.transcripts as _transcripts

    if not cwd:
        return ""
    metadata = _transcripts.find_pending(cwd)
    if not metadata:
        return ""

    transcript_path = metadata.get("transcript_path", "")
    export_session = metadata.get("session_id", session_id)
    cwd = metadata.get("project_path", cwd)  # the resolved (matching) project
    url = simba.hooks._memory_client.daemon_url()
    maxlen = simba.memory.config.resolve_max_content_length(pathlib.Path(cwd))

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
        'simba memory store --type "<TYPE>" --content "<LEARNING>" '
        '--context "<CONTEXT>" --confidence <SCORE> '
        f'--session-source "{export_session}" '
        f'--project-path "{cwd}"\n\n'
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
        f"- Keep content under {maxlen} characters, use context for details\n"
        "- Preserve proper nouns, file paths, and identifiers verbatim — "
        "never replace them with generic words\n"
        "- Preserve numeric precision: keep exact values exact; never weaken "
        "an exact number to a range or approximation\n"
        '- Resolve relative dates to absolute ones (e.g. "yesterday" -> the '
        "actual date)\n"
        "---\n"
        "</learning-extraction-required>"
    )


def run(hook_input: dict) -> CanonicalResult:
    """Run the SessionStart hook pipeline. Returns a CanonicalResult."""
    cwd_str = hook_input.get("cwd")
    # Path derives from payload only — never the process cwd (dispatch may run
    # in the daemon). gather_context / get_db_path accept None and handle it.
    cwd = pathlib.Path(cwd_str) if cwd_str else None
    session_id = hook_input.get("session_id", "")

    # Reset the rules-signal flag (spec 25): a fresh session has no prior
    # response, so the first prompt must re-inject the CORE block. Fail-soft.
    if session_id:
        import contextlib

        with contextlib.suppress(Exception):
            simba.guardian.signal_flag.reset_signal(session_id)

    parts: list[str] = []

    # 1. Memory daemon health check + auto-start
    health = _check_health()
    if health is None:
        started = _auto_start_daemon(cwd)
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

        # Lifecycle nudges (spec 33 Phase 5): heartbeat + promotion inbox +
        # stale curated re-import (Phase 5 + R4). Default-off → "" (byte-
        # identical). Fail-soft.
        with contextlib.suppress(Exception):
            nudges = _lifecycle_nudges(_hooks_cfg(), cwd=cwd)
            if nudges:
                parts.append(nudges)

    # 2/3. Project-scoped context — only when the payload carried a cwd.
    #      gather_context / get_db_path / get_stats all fall back to Path.cwd()
    #      on None, which inside the daemon is the wrong project.
    if cwd is not None:
        # 2. Tailor session context (time, git, marks)
        tailor_ctx = simba.tailor.session_start.gather_context(cwd=cwd)
        if tailor_ctx:
            parts.append(tailor_ctx)

        # 3. Project memory status
        try:
            if simba.db.get_db_path(cwd).exists():
                stats = simba.search.project_memory.get_stats(cwd)
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
        extraction = _check_pending_extraction(session_id, cwd=cwd_str or "")
        if extraction:
            parts.append(extraction)

    combined = "\n\n".join(parts)
    return CanonicalResult(additional_context=combined)


def main(hook_input: dict) -> str:
    """Run the SessionStart hook and render the Claude/Codex envelope."""
    import simba.harness.adapters.claude as claude

    return claude.render("SessionStart", run(hook_input))
