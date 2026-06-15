"""PreToolUse hook — tool-rule checking, thinking-based memory recall, truth DB.

Reads stdin JSON with tool_name, tool_input, and transcript_path.

Pipeline (in order):
1. Context-low warning (bytes since last compaction; once per compaction segment)
2. Tool-rule check (query TOOL_RULE memories matching current tool call)
3. Truth DB check (query proven facts for Bash commands)
4. Memory recall (extract thinking block, query general memories)
"""

from __future__ import annotations

import calendar
import contextlib
import hashlib
import json
import pathlib
import time

import simba.config
import simba.db
import simba.hooks._io
import simba.hooks._kg_client
import simba.hooks._memory_client
import simba.redirect.check

_HASH_CACHE = pathlib.Path("/tmp/claude-memory-hash-cache.json")
_CONTEXT_LOW_FLAG = pathlib.Path("/tmp/claude-context-low-flag.json")
_TOOL_RULE_COUNT_CACHE = pathlib.Path("/tmp/claude-toolrule-count-cache.json")
# Separate dedup cache for the pitfall gate so it fires once per reasoning turn
# without colliding with the general-recall hash cache (shared cache would let one
# consumer suppress the other on the same thinking block).
_PITFALL_DEDUP_CACHE = pathlib.Path("/tmp/claude-pitfall-dedup-cache.json")

_ENABLED_TOOLS = frozenset(
    ["Read", "Grep", "Glob", "Task", "WebSearch", "WebFetch", "Bash"]
)


def _hooks_cfg():
    import simba.hooks.config

    return simba.config.load("hooks")


def _extract_thinking(transcript_path: pathlib.Path) -> str:
    """Extract last thinking block from transcript JSONL."""
    if not transcript_path.exists():
        return ""

    try:
        lines = transcript_path.read_text().strip().split("\n")
    except OSError:
        return ""

    # Read from end to find last assistant thinking
    for line in reversed(lines):
        if not line:
            continue
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        message = entry.get("message", {})
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue

        content = message.get("content", [])
        if not isinstance(content, list):
            continue

        for item in reversed(content):
            if isinstance(item, dict) and item.get("type") == "thinking":
                thinking = item.get("thinking", "")
                return thinking[-_hooks_cfg().thinking_chars :]

    return ""


def _check_dedup(text: str, cache_path: pathlib.Path = _HASH_CACHE) -> bool:
    """Return True if this text was already processed recently (per ``cache_path``)."""
    text_hash = hashlib.md5(text.encode()).hexdigest()

    try:
        cache = json.loads(cache_path.read_text())
        if (
            cache.get("lastHash") == text_hash
            and (time.time() - cache.get("timestamp", 0)) < _hooks_cfg().dedup_ttl
        ):
            return True
    except (json.JSONDecodeError, OSError):
        pass

    return False


def _save_hash(text: str, cache_path: pathlib.Path = _HASH_CACHE) -> None:
    """Save hash to cache file (per ``cache_path``)."""
    text_hash = hashlib.md5(text.encode()).hexdigest()
    with contextlib.suppress(OSError):
        cache_path.write_text(
            json.dumps({"lastHash": text_hash, "timestamp": time.time()})
        )


_COMPACT_MARKER = b'"isCompactSummary":true'


def _post_compaction_tail_bytes(transcript_path: pathlib.Path) -> tuple[int, int]:
    """Bytes of the transcript *since the last compaction* (the live-context proxy).

    The transcript JSONL is append-only, so its total size keeps growing across
    compactions and badly overcounts the in-context window. We return
    ``(tail_bytes, last_compaction_offset)`` where the tail is measured from the
    start of the last ``isCompactSummary`` line to EOF (0 offset = never
    compacted, tail == total).
    """
    try:
        data = transcript_path.read_bytes()
    except OSError:
        return (0, 0)
    total = len(data)
    idx = data.rfind(_COMPACT_MARKER.replace(b" ", b""))
    if idx == -1:
        # tolerate whitespace variants ("isCompactSummary": true)
        loose = data.rfind(b'"isCompactSummary"')
        idx = loose if loose != -1 else -1
    if idx == -1:
        return (total, 0)
    line_start = data.rfind(b"\n", 0, idx) + 1  # 0 if marker is on the first line
    return (total - line_start, line_start)


def _check_context_low(transcript_path: pathlib.Path) -> str | None:
    """Warn when the *live* context (bytes since the last compaction) nears the
    window. Re-arms after each new compaction so it can fire again per segment.

    Threshold is ``hooks.context_low_bytes`` (configurable). Cheap-gated on total
    size — only reads the file once the total could possibly exceed the threshold.
    """
    threshold = _hooks_cfg().context_low_bytes
    try:
        size = transcript_path.stat().st_size
    except OSError:
        return None

    # tail <= total, so if total < threshold the tail can't exceed it (cheap path).
    if size < threshold:
        return None

    tail, offset = _post_compaction_tail_bytes(transcript_path)
    if tail < threshold:
        return None  # a compaction shrank the live context — no false alarm

    # Warn once per (transcript, compaction boundary); a new compaction re-arms.
    try:
        flag = json.loads(_CONTEXT_LOW_FLAG.read_text())
        if (
            flag.get("transcript") == str(transcript_path)
            and flag.get("offset") == offset
        ):
            return None
    except (json.JSONDecodeError, OSError):
        pass

    with contextlib.suppress(OSError):
        _CONTEXT_LOW_FLAG.write_text(
            json.dumps(
                {
                    "transcript": str(transcript_path),
                    "offset": offset,
                    "timestamp": time.time(),
                }
            )
        )

    tail_mb = tail / 1_000_000
    return (
        "<context-low-warning>\n"
        f"~{tail_mb:.1f}MB of transcript since last compaction — "
        "the live context is getting large.\n\n"
        "RECOMMENDED: Prepare for context compaction now.\n"
        "1. Summarize your current work state "
        "(what's done, what's pending)\n"
        "2. Note the current branch, files being modified, "
        "and any in-progress changes\n"
        "3. If there are pending tasks, document them clearly\n"
        "4. The pre-compact hook will automatically export the "
        "transcript for learning extraction\n"
        "</context-low-warning>"
    )


def _within_max_age(created_at: str | None, max_age_days: int) -> bool:
    """True if *created_at* (ISO ``...Z``) is within ``max_age_days`` of now.

    Missing or unparseable timestamps are treated as fresh (kept) so we never
    silently drop a rule just because it lacks a timestamp.
    """
    if not created_at:
        return True
    try:
        parsed = time.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        return True
    age_seconds = time.time() - calendar.timegm(parsed)
    return age_seconds <= max_age_days * 86400


def _project_has_tool_rules(project_id: str, cfg) -> bool:
    """True if the project has >=1 ``TOOL_RULE`` memory (TTL-cached, fail-open).

    Most projects have no learned rules, so the per-tool-call embed+recall the
    rule check would otherwise run is pure waste (a guaranteed miss). We cache the
    project's count for ``cfg.rule_count_ttl`` seconds and skip the recall when it
    is zero. A ``ttl`` of 0 disables the skip. On any uncertainty (the daemon
    can't be reached → count is ``None``) we return ``True`` so a real rule is
    never silently suppressed.
    """
    ttl = getattr(cfg, "rule_count_ttl", 0)
    if not ttl or ttl <= 0:
        return True

    now = time.time()
    cache: dict = {}
    try:
        loaded = json.loads(_TOOL_RULE_COUNT_CACHE.read_text())
        if isinstance(loaded, dict):
            cache = loaded
    except (json.JSONDecodeError, OSError):
        cache = {}

    entry = cache.get(project_id)
    if isinstance(entry, dict) and (now - entry.get("ts", 0)) < ttl:
        return entry.get("count", 0) > 0

    count = simba.hooks._memory_client.count_memories(
        memory_type="TOOL_RULE", project_path=project_id
    )
    if count is None:
        return True  # fail-open: couldn't determine → do the check

    cache[project_id] = {"count": count, "ts": now}
    with contextlib.suppress(OSError, TypeError):
        _TOOL_RULE_COUNT_CACHE.write_text(json.dumps(cache))
    return count > 0


def _check_tool_rules(
    tool_name: str, tool_input: dict, cwd_str: str | None
) -> str | None:
    """Query TOOL_RULE memories matching this tool call."""
    cfg = _hooks_cfg()
    if not cfg.rule_check_enabled:
        return None

    # Build a query from the tool input
    if tool_name == "Bash":
        query = tool_input.get("command", "")[:200]
    elif tool_name in ("Read", "Write", "Edit"):
        query = tool_input.get("file_path", "")
    else:
        return None

    if not query:
        return None

    # Scope to the opaque, worktree-robust project id the learner stores under,
    # so another repo's rules never surface here (and a repo's worktrees share).
    project_id = simba.db.resolve_project_id(pathlib.Path(cwd_str) if cwd_str else None)

    # Skip the embed+recall entirely when the project has no learned rules.
    if not _project_has_tool_rules(project_id, cfg):
        return None

    memories = simba.hooks._memory_client.recall_memories(
        query,
        project_path=project_id,
        min_similarity=cfg.rule_min_similarity,
        max_results=2,
        filters={"types": ["TOOL_RULE"]},
    )
    if not memories:
        return None

    # Recency gate: stale rules (e.g. a "no such file" probe recorded weeks ago
    # against a since-moved path) age out of the warning injection.
    max_age_days = getattr(cfg, "rule_max_age_days", 0)
    if max_age_days and max_age_days > 0:
        memories = [
            m for m in memories if _within_max_age(m.get("createdAt"), max_age_days)
        ]
        if not memories:
            return None

    lines = ["<tool-rule-warning>"]
    for m in memories:
        ctx_str = m.get("context", "{}")
        with contextlib.suppress(json.JSONDecodeError):
            ctx = json.loads(ctx_str)
            correction = ctx.get("correction", "")
            lines.append(f"  WARNING: {m.get('content', '')}")
            if correction:
                lines.append(f"  INSTEAD: {correction}")
    lines.append("</tool-rule-warning>")
    return "\n".join(lines) if len(lines) > 2 else None


def _check_truth_constraints(
    tool_name: str, tool_input: dict, cwd_str: str | None = None
) -> str | None:
    """Check the KG for facts relevant to this tool call (project-scoped)."""
    if tool_name != "Bash":
        return None
    command = tool_input.get("command", "")
    if not command:
        return None
    return simba.hooks._kg_client.query_kg(command, cwd=cwd_str) or None


def _pitfall_llm_client(cfg):
    """Return an available LLM client for violation-mode, or None (→ fallback).

    None whenever the gate isn't in violation mode, the llm subsystem is missing, or
    the client reports unavailable — the orchestrator then takes the configured
    fallback. Never raises (fail-open)."""
    if getattr(cfg, "pitfall_gate_mode", "violation") != "violation":
        return None
    try:
        import simba.llm.client

        client = simba.llm.client.get_client()
        return client if client.available() else None
    except Exception:
        return None


def _check_pitfall(thinking: str, cwd_str: str | None) -> str | None:
    """Pitfall/doctrine enforcement gate: fire a STOP-and-confirm directive when the
    pending move (``thinking``) VIOLATES a stored doctrine/scar/trap.

    Recalls only the doctrine TYPES (``hooks.pitfall_gate_types``) for the project,
    then ``simba.memory.pitfall.pitfall_note`` decides: in violation mode it LLM-checks
    whether the move violates a topically-close doctrine (vs merely sharing its topic);
    with no LLM it falls back per ``hooks.pitfall_gate_fallback``. Fail-open: disabled,
    no thinking, or any failure returns ``None`` — never raises.
    """
    cfg = _hooks_cfg()
    if not getattr(cfg, "pitfall_gate_enabled", False):
        return None
    if not thinking:
        return None

    import simba.memory.pitfall

    types = [t.strip().upper() for t in cfg.pitfall_gate_types.split(",") if t.strip()]
    if not types:
        return None
    memories = simba.hooks._memory_client.recall_memories(
        thinking,
        project_path=cwd_str if cwd_str else None,
        max_results=cfg.pitfall_gate_max_results,
        filters={"types": types},
    )
    directive = simba.memory.pitfall.pitfall_note(
        memories, thinking, cfg=cfg, llm_client=_pitfall_llm_client(cfg)
    )
    return directive or None


def main(hook_input: dict) -> str:
    """Run the PreToolUse hook pipeline. Returns JSON output string."""
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    transcript_path_str = hook_input.get("transcript_path", "")
    cwd_str = hook_input.get("cwd")

    # --- Tool-call redirect (Bash): deny-with-correction or silent rewrite ---
    # Runs first so a redirected command never proceeds to recall/injection.
    if tool_name == "Bash" and isinstance(tool_input, dict):
        decision = simba.redirect.check.check_command(
            tool_input.get("command", ""), cwd_str
        )
        if decision is not None:
            if decision.action == "rewrite":
                return simba.hooks._io.pretool_rewrite(
                    decision.command, decision.reason
                )
            return simba.hooks._io.pretool_deny(decision.reason)

    parts: list[str] = []

    # --- Context-low check (fires for any tool, once per session) ---
    if transcript_path_str:
        warning = _check_context_low(pathlib.Path(transcript_path_str))
        if warning:
            parts.append(warning)

    # --- Tool-rule check (fires before thinking recall) ---
    if tool_input:
        rule_warning = _check_tool_rules(tool_name, tool_input, cwd_str)
        if rule_warning:
            parts.append(rule_warning)

        truth_warning = _check_truth_constraints(tool_name, tool_input, cwd_str)
        if truth_warning:
            parts.append(truth_warning)

    # --- Thinking-based recall + pitfall gate ---
    # General recall stays scoped to _ENABLED_TOOLS. The pitfall gate fires only before
    # MUTATING tools (hooks.pitfall_gate_tools, default Edit/Write/Bash) — it is about
    # "you're about to TAKE an action", and its measured false fires were all on read/
    # search/extract moves. Both paths read the same last thinking block, so extract it
    # once when either is live.
    _hcfg = _hooks_cfg()
    pitfall_tools = {
        t.strip()
        for t in getattr(_hcfg, "pitfall_gate_tools", "Edit,Write,Bash").split(",")
        if t.strip()
    }
    run_pitfall = (
        getattr(_hcfg, "pitfall_gate_enabled", False) and tool_name in pitfall_tools
    )
    if transcript_path_str and (tool_name in _ENABLED_TOOLS or run_pitfall):
        transcript_path = pathlib.Path(transcript_path_str)
        thinking = _extract_thinking(transcript_path)

        # Pitfall/doctrine enforcement gate (own dedup so it fires once per turn and
        # does not collide with general recall's hash cache).
        if (
            run_pitfall
            and thinking
            and not _check_dedup(thinking, _PITFALL_DEDUP_CACHE)
        ):
            directive = _check_pitfall(thinking, cwd_str)
            _save_hash(thinking, _PITFALL_DEDUP_CACHE)
            if directive:
                parts.append(directive)

        # General thinking-block recall (unchanged; only for the enabled tool set).
        if tool_name in _ENABLED_TOOLS and thinking and not _check_dedup(thinking):
            project_path = cwd_str if cwd_str else None
            # Defer the cosine floor to the daemon's intent-aware selection.
            memories = simba.hooks._memory_client.recall_memories(
                thinking,
                project_path=project_path,
            )

            if memories:
                _save_hash(thinking)

            formatted = simba.hooks._memory_client.format_memories(
                memories, source="thinking-block", query=thinking
            )
            if formatted:
                parts.append(formatted)

    if not parts:
        return simba.hooks._io.empty("PreToolUse")

    combined = "\n\n".join(parts)
    tokens = len(combined) // 4
    combined += f"\n[simba: ~{tokens} tokens injected]"
    return simba.hooks._io.context("PreToolUse", combined)
