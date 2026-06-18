"""UserPromptSubmit hook — CORE extraction + memory recall.

Reads stdin JSON with user prompt, extracts CORE blocks from CLAUDE.md,
queries memory daemon for relevant memories, outputs combined context.
"""

from __future__ import annotations

import contextlib
import pathlib
import sys

import simba.guardian.extract_core
import simba.hooks._memory_client
import simba.hooks.context_lanes
import simba.hooks.recall_triage
import simba.search.rag_context
from simba.harness.core import CanonicalResult


def _rlm_pointer_context(memories: list[dict], cwd_str: str | None) -> str:
    """Return an <rlm-pointers> block when rlm.inject_pointers is enabled.

    Reuses the memories already recalled this turn (no second recall) and
    surfaces navigable transcripts so the agent knows it can rlm_grep/rlm_peek
    them for lossless detail. Never raises into the hook.

    TODO(rlm): reusing the turn's recall (top-N at the hook's higher similarity
    bar) makes the nudge sparse — it only fires when a top hit is navigable. To
    surface pointers more reliably, do a dedicated wider route() here (top-5 at
    ~0.35) at the cost of one extra recall per prompt. Deferred.
    """
    import simba.config
    import simba.rlm.config  # registers the "rlm" section
    import simba.rlm.recall

    if not simba.config.load("rlm").inject_pointers:
        return ""
    pointers = simba.rlm.recall.pointers_from_memories(memories, cwd_str)
    nav = [p for p in pointers if p.available]
    if not nav:
        return ""
    lines = [
        "<rlm-pointers>",
        "Lossless transcripts available — call rlm_grep/rlm_peek on these ids "
        "if the recalled snippets aren't enough:",
    ]
    lines += [f"  - {p.transcript_id} :: {p.snippet[:70]}" for p in nav[:3]]
    lines.append("</rlm-pointers>")
    return "\n".join(lines)


def _cfg():
    """Load the hooks config section (registers it on first access)."""
    import simba.config
    import simba.hooks.config

    _ = simba.hooks.config  # ensure "hooks" section is registered
    return simba.config.load("hooks")


def _intent_prime(prompt: str, cwd_str: str | None, cfg, session_id: str) -> str:
    """Intent-primed doctrine injection (spec 28 Phase B). Default-OFF → "".

    When ``hooks.intent_priming_enabled`` is on, classify the prompt against the
    project's doctrine triggers (cheap embedding match via the daemon's loaded
    embedder — no LLM on the hot path) and return an ``<intent-priming>`` block
    naming the matched doctrine + applicable gates. A matched risk-tier doctrine
    additionally arms the preflight mandate for the turn (so the PreToolUse gate
    fires) and appends the mandate instruction. Fail-open: any error returns ""
    (priming is advisory). OFF → byte-identical to today (no doctrine load).
    """
    if not getattr(cfg, "intent_priming_enabled", False):
        return ""
    if not prompt or not cwd_str:
        return ""
    try:
        import simba.doctrine.priming
        import simba.doctrine.store
        import simba.hooks._memory_client

        doctrines = simba.doctrine.store.list_doctrines(
            project_path=cwd_str, cwd=pathlib.Path(cwd_str)
        )
        if not doctrines:
            return ""
        result = simba.doctrine.priming.prime(
            prompt,
            doctrines=doctrines,
            embed_fn=simba.hooks._memory_client.embed_text,
            min_similarity=getattr(cfg, "intent_priming_min_similarity", 0.55),
            max_doctrines=getattr(cfg, "intent_priming_max_doctrines", 3),
        )
    except Exception:
        return ""
    if not result.text:
        return ""

    parts = [result.text]
    # MANDATE: when a risk-tier doctrine was primed and the mandate is on, arm the
    # gate for this turn and tell the agent its first action is `simba preflight`.
    if result.risk_primed and getattr(cfg, "preflight_mandate_enabled", False):
        with contextlib.suppress(Exception):
            import simba.guardian.preflight_flag

            simba.guardian.preflight_flag.arm_mandate(session_id)
        sid = f" --session {session_id}" if session_id else ""
        parts.append(
            "<preflight-mandate>\n"
            f"Your first action this turn is `simba preflight{sid} "
            '"<your task>"` — it surfaces the doctrine + applicable rules and is '
            "required before any mutating tool (the gate blocks otherwise).\n"
            "</preflight-mandate>"
        )
    return "\n\n".join(parts)


def _task_snapshot_context(cwd_str: str | None, cfg, session_id: str) -> str:
    """Return the compact active task snapshot for this project/session."""
    if not getattr(cfg, "task_snapshot_injection_enabled", True):
        return ""
    if not cwd_str:
        return ""
    try:
        import simba.db
        import simba.task_snapshot as snapshots

        cwd = pathlib.Path(cwd_str)
        project_path = str(cwd.resolve())
        with simba.db.connect(cwd):
            row = snapshots.latest(project_path=project_path, session_id=session_id)
            if row is None and session_id:
                row = snapshots.latest(project_path=project_path)
            return snapshots.render(row) if row is not None else ""
    except Exception:
        return ""


def _reset_turn_flags(session_id: str) -> None:
    """Clear the per-turn preflight + mandate flags at the turn boundary (spec 28).

    UserPromptSubmit is the start of a turn, so any preflight/arm flag from the
    previous turn is stale. Fail-soft (best-effort tempfile unlinks).
    """
    if not session_id:
        return
    with contextlib.suppress(Exception):
        import simba.guardian.preflight_flag

        simba.guardian.preflight_flag.reset_preflight(session_id)
        simba.guardian.preflight_flag.reset_mandate(session_id)


def _engagement_marker(cfg, memories: list[dict], session_id: str) -> str:
    """Tier-1 engagement ledger (spec 27): the simba-EMITTED ``🦁☑`` line.

    Built deterministically from THIS turn's recall (count + top similarity), so
    even a zero-recall turn emits ``🦁☑ idle``. Records the ledger for the Stop
    echo-verify. Default-OFF → "" (byte-identical to today). Fail-soft.
    """
    if not getattr(cfg, "engagement_marker_enabled", False):
        return ""
    try:
        import simba.hooks.engagement

        top = float(memories[0].get("similarity", 0.0)) if memories else 0.0
        ledger = simba.hooks.engagement.prompt_ledger(
            memory_count=len(memories), top_similarity=top
        )
    except Exception:
        return ""
    with contextlib.suppress(Exception):
        import simba.guardian.engagement_flag

        simba.guardian.engagement_flag.reset_engagement(session_id)
        simba.guardian.engagement_flag.record_engagement(session_id, ledger=ledger)
    # The echo contract rides with the marker (config-gated, so off ⇒ byte-
    # identical). The bare ``ledger`` line is what Stop verifies the echo of.
    return (
        "<engagement-marker>\n"
        f"{ledger}\n"
        "Echo the 🦁☑ line above verbatim in your response so this simba "
        "interaction is auditable.\n"
        "</engagement-marker>"
    )


def _should_inject_core(cfg, session_id: str) -> bool:
    """Decide whether to (re)inject the CORE block this prompt (spec 25).

    When the ``guardian_signal_gated`` lever is OFF (default), always inject —
    byte-identical to today's behavior. When ON, defer to the per-session signal
    flag (fail-open: any error → inject, never silently drop the rules).
    """
    if not getattr(cfg, "guardian_signal_gated", False):
        return True
    try:
        import simba.guardian.signal_flag

        return simba.guardian.signal_flag.should_inject(session_id)
    except Exception:
        return True


def run(hook_input: dict) -> CanonicalResult:
    """Run the UserPromptSubmit hook pipeline. Returns a CanonicalResult."""
    prompt = hook_input.get("prompt", "")
    cwd_str = hook_input.get("cwd")
    session_id = hook_input.get("session_id", "")
    # Path derives from payload only \u2014 dispatch may run in the daemon process
    # whose own cwd differs from the agent's.
    cwd = pathlib.Path(cwd_str) if cwd_str else None

    cfg = _cfg()
    lanes: list[simba.hooks.context_lanes.ContextLane] = []

    # 0. Turn boundary (spec 28): clear last turn's per-turn preflight/mandate flags
    #    so this turn starts un-armed. No-op (and no output change) unless either
    #    intent-priming or the mandate is enabled.
    if getattr(cfg, "intent_priming_enabled", False) or getattr(
        cfg, "preflight_mandate_enabled", False
    ):
        _reset_turn_flags(session_id)

    # 1. Guardian: extract CORE blocks from CLAUDE.md.
    #    Only when the payload carried a cwd — extract_core.main(cwd=None) falls
    #    back to Path.cwd(), which inside the daemon is the wrong project.
    #    When guardian_signal_gated is ON (spec 25), skip the block while the
    #    rules are still present (prior response carried [✓ rules]); re-inject
    #    when they've decayed / on the first prompt. Fail-open: error → inject.
    core_blocks = ""
    if cwd is not None and _should_inject_core(cfg, session_id):
        core_blocks = simba.guardian.extract_core.main(cwd=cwd)
    if core_blocks:
        lanes.append(
            simba.hooks.context_lanes.ContextLane(
                "guardian",
                core_blocks,
                getattr(cfg, "context_lane_guardian_chars", 12000),
                protected=True,
            )
        )

    task_ctx = _task_snapshot_context(cwd_str, cfg, session_id)
    if task_ctx:
        lanes.append(
            simba.hooks.context_lanes.ContextLane(
                "task",
                task_ctx,
                getattr(cfg, "context_lane_task_chars", 800),
            )
        )

    triage = None
    retrieval_allowed = True
    if prompt and len(prompt) >= cfg.prompt_min_length and getattr(
        cfg, "recall_triage_enabled", False
    ):
        triage = simba.hooks.recall_triage.classify(prompt)
        retrieval_allowed = triage.should_retrieve
        if getattr(cfg, "recall_triage_emit_diagnostics", False):
            lanes.append(
                simba.hooks.context_lanes.ContextLane(
                    "diagnostics",
                    simba.hooks.recall_triage.render(triage),
                    getattr(cfg, "context_lane_diagnostics_chars", 800),
                )
            )

    # 2. Memory: recall relevant memories using prompt
    memories: list[dict] = []
    if retrieval_allowed and prompt and len(prompt) >= cfg.prompt_min_length:
        project_path = str(cwd) if cwd_str else None
        memories = simba.hooks._memory_client.recall_memories(
            prompt, project_path=project_path, min_similarity=cfg.prompt_min_similarity
        )
        formatted = simba.hooks._memory_client.format_memories(
            memories, source="user-prompt", query=prompt
        )
        if formatted:
            lanes.append(
                simba.hooks.context_lanes.ContextLane(
                    "recall",
                    formatted,
                    getattr(cfg, "context_lane_recall_chars", 4000),
                )
            )

    # 2b. Intent priming (spec 28): prime matched doctrine + applicable gates from
    #     the stated intent. Default-OFF → "" (byte-identical to today).
    prime_ctx = _intent_prime(prompt, cwd_str, cfg, session_id)
    if prime_ctx:
        lanes.append(
            simba.hooks.context_lanes.ContextLane(
                "doctrine",
                prime_ctx,
                getattr(cfg, "context_lane_doctrine_chars", 2000),
            )
        )

    # 3. Search: project memory + QMD context
    if (
        retrieval_allowed
        and cwd is not None
        and prompt
        and len(prompt) >= cfg.prompt_min_length
    ):
        try:
            search_ctx = simba.search.rag_context.build_context(prompt, cwd)
            if search_ctx:
                lanes.append(
                    simba.hooks.context_lanes.ContextLane(
                        "rag",
                        search_ctx,
                        getattr(cfg, "context_lane_rag_chars", 2500),
                    )
                )
        except Exception:
            pass

    # 4. RLM: surface navigable transcript pointers (opt-in via rlm.inject_pointers)
    if memories:
        with contextlib.suppress(Exception):
            rlm_ctx = _rlm_pointer_context(memories, cwd_str)
            if rlm_ctx:
                lanes.append(
                    simba.hooks.context_lanes.ContextLane(
                        "rlm",
                        rlm_ctx,
                        getattr(cfg, "context_lane_rlm_chars", 1500),
                    )
                )

    # 5. Engagement marker (spec 27): the simba-EMITTED 🦁☑ ledger of what this
    #    turn surfaced. Leads the injected context so it is the glanceable first
    #    line. Default-OFF → "" (byte-identical to today).
    marker = _engagement_marker(cfg, memories, session_id)
    if marker:
        lanes.insert(
            0,
            simba.hooks.context_lanes.ContextLane(
                "diagnostics",
                marker,
                getattr(cfg, "context_lane_diagnostics_chars", 800),
                protected=True,
            ),
        )

    rendered = simba.hooks.context_lanes.render(
        lanes, enabled=getattr(cfg, "context_lanes_enabled", False)
    )
    combined = rendered.text
    if combined:
        tokens = len(combined) // 4
        tags = f"~{tokens} tokens"
        if core_blocks:
            tags += " | \u2713 rules"
        combined += f"\n[simba: {tags}]"
        print(f"[simba: {tags}]", file=sys.stderr)
    return CanonicalResult(additional_context=combined, memory_count=len(memories))


def main(hook_input: dict) -> str:
    """Run the UserPromptSubmit hook and render the Claude/Codex envelope."""
    import simba.harness.adapters.claude as claude

    return claude.render("UserPromptSubmit", run(hook_input))
