"""Stop hook — guardian signal check + tailor error capture.

Reads stdin JSON, checks response for [✓ rules] signal marker,
runs tailor error capture pipeline on transcript.
"""

from __future__ import annotations

import contextlib
import json
import pathlib

import simba.config
import simba.guardian.check_signal
import simba.guardian.signal_flag
import simba.memory.continuous
import simba.tailor.hook
from simba.harness.core import CanonicalResult


def _hooks_cfg():
    """Load the hooks config section (registers it on first access)."""
    import simba.hooks.config

    _ = simba.hooks.config  # ensure the "hooks" section is registered
    return simba.config.load("hooks")


def _usage_signal_feedback(response: str, session_id: str, cfg) -> None:
    """Spec 33 Phase 1: convert this turn's injections into use/noise signals.

    Citation (distinctive-term whole-token overlap in the response) → POST
    feedback ``good``; repeat-injected-never-used → ONE weak ``bad`` per
    session. Consumes the per-turn record. Entirely fail-soft; default-off ⇒
    no files read, no POSTs (byte-identical to today).
    """
    if not getattr(cfg, "usage_signals_enabled", False) or not session_id:
        return
    try:
        import simba.hooks.usage_signals as usage_signals

        usage_signals.process_turn_outcome(session_id, response, cfg)
    except Exception:
        pass


def _verify_engagement_echo(response: str, session_id: str, cfg) -> str:
    """Tier-1 echo-verify (spec 27, Phase B): flag a marker-missing turn.

    simba EMITTED a ``🦁☑`` ledger this turn (recorded per session at
    UserPromptSubmit/PreToolUse) and the agent was asked to echo it. If simba
    surfaced activity but the response carries no marker, return a nudge naming the
    ledger the agent should have echoed. Reads then ages out the per-turn record.
    Default-OFF → "" (byte-identical to today). Fail-soft (advisory)."""
    if not getattr(cfg, "engagement_marker_enabled", False) or not session_id:
        return ""
    try:
        import simba.guardian.engagement_flag as ef
        import simba.hooks.engagement as eng

        if not ef.engaged(session_id):
            return ""  # simba did not surface activity → nothing to verify
        ledger = ef.last_ledger(session_id)
        echoed = eng.has_marker(response)
        ef.reset_engagement(session_id)  # per-turn record ages out
        if echoed:
            return ""
        return (
            "⚠️ ENGAGEMENT: simba surfaced activity this turn but your "
            "response did not echo the marker. Echo this ledger so the interaction "
            f"is auditable:\n{ledger}"
        )
    except Exception:
        return ""


def run(hook_input: dict) -> CanonicalResult:
    """Run the Stop hook pipeline. Returns a CanonicalResult."""
    cwd_str = hook_input.get("cwd")
    cwd = pathlib.Path(cwd_str) if cwd_str else None
    session_id = hook_input.get("session_id", "")

    parts: list[str] = []

    cfg = _hooks_cfg()

    # 1. Guardian: check for [✓ rules] signal in response
    response = hook_input.get("response", "")
    if response:
        signal_result = simba.guardian.check_signal.main(response=response, cwd=cwd)
        if signal_result:
            parts.append(signal_result)

    # 1a. Engagement echo-verify (spec 27, Phase B): flag a turn that surfaced
    #     simba activity but did not echo the 🦁☑ marker. Default-OFF → "".
    echo_nudge = _verify_engagement_echo(response, session_id, cfg)
    if echo_nudge:
        parts.append(echo_nudge)

    # 1b. Tier-2 doctrine-verify (spec 27, Phase E): when reasoning_verify is on,
    #     check the finalized message against stored doctrine and BLOCK-to-reconsider
    #     on a violation (the adapter maps a Stop block_reason to decision:block).
    #     Default-OFF → None (Stop stays observe-only, byte-identical). Fail-open.
    #     Compute the decision but DON'T early-return: the capture steps below
    #     (signal flag, tailor, continuous) must still run on a blocked turn, or the
    #     blocked response's transcript window is silently dropped.
    block_reason: str | None = None
    if getattr(cfg, "reasoning_verify_enabled", False) and response:
        from simba.hooks import reasoning_verify

        block_reason = reasoning_verify.verify(response, cwd_str, cfg) or None

    # 1c. Record the per-session signal flag so UserPromptSubmit can gate CORE
    #     re-injection next turn (spec 25, guardian_signal_gated). Fail-soft —
    #     the reader fail-opens to inject when the flag is missing/unreadable.
    if session_id:
        with contextlib.suppress(Exception):
            simba.guardian.signal_flag.record_signal(
                session_id,
                present=simba.guardian.signal_flag.signal_in_response(response),
            )

    # 1d. Usage signals (spec 33): citation → use, repeat-unused → noise.
    #     Default-OFF → no-op.
    _usage_signal_feedback(response, session_id, cfg)

    # 2. Tailor: error capture from transcript
    #    Side effect: writes reflections under <cwd>/.simba/ (cwd from payload).
    simba.tailor.hook.process_hook(json.dumps(hook_input))

    # 3. Continuous extraction (default-off): read only the NEW transcript window via
    # the incremental cursor and enqueue it for the scored worker. Fail-soft.
    with contextlib.suppress(Exception):
        simba.memory.continuous.on_stop(
            hook_input, simba.config.load("memory"), cwd=cwd
        )

    # Rendering (see simba.harness.adapters.claude): claude (default) renders
    # non-empty additional_context as hookSpecificOutput.additionalContext;
    # codex keeps the legacy top-level stopReason. The tailor error capture
    # writes to disk as a side effect. A non-None block_reason renders as
    # {"decision":"block","reason":…} (reconsider) regardless of client.
    return CanonicalResult(
        additional_context="\n\n".join(parts), block_reason=block_reason
    )


def main(hook_input: dict) -> str:
    """Run the Stop hook and render the Claude/Codex envelope."""
    import simba.harness.adapters.claude as claude

    return claude.render("Stop", run(hook_input))
