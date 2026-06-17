"""Configuration for simba hooks."""

from __future__ import annotations

import dataclasses

import simba.config


@simba.config.configurable("hooks")
@dataclasses.dataclass
class HooksConfig:
    # Session start — daemon polling
    health_timeout: float = 0.5
    poll_attempts: int = 15
    poll_interval: float = 0.3

    # Memory client
    daemon_host: str = "localhost"
    daemon_port: int = 8741
    default_max_results: int = 3
    default_timeout: float = 2.0
    # Route canonicalized hooks through the daemon when it's up (warm path);
    # fall back to running them inline. Output is byte-identical either way.
    dispatch_via_daemon: bool = True

    # User-prompt-submit recall
    prompt_min_length: int = 10  # skip recall for prompts shorter than this
    prompt_min_similarity: float = 0.45  # stricter floor than the daemon default

    # Guardian CORE re-injection (spec 25). The CLAUDE.md guardian re-injects the
    # SIMBA:core block on EVERY UserPromptSubmit (~2k tokens of per-turn overhead
    # even when the model still has the rules). When True, inject the block only
    # when the rules look decayed — i.e. the model's PREVIOUS response was MISSING
    # the [✓ rules] signal, or there is no recorded signal yet (first prompt /
    # post-compaction). Skip otherwise. Always fail-open: any uncertainty or error
    # re-injects (the block is the safety layer). The Stop hook records the
    # per-session signal flag (simba.guardian.signal_flag) that this reads;
    # PreCompact/SessionStart reset it. UNMEASURED token-saver → DEFAULT-OFF, which
    # is byte-identical to today (block injected every prompt). Enable:
    # `simba config set hooks.guardian_signal_gated 1`.
    guardian_signal_gated: bool = False

    # Pre-tool-use. General thinking-block recall defers to the daemon's
    # intent-aware floor (memory.min_similarity / min_similarity_broad), so
    # there is no general-recall floor here — only the strict rule gate below.
    thinking_chars: int = 1500
    dedup_ttl: int = 60
    # Context-low warning threshold, measured as transcript bytes SINCE the last
    # compaction (the live-context proxy), NOT cumulative file size. Calibrated for
    # a large (~1M-token) window; tune via `simba config set hooks.context_low_bytes`.
    context_low_bytes: int = 8_000_000

    # Tool rules (auto-learning from failures)
    rule_check_enabled: bool = True
    rule_min_similarity: float = 0.6
    rule_max_per_session: int = 10
    # Skip the per-tool-call TOOL_RULE embed+recall when the project has no
    # learned rules (the common case). The project's TOOL_RULE count is cached
    # for this many seconds; 0 disables the skip (always do the recall).
    rule_count_ttl: int = 300
    auto_learn_from_failures: bool = True
    # Recency gate: drop TOOL_RULE recall matches older than this many days
    # (0 disables the gate).  Stale "no such file" probes age out of warnings.
    rule_max_age_days: int = 14
    # Don't auto-learn "no such file" failures from read-only probe commands —
    # a discovery miss (ls/bfs/find on a guessed path) isn't a mistake to warn
    # about.  Comma-separated leading verbs treated as probes.
    learn_skip_probe_not_found: bool = True
    learn_probe_commands: str = (
        "ls,bfs,find,fd,tree,stat,test,realpath,readlink,which,type,file"
    )
    # Reader/echo commands emit file or echoed *content*, not their own failure —
    # an error word in their output is almost always a false positive, so never
    # auto-learn from them.  Comma-separated leading verbs.
    learn_reader_commands: str = (
        "grep,rg,ugrep,ag,ack,cat,bat,head,tail,less,more,sed,awk,"
        "echo,printf,jq,yq,fd,xxd,strings"
    )
    # Only auto-learn a Bash failure when the command reported a non-zero exit
    # code; when no exit code is reported, require the error on stderr (stdout
    # often merely *mentions* error words, e.g. pytest collection notices).
    learn_require_nonzero_exit: bool = True

    # PermissionRequest (Codex-only) — deny when a TOOL_RULE matches
    # the proposed call above ``permission_deny_similarity``.  Weaker
    # matches fall through to Codex's normal approval prompt.
    permission_check_enabled: bool = True
    permission_min_similarity: float = 0.6
    permission_deny_similarity: float = 0.78

    # memories-learn skill dispatch mode
    # False (default): dispatch synchronously, Claude waits for extraction to complete
    # True: dispatch in background, Claude continues immediately
    learn_async: bool = False

    # Tool-call redirect: steer bare commands to better tooling (cargo->soldr,
    # python->uv run, ...). Rules from .simba/redirects.toml + the `simba rule
    # redirect` store. No-op when there are no rules. "deny" blocks with the
    # corrected command (model retries); "rewrite" substitutes it silently
    # (PreToolUse updatedInput) for simple leading-program commands.
    redirect_enabled: bool = True
    redirect_mode: str = "deny"  # deny | rewrite

    # Pitfall/doctrine enforcement gate (src/simba/memory/pitfall.py). The
    # ENFORCEMENT half of the memory-surfacing cure: when the agent's pending move
    # (its last thinking block, before a MUTATING tool) would VIOLATE a stored
    # doctrine/scar/trap, fire it as a STOP-and-confirm DIRECTIVE — "you're about to
    # take the workaround you told me not to" — instead of leaving it as passive
    # recalled context. Fires once per reasoning turn (own dedup); fail-open.
    # MEASUREMENT (real acme moves, 2026-06-15): the naive top-candidate-similarity
    # gate fired on 29-46% of moves (topical match != violation). Adding the LLM
    # violation check dropped that to 17%; an abstention-biased prompt + the
    # mutating-tool gate took false fires to 0/31 on real moves while KEEPING 3/3
    # labeled violations — i.e. no-over-fire is now well measured.
    # STILL DEFAULT-OFF: (1) recall on real in-the-wild pitfalls is unproven (the
    # sampled moves contained none — only the 3 hand-crafted moments exercise the
    # true-positive side), and (2) violation mode costs an LLM call on the PreToolUse
    # hot path. Dogfood-ready; graduation needs the false-NEGATIVE measurement + a
    # hot-path-cost decision. Enable: `simba config set hooks.pitfall_gate_enabled 1`.
    pitfall_gate_enabled: bool = False
    # Detection strategy. "violation" (default) asks the LLM, for each topically-close
    # candidate, whether the pending move would VIOLATE the doctrine (do what it warns
    # against / contradict it / repeat the failure) vs merely share its topic — firing
    # only on a violation. "similarity" is the legacy top-candidate-over-floor gate,
    # kept for ablation: a sweep over real acme moves (2026-06-15) measured it firing on
    # 29-46% of moves — in a dense domain almost every move is topically close to SOME
    # doctrine, and topical closeness is not violation. Violation mode fixes that.
    pitfall_gate_mode: str = "violation"  # "violation" | "similarity"
    # When mode="violation" but no llm_client is wired, fall back to: "failure_only"
    # (the conservative FAILURE-type similarity gate at pitfall_gate_min_similarity —
    # FAILURE is the one pitfall-shaped type) or "off" (fire nothing).
    pitfall_gate_fallback: str = "failure_only"  # "failure_only" | "off"
    # Candidate floor for violation mode: similarity needed to be worth an LLM check
    # (permissive — the LLM supplies precision). Lower than the fire floor below.
    pitfall_gate_topical_floor: float = 0.70
    pitfall_gate_max_checks: int = 3  # max candidates LLM-checked per move (cost bound)
    # Tools the gate fires before. The gate is about "you're about to TAKE a workaround/
    # action" — measured (2026-06-15) its false fires were all on read/search/extract
    # moves, so it only runs before state-changing tools, not exploration (Read/Grep/
    # Glob/WebSearch). Comma-separated tool names.
    pitfall_gate_tools: str = "Edit,Write,Bash"
    # Fire floor for "similarity" mode and the "failure_only" fallback — stricter than
    # recall's min_similarity (a directive interrupts, so it must be a strong match).
    pitfall_gate_min_similarity: float = 0.78
    # Doctrine/scar/trap types recalled for the gate. Measured correction to the
    # original FAILURE+PREFERENCE guess: GOTCHA carries a high-information trap memory
    # central to 2 of the 3 labeled recurrence moments, so it is required.
    pitfall_gate_types: str = "FAILURE,PREFERENCE,GOTCHA"
    pitfall_gate_max_results: int = (
        5  # candidate pool recalled (top candidates checked)
    )

    # Intent-primed doctrine + mandated preflight (spec 28). UserPromptSubmit is the
    # only hook that sees the user's intent BEFORE any action — the cheapest, earliest
    # prevention point. These levers front-load the right approach from stated intent.
    #
    # PRIME: classify the prompt CHEAPLY (embedding-match against doctrine triggers —
    # no LLM on the hot path) and inject the matched doctrine + which TOOL_RULEs /
    # redirects apply to this project. Default-OFF (UNMEASURED false-prime rate); OFF
    # ⇒ byte-identical to today (recall + CORE only).
    intent_priming_enabled: bool = False
    # Cosine floor for a doctrine trigger to count as a match (stricter than recall's
    # min_similarity — a primed doctrine steers the agent, so it must be a real match).
    intent_priming_min_similarity: float = 0.55
    # Cap on primed doctrines injected per prompt (keep the injection small).
    intent_priming_max_doctrines: int = 3
    # MANDATE + ENFORCE: the teeth. When on, PreToolUse BLOCKS any mutating tool that
    # runs without a `simba preflight` having fired this turn (per-turn flag, reuses
    # the spec-25 signal-flag plumbing). Read-only tools are unaffected; a preflight
    # clears the gate. Without this half, priming is merely advisory. Default-OFF.
    preflight_mandate_enabled: bool = False
    # Mandate preflight only for RISK-TIER intents (a matched risk-tier doctrine
    # trigger), not every task — the over-fire guard. When True (default) the gate
    # is armed only after a risk-tier prime this turn; when False it is armed for
    # every turn (mutating tools always require a preflight). No effect unless
    # ``preflight_mandate_enabled`` is on.
    preflight_mandate_risk_only: bool = True
    # Mutating tools the preflight gate fires before (read-only tools are allowed
    # without a preflight). Comma-separated; mirrors the pitfall gate's tool set.
    preflight_mandate_tools: str = "Edit,Write,Bash"
