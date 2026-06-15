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

    # User-prompt-submit recall
    prompt_min_length: int = 10  # skip recall for prompts shorter than this
    prompt_min_similarity: float = 0.45  # stricter floor than the daemon default

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
    # ENFORCEMENT half of the acme "lobotomy" cure: when the agent's pending move
    # (its last thinking block) strongly matches a stored doctrine/scar/trap, fire
    # it as a STOP-and-confirm DIRECTIVE — "you're about to take the workaround you
    # told me not to" — instead of leaving it as passive recalled context. Recalls
    # only the doctrine TYPES, judges the TOP candidate against a STRICT floor (a
    # directive interrupts, so it must be a strong, specific match), and fires once
    # per reasoning turn (own dedup cache). Fail-open; fires for any tool (incl.
    # Edit/Write/Bash), not just the general-recall tool set.
    # DEFAULT-OFF: measured retrieval-side (probe 2026-06-15 on the live acme store
    # — 3/3 labeled moments fire their scar at sim>=0.82, 0/6 benign moves fire,
    # floor 0.78 sits in the gap) but graduation needs the BEHAVIORAL A/B
    # (recurrence prevention with the agent in the loop), which a retrieval probe
    # can't establish. Enable via `simba config set hooks.pitfall_gate_enabled true`.
    pitfall_gate_enabled: bool = False
    # Directive floor — stricter than recall's min_similarity. Calibrated on the
    # live acme store: labeled fires >= 0.82, benign tops <= 0.73; 0.78 is the
    # mid-gap operating point (0/6 false positives, 3/3 fire).
    pitfall_gate_min_similarity: float = 0.78
    # Doctrine/scar/trap types recalled for the gate. Measured correction to the
    # original FAILURE+PREFERENCE guess: GOTCHA carries the operand-presence root
    # cause (mem_49970a37) central to 2 of the 3 labeled moments, so it is required.
    pitfall_gate_types: str = "FAILURE,PREFERENCE,GOTCHA"
    pitfall_gate_max_results: int = 5  # candidate pool recalled (only the top fires)
