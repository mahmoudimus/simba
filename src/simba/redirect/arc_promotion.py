"""Failure-arc -> redirect-rule candidate promotion (deterministic, LLM-free).

Turns the transcript distiller's failure->fix arcs (``transcripts/arcs.py``'s
``failure_arc`` table) into candidate redirect rules for human review via
``simba rule promote`` -- never auto-applied. An arc only yields a candidate
when its failed->fixed shell-command pair is a purely mechanical,
token-level transformation (see ``classify_transform``) AND there's enough
cross-session evidence AND no other arc with the same failure signature
proposes a *different* fix (a contradiction means the "fix" is
context-dependent, not a universal rule).

Even then, the produced rule is only ever written in DENY mode
(``redirect/candidates.py``'s ``approve()``) -- a wrong deny just blocks a
command and is self-correcting (the human sees it and rejects the rule); a
wrong auto-rewrite would silently corrupt a command. That DENY default is
also the safety net for a residual risk this module cannot itself verify:
arcs only record FAILURES, so there's no way to confirm from arc data alone
that the failed form never legitimately succeeds elsewhere (e.g. a different
project, a different flag meaning) -- deny-by-default means a false-positive
rule merely nags instead of silently breaking something.
"""

from __future__ import annotations

import dataclasses
import re
import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Iterable

# Shell-class tool names a FailureArc's `tool` field takes across harnesses.
# Confirmed from the distiller's own test fixtures (not a hardcoded harness
# convention -- transcripts/distill.py just passes through whatever the
# harness's tool-call `name` carries): Claude Code's Bash tool reports
# "Bash" (tests/transcripts/test_distill.py), Codex's shell-executing
# custom_tool_call reports "exec" (tests/test_transcript_distill_cli.py).
# "bash"/"shell"/"exec_command"/"local_shell" are included defensively for
# harness variants not pinned down anywhere else in this repo; comparison is
# case-insensitive.
_SHELL_TOOLS = frozenset({"bash", "shell", "exec", "exec_command", "local_shell"})


def is_shell_tool(tool: str) -> bool:
    return tool.strip().lower() in _SHELL_TOOLS


# ── classification ──────────────────────────────────────────────────────────


@dataclasses.dataclass
class Classification:
    shape: str  # "flag-drop" | "token-replace" | "prefix-insert"
    rule_kind: str  # "program" | "pattern"
    before: str  # human-readable summary of what changed (for the reason line)
    after: str
    rule_program: str = ""
    rule_replacement: str = ""
    rule_pattern: str = ""
    rule_rewrite: str = ""


def _is_subsequence(needle: str, haystack: str) -> bool:
    """True if every character of ``needle`` appears in ``haystack`` in
    order (not necessarily contiguous) -- "needle is haystack with some
    characters dropped"."""
    it = iter(haystack)
    return all(ch in it for ch in needle)


def _token_pattern_rule(
    index: int, program: str, before: str, after: str
) -> tuple[str, str]:
    """Build a (pattern, rewrite) pair that swaps ``before`` -> ``after`` at
    token position ``index`` in future occurrences of the same command
    shape, anchored so it doesn't fire on an unrelated command.

    ``index == 0`` (the leading/program token itself changed): anchor on
    start-of-command directly -- anchoring on "program ... before" would be
    self-referential when program == before.  Otherwise: anchor on the
    leading program name, then match ``before`` as a whole word anywhere
    after it (non-greedy filler, mirrors the existing BUILTIN_RULES style).
    """
    if index == 0:
        pattern = rf"^(\s*){re.escape(before)}\b"
        rewrite = rf"\1{after}"
        return pattern, rewrite
    pattern = rf"(\b{re.escape(program)}\b[^\n]*?\s){re.escape(before)}(?=\s|$)"
    rewrite = rf"\1{after}"
    return pattern, rewrite


def _classify_equal_length(
    failed: list[str], fixed: list[str]
) -> Classification | None:
    diffs = [i for i, (a, b) in enumerate(zip(failed, fixed, strict=True)) if a != b]
    if len(diffs) != 1:
        return None  # zero diffs (no-op) or a multi-token diff -- reject
    index = diffs[0]
    before, after = failed[index], fixed[index]
    is_flag_before = before.startswith("-")
    is_flag_after = after.startswith("-")

    # Flag-vs-operand heuristic: flags start with '-'; program is token 0; a
    # changed operand (neither) is a value/path change (e.g. cd /a -> cd
    # /b) -- context-dependent, not mechanical. Reject.
    if index != 0 and not is_flag_before and not is_flag_after:
        return None

    program = failed[0]
    pattern, rewrite = _token_pattern_rule(index, program, before, after)

    # flag-drop (combined short-flag bundle): `after` is obtainable by
    # deleting characters from `before` once leading dashes are stripped
    # (e.g. "-rn" -> "-n": drop 'r'). A flag that changes to a DIFFERENT,
    # unrelated flag (e.g. "--summarize" -> "-summary") is not a subsequence
    # of the original and falls through to token-replace instead.
    if is_flag_before and is_flag_after:
        stripped_before = before.lstrip("-")
        stripped_after = after.lstrip("-")
        if len(stripped_after) < len(stripped_before) and _is_subsequence(
            stripped_after, stripped_before
        ):
            return Classification(
                shape="flag-drop",
                rule_kind="pattern",
                before=before,
                after=after,
                rule_pattern=pattern,
                rule_rewrite=rewrite,
            )

    return Classification(
        shape="token-replace",
        rule_kind="pattern",
        before=before,
        after=after,
        rule_pattern=pattern,
        rule_rewrite=rewrite,
    )


def _classify_removed_token(
    failed: list[str], fixed: list[str]
) -> Classification | None:
    """``len(fixed) == len(failed) - 1``: exactly one token removed, all
    others identical in order. Only a removed FLAG token (starts with '-')
    is treated as mechanical; a removed non-flag token is an argument-shape
    change, not a flag-drop -- reject."""
    for i in range(len(failed)):
        if failed[:i] + failed[i + 1 :] == fixed:
            dropped = failed[i]
            if not dropped.startswith("-"):
                return None
            program = failed[0]
            pattern = (
                rf"(\b{re.escape(program)}\b[^\n]*?)\s{re.escape(dropped)}(?=\s|$)"
            )
            rewrite = r"\1"
            return Classification(
                shape="flag-drop",
                rule_kind="pattern",
                before=dropped,
                after="",
                rule_pattern=pattern,
                rule_rewrite=rewrite,
            )
    return None


def _classify_prefix_insert(
    failed: list[str], fixed: list[str]
) -> Classification | None:
    """``fixed`` is ``failed`` with 1-3 tokens prepended before the (possibly
    also corrected) leading token, remainder identical -- e.g.
    ``python3 -m pytest ...`` -> ``uv run python -m pytest ...``: 2 tokens
    ("uv", "run") prepended, leading token corrected python3 -> python,
    remainder ("-m pytest ...") untouched.

    Purely a leading-token-region transformation by construction (the
    remainder match is exact), so this always derives a program rule."""
    n = len(failed)
    if n == 0:
        return None
    tail = failed[1:]
    tail_len = len(tail)
    for k in (1, 2, 3):
        if len(fixed) != n + k:
            continue
        fixed_tail = fixed[len(fixed) - tail_len :] if tail_len else []
        if fixed_tail != tail:
            continue
        prefix_head = fixed[: len(fixed) - tail_len]
        return Classification(
            shape="prefix-insert",
            rule_kind="program",
            before=failed[0],
            after=" ".join(prefix_head),
            rule_program=failed[0],
            rule_replacement=" ".join(prefix_head),
        )
    return None


def classify_transform(failed_args: str, fixed_args: str) -> Classification | None:
    """Classify a failed->fixed shell-command pair into one of three
    deterministic, mechanical shapes, or ``None`` (reject) if it doesn't
    cleanly fit any of them. shlex-splits both sides; a shlex failure on
    either rejects (never guesses at malformed shell syntax)."""
    try:
        failed = shlex.split(failed_args)
        fixed = shlex.split(fixed_args)
    except ValueError:
        return None
    if not failed or not fixed or failed == fixed:
        return None

    if len(fixed) > len(failed):
        return _classify_prefix_insert(failed, fixed)
    if len(fixed) == len(failed) - 1:
        return _classify_removed_token(failed, fixed)
    if len(fixed) == len(failed):
        return _classify_equal_length(failed, fixed)
    return None  # shrank by more than one token -- multi-token diff, reject


# ── evidence aggregation + contradiction check ──────────────────────────────


@dataclasses.dataclass
class CandidateProposal:
    signature: str
    tool: str
    failed_example: str
    fixed_example: str
    classification: Classification
    reason: str
    evidence_count: int
    session_count: int
    project_path: str


def _rule_key(c: Classification) -> tuple[str, str, str, str, str]:
    return (
        c.rule_kind,
        c.rule_program,
        c.rule_replacement,
        c.rule_pattern,
        c.rule_rewrite,
    )


def _build_reason(c: Classification, evidence_count: int, session_count: int) -> str:
    change = f"`{c.before}` -> `{c.after}`" if c.after else f"dropped `{c.before}`"
    return (
        f"{c.shape}: {change}; seen {evidence_count}x across "
        f"{session_count} session(s) (failure->fix arc mining)"
    )


def build_candidates(
    arcs: Iterable[object], *, min_evidence: int
) -> list[CandidateProposal]:
    """Pure aggregation: resolved, shell-tool arcs -> candidate rules.

    ``arcs`` is any iterable of objects with ``resolved``, ``tool``,
    ``signature``, ``session_source``, ``failed_args_head``,
    ``fix_args_head``, ``repeat_count``, ``project_path`` attributes (real
    ``FailureArc`` rows, or a ``types.SimpleNamespace``/similar in tests --
    attribute access, not dict-style).

    Groups eligible arcs by ``(tool, signature)`` -- the same pair the
    distiller itself upserts on, so it already correlates "the same
    recurring failure" across sessions. Within a group: classify each arc's
    failed->fixed pair independently; arcs that fail to classify are ignored
    (noise, not a competing fix); if the classifiable arcs disagree on the
    resulting rule, the whole group is a contradiction and is rejected
    (a different "fix" for the same failure is context-dependent); otherwise
    evidence is tallied over the classifiable, agreeing arcs and the
    evidence-threshold gate (``sum(repeat_count) >= min_evidence`` OR
    ``>=2`` distinct sessions) decides eligibility.
    """
    groups: dict[tuple[str, str], list[object]] = {}
    for arc in arcs:
        if not getattr(arc, "resolved", False):
            continue
        tool = getattr(arc, "tool", "")
        if not is_shell_tool(tool):
            continue
        if not getattr(arc, "fix_args_head", None):
            continue
        key = (tool, getattr(arc, "signature", ""))
        groups.setdefault(key, []).append(arc)

    proposals: list[CandidateProposal] = []
    for (tool, signature), group_arcs in groups.items():
        classified: list[tuple[object, Classification]] = []
        for arc in group_arcs:
            c = classify_transform(arc.failed_args_head, arc.fix_args_head)
            if c is not None:
                classified.append((arc, c))
        if not classified:
            continue

        distinct_rules = {_rule_key(c) for _, c in classified}
        if len(distinct_rules) > 1:
            continue  # contradiction: different fixes for the same failure

        winning_arcs = [arc for arc, _ in classified]
        classification = classified[0][1]
        evidence_count = sum(getattr(a, "repeat_count", 1) for a in winning_arcs)
        session_count = len({getattr(a, "session_source", "") for a in winning_arcs})
        if evidence_count < min_evidence and session_count < 2:
            continue

        best_arc = max(winning_arcs, key=lambda a: getattr(a, "repeat_count", 1))
        project_paths = {getattr(a, "project_path", "") for a in winning_arcs}
        project_path = next(iter(project_paths)) if len(project_paths) == 1 else ""

        proposals.append(
            CandidateProposal(
                signature=signature,
                tool=tool,
                failed_example=best_arc.failed_args_head,
                fixed_example=best_arc.fix_args_head or "",
                classification=classification,
                reason=_build_reason(classification, evidence_count, session_count),
                evidence_count=evidence_count,
                session_count=session_count,
                project_path=project_path,
            )
        )
    return proposals


# ── DB-backed orchestration ─────────────────────────────────────────────────


@dataclasses.dataclass
class ScanSummary:
    total: int
    new: int


def scan(cwd: pathlib.Path | None = None, *, min_evidence: int = 3) -> ScanSummary:
    """Mine ``failure_arc`` for candidates and upsert them into
    ``rule_candidate`` (redirect/candidates.py). Idempotent -- see
    ``candidates.upsert_candidate``."""
    import simba.redirect.candidates as candidates_store
    import simba.transcripts.arcs as arcs_store

    all_arcs = arcs_store.list_all(cwd=cwd)
    proposals = build_candidates(all_arcs, min_evidence=min_evidence)

    new = 0
    for p in proposals:
        result = candidates_store.upsert_candidate(
            signature=p.signature,
            tool=p.tool,
            failed_example=p.failed_example,
            fixed_example=p.fixed_example,
            rule_kind=p.classification.rule_kind,
            rule_program=p.classification.rule_program,
            rule_replacement=p.classification.rule_replacement,
            rule_pattern=p.classification.rule_pattern,
            rule_rewrite=p.classification.rule_rewrite,
            reason=p.reason,
            evidence_count=p.evidence_count,
            session_count=p.session_count,
            project_path=p.project_path,
            cwd=cwd,
        )
        if result == "new":
            new += 1
    return ScanSummary(total=len(proposals), new=new)
