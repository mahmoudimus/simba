"""Redirect rule model + the deny/rewrite decision (pure)."""

from __future__ import annotations

import dataclasses
import re

import simba.redirect.parse as parse


@dataclasses.dataclass
class RedirectRule:
    program: str  # program name to match (normalized, e.g. "cargo")
    replacement: str  # corrected leading command, e.g. "soldr cargo"
    reason: str = ""
    source: str = ""  # "toml" | "store" (provenance, for diagnostics)


@dataclasses.dataclass
class Decision:
    action: str  # "deny" | "rewrite"
    reason: str = ""
    command: str = ""  # the rewritten command (action == "rewrite")


def _rule_for(program: str, rules: list[RedirectRule]) -> RedirectRule | None:
    for rule in rules:
        if parse.program_name(program) == parse.program_name(rule.program):
            return rule
    return None


def _deny_message(program: str, rule: RedirectRule) -> str:
    msg = f"Use `{rule.replacement} ...` instead of `{program} ...`."
    return f"{msg} {rule.reason}".strip()


def _simple_leading_token(command: str, rule: RedirectRule) -> str | None:
    """Return the raw leading token if ``command`` is a single, simple invocation
    of ``rule.program`` at the very start (so a string rewrite is safe)."""
    segments = parse.split_segments(command)
    if len(segments) != 1:
        return None
    words = parse.tokenize(segments[0])
    if not words:
        return None
    if parse.program_name(words[0]) != parse.program_name(rule.program):
        return None
    # must start with that token verbatim (no env prefix, wrapper, etc.)
    if command.lstrip().startswith(words[0]):
        return words[0]
    return None


def evaluate(
    command: str, rules: list[RedirectRule], *, mode: str
) -> Decision | None:
    """Return a redirect Decision for ``command``, or None if no rule matches.

    ``mode`` is "deny" (block + suggest the corrected command) or "rewrite"
    (substitute it silently when the shape is simple, else fall back to deny).
    """
    matches = [
        (inv.program, rule)
        for inv in parse.invoked_programs(command)
        if (rule := _rule_for(inv.program, rules)) is not None
    ]
    if not matches:
        return None

    program, rule = matches[0]
    deny = Decision(action="deny", reason=_deny_message(program, rule))

    if mode != "rewrite":
        return deny

    # Only rewrite a single, simple, leading-program invocation — anything
    # fancier (env prefix, multiple segments, uv run, nested shell) is denied
    # so we never synthesize a broken command.
    if len(matches) == 1:
        token = _simple_leading_token(command, rule)
        if token is not None:
            rewritten = re.sub(
                r"^(\s*)" + re.escape(token), r"\1" + rule.replacement, command, count=1
            )
            return Decision(action="rewrite", command=rewritten)
    return deny
