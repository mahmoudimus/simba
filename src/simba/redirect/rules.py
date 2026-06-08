"""Redirect rule model + the deny/rewrite decision (pure)."""

from __future__ import annotations

import dataclasses
import re

import simba.redirect.parse as parse


@dataclasses.dataclass
class RedirectRule:
    # Program rule: match a program name + swap the leading token.
    program: str = ""  # program name to match (normalized, e.g. "cargo")
    replacement: str = ""  # corrected leading command, e.g. "soldr cargo"
    reason: str = ""
    source: str = ""  # "toml" | "store" (provenance, for diagnostics)
    # Pattern rule (flag-level fixes): match a regex over the whole command and
    # rewrite it via re.sub (backrefs allowed), e.g. pattern=r"\brg\s+-rln\b",
    # rewrite="rg -l". Checked before program rules; invalid regex is skipped.
    pattern: str = ""
    rewrite: str = ""
    # Per-rule mode override ("deny" | "rewrite"); empty ⇒ use the global mode. A
    # safe, unambiguous auto-fix can set "rewrite" so it applies even when the
    # project default is "deny".
    mode: str = ""


@dataclasses.dataclass
class Decision:
    action: str  # "deny" | "rewrite"
    reason: str = ""
    command: str = ""  # the rewritten command (action == "rewrite")


def _rule_for(program: str, rules: list[RedirectRule]) -> RedirectRule | None:
    for rule in rules:
        if not rule.program:  # pattern-only rule, not a program rule
            continue
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


def _pattern_suggestion(rule: RedirectRule, command: str) -> str:
    if not rule.rewrite:
        return ""
    try:
        new = re.sub(rule.pattern, rule.rewrite, command)
    except re.error:
        return ""
    return new if new != command else ""


def _evaluate_pattern_rules(
    command: str, rules: list[RedirectRule], *, mode: str
) -> Decision | None:
    """Regex rules over the whole command (flag-level fixes). First match wins."""
    for rule in rules:
        if not rule.pattern:
            continue
        try:
            if re.search(rule.pattern, command) is None:
                continue
        except re.error:
            continue  # malformed rule -> skip, never raise
        suggestion = _pattern_suggestion(rule, command)
        effective_mode = rule.mode or mode
        if effective_mode == "rewrite" and suggestion:
            return Decision(action="rewrite", command=suggestion, reason=rule.reason)
        base = (
            f"Use `{suggestion}` instead."
            if suggestion
            else "This command is discouraged."
        )
        return Decision(action="deny", reason=f"{base} {rule.reason}".strip())
    return None


def evaluate(
    command: str, rules: list[RedirectRule], *, mode: str
) -> Decision | None:
    """Return a redirect Decision for ``command``, or None if no rule matches.

    ``mode`` is "deny" (block + suggest the corrected command) or "rewrite"
    (substitute it silently when the shape is simple, else fall back to deny).
    Pattern (regex) rules are checked first, then program rules.
    """
    pattern_decision = _evaluate_pattern_rules(command, rules, mode=mode)
    if pattern_decision is not None:
        return pattern_decision

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


# Built-in rules every project gets with no config (appended after project rules,
# so a project's own rule still wins on first-match). Disable all redirects via
# hooks.redirect_enabled=false.
#
# The universal grep->rg `-r` trap: in ripgrep `-r` is --replace (it eats the next
# token) and recursion + line numbers are default, so a grep-style short-flag
# bundle that packs `-r` with other flags (`-rn`, `-rl`, `-rln`, `-nr`, `-rc`, …)
# silently becomes --replace and corrupts the output. Match a single-dash clump
# (≥2 chars) of common grep flags {r,l,n,i,c,w,o,h,v} that contains an r, and drop
# the r. The charset is the false-positive guard: a real --replace uses an
# arbitrary word (`rg -rnew old`) or a separate arg (`rg -r new`) — neither is all
# grep-flag letters — so both are left untouched. mode="rewrite" so the safe fix
# auto-applies even under the default "deny".
BUILTIN_RULES: list[RedirectRule] = [
    RedirectRule(
        pattern=r"(\brg\b[^|&;]*?\s)-(?=[rlnicwohv]{2,}\b)([rlnicwohv]*)r([rlnicwohv]*)\b",
        rewrite=r"\1-\2\3",
        mode="rewrite",
        reason=(
            "rg -r is --replace (consumes the next token); recursion + line "
            "numbers are default. Dropped the bundled -r."
        ),
        source="builtin",
    ),
]
