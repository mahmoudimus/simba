"""Tests for regex *pattern* redirect rules (flag-level fixes, e.g. `rg -rln`).

Program rules match a program name + swap the leading token; pattern rules match a
regex over the whole command and rewrite via re.sub — so a flag mistake like
`rg -rln` (in ripgrep `-r` is --replace) can be corrected to `rg -l`.
"""

from __future__ import annotations

import simba.redirect.rules as r
import simba.redirect.store as store

RG = r.RedirectRule(
    pattern=r"\brg\s+-rln\b",
    rewrite="rg -l",
    reason="rg -r is --replace; recursion is default",
)


def test_pattern_no_match_returns_none() -> None:
    assert r.evaluate("rg -l foo src/", [RG], mode="rewrite") is None
    assert r.evaluate("ls -la", [RG], mode="deny") is None


def test_pattern_rewrite_substitutes_via_regex() -> None:
    d = r.evaluate("rg -rln 'CONDITIONAL_RETURN' src/", [RG], mode="rewrite")
    assert d is not None and d.action == "rewrite"
    assert d.command == "rg -l 'CONDITIONAL_RETURN' src/"


def test_pattern_deny_suggests_the_fix() -> None:
    d = r.evaluate("rg -rln X src/", [RG], mode="deny")
    assert d is not None and d.action == "deny"
    assert "rg -l X src/" in d.reason
    assert "--replace" in d.reason


def test_pattern_with_backref_rewrite() -> None:
    rule = r.RedirectRule(pattern=r"\bgrep -rn (\S+)", rewrite=r"rg \1")
    d = r.evaluate("grep -rn TODO src/", [rule], mode="rewrite")
    assert d.action == "rewrite" and d.command == "rg TODO src/"


def test_invalid_regex_is_skipped_not_raised() -> None:
    bad = r.RedirectRule(pattern=r"rg (", rewrite="rg")
    assert r.evaluate("rg -rln X", [bad], mode="rewrite") is None  # no crash


def test_program_rules_still_work_alongside_pattern_rules() -> None:
    cargo = r.RedirectRule(program="cargo", replacement="soldr cargo")
    rules = [RG, cargo]
    d = r.evaluate("cargo build", rules, mode="rewrite")
    assert d.action == "rewrite" and d.command == "soldr cargo build"


def test_toml_loads_pattern_rule(tmp_path) -> None:
    toml = tmp_path / "redirects.toml"
    toml.parent.mkdir(parents=True, exist_ok=True)
    toml.write_text(
        '[[redirect]]\npattern = "\\\\brg\\\\s+-rln\\\\b"\n'
        'rewrite = "rg -l"\nreason = "rg -r is --replace"\n'
    )
    rules = store.load_toml(toml)
    assert len(rules) == 1
    assert rules[0].pattern == r"\brg\s+-rln\b"
    assert rules[0].rewrite == "rg -l"
    assert rules[0].source == "toml"
