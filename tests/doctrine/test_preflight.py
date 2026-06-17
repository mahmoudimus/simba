"""Tests for the preflight brief builder (spec 28 Phase C)."""

from __future__ import annotations

import simba.doctrine.preflight as preflight


class TestBuildBrief:
    def test_includes_doctrine_and_rules(self) -> None:
        brief = preflight.build_brief(
            task="review PR #42",
            doctrine_lines=["Use the worktree skill for PR review."],
            tool_rules=["WARNING: never git show pr-N: to review in place"],
            redirects=["git show pr-N -> worktree skill"],
        )
        assert "🦁☑" in brief
        assert "review PR #42" in brief
        assert "Use the worktree skill" in brief
        assert "never git show pr-N" in brief
        assert "worktree skill" in brief

    def test_empty_sections_still_emits_marker(self) -> None:
        brief = preflight.build_brief(
            task="some task", doctrine_lines=[], tool_rules=[], redirects=[]
        )
        # Even with nothing matched, the preflight is logged (the 🦁☑ ledger).
        assert "🦁☑" in brief
        assert "some task" in brief

    def test_ledger_line_summarizes_counts(self) -> None:
        brief = preflight.build_brief(
            task="t",
            doctrine_lines=["a", "b"],
            tool_rules=["r1"],
            redirects=[],
        )
        # The 🦁☑ line is a one-line ledger of what preflight surfaced.
        ledger = next(line for line in brief.splitlines() if "🦁☑" in line)
        assert "2" in ledger  # 2 doctrines
        assert "1" in ledger  # 1 rule


class TestApplicableRules:
    def test_tool_rule_lines_from_memories(self) -> None:
        memories = [
            {"type": "TOOL_RULE", "content": "do not hand-edit init-schema"},
            {"type": "TOOL_RULE", "content": "use the regen script"},
        ]
        lines = preflight.tool_rule_lines(memories)
        assert lines == [
            "do not hand-edit init-schema",
            "use the regen script",
        ]

    def test_redirect_lines_from_rules(self) -> None:
        class _R:
            def __init__(self, program, replacement):
                self.program = program
                self.replacement = replacement
                self.pattern = ""
                self.rewrite = ""
                self.reason = ""

        lines = preflight.redirect_lines([_R("cargo", "soldr cargo")])
        assert lines == ["cargo -> soldr cargo"]
