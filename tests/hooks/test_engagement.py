"""Tier-1 engagement marker (spec 27): the simba-emitted 🦁☑ ledger."""

from __future__ import annotations

import simba.hooks.engagement as eng


class TestLedgerBuilder:
    def test_recalled_with_top_similarity(self) -> None:
        line = eng.prompt_ledger(memory_count=3, top_similarity=0.74)
        assert line == "🦁☑ recalled 3 (top 0.74)"

    def test_idle_when_nothing_matched(self) -> None:
        # Zero recall, no gate action → idle marker (still emitted every turn).
        assert eng.prompt_ledger(memory_count=0, top_similarity=0.0) == (
            "🦁☑ idle (nothing matched)"
        )

    def test_recalled_zero_with_top_zero_is_idle(self) -> None:
        assert eng.prompt_ledger(memory_count=0, top_similarity=0.0).startswith(
            "🦁☑ idle"
        )

    def test_gate_action_appended(self) -> None:
        # PreToolUse appends the gate action onto an existing ledger line.
        line = eng.append_gate_action(
            "🦁☑ recalled 0", "rule-warned: Edit drizzle/meta"
        )
        assert line == "🦁☑ recalled 0 · rule-warned: Edit drizzle/meta"

    def test_gate_action_on_idle(self) -> None:
        line = eng.append_gate_action(
            "🦁☑ idle (nothing matched)", "rewrote: foo.sh"
        )
        assert line == "🦁☑ idle (nothing matched) · rewrote: foo.sh"

    def test_marker_constant_matches_preflight(self) -> None:
        import simba.doctrine.preflight as pf

        assert eng.MARKER == pf.MARKER

    def test_gate_action_for_rewrite(self) -> None:
        assert eng.gate_action_label("rewrite", "foo.sh") == "rewrote: foo.sh"

    def test_gate_action_for_block(self) -> None:
        assert eng.gate_action_label("block", "git push") == "blocked: git push"

    def test_gate_action_for_warn(self) -> None:
        assert eng.gate_action_label("warn", "Edit x") == "rule-warned: Edit x"

    def test_is_marker_line(self) -> None:
        assert eng.has_marker("some text\n🦁☑ recalled 2 (top 0.5)\nmore")
        assert not eng.has_marker("no marker here")
