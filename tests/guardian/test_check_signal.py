"""Tests for guardian check_signal module."""

from __future__ import annotations

import pathlib

import simba.guardian.check_signal


class TestCheckSignal:
    def test_signal_present_returns_true(self):
        assert (
            simba.guardian.check_signal.check_signal("Some response text [✓ rules] end")
            is True
        )

    def test_signal_missing_returns_false(self):
        assert (
            simba.guardian.check_signal.check_signal("Some response without signal")
            is False
        )

    def test_empty_response_returns_false(self):
        assert simba.guardian.check_signal.check_signal("") is False

    def test_signal_at_end_of_response(self):
        assert simba.guardian.check_signal.check_signal("Done.\n[✓ rules]") is True

    def test_partial_signal_returns_false(self):
        assert (
            simba.guardian.check_signal.check_signal("Some text [✓ rule] end") is False
        )

    def test_signal_with_surrounding_whitespace(self):
        assert simba.guardian.check_signal.check_signal("text  [✓ rules]  more") is True


class TestMain:
    def test_signal_present_returns_empty(self, claude_md_with_core):
        cwd = claude_md_with_core.parent
        result = simba.guardian.check_signal.main(response="Done. [✓ rules]", cwd=cwd)
        assert result == ""

    def test_signal_missing_returns_warning(self, claude_md_with_core):
        cwd = claude_md_with_core.parent
        result = simba.guardian.check_signal.main(
            response="Done without signal.", cwd=cwd
        )
        assert "MEMORY ALERT" in result
        assert "CLAUDE.md" in result

    def test_signal_missing_includes_claude_md_content(self, claude_md_with_core):
        cwd = claude_md_with_core.parent
        result = simba.guardian.check_signal.main(response="No signal here.", cwd=cwd)
        assert "Never delete files" in result

    def test_no_claude_md_returns_empty(self, tmp_path: pathlib.Path):
        result = simba.guardian.check_signal.main(
            response="No signal here.", cwd=tmp_path
        )
        assert result == ""

    def test_empty_response_with_claude_md(self, claude_md_with_core):
        cwd = claude_md_with_core.parent
        result = simba.guardian.check_signal.main(response="", cwd=cwd)
        assert "MEMORY ALERT" in result
