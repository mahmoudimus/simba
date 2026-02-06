"""Tests for search.deps — external tool dependency checker."""

from __future__ import annotations

import unittest
import unittest.mock

import simba.search.deps

# ---------------------------------------------------------------------------
# TestCheckDependency
# ---------------------------------------------------------------------------


class TestCheckDependency:
    def test_returns_true_and_version_when_found(self) -> None:
        mock_result = unittest.mock.Mock()
        mock_result.stdout = "ripgrep 14.1.0\n"
        mock_result.stderr = ""
        with (
            unittest.mock.patch(
                "simba.search.deps.shutil.which", return_value="/usr/bin/rg"
            ),
            unittest.mock.patch(
                "simba.search.deps.subprocess.run", return_value=mock_result
            ),
        ):
            found, version = simba.search.deps.check_dependency("rg")
        assert found is True
        assert version == "ripgrep 14.1.0"

    def test_returns_false_when_missing(self) -> None:
        with unittest.mock.patch("simba.search.deps.shutil.which", return_value=None):
            found, version = simba.search.deps.check_dependency("nonexistent")
        assert found is False
        assert version == "not found"


# ---------------------------------------------------------------------------
# TestCheckAll
# ---------------------------------------------------------------------------


class TestCheckAll:
    def test_returns_dict_with_all_expected_keys(self) -> None:
        with unittest.mock.patch(
            "simba.search.deps.check_dependency",
            return_value=(True, "1.0"),
        ):
            result = simba.search.deps.check_all()
        assert set(result.keys()) == {"rg", "fzf", "jq", "qmd"}
        for _name, (found, version) in result.items():
            assert found is True
            assert version == "1.0"


# ---------------------------------------------------------------------------
# TestGetInstallInstructions
# ---------------------------------------------------------------------------


class TestGetInstallInstructions:
    def test_returns_nonempty_for_each_known_tool(self) -> None:
        for tool in ("rg", "fzf", "jq", "qmd"):
            instructions = simba.search.deps.get_install_instructions(tool)
            assert isinstance(instructions, str)
            assert len(instructions) > 0

    def test_returns_generic_message_for_unknown_tool(self) -> None:
        instructions = simba.search.deps.get_install_instructions("unknown_tool_xyz")
        assert "unknown_tool_xyz" in instructions
