"""Tests for ``simba search index`` subcommand."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import unittest.mock

import simba.db
import simba.search.__main__
import simba.search.project_memory


class TestIndexCommand:
    def test_initializes_database(self, tmp_path: pathlib.Path) -> None:
        deps = {
            "rg": (False, "not found"),
            "fzf": (False, "not found"),
            "jq": (False, "not found"),
            "qmd": (False, "not found"),
        }
        db_path = tmp_path / ".simba" / "simba.db"
        with (
            unittest.mock.patch("simba.search.deps.check_all", return_value=deps),
            unittest.mock.patch(
                "simba.db.get_db_path",
                return_value=db_path,
            ),
        ):
            code = simba.search.__main__._cmd_index(tmp_path)

        assert code == 0
        assert db_path.exists()

    def test_skips_qmd_when_missing(
        self, tmp_path: pathlib.Path, capsys: object
    ) -> None:
        deps = {
            "rg": (False, "not found"),
            "fzf": (False, "not found"),
            "jq": (False, "not found"),
            "qmd": (False, "not found"),
        }
        db_path = tmp_path / ".simba" / "simba.db"
        with (
            unittest.mock.patch("simba.search.deps.check_all", return_value=deps),
            unittest.mock.patch(
                "simba.db.get_db_path",
                return_value=db_path,
            ),
        ):
            code = simba.search.__main__._cmd_index(tmp_path)

        assert code == 0
        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert "skipping semantic indexing" in captured.out.lower()

    def test_runs_qmd_when_available(self, tmp_path: pathlib.Path) -> None:
        deps = {
            "rg": (True, "14.0"),
            "fzf": (True, "0.50"),
            "jq": (True, "1.7"),
            "qmd": (True, "1.0"),
        }
        mock_result = unittest.mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = "a.py\nb.py\n"

        db_path = tmp_path / ".simba" / "simba.db"
        with (
            unittest.mock.patch("simba.search.deps.check_all", return_value=deps),
            unittest.mock.patch(
                "simba.db.get_db_path",
                return_value=db_path,
            ),
            unittest.mock.patch(
                "simba.search.__main__.subprocess.run",
                return_value=mock_result,
            ) as mock_run,
        ):
            code = simba.search.__main__._cmd_index(tmp_path)

        assert code == 0
        # Should call: qmd collection add, qmd context add, qmd embed, rg --files
        qmd_calls = [c for c in mock_run.call_args_list if c.args[0][0] == "qmd"]
        assert len(qmd_calls) == 3

    def test_qmd_failure_does_not_crash(self, tmp_path: pathlib.Path) -> None:
        deps = {
            "rg": (False, "not found"),
            "fzf": (False, "not found"),
            "jq": (False, "not found"),
            "qmd": (True, "1.0"),
        }
        db_path = tmp_path / ".simba" / "simba.db"
        with (
            unittest.mock.patch("simba.search.deps.check_all", return_value=deps),
            unittest.mock.patch(
                "simba.db.get_db_path",
                return_value=db_path,
            ),
            unittest.mock.patch(
                "simba.search.__main__.subprocess.run",
                side_effect=subprocess.SubprocessError("qmd crashed"),
            ),
        ):
            code = simba.search.__main__._cmd_index(tmp_path)

        assert code == 0

    def test_dispatched_from_main(self, tmp_path: pathlib.Path) -> None:
        """Verify the ``index`` command is reachable from ``main()``."""
        deps = {
            "rg": (False, "not found"),
            "fzf": (False, "not found"),
            "jq": (False, "not found"),
            "qmd": (False, "not found"),
        }
        db_path = tmp_path / ".simba" / "simba.db"
        with (
            unittest.mock.patch("simba.search.deps.check_all", return_value=deps),
            unittest.mock.patch(
                "simba.db.get_db_path",
                return_value=db_path,
            ),
            unittest.mock.patch.object(sys, "argv", ["simba search", "index"]),
            unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path),
        ):
            code = simba.search.__main__.main()

        assert code == 0
