"""Tests for search.qmd — wrapper around the qmd CLI tool."""

from __future__ import annotations

import json
import pathlib
import subprocess
import unittest
import unittest.mock

import simba.search.qmd

# ---------------------------------------------------------------------------
# TestIsAvailable
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_returns_true_when_qmd_found(self) -> None:
        with unittest.mock.patch(
            "simba.search.qmd.shutil.which", return_value="/usr/bin/qmd"
        ):
            assert simba.search.qmd.is_available() is True

    def test_returns_false_when_not_found(self) -> None:
        with unittest.mock.patch("simba.search.qmd.shutil.which", return_value=None):
            assert simba.search.qmd.is_available() is False


# ---------------------------------------------------------------------------
# TestExtractSearchTerms
# ---------------------------------------------------------------------------


class TestExtractSearchTerms:
    def test_removes_stop_words(self) -> None:
        result = simba.search.qmd.extract_search_terms("find the authentication module")
        assert "the" not in result.split()
        assert "find" not in result.split()
        assert "authentication" in result.split()
        assert "module" in result.split()

    def test_filters_short_words(self) -> None:
        result = simba.search.qmd.extract_search_terms("go do it now py")
        # "go", "do", "it", "py" are all < 3 chars and/or stop words
        assert result == "now"

    def test_limits_to_max_8_terms(self) -> None:
        prompt = " ".join(f"term{i}" for i in range(20))
        result = simba.search.qmd.extract_search_terms(prompt)
        assert len(result.split()) <= 8

    def test_returns_empty_for_all_stopword_input(self) -> None:
        result = simba.search.qmd.extract_search_terms("find the is are was were")
        assert result == ""


# ---------------------------------------------------------------------------
# TestSearch
# ---------------------------------------------------------------------------


class TestSearch:
    def test_returns_parsed_results(self) -> None:
        mock_data = [
            {"path": "qmd://src/foo.py", "snippet": "def foo():", "score": 0.95},
            {"path": "src/bar.py", "snippet": "class Bar:", "score": 0.8},
        ]
        mock_result = unittest.mock.Mock()
        mock_result.stdout = json.dumps(mock_data)
        with unittest.mock.patch(
            "simba.search.qmd.subprocess.run", return_value=mock_result
        ):
            results = simba.search.qmd.search("foo bar")
        assert len(results) == 2
        # qmd:// prefix should be stripped
        assert results[0]["path"] == "src/foo.py"
        assert results[0]["snippet"] == "def foo():"
        assert results[0]["score"] == "0.95"
        assert results[1]["path"] == "src/bar.py"

    def test_returns_empty_list_on_subprocess_error(self) -> None:
        with unittest.mock.patch(
            "simba.search.qmd.subprocess.run",
            side_effect=subprocess.SubprocessError("timeout"),
        ):
            results = simba.search.qmd.search("query")
        assert results == []

    def test_returns_empty_list_on_json_parse_error(self) -> None:
        mock_result = unittest.mock.Mock()
        mock_result.stdout = "not valid json"
        with unittest.mock.patch(
            "simba.search.qmd.subprocess.run", return_value=mock_result
        ):
            results = simba.search.qmd.search("query")
        assert results == []


# ---------------------------------------------------------------------------
# TestFormatFileSuggestions
# ---------------------------------------------------------------------------


class TestFormatFileSuggestions:
    def test_returns_xml_tagged_output(self) -> None:
        results = [
            {"path": "src/foo.py", "snippet": "def foo():", "score": "0.9"},
            {"path": "src/bar.py", "snippet": "class Bar:", "score": "0.8"},
        ]
        output = simba.search.qmd.format_file_suggestions(results)
        assert output.startswith("<qmd-context>")
        assert output.endswith("</qmd-context>")
        assert "src/foo.py" in output
        assert "src/bar.py" in output

    def test_returns_empty_string_for_empty_list(self) -> None:
        output = simba.search.qmd.format_file_suggestions([])
        assert output == ""


# ---------------------------------------------------------------------------
# Collection scoping
# ---------------------------------------------------------------------------

INDEX_YML = """\
collections:
  kirafin:
    path: {kirafin}
    pattern: "**/*.md"
  examples:
    path: "{examples}"
    pattern: "**/*.md"
  pi-stuff:
    path: {pi}
contexts: {{}}
"""


def _index(tmp_path: pathlib.Path) -> tuple[pathlib.Path, dict[str, pathlib.Path]]:
    dirs = {
        "kirafin": tmp_path / "kirafin",
        "examples": tmp_path / "kirafin" / "docs" / "examples",
        "pi": tmp_path / "pi-stuff",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "index.yml"
    config.write_text(INDEX_YML.format(**{k: str(v) for k, v in dirs.items()}))
    return config, dirs


class TestCollectionScoping:
    def test_reads_collection_paths(self, tmp_path: pathlib.Path) -> None:
        config, dirs = _index(tmp_path)
        assert simba.search.qmd.collection_paths(config) == {
            "kirafin": str(dirs["kirafin"]),
            "examples": str(dirs["examples"]),
            "pi-stuff": str(dirs["pi"]),
        }

    def test_missing_config_has_no_collections(self, tmp_path: pathlib.Path) -> None:
        assert simba.search.qmd.collection_paths(tmp_path / "absent.yml") == {}

    def test_innermost_collection_containing_cwd_wins(
        self, tmp_path: pathlib.Path
    ) -> None:
        config, dirs = _index(tmp_path)
        inner = dirs["examples"] / "deep"
        inner.mkdir()
        assert simba.search.qmd.collection_for(inner, config) == "examples"
        assert simba.search.qmd.collection_for(dirs["kirafin"], config) == "kirafin"
        assert simba.search.qmd.collection_for(dirs["pi"], config) == "pi-stuff"

    def test_unrelated_or_parent_directory_has_no_collection(
        self, tmp_path: pathlib.Path
    ) -> None:
        config, _ = _index(tmp_path)
        other = tmp_path / "simba"
        other.mkdir()
        assert simba.search.qmd.collection_for(other, config) is None
        assert simba.search.qmd.collection_for(tmp_path, config) is None

    def test_symlinked_cwd_matches_its_target(self, tmp_path: pathlib.Path) -> None:
        config, dirs = _index(tmp_path)
        link = tmp_path / "link"
        link.symlink_to(dirs["pi"])
        assert simba.search.qmd.collection_for(link, config) == "pi-stuff"

    def test_search_filters_by_collection_and_reads_file_field(self) -> None:
        mock_result = unittest.mock.Mock()
        mock_result.stdout = json.dumps(
            [{"file": "qmd://proj/docs/a.md", "snippet": "s", "score": 1}]
        )
        with unittest.mock.patch(
            "simba.search.qmd.subprocess.run", return_value=mock_result
        ) as run:
            results = simba.search.qmd.search("terms", collection="proj")
        cmd = run.call_args.args[0]
        assert cmd[cmd.index("-c") + 1] == "proj"
        assert results[0]["path"] == "proj/docs/a.md"
