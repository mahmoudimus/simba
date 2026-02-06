"""Tests for search.qmd — wrapper around the qmd CLI tool."""

from __future__ import annotations

import json
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
