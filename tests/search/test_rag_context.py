"""Tests for search.rag_context -- RAG context orchestration."""

from __future__ import annotations

import pathlib
import unittest.mock

import pytest

import simba.search.rag_context

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cwd(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return a temporary working directory for tests."""
    return tmp_path


# ---------------------------------------------------------------------------
# TestBuildContext
# ---------------------------------------------------------------------------


class TestBuildContext:
    def test_short_prompt_returns_empty(self, cwd: pathlib.Path) -> None:
        result = simba.search.rag_context.build_context("short", cwd)
        assert result == ""

    def test_command_like_prompt_returns_empty(self, cwd: pathlib.Path) -> None:
        for prompt in [
            "/commit all the changes now",
            "git status please show me",
            "yes I want to proceed with this",
            "hello can you help me out",
            "push the branch to remote",
        ]:
            result = simba.search.rag_context.build_context(prompt, cwd)
            assert result == "", f"Expected empty for: {prompt!r}"

    def test_all_stopword_prompt_returns_empty(self, cwd: pathlib.Path) -> None:
        # All words are stop words so extract_search_terms returns ""
        result = simba.search.rag_context.build_context(
            "please help me find and read this", cwd
        )
        assert result == ""

    def test_includes_memory_context_when_db_exists(self, cwd: pathlib.Path) -> None:
        mock_conn = unittest.mock.MagicMock()
        with (
            unittest.mock.patch(
                "simba.search.project_memory.get_connection",
                return_value=mock_conn,
            ),
            unittest.mock.patch(
                "simba.search.project_memory.get_context",
                return_value="## Project Facts\n- Use ruff for linting",
            ),
            unittest.mock.patch("simba.search.qmd.is_available", return_value=False),
        ):
            result = simba.search.rag_context.build_context(
                "how does the authentication module work", cwd
            )
        assert "# Memory Context" in result
        assert "Use ruff for linting" in result
        mock_conn.close.assert_called_once()

    def test_includes_code_context_when_qmd_available(self, cwd: pathlib.Path) -> None:
        qmd_results = [
            {
                "path": "src/auth.py",
                "snippet": "def authenticate(user):",
                "score": "0.85",
            }
        ]
        with (
            unittest.mock.patch(
                "simba.search.project_memory.get_connection",
                return_value=None,
            ),
            unittest.mock.patch("simba.search.qmd.is_available", return_value=True),
            unittest.mock.patch("simba.search.qmd.search", return_value=qmd_results),
        ):
            result = simba.search.rag_context.build_context(
                "how does the authentication module work", cwd
            )
        assert "# Code Context" in result
        assert "src/auth.py" in result
        assert "relevance: 0.85" in result
        assert "def authenticate(user):" in result

    def test_combines_both_sources(self, cwd: pathlib.Path) -> None:
        mock_conn = unittest.mock.MagicMock()
        qmd_results = [
            {
                "path": "src/db.py",
                "snippet": "class Database:",
                "score": "0.90",
            }
        ]
        with (
            unittest.mock.patch(
                "simba.search.project_memory.get_connection",
                return_value=mock_conn,
            ),
            unittest.mock.patch(
                "simba.search.project_memory.get_context",
                return_value="## Recent Work\n- Migrated database",
            ),
            unittest.mock.patch("simba.search.qmd.is_available", return_value=True),
            unittest.mock.patch("simba.search.qmd.search", return_value=qmd_results),
        ):
            result = simba.search.rag_context.build_context(
                "how does the database connection pooling work", cwd
            )
        assert "# Memory Context" in result
        assert "# Code Context" in result
        assert "Migrated database" in result
        assert "src/db.py" in result

    def test_no_crash_when_qmd_fails(self, cwd: pathlib.Path) -> None:
        with (
            unittest.mock.patch(
                "simba.search.project_memory.get_connection",
                return_value=None,
            ),
            unittest.mock.patch(
                "simba.search.qmd.is_available",
                side_effect=RuntimeError("qmd exploded"),
            ),
        ):
            result = simba.search.rag_context.build_context(
                "how does the authentication module work", cwd
            )
        assert result == ""

    def test_no_crash_when_db_connection_fails(self, cwd: pathlib.Path) -> None:
        with (
            unittest.mock.patch(
                "simba.search.project_memory.get_connection",
                side_effect=OSError("disk on fire"),
            ),
            unittest.mock.patch("simba.search.qmd.is_available", return_value=False),
        ):
            result = simba.search.rag_context.build_context(
                "how does the authentication module work", cwd
            )
        # Should not crash; returns empty because no context was gathered
        assert result == ""

    def test_xml_wrapper_present_in_output(self, cwd: pathlib.Path) -> None:
        mock_conn = unittest.mock.MagicMock()
        with (
            unittest.mock.patch(
                "simba.search.project_memory.get_connection",
                return_value=mock_conn,
            ),
            unittest.mock.patch(
                "simba.search.project_memory.get_context",
                return_value="## Project Facts\n- some fact",
            ),
            unittest.mock.patch("simba.search.qmd.is_available", return_value=False),
        ):
            result = simba.search.rag_context.build_context(
                "how does the authentication module work", cwd
            )
        assert result.startswith('<relevant-context source="project-search">')
        assert result.endswith("</relevant-context>")

    def test_search_terms_shown_in_output(self, cwd: pathlib.Path) -> None:
        mock_conn = unittest.mock.MagicMock()
        with (
            unittest.mock.patch(
                "simba.search.project_memory.get_connection",
                return_value=mock_conn,
            ),
            unittest.mock.patch(
                "simba.search.project_memory.get_context",
                return_value="## Project Facts\n- fact here",
            ),
            unittest.mock.patch("simba.search.qmd.is_available", return_value=False),
        ):
            result = simba.search.rag_context.build_context(
                "how does the authentication module work", cwd
            )
        assert "**Search terms:**" in result
        # "authentication" and "module" are stop words, but "authentication"
        # is not in the stop list so it should appear
        assert "authentication" in result
