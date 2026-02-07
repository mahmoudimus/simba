"""Tests for search.rag_context -- RAG context orchestration."""

from __future__ import annotations

import pathlib
import unittest.mock

import pytest

import simba.db
import simba.search.project_memory
import simba.search.rag_context

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cwd(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return a temporary working directory for tests."""
    return tmp_path


@pytest.fixture(autouse=True)
def _use_tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point simba.db at a temp directory so tests use an isolated SQLite DB."""
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


def _seed_db(cwd: pathlib.Path) -> None:
    """Populate the database with test data via the real project_memory API."""
    with simba.db.get_db(cwd) as conn:
        simba.search.project_memory.add_fact(conn, "Use ruff for linting", "tooling")
        simba.search.project_memory.add_fact(
            conn, "Authentication uses JWT tokens", "architecture"
        )
        simba.search.project_memory.add_knowledge(
            conn,
            "auth",
            "Authentication module handles JWT validation and session management",
            "decorator-based auth, middleware pattern",
        )
        simba.search.project_memory.add_knowledge(
            conn,
            "database",
            "Database layer uses SQLAlchemy with connection pooling",
            "repository pattern, unit of work",
        )
        simba.search.project_memory.add_session(
            conn,
            summary="Migrated database to async driver",
            files_touched="src/db.py,src/models.py",
            tools_used="Read,Write,Bash",
            topics="database,migration,async",
        )
        simba.search.project_memory.add_session(
            conn,
            summary="Fixed authentication token refresh bug",
            files_touched="src/auth.py",
            tools_used="Read,Edit",
            topics="authentication,bugfix",
        )


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
        _seed_db(cwd)
        with unittest.mock.patch("simba.search.qmd.is_available", return_value=False):
            result = simba.search.rag_context.build_context(
                "how does the authentication module work", cwd
            )
        assert "# Memory Context" in result
        # get_context returns facts, and the auth-related knowledge/session
        assert "## Project Facts" in result
        assert "Use ruff for linting" in result

    def test_includes_code_context_when_qmd_available(self, cwd: pathlib.Path) -> None:
        qmd_results = [
            {
                "path": "src/auth.py",
                "snippet": "def authenticate(user):",
                "score": "0.85",
            }
        ]
        # No DB seeded -- get_connection returns None when file doesn't exist
        with (
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
        _seed_db(cwd)
        qmd_results = [
            {
                "path": "src/db.py",
                "snippet": "class Database:",
                "score": "0.90",
            }
        ]
        with (
            unittest.mock.patch("simba.search.qmd.is_available", return_value=True),
            unittest.mock.patch("simba.search.qmd.search", return_value=qmd_results),
        ):
            result = simba.search.rag_context.build_context(
                "how does the database connection pooling work", cwd
            )
        assert "# Memory Context" in result
        assert "# Code Context" in result
        # Real get_context returns knowledge matching "database"
        has_db = "Database layer uses SQLAlchemy" in result
        assert has_db or "database" in result.lower()
        assert "src/db.py" in result

    def test_no_crash_when_qmd_fails(self, cwd: pathlib.Path) -> None:
        # DB not seeded, so get_connection returns None; qmd blows up
        with unittest.mock.patch(
            "simba.search.qmd.is_available",
            side_effect=RuntimeError("qmd exploded"),
        ):
            result = simba.search.rag_context.build_context(
                "how does the authentication module work", cwd
            )
        assert result == ""

    def test_no_crash_when_db_connection_fails(self, cwd: pathlib.Path) -> None:
        # Force get_connection to raise by making get_db_path raise
        with (
            unittest.mock.patch(
                "simba.db.get_connection",
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
        _seed_db(cwd)
        with unittest.mock.patch("simba.search.qmd.is_available", return_value=False):
            result = simba.search.rag_context.build_context(
                "how does the authentication module work", cwd
            )
        assert result.startswith('<relevant-context source="project-search">')
        assert result.endswith("</relevant-context>")

    def test_search_terms_shown_in_output(self, cwd: pathlib.Path) -> None:
        _seed_db(cwd)
        with unittest.mock.patch("simba.search.qmd.is_available", return_value=False):
            result = simba.search.rag_context.build_context(
                "how does the authentication module work", cwd
            )
        assert "**Search terms:**" in result
        # "authentication" is not in the stop list so it should appear
        assert "authentication" in result
