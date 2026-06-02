"""Tests for search.project_memory -- SQLite-backed per-project memory."""

from __future__ import annotations

import pathlib
import sqlite3

import pytest

import simba.db
import simba.search.project_memory as pm

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cwd(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Point simba.db.get_db_path at a tmp DB; functions use connect(cwd=None)."""
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda c=None: db_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FTS5_AVAILABLE: bool | None = None


def _has_fts5() -> bool:
    """Return True when the current SQLite build supports FTS5."""
    global _FTS5_AVAILABLE
    if _FTS5_AVAILABLE is None:
        try:
            c = sqlite3.connect(":memory:")
            c.execute("CREATE VIRTUAL TABLE _fts5_probe USING fts5(x)")
            _FTS5_AVAILABLE = True
            c.close()
        except sqlite3.OperationalError:
            _FTS5_AVAILABLE = False
    return _FTS5_AVAILABLE


# ---------------------------------------------------------------------------
# TestSchema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_creates_db_file(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = tmp_path / "sub" / ".simba" / "simba.db"
        monkeypatch.setattr(simba.db, "get_db_path", lambda c=None: db_path)
        with simba.db.connect(tmp_path):
            pass
        assert db_path.exists()

    def test_tables_exist(self, cwd: pathlib.Path) -> None:
        with simba.db.get_db(cwd) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        table_names = {row["name"] for row in rows}
        assert "sessions" in table_names
        assert "knowledge" in table_names
        assert "facts" in table_names

    def test_fts_table_exists_when_supported(self, cwd: pathlib.Path) -> None:
        with simba.db.get_db(cwd) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        table_names = {row["name"] for row in rows}
        if _has_fts5():
            assert "memory_fts" in table_names
        else:
            assert "memory_fts" not in table_names


# ---------------------------------------------------------------------------
# TestAddSession
# ---------------------------------------------------------------------------


class TestAddSession:
    def test_returns_positive_rowid(self, cwd: pathlib.Path) -> None:
        rowid = pm.add_session(
            summary="Fixed auth bug",
            files_touched="auth.py",
            tools_used="grep,read",
            topics="auth,security",
        )
        assert rowid > 0

    def test_data_retrievable(self, cwd: pathlib.Path) -> None:
        pm.add_session(
            summary="Refactored tests",
            files_touched="tests/test_foo.py",
            tools_used="edit",
            topics="testing",
        )
        with simba.db.get_db(cwd) as conn:
            row = conn.execute(
                "SELECT summary, files_touched, tools_used, topics "
                "FROM sessions WHERE id = 1"
            ).fetchone()
        assert row["summary"] == "Refactored tests"
        assert row["files_touched"] == "tests/test_foo.py"
        assert row["tools_used"] == "edit"
        assert row["topics"] == "testing"

    def test_multiple_sessions_get_different_ids(self, cwd: pathlib.Path) -> None:
        id1 = pm.add_session("s1", "f1", "t1", "topic1")
        id2 = pm.add_session("s2", "f2", "t2", "topic2")
        assert id1 != id2


# ---------------------------------------------------------------------------
# TestAddKnowledge
# ---------------------------------------------------------------------------


class TestAddKnowledge:
    def test_returns_positive_rowid(self, cwd: pathlib.Path) -> None:
        rowid = pm.add_knowledge(
            area="auth",
            summary="JWT-based authentication",
            patterns="middleware chain",
        )
        assert rowid > 0

    def test_upsert_updates_existing_area(self, cwd: pathlib.Path) -> None:
        pm.add_knowledge(area="auth", summary="v1", patterns="p1")
        pm.add_knowledge(area="auth", summary="v2", patterns="p2")
        with simba.db.get_db(cwd) as conn:
            row = conn.execute(
                "SELECT summary, patterns FROM knowledge WHERE area = 'auth'"
            ).fetchone()
        assert row["summary"] == "v2"
        assert row["patterns"] == "p2"

    def test_upsert_preserves_other_areas(self, cwd: pathlib.Path) -> None:
        pm.add_knowledge(area="auth", summary="auth-summary", patterns="ap")
        pm.add_knowledge(area="db", summary="db-summary", patterns="dp")
        pm.add_knowledge(area="auth", summary="auth-v2", patterns="ap2")
        with simba.db.get_db(cwd) as conn:
            db_row = conn.execute(
                "SELECT summary FROM knowledge WHERE area = 'db'"
            ).fetchone()
        assert db_row["summary"] == "db-summary"


# ---------------------------------------------------------------------------
# TestAddFact
# ---------------------------------------------------------------------------


class TestAddFact:
    def test_default_category(self, cwd: pathlib.Path) -> None:
        rowid = pm.add_fact(fact="Use ruff for linting")
        assert rowid > 0
        with simba.db.get_db(cwd) as conn:
            row = conn.execute(
                "SELECT fact, category FROM facts WHERE id = ?", (rowid,)
            ).fetchone()
        assert row["fact"] == "Use ruff for linting"
        assert row["category"] == "general"

    def test_custom_category(self, cwd: pathlib.Path) -> None:
        rowid = pm.add_fact(fact="Pin numpy to 1.26", category="dependency")
        with simba.db.get_db(cwd) as conn:
            row = conn.execute(
                "SELECT category FROM facts WHERE id = ?", (rowid,)
            ).fetchone()
        assert row["category"] == "dependency"


# ---------------------------------------------------------------------------
# TestSearchFts
# ---------------------------------------------------------------------------


class TestSearchFts:
    @pytest.mark.skipif(not _has_fts5(), reason="FTS5 not available")
    def test_finds_matching_content(self, cwd: pathlib.Path) -> None:
        pm.add_session(
            summary="Implemented caching layer for Redis",
            files_touched="cache.py",
            tools_used="edit",
            topics="caching,redis",
        )
        results = pm.search_fts("caching")
        assert len(results) >= 1
        assert results[0]["source_type"] == "session"

    def test_empty_query_returns_empty(self, cwd: pathlib.Path) -> None:
        assert pm.search_fts("") == []

    def test_special_characters_do_not_crash(self, cwd: pathlib.Path) -> None:
        # FTS5 special chars: * " ( ) : ^
        results = pm.search_fts('*(foo) "bar": ^baz')
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# TestGetContext
# ---------------------------------------------------------------------------


class TestGetContext:
    def test_includes_facts_section(self, cwd: pathlib.Path) -> None:
        pm.add_fact(fact="Always run linter before commit")
        ctx = pm.get_context(query="")
        assert "## Project Facts" in ctx
        assert "Always run linter before commit" in ctx

    def test_includes_knowledge_when_query_matches(self, cwd: pathlib.Path) -> None:
        pm.add_knowledge(area="auth", summary="JWT-based auth", patterns="middleware")
        ctx = pm.get_context(query="auth")
        assert "## Relevant Code Areas" in ctx
        assert "auth" in ctx

    def test_truncates_to_token_budget(self, cwd: pathlib.Path) -> None:
        for i in range(20):
            pm.add_fact(fact=f"Fact number {i} with extra padding words")
        ctx = pm.get_context(query="", token_budget=10)
        # 10 tokens * 4 chars = 40 chars max
        assert len(ctx) <= 40

    def test_empty_db_returns_empty_string(self, cwd: pathlib.Path) -> None:
        assert pm.get_context(query="test") == ""


# ---------------------------------------------------------------------------
# TestGetRecentSessions
# ---------------------------------------------------------------------------


class TestGetRecentSessions:
    def test_returns_sessions_reverse_chronological(self, cwd: pathlib.Path) -> None:
        # Insert with explicit timestamps to guarantee ordering.
        with simba.db.get_db(cwd) as conn:
            for summary, ts in [
                ("first", "2024-01-01 00:00:00"),
                ("second", "2024-01-02 00:00:00"),
                ("third", "2024-01-03 00:00:00"),
            ]:
                conn.execute(
                    "INSERT INTO sessions"
                    " (summary, files_touched, tools_used, topics, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (summary, "f", "t", "topic", ts),
                )
            conn.commit()

        sessions = pm.get_recent_sessions()
        assert len(sessions) == 3
        assert sessions[0]["summary"] == "third"
        assert sessions[1]["summary"] == "second"
        assert sessions[2]["summary"] == "first"

    def test_respects_limit(self, cwd: pathlib.Path) -> None:
        for i in range(10):
            pm.add_session(f"s{i}", f"f{i}", f"t{i}", f"topic{i}")
        sessions = pm.get_recent_sessions(limit=3)
        assert len(sessions) == 3


# ---------------------------------------------------------------------------
# TestGetStats
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_empty_db_returns_zeros(self, cwd: pathlib.Path) -> None:
        assert pm.get_stats(cwd) == {"sessions": 0, "knowledge": 0, "facts": 0}

    def test_counts_correct_after_inserts(self, cwd: pathlib.Path) -> None:
        pm.add_session("s", "f", "t", "topic")
        pm.add_session("s2", "f2", "t2", "topic2")
        pm.add_knowledge("auth", "summary", "patterns")
        pm.add_fact("fact1")
        pm.add_fact("fact2")
        pm.add_fact("fact3")
        assert pm.get_stats(cwd) == {"sessions": 2, "knowledge": 1, "facts": 3}


# ---------------------------------------------------------------------------
# TestEscapeFtsQuery
# ---------------------------------------------------------------------------


class TestEscapeFtsQuery:
    def test_plain_terms_quoted(self) -> None:
        assert pm._escape_fts_query("hello world") == '"hello" "world"'

    def test_special_chars_stripped(self) -> None:
        assert pm._escape_fts_query('"quoted"') == '"quoted"'

    def test_empty_string(self) -> None:
        assert pm._escape_fts_query("") == ""
