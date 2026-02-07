"""Tests for search.project_memory — SQLite-backed per-project memory."""

from __future__ import annotations

import collections.abc
import pathlib
import sqlite3

import pytest

import simba.search.project_memory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_conn(tmp_path: pathlib.Path) -> collections.abc.Generator[sqlite3.Connection]:
    """Create a temporary DB via init_db and return the connection."""
    db_path = tmp_path / "test_memory.db"
    conn = simba.search.project_memory.init_db(db_path)
    yield conn
    conn.close()


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
# TestInitDb
# ---------------------------------------------------------------------------


class TestInitDb:
    def test_creates_db_file(self, tmp_path: pathlib.Path) -> None:
        db_path = tmp_path / "sub" / "memory.db"
        conn = simba.search.project_memory.init_db(db_path)
        try:
            assert db_path.exists()
        finally:
            conn.close()

    def test_tables_exist(self, db_conn: sqlite3.Connection) -> None:
        rows = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {row["name"] for row in rows}
        assert "sessions" in table_names
        assert "knowledge" in table_names
        assert "facts" in table_names

    def test_fts_table_exists_when_supported(self, db_conn: sqlite3.Connection) -> None:
        rows = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {row["name"] for row in rows}
        if _has_fts5():
            assert "memory_fts" in table_names
        else:
            # FTS5 not available; the table should simply be absent.
            assert "memory_fts" not in table_names


# ---------------------------------------------------------------------------
# TestAddSession
# ---------------------------------------------------------------------------


class TestAddSession:
    def test_returns_positive_rowid(self, db_conn: sqlite3.Connection) -> None:
        rowid = simba.search.project_memory.add_session(
            db_conn,
            summary="Fixed auth bug",
            files_touched="auth.py",
            tools_used="grep,read",
            topics="auth,security",
        )
        assert rowid > 0

    def test_data_retrievable(self, db_conn: sqlite3.Connection) -> None:
        simba.search.project_memory.add_session(
            db_conn,
            summary="Refactored tests",
            files_touched="tests/test_foo.py",
            tools_used="edit",
            topics="testing",
        )
        row = db_conn.execute(
            "SELECT summary, files_touched, tools_used, topics "
            "FROM sessions WHERE id = 1"
        ).fetchone()
        assert row["summary"] == "Refactored tests"
        assert row["files_touched"] == "tests/test_foo.py"
        assert row["tools_used"] == "edit"
        assert row["topics"] == "testing"

    def test_multiple_sessions_get_different_ids(
        self, db_conn: sqlite3.Connection
    ) -> None:
        id1 = simba.search.project_memory.add_session(
            db_conn, "s1", "f1", "t1", "topic1"
        )
        id2 = simba.search.project_memory.add_session(
            db_conn, "s2", "f2", "t2", "topic2"
        )
        assert id1 != id2


# ---------------------------------------------------------------------------
# TestAddKnowledge
# ---------------------------------------------------------------------------


class TestAddKnowledge:
    def test_returns_positive_rowid(self, db_conn: sqlite3.Connection) -> None:
        rowid = simba.search.project_memory.add_knowledge(
            db_conn,
            area="auth",
            summary="JWT-based authentication",
            patterns="middleware chain",
        )
        assert rowid > 0

    def test_upsert_updates_existing_area(self, db_conn: sqlite3.Connection) -> None:
        simba.search.project_memory.add_knowledge(
            db_conn, area="auth", summary="v1", patterns="p1"
        )
        simba.search.project_memory.add_knowledge(
            db_conn, area="auth", summary="v2", patterns="p2"
        )
        row = db_conn.execute(
            "SELECT summary, patterns FROM knowledge WHERE area = 'auth'"
        ).fetchone()
        assert row["summary"] == "v2"
        assert row["patterns"] == "p2"

    def test_upsert_preserves_other_areas(self, db_conn: sqlite3.Connection) -> None:
        simba.search.project_memory.add_knowledge(
            db_conn, area="auth", summary="auth-summary", patterns="ap"
        )
        simba.search.project_memory.add_knowledge(
            db_conn, area="db", summary="db-summary", patterns="dp"
        )
        # Update auth only
        simba.search.project_memory.add_knowledge(
            db_conn, area="auth", summary="auth-v2", patterns="ap2"
        )
        db_row = db_conn.execute(
            "SELECT summary FROM knowledge WHERE area = 'db'"
        ).fetchone()
        assert db_row["summary"] == "db-summary"


# ---------------------------------------------------------------------------
# TestAddFact
# ---------------------------------------------------------------------------


class TestAddFact:
    def test_default_category(self, db_conn: sqlite3.Connection) -> None:
        rowid = simba.search.project_memory.add_fact(
            db_conn, fact="Use ruff for linting"
        )
        assert rowid > 0
        row = db_conn.execute(
            "SELECT fact, category FROM facts WHERE id = ?", (rowid,)
        ).fetchone()
        assert row["fact"] == "Use ruff for linting"
        assert row["category"] == "general"

    def test_custom_category(self, db_conn: sqlite3.Connection) -> None:
        rowid = simba.search.project_memory.add_fact(
            db_conn, fact="Pin numpy to 1.26", category="dependency"
        )
        row = db_conn.execute(
            "SELECT category FROM facts WHERE id = ?", (rowid,)
        ).fetchone()
        assert row["category"] == "dependency"


# ---------------------------------------------------------------------------
# TestSearchFts
# ---------------------------------------------------------------------------


class TestSearchFts:
    @pytest.mark.skipif(not _has_fts5(), reason="FTS5 not available")
    def test_finds_matching_content(self, db_conn: sqlite3.Connection) -> None:
        simba.search.project_memory.add_session(
            db_conn,
            summary="Implemented caching layer for Redis",
            files_touched="cache.py",
            tools_used="edit",
            topics="caching,redis",
        )
        results = simba.search.project_memory.search_fts(db_conn, "caching")
        assert len(results) >= 1
        assert results[0]["source_type"] == "session"

    def test_empty_query_returns_empty(self, db_conn: sqlite3.Connection) -> None:
        results = simba.search.project_memory.search_fts(db_conn, "")
        assert results == []

    def test_special_characters_do_not_crash(self, db_conn: sqlite3.Connection) -> None:
        # FTS5 special chars: * " ( ) : ^
        results = simba.search.project_memory.search_fts(db_conn, '*(foo) "bar": ^baz')
        # Should not raise; may return empty list
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# TestGetContext
# ---------------------------------------------------------------------------


class TestGetContext:
    def test_includes_facts_section(self, db_conn: sqlite3.Connection) -> None:
        simba.search.project_memory.add_fact(
            db_conn, fact="Always run linter before commit"
        )
        ctx = simba.search.project_memory.get_context(db_conn, query="")
        assert "## Project Facts" in ctx
        assert "Always run linter before commit" in ctx

    def test_includes_knowledge_when_query_matches(
        self, db_conn: sqlite3.Connection
    ) -> None:
        simba.search.project_memory.add_knowledge(
            db_conn, area="auth", summary="JWT-based auth", patterns="middleware"
        )
        ctx = simba.search.project_memory.get_context(db_conn, query="auth")
        assert "## Relevant Code Areas" in ctx
        assert "auth" in ctx

    def test_truncates_to_token_budget(self, db_conn: sqlite3.Connection) -> None:
        # Insert enough data to exceed a tiny budget
        for i in range(20):
            simba.search.project_memory.add_fact(
                db_conn, fact=f"Fact number {i} with extra padding words"
            )
        ctx = simba.search.project_memory.get_context(
            db_conn, query="", token_budget=10
        )
        # 10 tokens * 4 chars = 40 chars max
        assert len(ctx) <= 40

    def test_empty_db_returns_empty_string(self, db_conn: sqlite3.Connection) -> None:
        ctx = simba.search.project_memory.get_context(db_conn, query="test")
        assert ctx == ""


# ---------------------------------------------------------------------------
# TestGetRecentSessions
# ---------------------------------------------------------------------------


class TestGetRecentSessions:
    def test_returns_sessions_reverse_chronological(
        self, db_conn: sqlite3.Connection
    ) -> None:
        # Insert with explicit timestamps to guarantee ordering.
        db_conn.execute(
            "INSERT INTO sessions"
            " (summary, files_touched, tools_used, topics, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            ("first", "f1", "t1", "topic1", "2024-01-01 00:00:00"),
        )
        db_conn.execute(
            "INSERT INTO sessions"
            " (summary, files_touched, tools_used, topics, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            ("second", "f2", "t2", "topic2", "2024-01-02 00:00:00"),
        )
        db_conn.execute(
            "INSERT INTO sessions"
            " (summary, files_touched, tools_used, topics, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            ("third", "f3", "t3", "topic3", "2024-01-03 00:00:00"),
        )
        db_conn.commit()

        sessions = simba.search.project_memory.get_recent_sessions(db_conn)
        assert len(sessions) == 3
        assert sessions[0]["summary"] == "third"
        assert sessions[1]["summary"] == "second"
        assert sessions[2]["summary"] == "first"

    def test_respects_limit(self, db_conn: sqlite3.Connection) -> None:
        for i in range(10):
            simba.search.project_memory.add_session(
                db_conn, f"s{i}", f"f{i}", f"t{i}", f"topic{i}"
            )
        sessions = simba.search.project_memory.get_recent_sessions(db_conn, limit=3)
        assert len(sessions) == 3


# ---------------------------------------------------------------------------
# TestGetStats
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_empty_db_returns_zeros(self, db_conn: sqlite3.Connection) -> None:
        stats = simba.search.project_memory.get_stats(db_conn)
        assert stats == {"sessions": 0, "knowledge": 0, "facts": 0}

    def test_counts_correct_after_inserts(self, db_conn: sqlite3.Connection) -> None:
        simba.search.project_memory.add_session(db_conn, "s", "f", "t", "topic")
        simba.search.project_memory.add_session(db_conn, "s2", "f2", "t2", "topic2")
        simba.search.project_memory.add_knowledge(
            db_conn, "auth", "summary", "patterns"
        )
        simba.search.project_memory.add_fact(db_conn, "fact1")
        simba.search.project_memory.add_fact(db_conn, "fact2")
        simba.search.project_memory.add_fact(db_conn, "fact3")

        stats = simba.search.project_memory.get_stats(db_conn)
        assert stats == {"sessions": 2, "knowledge": 1, "facts": 3}


# ---------------------------------------------------------------------------
# TestFindRepoRoot
# ---------------------------------------------------------------------------


class TestFindRepoRoot:
    def test_finds_git_directory(self, tmp_path: pathlib.Path) -> None:
        # Arrange: create repo structure
        repo = tmp_path / "myrepo"
        repo.mkdir()
        (repo / ".git").mkdir()
        subdir = repo / "src" / "pkg"
        subdir.mkdir(parents=True)

        # Act
        result = simba.search.project_memory.find_repo_root(subdir)

        # Assert
        assert result is not None
        assert result == repo.resolve()

    def test_returns_none_when_no_git(self, tmp_path: pathlib.Path) -> None:
        # tmp_path has no .git
        result = simba.search.project_memory.find_repo_root(tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# TestGetDbPath
# ---------------------------------------------------------------------------


class TestGetDbPath:
    def test_returns_correct_path_structure(self, tmp_path: pathlib.Path) -> None:
        # Arrange: create a repo with .git
        repo = tmp_path / "myrepo"
        repo.mkdir()
        (repo / ".git").mkdir()

        # Act
        db_path = simba.search.project_memory.get_db_path(repo)

        # Assert
        assert db_path == repo.resolve() / ".simba" / "search" / "memory.db"


# ---------------------------------------------------------------------------
# TestGetConnection
# ---------------------------------------------------------------------------


class TestGetConnection:
    def test_returns_none_when_db_does_not_exist(self, tmp_path: pathlib.Path) -> None:
        conn = simba.search.project_memory.get_connection(tmp_path)
        assert conn is None

    def test_returns_connection_when_db_exists(self, tmp_path: pathlib.Path) -> None:
        # Arrange: create a repo with .git and an initialized DB
        repo = tmp_path / "myrepo"
        repo.mkdir()
        (repo / ".git").mkdir()
        db_path = repo / ".simba" / "search" / "memory.db"
        init_conn = simba.search.project_memory.init_db(db_path)
        init_conn.close()

        # Act
        conn = simba.search.project_memory.get_connection(repo)

        # Assert
        try:
            assert conn is not None
            # Verify it's a working connection
            row = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()
            assert row["c"] == 0
        finally:
            if conn is not None:
                conn.close()


# ---------------------------------------------------------------------------
# TestEscapeFtsQuery
# ---------------------------------------------------------------------------


class TestEscapeFtsQuery:
    def test_plain_terms_quoted(self) -> None:
        result = simba.search.project_memory._escape_fts_query("hello world")
        assert result == '"hello" "world"'

    def test_special_chars_stripped(self) -> None:
        result = simba.search.project_memory._escape_fts_query('"quoted"')
        assert result == '"quoted"'

    def test_empty_string(self) -> None:
        result = simba.search.project_memory._escape_fts_query("")
        assert result == ""
