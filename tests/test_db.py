"""Tests for the shared simba.db module."""

from __future__ import annotations

import pathlib
import sqlite3

import pytest

import simba.db


class TestFindRepoRoot:
    def test_finds_git_directory(self, tmp_path: pathlib.Path) -> None:
        repo = tmp_path / "myrepo"
        repo.mkdir()
        (repo / ".git").mkdir()
        subdir = repo / "src" / "pkg"
        subdir.mkdir(parents=True)

        result = simba.db.find_repo_root(subdir)
        assert result is not None
        assert result == repo.resolve()

    def test_returns_none_when_no_git(self, tmp_path: pathlib.Path) -> None:
        result = simba.db.find_repo_root(tmp_path)
        assert result is None


class TestGetDbPath:
    def test_under_repo_root(self, tmp_path: pathlib.Path) -> None:
        repo = tmp_path / "myrepo"
        repo.mkdir()
        (repo / ".git").mkdir()

        db_path = simba.db.get_db_path(repo)
        assert db_path == repo.resolve() / ".simba" / "simba.db"

    def test_fallback_to_cwd(self, tmp_path: pathlib.Path) -> None:
        db_path = simba.db.get_db_path(tmp_path)
        assert db_path == tmp_path / ".simba" / "simba.db"


class TestRegisterSchema:
    def test_initializer_runs_on_get_db(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[bool] = []

        def _init(conn: sqlite3.Connection) -> None:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS _test_table (id INTEGER PRIMARY KEY)"
            )
            calls.append(True)

        # Temporarily add our initializer
        monkeypatch.setattr(
            simba.db, "_SCHEMA_INITIALIZERS", [*simba.db._SCHEMA_INITIALIZERS, _init]
        )

        with simba.db.get_db(tmp_path) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='_test_table'"
            ).fetchall()

        assert len(calls) >= 1
        assert len(tables) == 1

    def test_register_adds_to_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = list(simba.db._SCHEMA_INITIALIZERS)
        monkeypatch.setattr(simba.db, "_SCHEMA_INITIALIZERS", original)

        def _noop(conn: sqlite3.Connection) -> None:
            pass

        simba.db.register_schema(_noop)
        assert _noop in simba.db._SCHEMA_INITIALIZERS


class TestGetDb:
    def test_creates_db_file(self, tmp_path: pathlib.Path) -> None:
        with simba.db.get_db(tmp_path):
            pass
        db_path = simba.db.get_db_path(tmp_path)
        assert db_path.exists()

    def test_creates_parent_dirs(self, tmp_path: pathlib.Path) -> None:
        simba_dir = tmp_path / ".simba"
        assert not simba_dir.exists()
        with simba.db.get_db(tmp_path):
            pass
        assert simba_dir.exists()

    def test_connection_has_row_factory(self, tmp_path: pathlib.Path) -> None:
        with simba.db.get_db(tmp_path) as conn:
            assert conn.row_factory is sqlite3.Row

    def test_connection_closed_after_exit(self, tmp_path: pathlib.Path) -> None:
        with simba.db.get_db(tmp_path) as conn:
            pass
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


class TestGetConnection:
    def test_returns_none_when_db_missing(self, tmp_path: pathlib.Path) -> None:
        conn = simba.db.get_connection(tmp_path)
        assert conn is None

    def test_returns_connection_when_db_exists(self, tmp_path: pathlib.Path) -> None:
        # Create the DB first
        with simba.db.get_db(tmp_path):
            pass

        conn = simba.db.get_connection(tmp_path)
        try:
            assert conn is not None
            assert conn.row_factory is sqlite3.Row
        finally:
            if conn is not None:
                conn.close()
