"""Tests for the shared simba.db module."""

from __future__ import annotations

import pathlib
import shutil
import sqlite3

import pytest

import simba._vendor.peewee as pw
import simba.config
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


class TestResolveProjectId:
    @pytest.fixture(autouse=True)
    def _isolate_global_config(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            simba.config, "_global_path", lambda: tmp_path / "global.toml"
        )

    def test_generates_and_persists(self, tmp_path: pathlib.Path) -> None:
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)

        pid = simba.db.resolve_project_id(repo)
        assert pid

        # Persisted locally → a second resolve returns the same id.
        assert simba.db.resolve_project_id(repo) == pid
        local = repo / ".simba" / "config.toml"
        assert local.exists()
        assert pid in local.read_text()

    def test_respects_configured_id(self, tmp_path: pathlib.Path) -> None:
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        simba.config.set_value(
            "project", "project_id", "custom-id", scope="local", root=repo
        )
        assert simba.db.resolve_project_id(repo) == "custom-id"

    def test_stable_across_folder_move(self, tmp_path: pathlib.Path) -> None:
        # `mv` carries .simba/config.toml with the repo → id survives, no
        # manual step needed (the move concern that motivated this design).
        old = tmp_path / "old"
        (old / ".git").mkdir(parents=True)
        pid = simba.db.resolve_project_id(old)

        new = tmp_path / "new"
        shutil.move(str(old), str(new))
        assert simba.db.resolve_project_id(new) == pid


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


class TestConnectModelRegistrationCache:
    """A peewee model registered (via ``register_model``) AFTER a path has
    already been through ``connect()`` once must still get its table created
    the next time that path connects.

    Real-world sequence this reproduces: the daemon process calls
    ``simba.db.connect(path)`` for the first time from a code path that
    hasn't yet imported a lazily-loaded subsystem module (e.g.
    ``simba/memory/usage_events.py``, which ``simba/memory/routes.py`` only
    imports inside individual handler bodies, not at module level). Import
    executing ``register_model(UsageEvent)`` moments later doesn't help: the
    schema-ready cache was keyed on *path alone*, so every later ``connect()``
    for that path skipped DDL for good -- the table never gets created for
    the rest of the process's life.
    """

    def test_model_registered_after_first_connect_still_gets_migrated(
        self, tmp_path: pathlib.Path
    ) -> None:
        class LateModel(simba.db.BaseModel):
            name = pw.TextField()

            class Meta:
                table_name = "test_late_registered_model"

        # 1) Something calls connect() BEFORE the lazy module has registered
        #    its model -- mirrors a /recall request's fire-and-forget
        #    `_bump_usage()` (imports only `simba.memory.usage`) winning the
        #    race against `_record_demand()`/the `/digest` handler (which
        #    import `simba.memory.demand` / `simba.memory.usage_events`).
        with simba.db.connect(tmp_path):
            pass

        # 2) The lazy import now happens -- mirrors routes.py's function-body
        #    `import simba.memory.usage_events`, which calls register_model()
        #    at module-import time, *after* the connect() above already ran
        #    DDL once for this path.
        simba.db.register_model(LateModel)

        # 3) A later request reuses the same (cached) path. On unfixed code
        #    this raises peewee.OperationalError("no such table") because the
        #    schema-ready cache keyed on path alone treats this path as fully
        #    migrated already.
        with simba.db.connect(tmp_path):
            LateModel.create(name="x")
            assert LateModel.select().where(LateModel.name == "x").count() == 1

    def test_no_new_model_skips_ddl_on_second_connect(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fix must stay cheap on the hot path: a second ``connect()`` for
        the same path with no newly-registered models must NOT re-run
        ``create_tables``."""
        with simba.db.connect(tmp_path):
            pass

        calls: list[int] = []
        original = simba.db.database.create_tables

        def _spy(models, **kwargs):
            calls.append(len(models))
            return original(models, **kwargs)

        monkeypatch.setattr(simba.db.database, "create_tables", _spy)

        with simba.db.connect(tmp_path):
            pass

        assert calls == []
