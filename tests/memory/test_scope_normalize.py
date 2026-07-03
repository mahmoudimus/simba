"""Worktree→root scope normalization (spec 33 Phase 3).

Strict exact-path scoping splits one repo's memories across its worktrees
(the audit found a project sharded 5,674/273/165/127 — the smaller shards
invisible from the main checkout). A linked worktree's ``.git`` is a FILE
(``gitdir: <main>/.git/worktrees/<name>``), so the main root is recoverable
with pure filesystem reads. Normalization is gated
(``memory.scope_normalize_worktrees``, default off); the migration endpoint
folds existing shards, dry-run by default.
"""

from __future__ import annotations

import pathlib

import httpx
import pytest

import simba.memory.config
import simba.memory.fts
import simba.memory.server
from simba.memory.vector_db import normalize_project_path, resolve_worktree_root


def _make_worktree(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """A fake main checkout + linked worktree (absolute gitdir pointer)."""
    main = tmp_path / "main"
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    wt = main / ".worktrees" / "wt"
    wt.mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'wt'}\n")
    return main, wt


# ---------------------------------------------------------------------------
# Pure resolution helpers
# ---------------------------------------------------------------------------


def test_resolve_worktree_root_detects_linked_worktree(tmp_path) -> None:
    main, wt = _make_worktree(tmp_path)
    assert resolve_worktree_root(wt) == main.resolve()


def test_resolve_worktree_root_from_subdirectory(tmp_path) -> None:
    main, wt = _make_worktree(tmp_path)
    sub = wt / "src" / "pkg"
    sub.mkdir(parents=True)
    assert resolve_worktree_root(sub) == main.resolve()


def test_resolve_worktree_root_primary_checkout_is_none(tmp_path) -> None:
    main = tmp_path / "repo"
    (main / ".git").mkdir(parents=True)
    assert resolve_worktree_root(main) is None


def test_resolve_worktree_root_relative_gitdir(tmp_path) -> None:
    main, wt = _make_worktree(tmp_path)
    (wt / ".git").write_text("gitdir: ../../.git/worktrees/wt\n")
    assert resolve_worktree_root(wt) == main.resolve()


def test_resolve_worktree_root_no_git_anywhere(tmp_path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    assert resolve_worktree_root(plain) is None


def test_normalize_project_path_folds_worktree_when_enabled(tmp_path) -> None:
    main, wt = _make_worktree(tmp_path)
    assert normalize_project_path(str(wt), resolve_worktrees=True) == str(
        main.resolve()
    )


def test_normalize_project_path_default_keeps_worktree_path(tmp_path) -> None:
    _main, wt = _make_worktree(tmp_path)
    assert normalize_project_path(str(wt)) == str(wt.resolve())


# ---------------------------------------------------------------------------
# Daemon wiring (store / recall / migration)
# ---------------------------------------------------------------------------


def _client(tmp_path, lance_table, mock_embed, cfg) -> httpx.AsyncClient:
    app = simba.memory.server.create_app(cfg)
    app.state.table = lance_table
    app.state.embed = mock_embed
    app.state.embed_query = mock_embed
    app.state.db_path = None
    app.state.cwd = tmp_path
    fts_path = tmp_path / simba.memory.fts.FTS_FILENAME
    simba.memory.fts.init(fts_path, tokenize=cfg.fts_tokenize)
    app.state.fts_path = str(fts_path)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


@pytest.mark.asyncio
async def test_store_folds_worktree_scope_when_enabled(
    tmp_path, lance_table, mock_embed
) -> None:
    main, wt = _make_worktree(tmp_path)
    cfg = simba.memory.config.MemoryConfig(
        scope_normalize_worktrees=True, duplicate_threshold=1.01
    )
    async with _client(tmp_path, lance_table, mock_embed, cfg) as ac:
        resp = await ac.post(
            "/store",
            json={
                "type": "GOTCHA",
                "content": "worktree fact",
                "projectPath": str(wt),
            },
        )
        assert resp.status_code == 200
    rows = await lance_table.query().where("content = 'worktree fact'").to_list()
    assert rows[0]["projectPath"] == str(main.resolve())


@pytest.mark.asyncio
async def test_store_keeps_worktree_scope_by_default(
    tmp_path, lance_table, mock_embed
) -> None:
    _main, wt = _make_worktree(tmp_path)
    cfg = simba.memory.config.MemoryConfig(duplicate_threshold=1.01)
    async with _client(tmp_path, lance_table, mock_embed, cfg) as ac:
        resp = await ac.post(
            "/store",
            json={
                "type": "GOTCHA",
                "content": "worktree fact",
                "projectPath": str(wt),
            },
        )
        assert resp.status_code == 200
    rows = await lance_table.query().where("content = 'worktree fact'").to_list()
    assert rows[0]["projectPath"] == str(wt.resolve())


@pytest.mark.asyncio
async def test_recall_from_worktree_finds_root_scoped_memory(
    tmp_path, lance_table, mock_embed
) -> None:
    main, wt = _make_worktree(tmp_path)
    cfg = simba.memory.config.MemoryConfig(
        scope_normalize_worktrees=True, duplicate_threshold=1.01
    )
    async with _client(tmp_path, lance_table, mock_embed, cfg) as ac:
        stored = await ac.post(
            "/store",
            json={
                "type": "GOTCHA",
                "content": "root scoped fact",
                "projectPath": str(main),
            },
        )
        assert stored.status_code == 200
        resp = await ac.post(
            "/recall", json={"query": "root scoped", "projectPath": str(wt)}
        )
        assert resp.status_code == 200
        assert [m["content"] for m in resp.json()["memories"]] == ["root scoped fact"]


@pytest.mark.asyncio
async def test_scope_migration_dry_run_then_run(
    tmp_path, lance_table, mock_embed
) -> None:
    main, wt = _make_worktree(tmp_path)
    cfg = simba.memory.config.MemoryConfig(duplicate_threshold=1.01)
    async with _client(tmp_path, lance_table, mock_embed, cfg) as ac:
        for content, path in (
            ("wt fact one", str(wt)),
            ("wt fact two", str(wt)),
            ("root fact", str(main)),
        ):
            resp = await ac.post(
                "/store",
                json={"type": "GOTCHA", "content": content, "projectPath": path},
            )
            assert resp.status_code == 200

        dry = await ac.post("/scopes/normalize", json={})
        assert dry.status_code == 200
        body = dry.json()
        assert body["run"] is False
        assert body["changed"] == 2
        assert body["folds"] == [
            {"from": str(wt.resolve()), "to": str(main.resolve()), "count": 2}
        ]
        # Dry run persisted nothing.
        rows = await lance_table.query().where("content = 'wt fact one'").to_list()
        assert rows[0]["projectPath"] == str(wt.resolve())

        run = await ac.post("/scopes/normalize", json={"run": True})
        assert run.status_code == 200
        assert run.json()["changed"] == 2
        rows = await lance_table.query().where("content = 'wt fact one'").to_list()
        assert rows[0]["projectPath"] == str(main.resolve())
        # FTS mirror retargeted too.
        with simba.memory.fts.connect(tmp_path / simba.memory.fts.FTS_FILENAME):
            n = (
                simba.memory.fts.MemoryFTS.select()
                .where(simba.memory.fts.MemoryFTS.project_path == str(main.resolve()))
                .count()
            )
        assert n == 3
