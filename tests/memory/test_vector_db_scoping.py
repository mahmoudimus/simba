"""Tests for hierarchical (ancestor-prefix) project scoping (spec 26).

Covers the pure scope-membership predicate and the ``search_memories``
ancestor-membership filter behind ``memory.hierarchical_recall``. The default-off
path stays strict exact-match (characterization).
"""

from __future__ import annotations

import typing

import pytest

import simba.memory.vector_db as vdb


class _FakeQuery:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def column(self, *_a: typing.Any, **_k: typing.Any) -> _FakeQuery:
        return self

    def distance_type(self, *_a: typing.Any, **_k: typing.Any) -> _FakeQuery:
        return self

    def select(self, *_a: typing.Any, **_k: typing.Any) -> _FakeQuery:
        return self

    def limit(self, *_a: typing.Any, **_k: typing.Any) -> _FakeQuery:
        return self

    async def to_list(self) -> list[dict]:
        return self._rows


class _FakeTable:
    """Minimal LanceDB table stand-in: no schema (skips the dim guard)."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        # No ``schema`` attribute -> _resolve_table_dim returns None (guard no-op).

    def vector_search(self, _embedding: list[float]) -> _FakeQuery:
        return _FakeQuery(self._rows)


def _row(mid: str, project: str, dist: float = 0.1) -> dict:
    return {
        "id": mid,
        "type": "PATTERN",
        "content": f"content {mid}",
        "context": "",
        "projectPath": project,
        "_distance": dist,
    }


class TestScopeMatch:
    def test_exact_path_matches(self) -> None:
        assert vdb._scope_match("/repo/api", ["/repo/api"], include_global=True)

    def test_ancestor_in_scope_matches(self) -> None:
        # /repo is an ancestor scope of cwd /repo/api -> inherits down.
        assert vdb._scope_match("/repo", ["/repo/api", "/repo"], include_global=True)

    def test_sibling_does_not_match(self) -> None:
        assert not vdb._scope_match(
            "/repo/web", ["/repo/api", "/repo"], include_global=True
        )

    def test_global_matches_when_included(self) -> None:
        assert vdb._scope_match("", ["/repo/api", "/repo"], include_global=True)

    def test_global_excluded_when_flag_off(self) -> None:
        assert not vdb._scope_match("", ["/repo/api", "/repo"], include_global=False)


class TestSearchMemoriesHierarchical:
    @pytest.mark.asyncio
    async def test_recall_api_includes_ancestor_and_global(self) -> None:
        rows = [
            _row("api1", "/repo/api"),
            _row("root1", "/repo"),
            _row("glob1", ""),
            _row("web1", "/repo/web"),
        ]
        table = _FakeTable(rows)
        results = await vdb.search_memories(
            table,
            [0.1] * 4,
            min_similarity=0.0,
            max_results=10,
            filters={
                "project_scopes": ["/repo/api", "/repo"],
                "hierarchical_recall": True,
                "hierarchical_recall_include_global": True,
            },
        )
        ids = {r["id"] for r in results}
        assert ids == {"api1", "root1", "glob1"}  # web1 (sibling) excluded

    @pytest.mark.asyncio
    async def test_recall_root_excludes_package_scoped(self) -> None:
        rows = [_row("api1", "/repo/api"), _row("root1", "/repo")]
        table = _FakeTable(rows)
        results = await vdb.search_memories(
            table,
            [0.1] * 4,
            min_similarity=0.0,
            max_results=10,
            filters={
                "project_scopes": ["/repo"],
                "hierarchical_recall": True,
                "hierarchical_recall_include_global": True,
            },
        )
        ids = {r["id"] for r in results}
        assert ids == {"root1"}  # /repo/api does NOT leak up to the root

    @pytest.mark.asyncio
    async def test_include_global_false_drops_globals(self) -> None:
        rows = [_row("root1", "/repo"), _row("glob1", "")]
        table = _FakeTable(rows)
        results = await vdb.search_memories(
            table,
            [0.1] * 4,
            min_similarity=0.0,
            max_results=10,
            filters={
                "project_scopes": ["/repo"],
                "hierarchical_recall": True,
                "hierarchical_recall_include_global": False,
            },
        )
        ids = {r["id"] for r in results}
        assert ids == {"root1"}

    @pytest.mark.asyncio
    async def test_legacy_exact_match_when_lever_off(self) -> None:
        # Characterization: with no hierarchical flag, the legacy strict exact match
        # on projectPath applies (drops other-project AND global rows).
        rows = [_row("api1", "/repo/api"), _row("root1", "/repo"), _row("glob1", "")]
        table = _FakeTable(rows)
        results = await vdb.search_memories(
            table,
            [0.1] * 4,
            min_similarity=0.0,
            max_results=10,
            filters={"projectPath": "/repo/api"},
        )
        ids = {r["id"] for r in results}
        assert ids == {"api1"}  # strict: only the exact project, no ancestor/global

    @pytest.mark.asyncio
    async def test_scopes_present_but_lever_off_is_noop(self) -> None:
        # project_scopes alone (lever off) must NOT widen — the daemon honors it
        # only when hierarchical_recall is also set in the filter.
        rows = [_row("api1", "/repo/api"), _row("root1", "/repo")]
        table = _FakeTable(rows)
        results = await vdb.search_memories(
            table,
            [0.1] * 4,
            min_similarity=0.0,
            max_results=10,
            filters={
                "projectPath": "/repo/api",
                "project_scopes": ["/repo/api", "/repo"],
                "hierarchical_recall": False,
            },
        )
        ids = {r["id"] for r in results}
        assert ids == {"api1"}
