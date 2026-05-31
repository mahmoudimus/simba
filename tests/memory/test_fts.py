"""Tests for the memory FTS5 keyword mirror (src/simba/memory/fts.py)."""

from __future__ import annotations

import pathlib

import pytest

import simba.memory.fts as fts


@pytest.fixture()
def conn(tmp_path: pathlib.Path):
    path = tmp_path / fts.FTS_FILENAME
    fts.init(path)
    c = fts.connect(path)
    yield c
    c.close()


def _mem(mid: str, content: str, **over) -> dict:
    base = {
        "id": mid,
        "type": "GOTCHA",
        "content": content,
        "context": "",
        "confidence": 0.85,
        "createdAt": "2026-01-01T00:00:00Z",
        "projectPath": "proj-1",
    }
    base.update(over)
    return base


class TestUpsertSearch:
    def test_upsert_then_search_finds_it(self, conn) -> None:
        fts.upsert(conn, _mem("m1", "ruff is the linter we use"))
        rows = fts.search(conn, "linter", project_path="proj-1")
        assert [r["memory_id"] for r in rows] == ["m1"]
        assert rows[0]["confidence"] == 0.85
        assert rows[0]["createdAt"] == "2026-01-01T00:00:00Z"

    def test_trigram_substring_match(self, conn) -> None:
        fts.upsert(conn, _mem("m1", "the kg_edges_fts table is trigram-tokenized"))
        rows = fts.search(conn, "kg_ed", project_path="proj-1")
        assert [r["memory_id"] for r in rows] == ["m1"]

    def test_idempotent_upsert(self, conn) -> None:
        fts.upsert(conn, _mem("m1", "alpha content"))
        fts.upsert(conn, _mem("m1", "alpha content updated"))
        assert fts.count(conn) == 1
        rows = fts.search(conn, "updated", project_path="proj-1")
        assert len(rows) == 1

    def test_returns_empty_for_short_query(self, conn) -> None:
        fts.upsert(conn, _mem("m1", "db things"))
        # "db" is shorter than the trigram minimum -> no usable terms.
        assert fts.search(conn, "db", project_path="proj-1") == []


class TestSystemExclusion:
    def test_system_never_indexed(self, conn) -> None:
        fts.upsert(conn, _mem("s1", "system marker", type="SYSTEM"))
        assert fts.count(conn) == 0
        assert fts.search(conn, "marker") == []


class TestDelete:
    def test_delete_removes(self, conn) -> None:
        fts.upsert(conn, _mem("m1", "deletable content here"))
        fts.delete(conn, "m1")
        assert fts.count(conn) == 0
        assert fts.search(conn, "deletable") == []


class TestScoping:
    def test_project_scope_isolates(self, conn) -> None:
        fts.upsert(conn, _mem("a", "shared alpha keyword", projectPath="proj-1"))
        fts.upsert(conn, _mem("b", "shared alpha keyword", projectPath="proj-2"))
        rows = fts.search(conn, "alpha", project_path="proj-1")
        assert [r["memory_id"] for r in rows] == ["a"]

    def test_type_filter(self, conn) -> None:
        fts.upsert(conn, _mem("g", "beta gotcha note", type="GOTCHA"))
        fts.upsert(conn, _mem("p", "beta pattern note", type="PATTERN"))
        rows = fts.search(conn, "beta", project_path="proj-1", types=["GOTCHA"])
        assert [r["memory_id"] for r in rows] == ["g"]


class TestSetProject:
    def test_set_project_moves_scope(self, conn) -> None:
        fts.upsert(conn, _mem("m1", "movable gamma content", projectPath="proj-1"))
        fts.set_project(conn, "m1", "proj-2")
        assert fts.search(conn, "gamma", project_path="proj-1") == []
        rows = fts.search(conn, "gamma", project_path="proj-2")
        assert [r["memory_id"] for r in rows] == ["m1"]


class TestRobustness:
    def test_malformed_match_returns_empty(self, conn) -> None:
        fts.upsert(conn, _mem("m1", "some content"))
        # A bare quote tokenizes to nothing usable -> "" -> [].
        assert fts.search(conn, '"', project_path="proj-1") == []


class TestRebuild:
    def test_rebuild_skips_system_and_counts(self, conn) -> None:
        memories = [
            _mem("m1", "first delta memory"),
            _mem("s1", "system row", type="SYSTEM"),
            _mem("m2", "second delta memory"),
        ]
        n = fts.rebuild(conn, memories)
        assert n == 2
        assert fts.count(conn) == 2
        rows = fts.search(conn, "delta", project_path="proj-1")
        assert {r["memory_id"] for r in rows} == {"m1", "m2"}
