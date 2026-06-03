"""Tests for multi-hop KG traversal (kg_neighbors + kg_query expand_hops)."""

from __future__ import annotations

import pathlib

import pytest

import simba.db
from simba.kg import kg_add, kg_invalidate, kg_query
from simba.kg.store import kg_neighbors


@pytest.fixture(autouse=True)
def _patch_db_path(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


def _chain(project="p"):
    # A -uses-> B -depends_on-> C -runs_on-> D ;  X -unrelated-> Y
    kg_add("A", "uses", "B", "pf", project_path=project)
    kg_add("B", "depends_on", "C", "pf", project_path=project)
    kg_add("C", "runs_on", "D", "pf", project_path=project)
    kg_add("X", "unrelated", "Y", "pf", project_path=project)


def _objs(rows):
    return {(r["subject"], r["predicate"], r["object"]) for r in rows}


class TestKgNeighbors:
    def test_one_hop_out(self) -> None:
        _chain()
        rows = kg_neighbors("A", project_path="p", depth=1, direction="out")
        assert _objs(rows) == {("A", "uses", "B")}

    def test_two_hops_out(self) -> None:
        _chain()
        rows = kg_neighbors("A", project_path="p", depth=2, direction="out")
        assert _objs(rows) == {("A", "uses", "B"), ("B", "depends_on", "C")}

    def test_deep_traversal_stops_at_component(self) -> None:
        _chain()
        rows = kg_neighbors("A", project_path="p", depth=10, direction="out")
        assert _objs(rows) == {
            ("A", "uses", "B"),
            ("B", "depends_on", "C"),
            ("C", "runs_on", "D"),
        }  # never reaches the disconnected X-Y edge

    def test_direction_both_includes_inbound(self) -> None:
        _chain()
        rows = kg_neighbors("C", project_path="p", depth=1, direction="both")
        assert _objs(rows) == {("B", "depends_on", "C"), ("C", "runs_on", "D")}

    def test_direction_in_only(self) -> None:
        _chain()
        rows = kg_neighbors("C", project_path="p", depth=1, direction="in")
        assert _objs(rows) == {("B", "depends_on", "C")}

    def test_hop_level_annotated(self) -> None:
        _chain()
        rows = kg_neighbors("A", project_path="p", depth=2, direction="out")
        by_edge = {r["object"]: r["hop"] for r in rows}
        assert by_edge["B"] == 1
        assert by_edge["C"] == 2

    def test_project_scoped(self) -> None:
        _chain(project="p")
        kg_add("A", "uses", "Z", "pf", project_path="other")
        rows = kg_neighbors("A", project_path="p", depth=1, direction="out")
        assert _objs(rows) == {("A", "uses", "B")}  # 'other' edge excluded

    def test_excludes_invalidated_by_default(self) -> None:
        _chain()
        kg_invalidate("B", "depends_on", "C", project_path="p")
        rows = kg_neighbors("A", project_path="p", depth=3, direction="out")
        # the retracted B->C edge is gone, so C/D are unreachable
        assert _objs(rows) == {("A", "uses", "B")}

    def test_max_edges_bound(self) -> None:
        _chain()
        rows = kg_neighbors(
            "A", project_path="p", depth=10, direction="out", max_edges=1
        )
        assert len(rows) == 1


class TestKgQueryExpand:
    def test_expand_hops_pulls_in_neighbors(self) -> None:
        _chain()
        # seed: direct edges for subject A (single hop); expand 1 more hop
        rows = kg_query(subject="A", project_path="p", expand_hops=1)
        got = _objs(rows)
        assert ("A", "uses", "B") in got
        assert ("B", "depends_on", "C") in got

    def test_expand_zero_is_single_hop(self) -> None:
        _chain()
        rows = kg_query(subject="A", project_path="p", expand_hops=0)
        assert _objs(rows) == {("A", "uses", "B")}
