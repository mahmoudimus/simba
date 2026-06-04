"""Tests for the KG-into-recall expansion core (HippoRAG/GraphRAG-style).

The pipeline borrowed from GraphQLite's llm-graphrag example: seed memories ->
their entities -> KG neighbor entities (the COOCCURS bridge) -> the memories those
entities source, ranked by how many entity-paths reach them (a poor-man's PPR).

``kg_expand`` takes an injected KG interface so it's unit-tested with a fake (no
LanceDB, no real kg_edges).
"""

from __future__ import annotations

import simba.memory.kg_recall as kr


class FakeKG:
    """In-memory stand-in for the kg_edges adapter."""

    def __init__(
        self,
        ents: dict[str, list[str]],
        nbrs: dict[str, list[str]],
        mems: dict[str, list[str]],
    ) -> None:
        self._ents = ents  # memory_id -> entities it mentions
        self._nbrs = nbrs  # entity -> neighbor entities
        self._mems = mems  # entity -> memory_ids that mention it
        self.hops_seen: list[int] = []

    def entities_of(self, memory_id: str) -> list[str]:
        return self._ents.get(memory_id, [])

    def neighbors(self, entity: str, hops: int) -> list[str]:
        self.hops_seen.append(hops)
        return self._nbrs.get(entity, [])

    def memories_of(self, entity: str) -> list[str]:
        return self._mems.get(entity, [])


def test_empty_seeds_returns_empty() -> None:
    kg = FakeKG({}, {}, {})
    assert kr.kg_expand([], kg) == []


def test_neighbor_entity_memory_is_surfaced_seeds_excluded() -> None:
    # m1 mentions Alice; Alice --KG--> Bob; Bob is mentioned by m2.
    kg = FakeKG(
        ents={"m1": ["Alice"]},
        nbrs={"Alice": ["Bob"]},
        mems={"Alice": ["m1"], "Bob": ["m2"]},
    )
    out = kr.kg_expand(["m1"], kg, hops=1)
    ids = [mid for mid, _ in out]
    assert "m2" in ids  # bridged memory surfaced
    assert "m1" not in ids  # seeds never returned


def test_more_paths_rank_higher() -> None:
    # m9 is reached via two entities (Bob, Carol); m8 via one (Bob).
    kg = FakeKG(
        ents={"m1": ["Alice", "Carol"]},
        nbrs={"Alice": ["Bob"], "Carol": ["Carol"]},
        mems={"Bob": ["m8", "m9"], "Carol": ["m9"]},
    )
    out = kr.kg_expand(["m1"], kg, hops=1)
    ranked = [mid for mid, _ in out]
    assert ranked[0] == "m9"  # two entity-paths reach m9
    assert ranked.index("m9") < ranked.index("m8")


def test_max_neighbors_caps() -> None:
    kg = FakeKG(
        ents={"m1": ["E"]},
        nbrs={"E": []},
        mems={"E": [f"m{i}" for i in range(20)]},
    )
    out = kr.kg_expand(["m1"], kg, hops=1, max_neighbors=5)
    assert len(out) == 5


def test_hops_passed_through() -> None:
    kg = FakeKG(ents={"m1": ["E"]}, nbrs={"E": []}, mems={"E": ["m1"]})
    kr.kg_expand(["m1"], kg, hops=3)
    assert kg.hops_seen and all(h == 3 for h in kg.hops_seen)
