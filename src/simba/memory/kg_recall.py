"""KG-into-recall expansion — borrowed from GraphQLite's llm-graphrag pipeline.

The benchmark finding: simba retrieves single-hop evidence well but misses
multi-hop / cross-session connections — exactly what the KG should bridge but
doesn't, because the KG is never consulted at recall time. This module supplies
the bridge, mirroring the GraphRAG recipe (vector seed → graph traversal →
merge), mapped onto simba's bitemporal ``kg_edges``:

    seed memories (top of the RRF fusion)
      → the entities each seed mentions          (kg.entities_of)
      → KG-connected neighbor entities           (kg.neighbors — the COOCCURS bridge)
      → the memories those entities source        (kg.memories_of)

Related memories are ranked by how many distinct entity-paths reach them — a
cheap stand-in for Personalized PageRank (deferred). The KG interface is injected
so the core is pure and unit-tested with a fake; the production adapter wires
``simba.kg.store``. Community detection (label-propagation) is a later step.
"""

from __future__ import annotations

import typing


@typing.runtime_checkable
class KGView(typing.Protocol):
    def entities_of(self, memory_id: str) -> list[str]: ...
    def neighbors(self, entity: str, hops: int) -> list[str]: ...
    def memories_of(self, entity: str) -> list[str]: ...


def kg_expand(
    seed_ids: list[str],
    kg: KGView,
    *,
    hops: int = 1,
    max_neighbors: int = 50,
) -> list[tuple[str, float]]:
    """Expand seed memory ids to KG-bridged related memories.

    Returns ``[(memory_id, score), ...]`` (seeds excluded), ranked by the number
    of distinct reaching entities, capped at ``max_neighbors``.
    """
    if not seed_ids:
        return []
    seed_set = set(seed_ids)

    # Entities the seeds mention, then their KG neighbors (the bridge).
    frontier: set[str] = set()
    for mid in seed_ids:
        frontier.update(kg.entities_of(mid))
    reached: set[str] = set(frontier)
    for ent in list(frontier):
        reached.update(kg.neighbors(ent, hops))

    # Memories those entities source, scored by reaching-entity count (PPR-ish).
    scores: dict[str, float] = {}
    for ent in reached:
        for mid in kg.memories_of(ent):
            if mid in seed_set:
                continue
            scores[mid] = scores.get(mid, 0.0) + 1.0

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:max_neighbors]
