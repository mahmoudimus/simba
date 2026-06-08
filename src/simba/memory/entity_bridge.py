"""Entity-bridge multi-hop links: connect memories that share a named entity.

The one multi-hop mechanism with a *positive* external result (YourMemory, +12pp
on HotpotQA) and distinct from simba's three negatives (C1 co-occurrence, Track B
PPR, Track A IRCoT): links are **sparse + high-precision** (shared *named*
entities, not co-occurrence/PPR density) and traversed from the retrieved seeds.

At index time each memory contributes its proper-noun entities (normalized, with a
2-word-prefix bridge key so "Shirley Temple Black" ↔ "Shirley Temple"). At recall
time we BFS depth-N from the top seeds over the memory—entity—memory bipartite
graph and fold the bridged memory ids into the candidate set. Pure + deterministic;
no NER dependency (capitalized-span regex + ``kg.entities.normalize_entity``).
"""

from __future__ import annotations

import collections
import dataclasses
import re
import typing

import simba.kg.entities

# Capitalized word-span → a candidate named entity (PERSON/ORG/GPE/work).
_PROPER_NOUN_RE = re.compile(r"[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*")
# How many leading words form the bridge key (2 → "Shirley Temple Black" and
# "Shirley Temple" collapse to "shirley temple"; "Ed Wood (film)" → "ed wood").
_PREFIX_WORDS = 2


def _bridge_key(entity_key: str) -> str:
    """Collapse a normalized entity to its leading-N-word bridge key."""
    words = entity_key.split()
    return " ".join(words[:_PREFIX_WORDS]) if len(words) > _PREFIX_WORDS else entity_key


def extract_entities(text: str) -> set[str]:
    """Normalized bridge keys for the proper-noun entities mentioned in ``text``."""
    keys: set[str] = set()
    for span in _PROPER_NOUN_RE.findall(text or ""):
        norm = simba.kg.entities.normalize_entity(span)
        if norm:
            keys.add(_bridge_key(norm))
    return keys


@dataclasses.dataclass
class EntityBridgeIndex:
    """Bipartite memory↔entity index for shared-named-entity traversal."""

    entity_to_memories: dict[str, set[str]]
    memory_to_entities: dict[str, set[str]]

    def is_empty(self) -> bool:
        return not self.entity_to_memories


def build_index(items: typing.Iterable[tuple[str, str]]) -> EntityBridgeIndex:
    """Build the index from ``(memory_id, text)`` pairs."""
    e2m: dict[str, set[str]] = collections.defaultdict(set)
    m2e: dict[str, set[str]] = collections.defaultdict(set)
    for mem_id, text in items:
        for key in extract_entities(text):
            e2m[key].add(mem_id)
            m2e[mem_id].add(key)
    return EntityBridgeIndex(entity_to_memories=dict(e2m), memory_to_entities=dict(m2e))


def bridged_ids(
    index: EntityBridgeIndex,
    seeds: typing.Iterable[str],
    *,
    hops: int = 1,
    max_out: int | None = None,
) -> list[str]:
    """Memory ids reachable within ``hops`` of a seed via shared entities.

    BFS over the memory—entity—memory bipartite graph. Seeds are excluded from the
    output (we want *new* bridged evidence). Deterministic order: by hop, then
    first-seen. ``max_out`` caps the result. Fail-open: unknown seeds → ``[]``.
    """
    seed_set = set(seeds)
    visited = set(seed_set)
    frontier = set(seed_set)
    out: list[str] = []
    for _ in range(max(0, hops)):
        nxt: list[str] = []
        for mem in sorted(frontier):
            for key in index.memory_to_entities.get(mem, ()):
                for neighbor in sorted(index.entity_to_memories.get(key, ())):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        nxt.append(neighbor)
                        out.append(neighbor)
                        if max_out is not None and len(out) >= max_out:
                            return out
        if not nxt:
            break
        frontier = set(nxt)
    return out


def fold_into_candidates(
    fused: list[dict[str, typing.Any]],
    bridged: list[str],
    *,
    record_lookup: dict[str, dict[str, typing.Any]],
    rrf_k: int,
    weight: float,
) -> list[dict[str, typing.Any]]:
    """Merge ``bridged`` ids into ``fused`` as a third RRF arm and re-sort.

    Each bridged id (ordered by traversal) contributes ``weight / (rrf_k + rank)``
    to its fusion score, so a graph-surfaced memory competes for the top-k instead
    of being appended below it. Already-present ids are boosted; new ids are pulled
    from ``record_lookup`` (skipped if missing). Mirrors ``rrf_fuse``'s math.
    """
    scores = {r["id"]: r.get("rrf_score", 0.0) for r in fused}
    records = {r["id"]: r for r in fused}
    for rank, mid in enumerate(bridged, start=1):
        contrib = weight / (rrf_k + rank)
        if mid in records:
            scores[mid] = scores.get(mid, 0.0) + contrib
        else:
            rec = record_lookup.get(mid)
            if rec is None:
                continue
            records[mid] = dict(rec)
            scores[mid] = scores.get(mid, 0.0) + contrib
    ordered = sorted(records.values(), key=lambda r: scores[r["id"]], reverse=True)
    for r in ordered:
        r["rrf_score"] = round(scores[r["id"]], 6)
    return ordered
