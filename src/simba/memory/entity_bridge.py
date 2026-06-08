"""Entity-bridge multi-hop links: connect memories that share a named entity.

The one multi-hop mechanism with a *positive* external result (YourMemory, +12pp
on HotpotQA) and distinct from simba's three negatives (C1 co-occurrence, Track B
PPR, Track A IRCoT): links are **sparse + high-precision** (shared *named*
entities, not co-occurrence/PPR density) and traversed from the retrieved seeds.

At index time each memory contributes its proper-noun entities (normalized, with a
2-word-prefix bridge key so "Shirley Temple Black" ↔ "Shirley Temple"). At recall
time we score candidate memories by how many of the *seed* entities they share
(precision signal — NOT arbitrary traversal order), keep those above
``min_shared``, and fold the **missed** ones (not already retrieved) into the
candidate set. ADD-only: a fully-retrieved corpus is a no-op (no scramble). Pure +
deterministic; no NER dependency (capitalized-span regex + ``normalize_entity``).
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
    min_shared: int = 1,
    max_df: int = 0,
    max_out: int | None = None,
) -> list[str]:
    """Memories sharing ≥``min_shared`` of the *seed* entities, ranked by overlap.

    BFS reaches candidates within ``hops`` of a seed, but each candidate is scored
    by how many of the original seed entities it shares (a precision signal, not
    arbitrary traversal order), so distractors that merely touch the graph are
    filtered by ``min_shared`` and genuine co-references rank first. ``max_df`` > 0
    drops low-specificity entities (those appearing in more than ``max_df``
    memories — e.g. regex-NER noise like sentence-initial "It"/"American") so the
    bridge keys on *rare, discriminative* entities. Seeds are excluded.
    Deterministic: by shared-count desc, then id. ``[]`` on empty/unknown.
    """
    seed_set = set(seeds)
    seed_ents: set[str] = set()
    for s in seed_set:
        seed_ents |= index.memory_to_entities.get(s, set())
    if max_df > 0:
        seed_ents = {
            e for e in seed_ents if len(index.entity_to_memories.get(e, ())) <= max_df
        }
    if not seed_ents:
        return []

    shared_count: dict[str, int] = {}
    visited = set(seed_set)
    frontier_ents = set(seed_ents)
    for _ in range(max(1, hops)):
        cand: set[str] = set()
        for ent in frontier_ents:
            cand |= index.entity_to_memories.get(ent, set())
        cand -= visited
        if not cand:
            break
        next_ents: set[str] = set()
        for mem in cand:
            ents = index.memory_to_entities.get(mem, set())
            shared = len(ents & seed_ents)
            if shared > shared_count.get(mem, 0):
                shared_count[mem] = shared
            next_ents |= ents
        visited |= cand
        frontier_ents = next_ents

    ranked = sorted(
        (m for m, sh in shared_count.items() if sh >= min_shared),
        key=lambda m: (-shared_count[m], m),
    )
    return ranked[:max_out] if max_out is not None else ranked


def fold_into_candidates(
    fused: list[dict[str, typing.Any]],
    bridged: list[str],
    *,
    record_lookup: dict[str, dict[str, typing.Any]],
    rrf_k: int,
    weight: float,
) -> list[dict[str, typing.Any]]:
    """ADD-only fold: insert *missed* bridged ids; never reorder retrieved items.

    Bridged ids already present in ``fused`` are left untouched (no boost → no
    scramble of the already-correct relevance order). Each genuinely-missed id is
    materialized from ``record_lookup`` and scored ``weight / (rrf_k + insert_rank)``
    so high-overlap bridges can enter the top-k. On a fully-retrieved corpus there
    is nothing to insert → a no-op.
    """
    scores = {r["id"]: r.get("rrf_score", 0.0) for r in fused}
    records = {r["id"]: r for r in fused}
    present = set(records)
    inserted = 0
    for mid in bridged:
        if mid in present:
            continue  # ADD-only: don't touch retrieved items
        rec = record_lookup.get(mid)
        if rec is None:
            continue
        inserted += 1
        records[mid] = dict(rec)
        scores[mid] = weight / (rrf_k + inserted)

    ordered = sorted(records.values(), key=lambda r: scores[r["id"]], reverse=True)
    for r in ordered:
        r["rrf_score"] = round(scores[r["id"]], 6)
    return ordered
