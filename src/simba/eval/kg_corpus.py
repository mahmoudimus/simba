"""Build a throwaway knowledge graph from a benchmark corpus.

Bench corpora (LoCoMo, LongMemEval) carry no KG, so the retrieval-time GraphRAG
lever (Track B: PPR + community over `kg_edges`) can't be measured against them
directly. This module extracts an in-memory KG from the corpus text via a
pluggable extractor and exposes graph traversal + density so we can ask the
gating question *before* building PPR: is any labelled gold reachable through the
graph that the vector arm doesn't already retrieve? If not, that is the finding
(KG/extraction density is the bottleneck), not a reason to tune PPR.

Pure + deterministic — the extractor is injected, so tests use a fake and the CLI
wires the regex (`sync.heuristics`) or LLM (`sync.llm_extract`) extractor.
"""

from __future__ import annotations

import collections
import dataclasses
import itertools
import typing

import simba.kg.entities
import simba.memory.keywords

if typing.TYPE_CHECKING:
    from simba.eval.dataset import Memory

# (subject, predicate, object) over raw surface forms.
Triple = tuple[str, str, str]
# Memory -> triples extracted from its content/context.
ExtractFn = typing.Callable[["Memory"], list[Triple]]
# (subject, predicate, object, source_memory_id) over normalized entities.
Edge = tuple[str, str, str, str]


@dataclasses.dataclass
class CorpusKG:
    """An undirected entity graph derived from a corpus, with source provenance.

    ``adjacency`` maps a normalized entity to its neighbor entities (self-loops
    excluded). ``entity_memories`` maps a normalized entity to the set of corpus
    memory ids it appears in — the bridge from graph nodes back to retrievable
    memories. ``edges`` keeps the directed, provenance-tagged triples.
    """

    edges: list[Edge]
    adjacency: dict[str, set[str]]
    entity_memories: dict[str, set[str]]

    def density(self) -> float:
        """Directed-graph edge density ``2E / (n(n-1))`` over the node set.

        ``0`` for fewer than two nodes. Self-loops are already excluded from the
        adjacency, so they don't inflate the count.
        """
        n = len(self.entity_memories)
        if n < 2:
            return 0.0
        edge_count = sum(len(v) for v in self.adjacency.values()) / 2
        return (2 * edge_count) / (n * (n - 1))

    def reachable_entities(
        self, seeds: typing.Iterable[str], max_hops: int
    ) -> set[str]:
        """BFS the entity graph from (normalized) ``seeds`` up to ``max_hops``."""
        start = {simba.kg.entities.normalize_entity(s) for s in seeds}
        start.discard("")
        visited = set(start)
        frontier = set(start)
        for _ in range(max(0, max_hops)):
            nxt: set[str] = set()
            for node in frontier:
                nxt |= self.adjacency.get(node, set())
            nxt -= visited
            if not nxt:
                break
            visited |= nxt
            frontier = nxt
        return visited

    def reachable_memories(
        self, seeds: typing.Iterable[str], max_hops: int
    ) -> set[str]:
        """Memory ids whose entities are within ``max_hops`` of a seed entity."""
        mems: set[str] = set()
        for ent in self.reachable_entities(seeds, max_hops):
            mems |= self.entity_memories.get(ent, set())
        return mems


def entities_of(text: str, *, max_terms: int = 12) -> list[str]:
    """High-signal terms from ``text`` — the entity surface forms.

    Reuses the keyword arm's ``focus_terms`` (stop-word filtered, salience-ranked)
    so the KG's entities and the query seeds are extracted the same way the live
    recall keys on, keeping the ceiling probe aligned with what actually ships.
    """
    return simba.memory.keywords.focus_terms(text, max_terms=max_terms)


def cooccurrence_extract(mem: Memory) -> list[Triple]:
    """All unordered entity pairs co-mentioned in a memory → ``(e, "co", e)``.

    The deliberately *dense* upper-bound extractor: it maximizes reachability, so
    if even this graph adds no recall headroom, a sparser typed KG can't either
    (it bounds Track B from above). Entities come from content + context.
    """
    text = mem.content + (f" {mem.context}" if mem.context else "")
    ents = entities_of(text)
    return [(a, "co", b) for a, b in itertools.combinations(ents, 2)]


def build_corpus_kg(corpus: list[Memory], extract: ExtractFn) -> CorpusKG:
    """Extract triples from each memory and assemble the undirected corpus KG."""
    edges: list[Edge] = []
    adjacency: dict[str, set[str]] = collections.defaultdict(set)
    entity_memories: dict[str, set[str]] = collections.defaultdict(set)

    for mem in corpus:
        for subject, predicate, obj in extract(mem):
            ns = simba.kg.entities.normalize_entity(subject)
            no = simba.kg.entities.normalize_entity(obj)
            if not ns or not no:
                continue
            entity_memories[ns].add(mem.id)
            entity_memories[no].add(mem.id)
            edges.append((ns, predicate, no, mem.id))
            if ns != no:  # skip self-loops in the traversal graph
                adjacency[ns].add(no)
                adjacency[no].add(ns)

    return CorpusKG(
        edges=edges,
        adjacency=dict(adjacency),
        entity_memories=dict(entity_memories),
    )
