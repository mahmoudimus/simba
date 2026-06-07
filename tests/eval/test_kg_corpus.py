"""Throwaway corpus KG — the measurable substrate for the Track B (GraphRAG)
recall-ceiling diagnostic. Bench corpora ship no KG, so we extract one from the
corpus text and ask: is any gold reachable via the graph that the vector arm
misses? (If not, that is the finding — extraction density, not PPR.)"""

from __future__ import annotations

import simba.eval.kg_corpus as kgc
from simba.eval.dataset import Memory


def _mem(mem_id: str, content: str, context: str = "") -> Memory:
    return Memory(id=mem_id, content=content, context=context)


def test_build_corpus_kg_adjacency_and_sources():
    corpus = [
        _mem("m1", "A and B"),
        _mem("m2", "B and C"),
        _mem("m3", "C and D"),
        _mem("m4", "isolated Z"),
    ]
    triples = {
        "m1": [("A", "r", "B")],
        "m2": [("B", "r", "C")],
        "m3": [("C", "r", "D")],
        "m4": [("Z", "r", "Z")],  # self-loop → no adjacency edge
    }
    kg = kgc.build_corpus_kg(corpus, lambda m: triples[m.id])

    # Undirected adjacency over normalized entities.
    assert "b" in kg.adjacency["a"]
    assert "a" in kg.adjacency["b"]
    # Each entity maps back to every memory it appears in.
    assert kg.entity_memories["b"] == {"m1", "m2"}
    # Self-loop adds the node but no edge.
    assert kg.adjacency.get("z", set()) == set()

    # From A: m1+m2 within 1 hop (B bridges them); m3 needs 2 hops (via C).
    r1 = kg.reachable_memories({"A"}, max_hops=1)
    assert "m2" in r1 and "m3" not in r1
    r2 = kg.reachable_memories({"A"}, max_hops=2)
    assert "m3" in r2
    # The isolated component is never reached from A.
    assert "m4" not in kg.reachable_memories({"A"}, max_hops=5)


def test_seeds_are_normalized_and_unknown_seeds_reach_nothing():
    corpus = [_mem("m1", "Bob"), _mem("m2", "x")]
    kg = kgc.build_corpus_kg(
        corpus,
        lambda m: [("Bob", "likes", "Tea")] if m.id == "m1" else [],
    )
    # "the Bob" normalizes to "bob" and still maps to its memory.
    assert kg.reachable_memories({"the Bob"}, max_hops=1) == {"m1"}
    # A seed absent from the graph reaches nothing (never raises).
    assert kg.reachable_memories({"nobody"}, max_hops=3) == set()


def test_density_of_corpus_kg():
    corpus = [_mem("m1", "x"), _mem("m2", "y")]
    triples = {"m1": [("A", "r", "B")], "m2": [("B", "r", "C")]}
    kg = kgc.build_corpus_kg(corpus, lambda m: triples[m.id])
    # nodes A,B,C (3); undirected edges A-B, B-C (2): density = 2E/(n(n-1)).
    assert abs(kg.density() - (2 * 2) / (3 * 2)) < 1e-9


def test_empty_extractor_yields_empty_kg():
    corpus = [_mem("m1", "x")]
    kg = kgc.build_corpus_kg(corpus, lambda m: [])
    assert kg.adjacency == {}
    assert kg.density() == 0.0
    assert kg.reachable_memories({"anything"}, max_hops=3) == set()


def test_entities_of_extracts_high_signal_terms():
    ents = {e.lower() for e in kgc.entities_of("Alice met Bob in the Paris cafe")}
    assert {"alice", "bob", "paris"} <= ents
    assert "in" not in ents and "the" not in ents  # stop words / short tokens out


def test_cooccurrence_extract_pairs_entities_within_a_memory():
    triples = kgc.cooccurrence_extract(_mem("m1", "Alice and Bob and Carol"))
    pairs = {frozenset((s.lower(), o.lower())) for s, _, o in triples}
    assert frozenset(("alice", "bob")) in pairs
    assert frozenset(("bob", "carol")) in pairs


def test_cooccurrence_single_entity_has_no_edges():
    assert kgc.cooccurrence_extract(_mem("m1", "Alice")) == []


def test_cooccurrence_uses_context_too():
    triples = kgc.cooccurrence_extract(_mem("m1", "Alice", context="Paris"))
    pairs = {frozenset((s.lower(), o.lower())) for s, _, o in triples}
    assert frozenset(("alice", "paris")) in pairs
