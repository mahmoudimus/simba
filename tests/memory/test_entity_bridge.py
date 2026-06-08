"""Entity-bridge index: link memories that share a named entity, traverse from
retrieved seeds (YourMemory-style). The mechanism distinct from C1/Track-B —
sparse, high-precision *named-entity* links, not co-occurrence/PPR density."""

from __future__ import annotations

import simba.memory.entity_bridge as eb


def test_extract_named_entities_normalized():
    ents = eb.extract_entities("Film X was directed by Jane Doe in Paris.")
    assert "jane doe" in ents and "paris" in ents
    assert "directed" not in ents  # lowercase / non-proper dropped


def test_exact_shared_entity_bridges():
    idx = eb.build_index(
        [
            ("m1", "Film X was directed by Jane Doe."),
            ("m2", "Jane Doe was born in Paris."),
            ("m3", "An unrelated note about Bob Smith."),
        ]
    )
    bridged = set(eb.bridged_ids(idx, ["m1"], hops=1))
    assert "m2" in bridged  # shares "jane doe"
    assert "m3" not in bridged
    assert "m1" not in bridged  # seeds excluded from the output


def test_two_word_prefix_match():
    # "Shirley Temple Black" should bridge to "Shirley Temple".
    idx = eb.build_index(
        [
            ("m1", "A documentary about Shirley Temple Black aired."),
            ("m2", "Shirley Temple was a child actress."),
            ("m3", "Something about Marlon Brando."),
        ]
    )
    bridged = set(eb.bridged_ids(idx, ["m1"], hops=1))
    assert "m2" in bridged and "m3" not in bridged


def test_depth_2_traversal():
    idx = eb.build_index(
        [
            ("m1", "Alice met Bob."),
            ("m2", "Bob knows Carol."),
            ("m3", "Carol visited Dave."),
        ]
    )
    h1 = set(eb.bridged_ids(idx, ["m1"], hops=1))
    assert "m2" in h1 and "m3" not in h1  # m3 is two hops away
    h2 = set(eb.bridged_ids(idx, ["m1"], hops=2))
    assert "m3" in h2


def test_empty_index_and_unknown_seed_fail_open():
    assert eb.bridged_ids(eb.build_index([]), ["x"], hops=2) == []
    idx = eb.build_index([("m1", "Jane Doe")])
    assert eb.bridged_ids(idx, ["nobody"], hops=2) == []


def test_fold_boosts_bridged_and_inserts_new():
    # Folds bridged ids as a 3rd RRF arm: score += weight/(rrf_k + rank).
    fused = [{"id": "v1", "rrf_score": 0.05}, {"id": "v2", "rrf_score": 0.02}]
    out = eb.fold_into_candidates(
        fused,
        ["g1", "v2"],  # g1 new (rank1), v2 existing (rank2)
        record_lookup={"g1": {"id": "g1", "content": "gold"}},
        rrf_k=20,
        weight=1.0,
    )
    ids = [r["id"] for r in out]
    # v2: 0.02 + 1/22 = 0.0655 → above v1 (0.05); g1: 1/21 = 0.0476 → last.
    assert ids == ["v2", "v1", "g1"]
    assert len(out) == 3  # g1 materialized from the lookup


def test_fold_zero_weight_preserves_order():
    fused = [{"id": "v1", "rrf_score": 0.05}, {"id": "v2", "rrf_score": 0.02}]
    out = eb.fold_into_candidates(
        fused, ["v2"], record_lookup={}, rrf_k=20, weight=0.0
    )
    assert [r["id"] for r in out] == ["v1", "v2"]
