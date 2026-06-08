"""Entity-bridge index: link memories that share a named entity, rank candidates
by shared-entity overlap with the seeds (precision, not traversal order), and fold
only the *missed* ones in (ADD-only). Distinct from C1/Track-B (sparse, high-
precision named-entity links)."""

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
    bridged = set(eb.bridged_ids(idx, ["m1"], min_shared=1))
    assert "m2" in bridged  # shares "jane doe"
    assert "m3" not in bridged  # shares nothing with the seed
    assert "m1" not in bridged  # seeds excluded


def test_two_word_prefix_match():
    idx = eb.build_index(
        [
            ("m1", "A documentary about Shirley Temple Black aired."),
            ("m2", "Shirley Temple was a child actress."),
            ("m3", "Something about Marlon Brando."),
        ]
    )
    bridged = set(eb.bridged_ids(idx, ["m1"], min_shared=1))
    assert "m2" in bridged and "m3" not in bridged


def test_ranked_by_shared_count_and_min_shared_threshold():
    # m2 shares 2 seed entities, m3 shares 1.
    idx = eb.build_index(
        [
            ("m1", "Alice Adams met Bob Brown."),
            ("m2", "Alice Adams and Bob Brown went to Paris."),
            ("m3", "Bob Brown likes coffee."),
        ]
    )
    ranked = eb.bridged_ids(idx, ["m1"], min_shared=1)
    assert ranked[0] == "m2"  # 2 shared entities ranks above 1
    assert set(ranked) == {"m2", "m3"}
    # min_shared=2 keeps only the high-overlap bridge.
    assert eb.bridged_ids(idx, ["m1"], min_shared=2) == ["m2"]


def test_pure_transitive_neighbor_excluded():
    # m3 shares an entity with m2 but NOT with the seed m1 → precision excludes it.
    idx = eb.build_index(
        [("m1", "Alice met Bob."), ("m2", "Bob knows Carol."), ("m3", "Carol saw Dave.")]
    )
    bridged = set(eb.bridged_ids(idx, ["m1"], hops=2, min_shared=1))
    assert "m2" in bridged  # shares Bob with the seed
    assert "m3" not in bridged  # only transitively linked → filtered out


def test_empty_index_and_unknown_seed_fail_open():
    assert eb.bridged_ids(eb.build_index([]), ["x"]) == []
    idx = eb.build_index([("m1", "Jane Doe")])
    assert eb.bridged_ids(idx, ["nobody"]) == []


def test_fold_inserts_missed_only_no_reorder_of_retrieved():
    fused = [{"id": "v1", "rrf_score": 0.05}, {"id": "v2", "rrf_score": 0.02}]
    out = eb.fold_into_candidates(
        fused,
        ["g1", "v2"],  # g1 missed (insert), v2 already retrieved (leave alone)
        record_lookup={"g1": {"id": "g1", "content": "gold"}},
        rrf_k=20,
        weight=1.0,
    )
    ids = [r["id"] for r in out]
    # g1 inserted at 1/21=0.0476 → between v1 (0.05) and v2 (0.02); v2 NOT boosted.
    assert ids == ["v1", "g1", "v2"]


def test_fold_is_noop_when_all_present():
    fused = [{"id": "v1", "rrf_score": 0.05}, {"id": "v2", "rrf_score": 0.02}]
    out = eb.fold_into_candidates(
        fused, ["v2", "v1"], record_lookup={}, rrf_k=20, weight=1.0
    )
    assert [r["id"] for r in out] == ["v1", "v2"]  # nothing missed → unchanged


def test_fold_skips_unmaterializable():
    fused = [{"id": "v1", "rrf_score": 0.05}]
    out = eb.fold_into_candidates(
        fused, ["ghost"], record_lookup={}, rrf_k=20, weight=1.0
    )
    assert [r["id"] for r in out] == ["v1"]
