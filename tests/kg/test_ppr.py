"""Personalized PageRank over the KG adjacency — the selective, mass-ranked
alternative to C1's non-discriminating BFS fold. Mass concentrates near the
query's seed entities, so a *budgeted* fold (top-N by mass) is selective where
raw reachability was not."""

from __future__ import annotations

import simba.kg.ppr as ppr


def test_mass_decreases_with_distance_from_seed():
    # Path a-b-c-d seeded at a. (a is degree-1, so it hands all its mass to b and
    # gets only half back — b can exceed a; that's a known undirected-PPR degree
    # effect, not a bug. The robust signal is decay with distance: b > c > d, and
    # the seed beats the farthest node.)
    adj = {"a": {"b"}, "b": {"a", "c"}, "c": {"b", "d"}, "d": {"c"}}
    r = ppr.personalized_pagerank(adj, ["a"])
    assert r["b"] > r["c"] > r["d"] > 0.0
    assert r["a"] > r["d"]


def test_seedless_and_unknown_seeds_return_empty():
    adj = {"a": {"b"}, "b": {"a"}}
    assert ppr.personalized_pagerank(adj, []) == {}
    assert ppr.personalized_pagerank(adj, ["zzz"]) == {}  # seed not a node


def test_favors_the_seed_cluster():
    # Two triangles joined by a single bridge (c-d); seed in the first triangle.
    adj = {
        "a": {"b", "c"},
        "b": {"a", "c"},
        "c": {"a", "b", "d"},
        "d": {"c", "e", "f"},
        "e": {"d", "f"},
        "f": {"d", "e"},
    }
    r = ppr.personalized_pagerank(adj, ["a"])
    assert min(r["a"], r["b"], r["c"]) > max(r["e"], r["f"])


def test_mass_sums_to_about_one():
    adj = {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}}
    r = ppr.personalized_pagerank(adj, ["a"])
    assert abs(sum(r.values()) - 1.0) < 1e-6


def test_multiple_seeds_are_symmetric():
    adj = {"a": {"b"}, "b": {"a", "c"}, "c": {"b", "d"}, "d": {"c"}}
    r = ppr.personalized_pagerank(adj, ["a", "d"])
    assert abs(r["a"] - r["d"]) < 1e-6
    assert abs(r["b"] - r["c"]) < 1e-6


def test_rank_memories_orders_by_mass_and_limits():
    # Star centered on the seed a; e is 2 hops out (only via b).
    adj = {"a": {"b", "c", "d"}, "b": {"a", "e"}, "c": {"a"}, "d": {"a"}, "e": {"b"}}
    em = {n: {f"m_{n}"} for n in adj}
    ranked = ppr.rank_memories(adj, em, ["a"], top=5)
    assert ranked[0] == "m_a"  # the seed's own memory ranks first
    assert ranked[-1] == "m_e"  # the farthest memory ranks last
    assert ppr.rank_memories(adj, em, ["a"], top=2) == ranked[:2]  # budget caps


def test_rank_memories_no_seed_is_empty():
    adj = {"a": {"b"}, "b": {"a"}}
    assert ppr.rank_memories(adj, {"a": {"m"}}, [], top=5) == []
