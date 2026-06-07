"""Personalized PageRank over the knowledge-graph adjacency (pure Python).

C1's KG-into-recall failed because it folded *raw* BFS neighbors — on a
near-complete conversational graph that reaches everything, so it added noise,
not signal ([[kg-into-recall-cooccurrence-negative]]). PPR is the selective
alternative: power-iteration from a restart distribution over the query's seed
entities, so stationary mass concentrates on nodes *close and well-connected to
the seeds*. A budgeted fold of the top-mass nodes is therefore discriminating
where raw reachability was not.

Operates on a plain ``{node: set(neighbors)}`` adjacency so it's storage-agnostic
(the eval throwaway KG and the live ``kg_edges`` both reduce to this) and trivially
testable on toy graphs. Undirected/symmetric adjacency is assumed; all nodes in
the adjacency have at least one neighbor (isolated nodes are simply absent).
"""

from __future__ import annotations

import typing


def personalized_pagerank(
    adjacency: dict[str, set[str]],
    seeds: typing.Iterable[str],
    *,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> dict[str, float]:
    """Return ``{node: stationary_mass}`` for a PPR restarted at ``seeds``.

    The restart distribution is uniform over the seeds that are present as nodes.
    Returns ``{}`` when there are no nodes or no seed lands on a node (fail-open:
    the caller folds nothing). Mass sums to ~1.
    """
    nodes = list(adjacency)
    seed_nodes = [s for s in dict.fromkeys(seeds) if s in adjacency]
    if not nodes or not seed_nodes:
        return {}

    restart = dict.fromkeys(nodes, 0.0)
    weight = 1.0 / len(seed_nodes)
    for s in seed_nodes:
        restart[s] = weight

    rank = dict(restart)
    for _ in range(max_iter):
        nxt = {n: (1.0 - damping) * restart[n] for n in nodes}
        for node, mass in rank.items():
            neighbors = adjacency[node]
            if not neighbors:
                continue
            share = damping * mass / len(neighbors)
            for m in neighbors:
                if m in nxt:
                    nxt[m] += share
        diff = sum(abs(nxt[n] - rank[n]) for n in nodes)
        rank = nxt
        if diff < tol:
            break
    return rank
