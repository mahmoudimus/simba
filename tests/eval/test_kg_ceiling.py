"""The Track B recall-ceiling diagnostic: how much gold could a graph lever add
beyond what the vector arm already retrieves? This is the gate — if the ceiling
is ~0, PPR cannot move recall@k no matter how well-tuned (the C1 lesson,
quantified)."""

from __future__ import annotations

import simba.eval.kg_ceiling as kgceil
import simba.eval.kg_corpus as kgc
from simba.eval.dataset import Memory


def _kg(triples_by_mem: dict[str, list[tuple[str, str, str]]]) -> kgc.CorpusKG:
    corpus = [Memory(id=mid, content="x") for mid in triples_by_mem]
    return kgc.build_corpus_kg(corpus, lambda m: triples_by_mem[m.id])


def _entities(query: str) -> list[str]:
    # Deterministic seed extractor for tests: whitespace tokens.
    return query.split()


def test_headroom_when_gold_reachable_but_not_retrieved():
    # Gold m3 is 2 hops from the query seed (A) via the graph, and NOT in topk.
    kg = _kg(
        {"m1": [("A", "r", "B")], "m2": [("B", "r", "C")], "m3": [("C", "r", "D")]}
    )
    base, ceil = kgceil.case_ceiling(
        query="A", gold=["m3"], topk=["m1"], kg=kg, max_hops=2, entities_of=_entities
    )
    assert base == 0.0  # m3 not retrieved
    assert ceil == 1.0  # m3 reachable via the graph → recoverable


def test_no_headroom_when_gold_already_retrieved():
    kg = _kg({"m1": [("A", "r", "B")], "m2": [("B", "r", "C")]})
    base, ceil = kgceil.case_ceiling(
        query="A", gold=["m2"], topk=["m2"], kg=kg, max_hops=3, entities_of=_entities
    )
    assert base == 1.0 and ceil == 1.0  # already in topk → graph adds nothing


def test_no_headroom_when_gold_unreachable():
    kg = _kg({"m1": [("A", "r", "B")], "m9": [("Y", "r", "Z")]})
    base, ceil = kgceil.case_ceiling(
        query="A", gold=["m9"], topk=["m1"], kg=kg, max_hops=5, entities_of=_entities
    )
    assert base == 0.0 and ceil == 0.0  # disconnected → graph can't help either


def test_aggregate_reports_net_new_headroom():
    rows = [(0.0, 1.0), (1.0, 1.0), (0.0, 0.0), (0.5, 1.0)]
    rep = kgceil.aggregate_ceiling(rows, density=0.04)
    assert rep.n_cases == 4
    assert rep.n_with_headroom == 2  # rows 0 and 3 gained
    assert abs(rep.baseline_recall - (0.0 + 1.0 + 0.0 + 0.5) / 4) < 1e-9
    assert abs(rep.ceiling_recall - (1.0 + 1.0 + 0.0 + 1.0) / 4) < 1e-9
    assert abs(rep.net_new_fraction - (rep.ceiling_recall - rep.baseline_recall)) < 1e-9
    assert rep.density == 0.04


def test_aggregate_empty():
    rep = kgceil.aggregate_ceiling([], density=0.0)
    assert rep.n_cases == 0
    assert rep.baseline_recall == 0.0
    assert rep.net_new_fraction == 0.0
