"""HotpotQA loader: distractor-setting JSON -> one Dataset per question.

corpus = the 10 context paragraphs (id = title); relevant_ids = the supporting-
fact paragraph titles (the bridge). Scored with bridge_recall@k (all hops in
top-k). Fully local — no LLM judge.
"""

from __future__ import annotations

import simba.eval.benchmarks.hotpotqa as hq

RAW = [
    {
        "_id": "q1",
        "question": "Where was the director of Film X born?",
        "answer": "Paris",
        "type": "bridge",
        "level": "hard",
        "context": [
            ["Film X", ["Film X is a 2000 movie.", "It was directed by Jane Doe."]],
            ["Jane Doe", ["Jane Doe is a director.", "She was born in Paris."]],
            ["Distractor", ["An unrelated sentence about nothing."]],
        ],
        "supporting_facts": [["Film X", 1], ["Jane Doe", 1]],
    },
    {
        "_id": "q2",
        "question": "Are A and B the same type?",
        "answer": "yes",
        "type": "comparison",
        "context": [["A", ["a fact."]], ["B", ["b fact."]]],
        "supporting_facts": [["A", 0], ["B", 0]],
    },
]


def test_one_dataset_per_question_with_bridge_gold():
    ds = hq.load_hotpotqa_data(RAW)
    assert len(ds) == 2
    d = ds[0]
    assert d.name == "q1"
    assert {m.id for m in d.corpus} == {"Film X", "Jane Doe", "Distractor"}
    case = d.cases[0]
    assert case.query.startswith("Where was the director")
    assert set(case.relevant_ids) == {"Film X", "Jane Doe"}  # the two bridge paras
    assert case.intent == "bridge"
    assert case.answer == "Paris"
    fx = next(m for m in d.corpus if m.id == "Film X")
    assert "directed by Jane Doe" in fx.content  # title + sentences joined


def test_comparison_intent():
    ds = hq.load_hotpotqa_data(RAW)
    assert ds[1].cases[0].intent == "comparison"
    assert set(ds[1].cases[0].relevant_ids) == {"A", "B"}


def test_drops_question_with_unresolvable_gold():
    raw = [
        {
            "_id": "q3",
            "question": "q",
            "answer": "a",
            "type": "bridge",
            "context": [["A", ["x."]]],
            "supporting_facts": [["Ghost", 0]],  # title not in context
        }
    ]
    assert hq.load_hotpotqa_data(raw) == []
