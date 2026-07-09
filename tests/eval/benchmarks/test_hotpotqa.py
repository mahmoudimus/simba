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


def test_pooled_shares_one_large_corpus_across_questions():
    # Pooling deduplicates paragraphs by title into ONE corpus so each question
    # must recall its gold from thousands of competitors (the fullwiki recall
    # regime), not a 10-paragraph haystack.
    pooled = hq.load_hotpotqa_pooled(RAW)
    assert len(pooled) == 1
    d = pooled[0]
    # q1 has Film X / Jane Doe / Distractor; q2 has A / B → 5 unique titles.
    assert {m.id for m in d.corpus} == {"Film X", "Jane Doe", "Distractor", "A", "B"}
    assert len(d.cases) == 2  # one case per question, shared corpus
    q1 = next(c for c in d.cases if c.id == "q1")
    assert set(q1.relevant_ids) == {"Film X", "Jane Doe"}


def test_pooled_dedupes_shared_titles():
    raw = [
        {
            "_id": "x",
            "question": "q",
            "answer": "a",
            "type": "bridge",
            "context": [["Shared", ["s1."]], ["OnlyX", ["x1."]]],
            "supporting_facts": [["Shared", 0], ["OnlyX", 0]],
        },
        {
            "_id": "y",
            "question": "q2",
            "answer": "b",
            "type": "bridge",
            "context": [["Shared", ["s1."]], ["OnlyY", ["y1."]]],
            "supporting_facts": [["Shared", 0], ["OnlyY", 0]],
        },
    ]
    d = hq.load_hotpotqa_pooled(raw)[0]
    assert {m.id for m in d.corpus} == {"Shared", "OnlyX", "OnlyY"}  # Shared once
    assert len(d.cases) == 2


def test_pooled_caps_questions():
    d = hq.load_hotpotqa_pooled(RAW, max_questions=1)[0]
    assert len(d.cases) == 1  # only the first question pooled


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
