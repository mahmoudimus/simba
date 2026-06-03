"""Tests for the LongMemEval loader (turn-level recall@k, per-question haystacks).

One Dataset per question; gold = has_answer turns; abstention excluded by default.
"""

from __future__ import annotations

from simba.eval.benchmarks import longmemeval as lme

_SAMPLE = [
    {
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "What car issue did I have?",
        "answer": "GPS not working",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "GPS stopped working", "has_answer": True},
                {"role": "assistant", "content": "Sorry to hear", "has_answer": False},
            ],
            [
                {"role": "user", "content": "Nice weather today", "has_answer": False},
            ],
        ],
        "answer_session_ids": ["s1"],
    },
    {
        # abstention variant that *does* carry an evidence turn
        "question_id": "q2_abs",
        "question_type": "temporal-reasoning",
        "question": "When did I buy a boat?",
        "answer": "no information available",
        "haystack_session_ids": ["s3"],
        "haystack_sessions": [
            [{"role": "user", "content": "I went sailing once", "has_answer": True}],
        ],
        "answer_session_ids": ["s3"],
    },
    {
        # no has_answer turn anywhere -> unresolvable -> always dropped
        "question_id": "q3",
        "question_type": "multi-session",
        "question": "Unanswerable from haystack",
        "answer": "x",
        "haystack_session_ids": ["s4"],
        "haystack_sessions": [
            [{"role": "user", "content": "small talk", "has_answer": False}],
        ],
        "answer_session_ids": ["s4"],
    },
]


def test_one_dataset_per_question_excludes_abstention_by_default() -> None:
    datasets = lme.load_longmemeval_data(_SAMPLE)
    # q2_abs excluded (abstention), q3 dropped (no gold) -> only q1
    assert [d.name for d in datasets] == ["q1"]


def test_include_abstention_keeps_resolvable_abs_questions() -> None:
    datasets = lme.load_longmemeval_data(_SAMPLE, include_abstention=True)
    # q1 + q2_abs (has gold); q3 still dropped (no gold)
    assert {d.name for d in datasets} == {"q1", "q2_abs"}


def test_turns_become_corpus_with_role_and_session_keyed_ids() -> None:
    ds = next(d for d in lme.load_longmemeval_data(_SAMPLE) if d.name == "q1")
    ids = {m.id for m in ds.corpus}
    assert ids == {"s1#0", "s1#1", "s2#0"}
    first = next(m for m in ds.corpus if m.id == "s1#0")
    assert first.content == "user: GPS stopped working"


def test_gold_is_has_answer_turns() -> None:
    ds = next(d for d in lme.load_longmemeval_data(_SAMPLE) if d.name == "q1")
    assert len(ds.cases) == 1
    assert ds.cases[0].relevant_ids == ["s1#0"]


def test_intent_is_question_type() -> None:
    ds = next(d for d in lme.load_longmemeval_data(_SAMPLE) if d.name == "q1")
    assert ds.cases[0].intent == "single-session-user"
