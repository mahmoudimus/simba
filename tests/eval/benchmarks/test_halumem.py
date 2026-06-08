"""HaluMem loader + QA aggregation (offline; no dataset/judge)."""

from __future__ import annotations

import json

import simba.eval.benchmarks.halumem as hm

_REC = {
    "uuid": "u1",
    "persona_info": "Name: Martin Mark; Gender: M",
    "sessions": [
        {
            "start_time": "Sep 04, 2025",
            "memory_points": [
                {
                    "index": 0,
                    "memory_content": "Martin's birth date is 1996-08-02",
                    "memory_type": "Persona Memory",
                    "is_update": False,
                    "original_memories": [],
                    "timestamp": "t0",
                },
                {
                    "index": 1,
                    "memory_content": "Martin now lives in Berlin",
                    "memory_type": "Event Memory",
                    "is_update": True,
                    "original_memories": [0],
                    "timestamp": "t1",
                },
            ],
            "questions": [
                {
                    "question": "What is Martin's birth date?",
                    "answer": "1996-08-02",
                    "evidence": [
                        {"memory_content": "Martin's birth date is 1996-08-02"}
                    ],
                    "question_type": "Basic Fact Recall",
                    "difficulty": "easy",
                },
                {
                    "question": "What is Martin's middle name?",
                    "answer": "Unknown; not provided by the user.",
                    "evidence": [],
                    "question_type": "Memory Boundary",
                    "difficulty": "easy",
                },
            ],
        }
    ],
}


def test_loads_structure(tmp_path):
    p = tmp_path / "halu.jsonl"
    p.write_text(json.dumps(_REC) + "\n")
    users = hm.load_halumem(p)
    assert len(users) == 1
    u = users[0]
    assert u.uuid == "u1" and "Martin" in u.persona
    sess = u.sessions[0]
    # memory points: the update carries original_memories (the supersession signal)
    assert len(sess.memory_points) == 2
    upd = sess.memory_points[1]
    assert upd.is_update is True and upd.original_memories == [0]
    assert sess.memory_points[0].is_update is False


def test_boundary_and_evidence(tmp_path):
    p = tmp_path / "halu.jsonl"
    p.write_text(json.dumps(_REC) + "\n")
    q_fact, q_boundary = hm.load_halumem(p)[0].sessions[0].questions
    assert q_fact.evidence == ["Martin's birth date is 1996-08-02"]
    assert q_fact.is_boundary is False
    # "Memory Boundary" + empty evidence => abstention case (answering = hallucination)
    assert q_boundary.is_boundary is True


def test_user_limit(tmp_path):
    p = tmp_path / "halu.jsonl"
    p.write_text((json.dumps(_REC) + "\n") * 3)
    assert len(hm.load_halumem(p, user_limit=2)) == 2


def test_aggregate_qa_rates():
    outcomes = [
        ("Basic Fact Recall", hm.QA_CORRECT),
        ("Basic Fact Recall", hm.QA_HALLUCINATION),
        ("Memory Boundary", hm.QA_CORRECT),  # correctly abstained
        ("Memory Boundary", hm.QA_HALLUCINATION),  # fabricated an answer
    ]
    agg = hm.aggregate_qa(outcomes)
    assert agg["overall"]["n"] == 4
    assert agg["overall"]["accuracy"] == 0.5
    assert agg["overall"]["hallucination_rate"] == 0.5
    # boundary block isolates abstention performance
    assert agg["boundary"]["n"] == 2
    assert agg["boundary"]["accuracy"] == 0.5
    assert agg["boundary"]["hallucination_rate"] == 0.5
    assert set(agg["by_type"]) == {"Basic Fact Recall", "Memory Boundary"}


def test_aggregate_empty():
    agg = hm.aggregate_qa([])
    assert agg["overall"]["n"] == 0
    assert agg["overall"]["accuracy"] == 0.0
