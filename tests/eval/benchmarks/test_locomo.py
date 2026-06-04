"""Tests for the LoCoMo loader (JSON -> simba eval Datasets)."""

from __future__ import annotations

import simba.eval.benchmarks.locomo as locomo

_SAMPLE = {
    "sample_id": "conv-1",
    "conversation": {
        "speaker_a": "Caroline",
        "speaker_b": "Melanie",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel!"},
            {"speaker": "Caroline", "dia_id": "D1:3", "text": "Went to group 7 May."},
        ],
        "session_2_date_time": "2:00 pm on 9 May, 2023",
        "session_2": [
            {"speaker": "Melanie", "dia_id": "D2:1", "text": "Nice to hear!"},
        ],
    },
    "qa": [
        {"question": "When did Caroline go?", "answer": "7 May",
         "evidence": ["D1:3"], "category": 2},
        {"question": "adversarial one", "evidence": ["D2:1"], "category": 5,
         "adversarial_answer": "x"},
        {"question": "dangling evidence", "answer": "y",
         "evidence": ["D9:9"], "category": 4},
    ],
}


def test_one_dataset_per_conversation() -> None:
    dsets = locomo.load_locomo_data([_SAMPLE])
    assert len(dsets) == 1
    assert dsets[0].name == "conv-1"


def test_turns_become_corpus_with_speaker_and_session_date() -> None:
    d = locomo.load_locomo_data([_SAMPLE])[0]
    assert d.corpus_ids() == {"D1:1", "D1:3", "D2:1"}
    d13 = next(m for m in d.corpus if m.id == "D1:3")
    assert "Caroline" in d13.content and "Went to group" in d13.content
    # session date is prefixed so relative time ("yesterday") can be grounded
    assert "8 May" in d13.content


def test_cases_keep_resolvable_evidence_drop_dangling() -> None:
    d = locomo.load_locomo_data([_SAMPLE])[0]
    golds = {c.relevant_ids[0] for c in d.cases}
    # D1:3 and D2:1 resolve; D9:9 is dangling -> that case dropped
    assert golds == {"D1:3", "D2:1"}


def test_category_recorded_as_intent() -> None:
    d = locomo.load_locomo_data([_SAMPLE])[0]
    by_gold = {c.relevant_ids[0]: c.intent for c in d.cases}
    assert by_gold["D1:3"] == "single-hop"  # category 2
    assert by_gold["D2:1"] == "adversarial"  # category 5


def test_exclude_adversarial_helper() -> None:
    d = locomo.load_locomo_data([_SAMPLE], include_adversarial=False)[0]
    assert {c.relevant_ids[0] for c in d.cases} == {"D1:3"}  # adversarial dropped
