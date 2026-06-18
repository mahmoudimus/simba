"""SubtleMemory loader + per-relation-slice aggregation (offline; no judge).

SubtleMemory (relational / contradiction memory eval) ships per-persona
``bench_instances.json`` (cases with relation labels + QA pairs) and
``history_sessions.json`` (the conversation corpus). The loader maps one persona
-> one simba ``Dataset``: every dialogue turn becomes a ``Memory``, every QA pair
becomes an ``EvalCase`` whose gold is all turns from the case's target
``session_ids`` and whose ``intent`` is the ``relation_type`` (so the recall /
QA harness groups by relation slice; ``contradictory`` is the headline).
"""

from __future__ import annotations

import json

import pytest

import simba.eval.benchmarks.subtlememory as sm

_BENCH = [
    {
        "instance_id": "rel-aaa",
        "case_id": "case-1",
        "persona_id": "0",
        "topic": "Literature",
        "source": "user-related",
        "case": "Amara leans toward African classics.",
        "facts": ["Amara enjoys African literature.", "She gravitates to classics."],
        "relation_type": "complementary",
        "relation_subtype": "any_one",
        "session_ids": ["s1", "s2"],
        "qas": [
            {
                "query": "Recommend two books.",
                "correct_answers": ["So Long a Letter; The Concubine", "alt gold"],
                "incorrect_answers": ["Open Water; Homegoing"],
            }
        ],
    },
    {
        "instance_id": "con-bbb",
        "case_id": "case-2",
        "persona_id": "0",
        "topic": "Diet",
        "source": "user-related",
        "case": "Amara's diet claims conflict.",
        "facts": ["Amara is vegetarian.", "Amara ate steak last week."],
        "relation_type": "contradictory",
        "relation_subtype": "a_user_vs_user",
        "session_ids": ["s3"],
        "qas": [
            {
                "query": "Is Amara vegetarian?",
                "correct_answers": ["The evidence is contradictory."],
                "incorrect_answers": ["Yes, strictly."],
            },
            {
                "query": "What conflict exists about her diet?",
                "correct_answers": ["She claims vegetarian but also ate steak."],
                "incorrect_answers": ["No conflict."],
            },
        ],
    },
    {
        "instance_id": "nua-ccc",
        "case_id": "case-3",
        "persona_id": "0",
        "topic": "Work",
        "source": "user-unrelated",
        "case": "Role depends on context.",
        "facts": ["A.", "B."],
        "relation_type": "nuanced",
        "relation_subtype": "Context",
        "session_ids": ["s4"],
        "qas": [],  # no QA -> contributes no EvalCase
    },
]

_HISTORY = [
    {
        "session_id": "s1",
        "persona_id": "0",
        "case_id": "case-1",
        "source": "user-related",
        "timestamp": "2025-04-20T20:49:34+08:00",
        "conversation_type": "learning",
        "order": 0,
        "history": [
            {"role": "user", "content": "I love African classics."},
            {"role": "assistant", "content": "Noted."},
        ],
    },
    {
        "session_id": "s2",
        "persona_id": "0",
        "case_id": "case-1",
        "source": "user-related",
        "timestamp": "2025-04-21T12:08:55+08:00",
        "conversation_type": "learning",
        "order": 1,
        "history": [
            {"role": "user", "content": "I gravitate to classics."},
        ],
    },
    {
        "session_id": "s3",
        "persona_id": "0",
        "case_id": "case-2",
        "source": "user-related",
        "timestamp": "2025-04-22T09:00:00+08:00",
        "conversation_type": "chat",
        "order": 2,
        "history": [
            {"role": "user", "content": "I am vegetarian. But I ate steak."},
        ],
    },
    {
        "session_id": "s4",
        "persona_id": "0",
        "case_id": "case-3",
        "source": "user-unrelated",
        "timestamp": "2025-04-23T09:00:00+08:00",
        "conversation_type": "chat",
        "order": 3,
        "history": [
            {"role": "user", "content": "Unrelated noise."},
        ],
    },
]


def _write_persona(root, pid, bench, history):
    pdir = root / f"persona_{pid}"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "bench_instances.json").write_text(json.dumps(bench))
    (pdir / "history_sessions.json").write_text(json.dumps(history))


# --- loader: corpus -----------------------------------------------------------


def test_one_dataset_per_persona() -> None:
    dsets = sm.load_subtlememory_data([("0", _BENCH, _HISTORY)])
    assert len(dsets) == 1
    assert dsets[0].name == "subtlememory_persona_0"


def test_every_turn_becomes_a_memory_with_timestamp() -> None:
    d = sm.load_subtlememory_data([("0", _BENCH, _HISTORY)])[0]
    # 2 + 1 + 1 + 1 = 5 turns across 4 sessions
    assert len(d.corpus) == 5
    # ids are session_id + turn index; globally unique
    ids = {m.id for m in d.corpus}
    assert "s1_0" in ids and "s1_1" in ids and "s2_0" in ids
    first = next(m for m in d.corpus if m.id == "s1_0")
    assert "user: I love African classics." in first.content
    assert first.created_at == "2025-04-20T20:49:34+08:00"
    assert first.session_source == "s1"


# --- loader: cases ------------------------------------------------------------


def test_each_qa_pair_becomes_one_case() -> None:
    d = sm.load_subtlememory_data([("0", _BENCH, _HISTORY)])[0]
    # case-1 has 1 qa, case-2 has 2 qas, case-3 has 0 -> 3 cases total
    assert len(d.cases) == 3
    ids = {c.id for c in d.cases}
    assert "0_rel-aaa_0" in ids
    assert "0_con-bbb_0" in ids and "0_con-bbb_1" in ids


def test_case_gold_is_all_turns_of_target_sessions() -> None:
    d = sm.load_subtlememory_data([("0", _BENCH, _HISTORY)])[0]
    c1 = next(c for c in d.cases if c.id == "0_rel-aaa_0")
    # session_ids s1 (2 turns) + s2 (1 turn) -> 3 gold ids
    assert set(c1.relevant_ids) == {"s1_0", "s1_1", "s2_0"}
    assert c1.query == "Recommend two books."
    assert c1.answer == "So Long a Letter; The Concubine"  # first correct answer


def test_intent_is_relation_type_and_note_is_subtype() -> None:
    d = sm.load_subtlememory_data([("0", _BENCH, _HISTORY)])[0]
    c1 = next(c for c in d.cases if c.id == "0_rel-aaa_0")
    assert c1.intent == "complementary"
    assert c1.note == "any_one"
    con = next(c for c in d.cases if c.id == "0_con-bbb_0")
    assert con.intent == sm.CONTRADICTORY
    assert con.note == "a_user_vs_user"


def test_contradictory_slice_identified() -> None:
    d = sm.load_subtlememory_data([("0", _BENCH, _HISTORY)])[0]
    contradictory = [c for c in d.cases if c.intent == sm.CONTRADICTORY]
    assert len(contradictory) == 2  # both qas from con-bbb
    assert all(c.id.startswith("0_con-bbb") for c in contradictory)


def test_dataset_load_validates_gold_resolves() -> None:
    # The simba Dataset invariant: every relevant_id must exist in the corpus.
    d = sm.load_subtlememory_data([("0", _BENCH, _HISTORY)])[0]
    corpus_ids = d.corpus_ids()
    for case in d.cases:
        for rid in case.relevant_ids:
            assert rid in corpus_ids


# --- loader: from disk + subsample knob --------------------------------------


def test_load_persona_from_disk(tmp_path) -> None:
    _write_persona(tmp_path, 0, _BENCH, _HISTORY)
    d = sm.load_persona(tmp_path, 0)
    assert d.name == "subtlememory_persona_0"
    assert len(d.cases) == 3


def test_persona_limit_subsamples(tmp_path) -> None:
    _write_persona(tmp_path, 0, _BENCH, _HISTORY)
    _write_persona(tmp_path, 1, _BENCH, _HISTORY)
    _write_persona(tmp_path, 2, _BENCH, _HISTORY)
    one = sm.load_subtlememory(tmp_path, persona_limit=1)
    assert len(one) == 1 and one[0].name == "subtlememory_persona_0"
    two = sm.load_subtlememory(tmp_path, persona_limit=2)
    assert {d.name for d in two} == {
        "subtlememory_persona_0",
        "subtlememory_persona_1",
    }
    every = sm.load_subtlememory(tmp_path, persona_limit=0)
    assert len(every) == 3
    heldout = sm.load_subtlememory(tmp_path, persona_start=1, persona_limit=2)
    assert [d.name for d in heldout] == [
        "subtlememory_persona_1",
        "subtlememory_persona_2",
    ]


def test_missing_session_id_is_dropped_not_crash() -> None:
    bad = [
        {
            "instance_id": "x",
            "case_id": "c",
            "persona_id": "0",
            "relation_type": "complementary",
            "relation_subtype": "any_one",
            "session_ids": ["s1", "ghost"],  # ghost not in history
            "qas": [{"query": "q", "correct_answers": ["a"], "incorrect_answers": []}],
        }
    ]
    d = sm.load_subtlememory_data([("0", bad, _HISTORY)])[0]
    c = d.cases[0]
    # ghost dropped; s1's turns kept -> still resolvable, no crash
    assert set(c.relevant_ids) == {"s1_0", "s1_1"}


def test_case_with_no_resolvable_gold_is_dropped() -> None:
    bad = [
        {
            "instance_id": "x",
            "case_id": "c",
            "persona_id": "0",
            "relation_type": "complementary",
            "relation_subtype": "any_one",
            "session_ids": ["ghost"],  # nothing resolves
            "qas": [{"query": "q", "correct_answers": ["a"], "incorrect_answers": []}],
        }
    ]
    d = sm.load_subtlememory_data([("0", bad, _HISTORY)])[0]
    assert d.cases == []  # no gold -> not scoreable -> dropped


# --- per-relation-slice aggregation ------------------------------------------


def test_aggregate_by_relation_splits_slices() -> None:
    rows = [
        ("complementary", True),
        ("complementary", False),
        (sm.CONTRADICTORY, True),
        (sm.CONTRADICTORY, True),
        ("nuanced", False),
    ]
    agg = sm.aggregate_by_relation(rows)
    assert agg["overall"]["n"] == 5
    assert agg["overall"]["accuracy"] == 0.6
    assert agg["by_relation"]["complementary"]["n"] == 2
    assert agg["by_relation"]["complementary"]["accuracy"] == 0.5
    assert agg["by_relation"][sm.CONTRADICTORY]["n"] == 2
    assert agg["by_relation"][sm.CONTRADICTORY]["accuracy"] == 1.0


def test_aggregate_flags_contradictory_headline() -> None:
    rows = [(sm.CONTRADICTORY, True), ("complementary", False)]
    agg = sm.aggregate_by_relation(rows)
    # the contradictory slice is the headline differentiator
    assert agg["contradictory"]["n"] == 1
    assert agg["contradictory"]["accuracy"] == 1.0
    assert agg["by_relation"][sm.CONTRADICTORY]["is_headline"] is True


def test_aggregate_empty() -> None:
    agg = sm.aggregate_by_relation([])
    assert agg["overall"]["n"] == 0
    assert agg["overall"]["accuracy"] == 0.0
    assert agg["contradictory"]["n"] == 0


# --- readback ceiling ---------------------------------------------------------


def test_readback_recall_uses_session_sources_as_ceiling() -> None:
    d = sm.load_subtlememory_data([("0", _BENCH, _HISTORY)])[0]
    report = sm.run_readback_recall([d])

    assert report["mode"] == "session_readback_ceiling"
    assert report["n_conversations"] == 1
    assert report["n_cases"] == 3
    # complementary has 3 gold turns, each contradictory QA has 1 gold turn:
    # (1/3 + 1 + 1) / 3
    assert report["overall"]["recall@1"] == pytest.approx(7 / 9)
    assert report["overall"]["recall@3"] == 1.0
    assert report["by_category"]["complementary"]["n"] == 1
    assert report["by_category"][sm.CONTRADICTORY]["n"] == 2
    assert report["diagnostics"]["max_gold_ids"] == 3
    assert report["diagnostics"]["gold_gt_k"]["1"] == 1
    assert report["diagnostics"]["gold_gt_k"]["3"] == 0


def test_readback_recall_falls_back_to_gold_without_session_source() -> None:
    d = sm.load_subtlememory_data([("0", _BENCH, _HISTORY)])[0]
    for mem in d.corpus:
        mem.session_source = ""
    case = next(c for c in d.cases if c.id == "0_rel-aaa_0")
    ranked = sm._readback_ranked_ids(d, case)

    assert ranked == ["s1_0", "s1_1", "s2_0"]


def test_compare_readback_ceiling_reports_delta_by_relation() -> None:
    d = sm.load_subtlememory_data([("0", _BENCH, _HISTORY)])[0]
    normal = {
        "overall": {"recall@1": 0.2, "recall@3": 0.5, "mrr": 0.4},
        "by_category": {
            "complementary": {"n": 1, "recall@1": 0.0, "mrr": 0.0},
            sm.CONTRADICTORY: {"n": 2, "recall@1": 0.5, "mrr": 0.5},
        },
    }
    comparison = sm.compare_readback_ceiling(normal, [d])

    ceiling = comparison["ceiling"]
    delta = comparison["delta_vs_recall"]
    assert ceiling["by_category"][sm.CONTRADICTORY]["n"] == 2
    assert delta["overall"]["recall@1"] == pytest.approx(
        ceiling["overall"]["recall@1"] - 0.2
    )
    assert delta["by_category"]["complementary"]["mrr"] == 1.0


# --- failure ledger -----------------------------------------------------------


def test_failure_ledger_classifies_no_session_hit() -> None:
    d = sm.load_subtlememory_data([("0", _BENCH, _HISTORY)])[0]
    case = next(c for c in d.cases if c.id == "0_con-bbb_0")
    report = sm.build_failure_ledger([d], {case.id: ["s1_0"]})

    row = next(r for r in report["cases"] if r["case_id"] == case.id)
    assert row["gap_label"] == "no_session_hit"
    assert report["summary"]["gap_counts"]["no_session_hit"] >= 1


def test_failure_ledger_classifies_partial_and_content_gaps() -> None:
    d = sm.load_subtlememory_data([("0", _BENCH, _HISTORY)])[0]
    case = next(c for c in d.cases if c.id == "0_rel-aaa_0")

    partial = sm.build_failure_ledger([d], {case.id: ["s1_0"]})
    partial_row = next(r for r in partial["cases"] if r["case_id"] == case.id)
    assert partial_row["gap_label"] == "partial_session_hit"

    content = sm.build_failure_ledger([d], {case.id: ["s1_0", "s2_0"]})
    content_row = next(r for r in content["cases"] if r["case_id"] == case.id)
    assert content_row["gap_label"] == "session_content_gap"
    assert content["summary"]["gap_counts"]["session_content_gap"] == 1


def test_write_failure_ledger_creates_parent(tmp_path) -> None:
    out = sm.write_failure_ledger(
        {"summary": {"n_cases": 0}, "cases": []}, tmp_path / "nested" / "driver.json"
    )

    assert out.exists()
    assert json.loads(out.read_text())["summary"]["n_cases"] == 0


def test_driver_promotion_gate_passes_positive_recall_with_guarded_mrr() -> None:
    gate = sm.driver_promotion_gate(
        winner_positive=True,
        winner_delta={
            "overall": {"recall@10": 0.01, "mrr": -0.001},
            sm.CONTRADICTORY: {"recall@10": 0.02, "mrr": -0.009},
        },
    )

    assert gate["passed"] is True
    assert {check["name"] for check in gate["checks"]} == {
        "objective_positive",
        "contradictory_recall@10_lift",
        "overall_recall@10_lift",
        "contradictory_mrr_guard",
        "overall_mrr_guard",
    }


def test_driver_promotion_gate_fails_material_mrr_regression() -> None:
    gate = sm.driver_promotion_gate(
        winner_positive=True,
        winner_delta={
            "overall": {"recall@10": 0.01, "mrr": -0.001},
            sm.CONTRADICTORY: {"recall@10": 0.02, "mrr": -0.02},
        },
    )

    assert gate["passed"] is False
    failed = {check["name"] for check in gate["checks"] if not check["passed"]}
    assert failed == {"contradictory_mrr_guard"}
