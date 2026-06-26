from __future__ import annotations

import json
import pathlib

import simba.eval.interpretation_metrics as interpretation_metrics


def test_fail18_baseline_records_saved_fixture_and_live_prompt_gap(
    tmp_path: pathlib.Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    corpus = tmp_path / "corpus.json"
    saved = tmp_path / "saved.json"
    live = tmp_path / "live.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "How many things happened?",
                    "gold_answer": 2,
                    "gold_count": 2,
                    "failure_mode": "test",
                    "clingo_certain": 1,
                    "clingo_possible": 3,
                }
            ]
        ),
        encoding="utf-8",
    )
    corpus.write_text("[]", encoding="utf-8")
    saved.write_text(
        json.dumps(
            {
                "name": "saved",
                "prompt_version": "v1",
                "tool": "claude -p",
                "compiler_contract": ["answer_variable", "candidate_units"],
                "score": {"matches": 1, "total": 1},
                "cases": [
                    {
                        "id": "q1",
                        "answer_variable": "entity",
                        "individuation_policy": "canonical_entity",
                        "aggregation": "count_distinct",
                        "computed_answer": 2,
                        "gold": 2,
                        "match": True,
                        "candidate_units": [
                            {
                                "unit_id": "u1",
                                "status": "included",
                                "merge_target": None,
                                "reason": "supported",
                            }
                        ],
                        "facts": ["action(user,u1,complete)."],
                        "query": "answer(2).",
                        "rationale": "supported",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    live.write_text(
        json.dumps(
            {
                "tool": "claude -p",
                "prompt_version": "v1",
                "matches": 0,
                "total": 1,
                "results": [
                    {"id": "q1", "gold": 2, "pred": 1, "match": False}
                ],
            }
        ),
        encoding="utf-8",
    )

    baseline = interpretation_metrics.build_fail18_baseline(
        manifest_path=manifest,
        corpus_path=corpus,
        candidate_fixture_path=saved,
        live_candidate_path=live,
        include_repair=False,
    )

    assert baseline["summary"]["clingo_manifest_range"] == {
        "total": 1,
        "gold_known": 1,
        "contains_gold": 1,
        "misses_gold": 0,
    }
    assert (
        baseline["modes"]["clingo_manifest_range"]["source"]
        == "clingo_manifest_range"
    )
    assert (
        baseline["modes"]["clingo_manifest_range"]["executor"]
        == "python_ambiguity_backend"
    )
    assert baseline["saved_candidate_unit_fixture"]["matches"] == 1
    assert baseline["live_candidate_unit_prompt"]["matches"] == 0
    assert baseline["saved_fixture_succeeds_live_prompt_fails"] == ["q1"]
