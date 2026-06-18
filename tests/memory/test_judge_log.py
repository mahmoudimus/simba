from __future__ import annotations

import pathlib

import simba.db
import simba.memory.conflict as conflict
import simba.memory.judge_log as judge_log


class FakeLlm:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def complete_json(self, prompt: str):
        self.calls += 1
        return self.reply


def test_judge_log_record_is_idempotent(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        row1 = judge_log.record(
            decision_key="decision-1",
            project_path="/repo",
            strategy="heuristic",
            input_memory_ids=["mem_a", "mem_b"],
            winner_id="mem_a",
            loser_ids=["mem_b"],
            judge_kind="heuristic",
            judge_model="rules-v1",
            prompt_hash="p",
            config_hash="c",
            decision={"conflict": True, "description": "two claims"},
            now=100.0,
        )
        row2 = judge_log.record(
            decision_key="decision-1",
            project_path="/repo",
            strategy="heuristic",
            input_memory_ids=["mem_a", "mem_b"],
            winner_id="mem_b",
            loser_ids=["mem_a"],
            judge_kind="heuristic",
            judge_model="rules-v2",
            prompt_hash="p2",
            config_hash="c2",
            decision={"conflict": False},
            now=200.0,
        )

    assert row1.id == row2.id
    assert row2.winner_id == "mem_a"
    assert judge_log.decision_payload(row2)["conflict"] is True
    assert judge_log.loser_ids(row2) == ["mem_b"]


def test_logged_write_conflict_reuses_decision_without_judge(
    tmp_path: pathlib.Path,
) -> None:
    llm = FakeLlm({"conflict": True, "description": "two cities"})
    with simba.db.connect(tmp_path):
        first = conflict.detect_conflicts_on_write_logged(
            "mem_new",
            "Alice lives in Berlin.",
            [("mem_old", "Alice lives in Paris.")],
            llm_client=llm,
            project_path="/repo",
            now=100.0,
        )
        second = conflict.detect_conflicts_on_write_logged(
            "mem_new",
            "Alice lives in Berlin.",
            [("mem_old", "Alice lives in Paris.")],
            llm_client=llm,
            project_path="/repo",
            now=200.0,
        )
        rows = judge_log.recent(project_path="/repo")

    assert first == [("mem_old", "two cities")]
    assert second == [("mem_old", "two cities")]
    assert llm.calls == 1
    assert len(rows) == 1
    assert rows[0].winner_id == "mem_new"
    assert judge_log.loser_ids(rows[0]) == ["mem_old"]
