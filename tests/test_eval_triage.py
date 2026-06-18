"""Tests for ``simba eval triage``."""

from __future__ import annotations

import json

import simba.__main__ as cli
import simba.eval.recall_triage as recall_triage_eval


def test_recall_triage_eval_default_cases_pass_gate() -> None:
    result = recall_triage_eval.evaluate(recall_triage_eval.load_cases())

    assert result["gate"] == "pass"
    assert result["false_negatives"] == 0
    assert result["n"] >= 5


def test_eval_triage_prints_summary(capsys) -> None:
    rc = cli._eval_triage([])

    assert rc == 0
    out = capsys.readouterr().out
    assert "recall-triage eval:" in out
    assert "gate=pass" in out


def test_eval_triage_json(capsys) -> None:
    rc = cli._eval_triage(["--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["false_negatives"] == 0
    assert payload["gate"] == "pass"


def test_eval_triage_loads_jsonl_path(tmp_path, capsys) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": "what is next from the borrow roadmap?",
                "expected": "retrieve",
                "category": "memory",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rc = cli._eval_triage(["--path", str(path), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n"] == 1
    assert payload["cases"][0]["actual"] == "retrieve"


def test_eval_triage_unknown_option_returns_1() -> None:
    assert cli._eval_triage(["--bad"]) == 1
