"""Tests for the results store + leaderboard (A5)."""

from __future__ import annotations

import simba.__main__ as cli
import simba.config
import simba.db
import simba.eval.bench_config as bench_config
from simba.eval.bench_results import (
    append_result,
    latest_two_by_group,
    load_results,
)
from simba.eval.leaderboard import _delta_str, render_markdown, write_leaderboard


def _record(dataset: str, timestamp: float, recall5: float, *, qa=None, sha="abc1234"):
    rec: dict[str, object] = {
        "dataset": dataset,
        "timestamp": timestamp,
        "git_sha": sha,
        "split": None,
        "recall": {
            "overall": {
                "recall@1": recall5 - 0.1,
                "recall@3": recall5 - 0.05,
                "recall@5": recall5,
                "recall@10": recall5 + 0.05,
                "mrr": recall5 - 0.02,
            }
        },
        "qa": qa,
    }
    return rec


# --------------------------------------------------------------------------
# load_results
# --------------------------------------------------------------------------


def test_load_results_empty_when_file_missing(tmp_path) -> None:
    assert load_results(tmp_path / "nope.jsonl") == []


def test_load_results_skips_malformed_lines(tmp_path) -> None:
    p = tmp_path / "results.jsonl"
    p.write_text('{"dataset": "locomo", "timestamp": 1.0}\nnot-json\n')
    records = load_results(p)
    assert len(records) == 1
    assert records[0]["dataset"] == "locomo"


def test_append_result_is_loadable_back(tmp_path) -> None:
    p = tmp_path / "results.jsonl"
    rec = {"dataset": "locomo", "timestamp": 1.0, "x": 7}
    append_result(p, rec)
    loaded = load_results(p)
    assert loaded == [rec]


# --------------------------------------------------------------------------
# latest_two_by_group
# --------------------------------------------------------------------------


def test_latest_two_by_group_returns_correct_order() -> None:
    r1 = _record("locomo", 1.0, 0.5)
    r2 = _record("locomo", 2.0, 0.6)
    r3 = _record("locomo", 3.0, 0.7)
    groups = latest_two_by_group([r1, r2, r3])
    assert len(groups) == 1
    (latest, prev) = next(iter(groups.values()))
    assert latest is r3
    assert prev is r2


def test_latest_two_by_group_two_datasets_separate_groups() -> None:
    groups = latest_two_by_group(
        [_record("locomo", 1.0, 0.5), _record("longmemeval", 1.0, 0.4)]
    )
    assert len(groups) == 2


# --------------------------------------------------------------------------
# _delta_str
# --------------------------------------------------------------------------


def test_delta_str_positive() -> None:
    assert _delta_str(0.570, 0.558) == "+0.012"


def test_delta_str_negative() -> None:
    assert _delta_str(0.490, 0.493) == "-0.003"


def test_delta_str_no_previous() -> None:
    assert _delta_str(0.5, None) == ""


# --------------------------------------------------------------------------
# render_markdown
# --------------------------------------------------------------------------


def test_render_markdown_contains_recall5() -> None:
    groups = latest_two_by_group([_record("locomo", 1.0, 0.57)])
    md = render_markdown(groups)
    assert "recall@5" in md
    assert "## locomo" in md


def test_render_markdown_no_qa_section_when_qa_is_none() -> None:
    groups = latest_two_by_group([_record("locomo", 1.0, 0.57, qa=None)])
    md = render_markdown(groups)
    assert "accuracy" not in md


def test_write_leaderboard_creates_file(tmp_path) -> None:
    results = tmp_path / "results.jsonl"
    append_result(results, _record("locomo", 1.0, 0.5))
    append_result(results, _record("locomo", 2.0, 0.6))
    out = tmp_path / "BENCHMARKS.md"
    write_leaderboard(results, out)
    assert out.exists()
    assert "# Benchmark Results" in out.read_text()


# --------------------------------------------------------------------------
# CLI helper
# --------------------------------------------------------------------------


def test_leaderboard_cmd_no_write_prints_table(monkeypatch, tmp_path, capsys) -> None:
    results = tmp_path / ".simba" / "eval" / "results.jsonl"
    append_result(results, _record("locomo", 1.0, 0.5))

    bcfg = bench_config.BenchConfig(
        results_path=str(results), leaderboard_path=str(tmp_path / "BENCHMARKS.md")
    )
    monkeypatch.setattr(simba.config, "load", lambda section, *a, **k: bcfg)
    monkeypatch.setattr(simba.db, "find_repo_root", lambda cwd: tmp_path)

    rc = cli._eval_leaderboard(["--no-write"])
    assert rc == 0
    assert capsys.readouterr().out.strip() != ""
    # --no-write must not create the leaderboard file
    assert not (tmp_path / "BENCHMARKS.md").exists()


def test_leaderboard_cmd_returns_1_when_no_results(monkeypatch, tmp_path) -> None:
    bcfg = bench_config.BenchConfig(
        results_path=str(tmp_path / "missing.jsonl"),
        leaderboard_path=str(tmp_path / "BENCHMARKS.md"),
    )
    monkeypatch.setattr(simba.config, "load", lambda section, *a, **k: bcfg)
    monkeypatch.setattr(simba.db, "find_repo_root", lambda cwd: tmp_path)
    assert cli._eval_leaderboard([]) == 1
