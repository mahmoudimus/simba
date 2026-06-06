"""Tests for ``simba eval bench`` (A4) and its results-store helpers."""

from __future__ import annotations

import json
import pathlib

import simba.__main__ as cli
import simba.config
import simba.eval.bench_config as bench_config
import simba.eval.bench_results as bench_results
import simba.eval.benchmarks.locomo as locomo
import simba.eval.benchmarks.run as bench_run
import simba.eval.run as run
import simba.memory.embedding_cache as ec
from simba.eval.bench_results import append_result, current_git_sha
from simba.eval.dataset import Dataset, EvalCase, Memory

_OVERALL = {
    "recall@1": 0.5,
    "recall@3": 0.6,
    "recall@5": 0.7,
    "recall@10": 0.8,
    "mrr": 0.55,
    "ndcg@1": 0.5,
    "ndcg@3": 0.6,
    "ndcg@5": 0.7,
    "ndcg@10": 0.8,
}


def _fake_sync_embedders(cfg, *, cache=None):
    return (lambda t: [0.1] * 4), (lambda t: [0.2] * 4)


def _fake_run_recall(datasets, *, embed_doc, embed_query, cfg):
    return {
        "n_conversations": len(datasets),
        "n_cases": 2,
        "overall": dict(_OVERALL),
        "by_category": {},
    }


def _fake_dataset(name: str) -> Dataset:
    return Dataset(
        name=name,
        corpus=[Memory(id=f"{name}-m1", content="hello world")],
        cases=[EvalCase(id=f"{name}-q1", query="hi", relevant_ids=[f"{name}-m1"])],
    )


def _install_common_fakes(monkeypatch, tmp_path, **bench_overrides):
    """Wire the standard fakes for a successful locomo recall run."""
    bcfg = bench_config.BenchConfig(
        locomo_path="/fake/locomo.json",
        results_path=str(tmp_path / ".simba" / "eval" / "results.jsonl"),
        embedding_cache_path=str(tmp_path / ".simba" / "eval" / "embedding_cache.db"),
        judge_cache_path=str(tmp_path / ".simba" / "eval" / "judge_cache.db"),
        **bench_overrides,
    )
    mcfg = simba.config.load("memory")

    def _fake_load(section, *a, **k):
        if section == "bench":
            return bcfg
        if section == "memory":
            return mcfg
        raise KeyError(section)

    monkeypatch.setattr(simba.config, "load", _fake_load)
    monkeypatch.setattr(
        locomo, "load_locomo", lambda path: [_fake_dataset("a"), _fake_dataset("b")]
    )
    monkeypatch.setattr(run, "sync_embedders", _fake_sync_embedders)
    monkeypatch.setattr(bench_run, "run_recall", _fake_run_recall)
    monkeypatch.setattr(bench_results, "current_git_sha", lambda: "abc1234")
    return bcfg


# --------------------------------------------------------------------------
# arg-parse / validation
# --------------------------------------------------------------------------


def test_bench_missing_dataset_name_returns_1(monkeypatch) -> None:
    assert cli._eval_bench([]) == 1


def test_bench_unknown_dataset_returns_1(monkeypatch) -> None:
    assert cli._eval_bench(["badname"]) == 1


def test_bench_unknown_flag_returns_1(monkeypatch) -> None:
    assert cli._eval_bench(["locomo", "--notaflags"]) == 1


def test_bench_locomo_missing_path_returns_1(monkeypatch, tmp_path) -> None:
    bcfg = bench_config.BenchConfig(locomo_path="")
    mcfg = simba.config.load("memory")

    def _fake_load(section, *a, **k):
        return bcfg if section == "bench" else mcfg

    monkeypatch.setattr(simba.config, "load", _fake_load)
    assert cli._eval_bench(["locomo"]) == 1


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------


def test_bench_locomo_recall_runs_and_appends_result(monkeypatch, tmp_path) -> None:
    bcfg = _install_common_fakes(monkeypatch, tmp_path)
    rc = cli._eval_bench(["locomo"])
    assert rc == 0

    results = pathlib.Path(bcfg.results_path)
    assert results.exists()
    lines = results.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["git_sha"] == "abc1234"
    assert record["dataset"] == "locomo"


def test_bench_n_flag_slices_datasets(monkeypatch, tmp_path) -> None:
    _install_common_fakes(monkeypatch, tmp_path)
    seen: dict[str, int] = {}

    def _spy_run_recall(datasets, *, embed_doc, embed_query, cfg):
        seen["count"] = len(datasets)
        return _fake_run_recall(
            datasets, embed_doc=embed_doc, embed_query=embed_query, cfg=cfg
        )

    monkeypatch.setattr(bench_run, "run_recall", _spy_run_recall)
    rc = cli._eval_bench(["locomo", "--n", "1"])
    assert rc == 0
    assert seen["count"] == 1


def test_bench_json_flag_prints_json(monkeypatch, tmp_path, capsys) -> None:
    _install_common_fakes(monkeypatch, tmp_path)
    rc = cli._eval_bench(["locomo", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "recall" in parsed


def test_bench_config_memory_overrides_applied(monkeypatch, tmp_path) -> None:
    _install_common_fakes(monkeypatch, tmp_path)
    seen: dict[str, object] = {}

    def _spy_run_recall(datasets, *, embed_doc, embed_query, cfg):
        seen["cfg"] = cfg
        return _fake_run_recall(
            datasets, embed_doc=embed_doc, embed_query=embed_query, cfg=cfg
        )

    monkeypatch.setattr(bench_run, "run_recall", _spy_run_recall)
    rc = cli._eval_bench(["locomo"])
    assert rc == 0
    cfg = seen["cfg"]
    assert cfg.llm_rerank_enabled is False
    assert cfg.max_results == 20


def test_bench_embedding_cache_passed_to_sync_embedders(monkeypatch, tmp_path) -> None:
    _install_common_fakes(monkeypatch, tmp_path)
    seen: dict[str, object] = {}

    def _spy_sync(cfg, *, cache=None):
        seen["cache"] = cache
        return _fake_sync_embedders(cfg, cache=cache)

    monkeypatch.setattr(run, "sync_embedders", _spy_sync)
    rc = cli._eval_bench(["locomo"])
    assert rc == 0
    assert isinstance(seen["cache"], ec.EmbeddingCache)


# --------------------------------------------------------------------------
# bench_results helpers
# --------------------------------------------------------------------------


def test_current_git_sha_returns_string() -> None:
    assert isinstance(current_git_sha(), str)


def test_append_result_creates_file_and_appends(tmp_path) -> None:
    path = tmp_path / "sub" / "results.jsonl"
    append_result(path, {"dataset": "locomo", "x": 1})
    append_result(path, {"dataset": "longmemeval", "x": 2})
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["dataset"] == "locomo"
    assert json.loads(lines[1])["dataset"] == "longmemeval"
