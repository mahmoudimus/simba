"""CI smoke test: exercise the bench code path on a synthetic 2-doc dataset.

No GGUF load, no live LLM, no LanceDB — ``build_retriever`` is monkeypatched
to a deterministic fake that ranks each query's gold document first.
"""

from __future__ import annotations

import json
import pathlib

import simba.__main__ as cli
import simba.config
import simba.eval.bench_config as bench_config
import simba.eval.bench_results as bench_results
import simba.eval.benchmarks.locomo as locomo
import simba.eval.benchmarks.run as bench_run
import simba.eval.dataset
import simba.eval.recall_adapter
import simba.eval.run as run
from simba.eval.bench_results import append_result, load_results
from simba.memory.config import MemoryConfig

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "smoke_bench.json"

_QUERY_GOLD = {"Alice": "m1", "Bob": "m2"}


def _embed_doc(text: str) -> list[float]:
    if "Alice" in text:
        return [1.0, 0.0, 0.0, 0.0]
    if "Bob" in text:
        return [0.0, 1.0, 0.0, 0.0]
    return [0.25, 0.25, 0.25, 0.25]


def _embed_query(text: str) -> list[float]:
    if "Alice" in text:
        return [1.0, 0.0, 0.0, 0.0]
    if "Bob" in text:
        return [0.0, 1.0, 0.0, 0.0]
    return [0.25, 0.25, 0.25, 0.25]


def _perfect_retriever_factory(*_a, **_k):
    def _retrieve(query: str) -> list[str]:
        for token, gold in _QUERY_GOLD.items():
            if token in query:
                return [gold]
        return []

    return _retrieve


def _empty_retriever_factory(*_a, **_k):
    return lambda query: []


# --------------------------------------------------------------------------
# fixture validity
# --------------------------------------------------------------------------


def test_smoke_bench_fixture_is_valid_json() -> None:
    data = json.loads(FIXTURE.read_text())
    assert len(data["corpus"]) == 2
    assert len(data["cases"]) == 2


def test_smoke_bench_fixture_loads_as_dataset() -> None:
    dataset = simba.eval.dataset.load_dataset(FIXTURE)
    assert dataset.name == "smoke"
    assert len(dataset.corpus) == 2


# --------------------------------------------------------------------------
# run_recall aggregation
# --------------------------------------------------------------------------


def test_smoke_run_recall_perfect_retriever(monkeypatch) -> None:
    monkeypatch.setattr(
        simba.eval.recall_adapter, "build_retriever", _perfect_retriever_factory
    )
    dataset = simba.eval.dataset.load_dataset(FIXTURE)
    report = bench_run.run_recall(
        [dataset], embed_doc=_embed_doc, embed_query=_embed_query, cfg=MemoryConfig()
    )
    assert report["overall"]["recall@1"] == 1.0
    assert report["n_cases"] == 2


def test_smoke_run_recall_zero_retriever(monkeypatch) -> None:
    monkeypatch.setattr(
        simba.eval.recall_adapter, "build_retriever", _empty_retriever_factory
    )
    dataset = simba.eval.dataset.load_dataset(FIXTURE)
    report = bench_run.run_recall(
        [dataset], embed_doc=_embed_doc, embed_query=_embed_query, cfg=MemoryConfig()
    )
    assert report["overall"]["recall@1"] == 0.0


# --------------------------------------------------------------------------
# results store roundtrip
# --------------------------------------------------------------------------


def test_smoke_append_and_load_roundtrip(tmp_path) -> None:
    p = tmp_path / ".simba" / "eval" / "results.jsonl"
    append_result(p, {"dataset": "locomo", "timestamp": 1.0})
    append_result(p, {"dataset": "longmemeval", "timestamp": 2.0})
    records = load_results(p)
    assert len(records) == 2
    assert {r["dataset"] for r in records} == {"locomo", "longmemeval"}


# --------------------------------------------------------------------------
# full CLI dispatch path
# --------------------------------------------------------------------------


def test_smoke_bench_cmd_end_to_end(monkeypatch, tmp_path) -> None:
    bcfg = bench_config.BenchConfig(
        results_path=str(tmp_path / ".simba" / "eval" / "results.jsonl"),
        embedding_cache_path=str(tmp_path / ".simba" / "eval" / "embedding_cache.db"),
        judge_cache_path=str(tmp_path / ".simba" / "eval" / "judge_cache.db"),
    )
    mcfg = simba.config.load("memory")

    def _fake_load(section, *a, **k):
        return bcfg if section == "bench" else mcfg

    monkeypatch.setattr(simba.config, "load", _fake_load)
    monkeypatch.setattr(
        locomo, "load_locomo", lambda path: [simba.eval.dataset.load_dataset(path)]
    )
    monkeypatch.setattr(
        run, "sync_embedders", lambda cfg, *, cache=None: (_embed_doc, _embed_query)
    )
    monkeypatch.setattr(
        simba.eval.recall_adapter, "build_retriever", _perfect_retriever_factory
    )
    monkeypatch.setattr(bench_results, "current_git_sha", lambda: "smoke00")

    rc = cli._eval_bench(["locomo", "--path", str(FIXTURE), "--json"])
    assert rc == 0

    results = pathlib.Path(bcfg.results_path)
    assert results.exists()
    records = load_results(results)
    assert len(records) == 1
    assert records[0]["dataset"] == "locomo"
    assert records[0]["recall"]["overall"]["recall@1"] == 1.0
