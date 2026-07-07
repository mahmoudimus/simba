"""Tests for ``simba eval bench`` (A4) and its results-store helpers."""

from __future__ import annotations

import json
import pathlib
import types as _types

import simba.__main__ as cli
import simba.config
import simba.eval.bench_config as bench_config
import simba.eval.bench_results as bench_results
import simba.eval.benchmarks.locomo as locomo
import simba.eval.benchmarks.run as bench_run
import simba.eval.benchmarks.subtlememory as subtlememory
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


class _FakeClient:
    """Minimal stand-in for an llm/judge client (has the ``_cfg.model`` the
    bench reads for the results record)."""

    _cfg = _types.SimpleNamespace(model="fake-model", provider="fake")

    def available(self) -> bool:
        return True


_SENTINEL_CLIENT = _FakeClient()


def _fake_run_recall(datasets, *, embed_doc, embed_query, cfg, llm_client=None):
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
    import simba.eval.config as eval_config

    mcfg = simba.config.load("memory")
    ecfg = eval_config.EvalConfig()

    def _fake_load(section, *a, **k):
        if section == "bench":
            return bcfg
        if section == "memory":
            return mcfg
        if section == "eval":
            return ecfg
        raise KeyError(section)

    import simba.llm.client as llm_client

    monkeypatch.setattr(simba.config, "load", _fake_load)
    monkeypatch.setattr(
        locomo, "load_locomo", lambda path: [_fake_dataset("a"), _fake_dataset("b")]
    )
    monkeypatch.setattr(run, "sync_embedders", _fake_sync_embedders)
    monkeypatch.setattr(bench_run, "run_recall", _fake_run_recall)
    monkeypatch.setattr(bench_results, "current_git_sha", lambda: "abc1234")
    # The bench builds an llm client and threads it into retrieval (for the
    # reranker / LLM-HyDE levers); stub it so no real client is constructed.
    monkeypatch.setattr(llm_client, "get_client", lambda *a, **k: _SENTINEL_CLIENT)
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

    def _spy_run_recall(datasets, *, embed_doc, embed_query, cfg, llm_client=None):
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

    def _spy_run_recall(datasets, *, embed_doc, embed_query, cfg, llm_client=None):
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


def test_bench_passes_llm_client_to_run_recall(monkeypatch, tmp_path) -> None:
    # The reranker + LLM-HyDE levers can only be measured if the bench threads a
    # real llm client into the retrieval path.
    _install_common_fakes(monkeypatch, tmp_path)
    seen: dict[str, object] = {}

    def _spy_run_recall(datasets, *, embed_doc, embed_query, cfg, llm_client=None):
        seen["llm_client"] = llm_client
        return _fake_run_recall(
            datasets, embed_doc=embed_doc, embed_query=embed_query, cfg=cfg
        )

    monkeypatch.setattr(bench_run, "run_recall", _spy_run_recall)
    rc = cli._eval_bench(["locomo"])
    assert rc == 0
    assert seen["llm_client"] is _SENTINEL_CLIENT


def test_bench_compare_readback_is_subtlememory_only(monkeypatch, tmp_path) -> None:
    _install_common_fakes(monkeypatch, tmp_path)
    rc = cli._eval_bench(["locomo", "--compare-readback"])
    assert rc == 1


def test_bench_subtlememory_compare_readback_records_report(
    monkeypatch, tmp_path, capsys
) -> None:
    bcfg = _install_common_fakes(
        monkeypatch, tmp_path, subtlememory_path="/fake/subtlememory"
    )
    dset = Dataset(
        name="subtle",
        corpus=[
            Memory(id="m1", content="first", session_source="s1"),
            Memory(id="m2", content="second", session_source="s1"),
        ],
        cases=[
            EvalCase(
                id="q1",
                query="what happened?",
                relevant_ids=["m1", "m2"],
                intent="contradictory",
            )
        ],
    )
    monkeypatch.setattr(
        subtlememory,
        "load_subtlememory",
        lambda path, *, persona_limit=0, persona_start=0: [dset],
    )

    rc = cli._eval_bench(["subtlememory", "--compare-readback", "--json"])

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["readback"]["mode"] == "session_readback_ceiling"
    assert out["readback"]["ceiling"]["overall"]["recall@3"] == 1.0
    results = pathlib.Path(bcfg.results_path).read_text().splitlines()
    record = json.loads(results[0])
    assert record["dataset"] == "subtlememory"
    assert record["readback"]["ceiling"]["diagnostics"]["max_gold_ids"] == 2


def test_bench_subtlememory_driver_report_records_summary(
    monkeypatch, tmp_path, capsys
) -> None:
    bcfg = _install_common_fakes(
        monkeypatch, tmp_path, subtlememory_path="/fake/subtlememory"
    )
    dset = Dataset(
        name="subtle",
        corpus=[Memory(id="m1", content="first", session_source="s1")],
        cases=[
            EvalCase(
                id="q1",
                query="what happened?",
                relevant_ids=["m1"],
                intent="contradictory",
            )
        ],
    )
    driver_path = tmp_path / "driver.json"
    monkeypatch.setattr(
        subtlememory,
        "load_subtlememory",
        lambda path, *, persona_limit=0, persona_start=0: [dset],
    )
    monkeypatch.setattr(
        subtlememory,
        "run_recall_with_ranked",
        lambda datasets, **kwargs: (
            {
                "n_conversations": 1,
                "n_cases": 1,
                "overall": dict(_OVERALL),
                "by_category": {},
            },
            {"q1": ["m1"]},
        ),
    )

    rc = cli._eval_bench(
        ["subtlememory", "--driver-report", str(driver_path), "--json"]
    )

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["driver"]["summary"]["recommendation"] == "answer_time_or_cutoff"
    assert driver_path.exists()
    record = json.loads(pathlib.Path(bcfg.results_path).read_text().splitlines()[0])
    assert record["driver"]["path"] == str(driver_path)


def test_bench_subtlememory_driver_loop_picks_winning_variant(
    monkeypatch, tmp_path, capsys
) -> None:
    bcfg = _install_common_fakes(
        monkeypatch, tmp_path, subtlememory_path="/fake/subtlememory"
    )
    dset = Dataset(
        name="subtle",
        corpus=[Memory(id="m1", content="first", session_source="s1")],
        cases=[
            EvalCase(
                id="q1",
                query="what happened?",
                relevant_ids=["m1"],
                intent="contradictory",
            )
        ],
    )
    loop_path = tmp_path / "loop.json"
    monkeypatch.setattr(
        subtlememory,
        "load_subtlememory",
        lambda path, *, persona_limit=0, persona_start=0: [dset],
    )

    def _report(overall_r10: float, contra_r10: float) -> dict:
        overall = dict(_OVERALL)
        overall["recall@10"] = overall_r10
        return {
            "n_conversations": 1,
            "n_cases": 1,
            "overall": overall,
            "by_category": {
                "contradictory": {
                    "n": 1,
                    "recall@5": contra_r10 / 2,
                    "recall@10": contra_r10,
                    "mrr": contra_r10,
                }
            },
        }

    def _fake_ranked(datasets, **kwargs):
        cfg = kwargs["cfg"]
        if getattr(cfg, "session_expansion_enabled", False):
            top = getattr(cfg, "session_expansion_top_sessions", 0)
            weight = getattr(cfg, "session_expansion_weight", 0.0)
            if top == 2 and weight == 2.0:
                return _report(0.8, 0.9), {"q1": ["m1"]}
            return _report(0.7, 0.7), {"q1": ["m1"]}
        return _report(0.5, 0.5), {"q1": ["m1"]}

    monkeypatch.setattr(subtlememory, "run_recall_with_ranked", _fake_ranked)

    rc = cli._eval_bench(["subtlememory", "--driver-loop", str(loop_path), "--json"])

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["driver_loop"]["summary"]["winner"] == "session_top2_w2"
    assert out["driver_loop"]["summary"]["winner_positive"] is True
    assert out["driver_loop"]["summary"]["promotion_gate_passed"] is True
    artifact = json.loads(loop_path.read_text())
    assert (
        artifact["summary"]["winner_config_overrides"]["session_expansion_top_sessions"]
        == 2
    )
    assert artifact["summary"]["promotion_gate"]["passed"] is True
    record = json.loads(pathlib.Path(bcfg.results_path).read_text().splitlines()[0])
    assert record["driver_loop"]["path"] == str(loop_path)
    assert record["driver_loop"]["summary"]["promotion_gate_passed"] is True


def test_bench_qa_passes_eval_cfg_to_run_qa(monkeypatch, tmp_path) -> None:
    # IRCoT routing in run_qa only fires when the bench passes the eval config
    # (eval.ircot_enabled). Verify --qa threads it through.
    import simba.eval.benchmarks.judge as judge
    import simba.llm.judge_config as jcfg

    _install_common_fakes(monkeypatch, tmp_path)
    seen: dict[str, object] = {}

    def _spy_run_qa(datasets, **kwargs):
        seen["eval_cfg"] = kwargs.get("eval_cfg")
        seen["judge"] = kwargs.get("judge")
        return {
            "n_graded": 1,
            "n_skipped": 0,
            "overall": {"accuracy": 0.5},
            "by_category": {},
        }

    monkeypatch.setattr(judge, "run_qa", _spy_run_qa)
    monkeypatch.setattr(judge, "sample_cases", lambda ds, **k: ds)
    monkeypatch.setattr(jcfg, "get_judge_client", lambda *a, **k: _SENTINEL_CLIENT)
    rc = cli._eval_bench(["locomo", "--qa"])
    assert rc == 0
    # eval_cfg must be the loaded "eval" section (has ircot_enabled), not None.
    assert seen["eval_cfg"] is not None
    assert hasattr(seen["eval_cfg"], "ircot_enabled")


def test_bench_config_judge_style_defaults_official() -> None:
    # Canonical axis: the official LongMemEval per-type judge is the default
    # (measured +3.6pp vs the generic JSON judge on simba outputs, p=5e-4).
    assert bench_config.BenchConfig().judge_style == "official"


def test_bench_qa_threads_judge_style_to_run_qa(monkeypatch, tmp_path) -> None:
    import simba.eval.benchmarks.judge as judge
    import simba.llm.judge_config as jcfg

    _install_common_fakes(monkeypatch, tmp_path, judge_style="generic")
    seen: dict[str, object] = {}

    def _spy_run_qa(datasets, **kwargs):
        seen["judge_style"] = kwargs.get("judge_style")
        return {
            "n_graded": 1,
            "n_skipped": 0,
            "overall": {"accuracy": 0.5},
            "by_category": {},
        }

    monkeypatch.setattr(judge, "run_qa", _spy_run_qa)
    monkeypatch.setattr(judge, "sample_cases", lambda ds, **k: ds)
    monkeypatch.setattr(jcfg, "get_judge_client", lambda *a, **k: _SENTINEL_CLIENT)
    rc = cli._eval_bench(["locomo", "--qa"])
    assert rc == 0
    # run_qa must receive bench.judge_style (not the function-level default).
    assert seen["judge_style"] == "generic"


def test_bench_config_reader_levers_default_to_current_behavior() -> None:
    # The 0.823 stack is opt-in: defaults must preserve current bench behavior.
    cfg = bench_config.BenchConfig()
    assert cfg.reader_style == "minimal"
    assert cfg.preference_synthesis is False
    assert cfg.temporal_codegen is False


def test_bench_qa_threads_reader_levers_to_run_qa(monkeypatch, tmp_path) -> None:
    import simba.eval.benchmarks.judge as judge
    import simba.llm.judge_config as jcfg

    _install_common_fakes(
        monkeypatch,
        tmp_path,
        reader_style="rules",
        preference_synthesis=True,
        temporal_codegen=True,
    )
    seen: dict[str, object] = {}

    def _spy_run_qa(datasets, **kwargs):
        seen.update(kwargs)
        return {
            "n_graded": 1,
            "n_skipped": 0,
            "overall": {"accuracy": 0.5},
            "by_category": {},
        }

    monkeypatch.setattr(judge, "run_qa", _spy_run_qa)
    monkeypatch.setattr(judge, "sample_cases", lambda ds, **k: ds)
    monkeypatch.setattr(jcfg, "get_judge_client", lambda *a, **k: _SENTINEL_CLIENT)
    rc = cli._eval_bench(["locomo", "--qa"])
    assert rc == 0
    assert seen["reader_style"] == "rules"
    assert seen["preference_synthesis"] is True
    assert seen["temporal_codegen"] is True


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
