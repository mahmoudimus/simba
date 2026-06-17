"""Tests for the eval results store helpers."""

from __future__ import annotations

import dataclasses

import simba.eval.bench_results as br


@dataclasses.dataclass
class _Cfg:
    a: int = 1
    model: str = ""


def test_config_snapshot_memory_and_bench():
    snap = br.config_snapshot(_Cfg(a=2), _Cfg(a=3))
    assert snap["memory"] == {"a": 2, "model": ""}
    assert snap["bench"] == {"a": 3, "model": ""}


def test_config_snapshot_records_answerer_and_judge():
    # QA numbers depend on which answerer/judge produced them; the snapshot must
    # capture both so a record is attributable to a model (e.g. gpt-oss vs Qwen).
    snap = br.config_snapshot(
        _Cfg(),
        _Cfg(),
        llm_cfg=_Cfg(model="gpt-oss"),
        judge_cfg=_Cfg(model="qwen-judge"),
    )
    assert snap["llm"]["model"] == "gpt-oss"
    assert snap["judge"]["model"] == "qwen-judge"


def test_config_snapshot_omits_llm_judge_when_not_given():
    snap = br.config_snapshot(_Cfg(), _Cfg())
    assert "llm" not in snap and "judge" not in snap


def test_path_digest_hashes_files(tmp_path):
    p = tmp_path / "dataset.json"
    p.write_text('{"name":"tiny"}')
    meta = br.path_digest(p)
    assert meta["exists"] is True
    assert meta["kind"] == "file"
    assert meta["file_count"] == 1
    assert len(meta["sha256"]) == 64


def test_build_provenance_records_dataset_config_and_model_identity(tmp_path):
    p = tmp_path / "dataset.json"
    p.write_text("{}")
    config = br.config_snapshot(
        _Cfg(a=2),
        _Cfg(a=3),
        llm_cfg=_Cfg(model="answerer"),
        judge_cfg=_Cfg(model="judge"),
    )
    prov = br.build_provenance(
        dataset_name="locomo",
        dataset_path=p,
        split="test",
        config=config,
        git_sha="abc1234",
        answerer_cfg=_Cfg(model="answerer"),
        judge_cfg=_Cfg(model="judge"),
        excluded_count=2,
        abstained_count=1,
    )
    assert prov["schema_version"] == 1
    assert prov["dataset"]["name"] == "locomo"
    assert prov["dataset"]["split"] == "test"
    assert len(prov["dataset"]["sha256"]) == 64
    assert len(prov["config_hash"]) == 64
    assert prov["answerer"]["model"] == "answerer"
    assert prov["judge"]["model"] == "judge"
    assert prov["counts"] == {"excluded": 2, "abstained": 1, "contaminated": 0}
