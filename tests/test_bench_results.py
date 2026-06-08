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
