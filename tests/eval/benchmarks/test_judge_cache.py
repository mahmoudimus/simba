"""Tests for the persistent judge-verdict cache + its use in score_case."""

from __future__ import annotations

import pathlib

import simba.eval.benchmarks.judge as judge
import simba.eval.benchmarks.judge_cache as jc
from simba.eval.dataset import EvalCase


def test_key_varies_by_all_fields() -> None:
    base = jc.JudgeCache.key("m", "q", "gold", "pred")
    assert base == jc.JudgeCache.key("m", "q", "gold", "pred")
    assert base != jc.JudgeCache.key("m2", "q", "gold", "pred")
    assert base != jc.JudgeCache.key("m", "q2", "gold", "pred")
    assert base != jc.JudgeCache.key("m", "q", "gold2", "pred")
    assert base != jc.JudgeCache.key("m", "q", "gold", "pred2")


def test_get_miss_then_hit_persists(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "verdicts.db"
    c1 = jc.JudgeCache(path)
    assert c1.get("m", "q", "g", "p") is None
    c1.put("m", "q", "g", "p", True)
    c1.close()
    c2 = jc.JudgeCache(path)
    assert c2.get("m", "q", "g", "p") is True


class _Llm:
    def __init__(self) -> None:
        self.judged: list[str] = []

    def complete(self, prompt: str) -> str:
        return "7 May 2023"

    def complete_json(self, prompt: str) -> object:
        self.judged.append(prompt)
        return {"correct": True}


def test_score_case_uses_cache_and_skips_second_judge(tmp_path: pathlib.Path) -> None:
    cache = jc.JudgeCache(tmp_path / "v.db")
    case = EvalCase(id="q1", query="When?", relevant_ids=["c1"], answer="7 May 2023")
    llm = _Llm()
    kw = {"cache": cache, "judge_model": "m"}
    r1 = judge.score_case(case, lambda q: ["c1"], {"c1": "x"}, llm, k=5, **kw)
    r2 = judge.score_case(case, lambda q: ["c1"], {"c1": "x"}, llm, k=5, **kw)
    assert r1 is True and r2 is True
    assert len(llm.judged) == 1  # second verdict served from cache
