"""Tests for deterministic dev/test splitting of eval cases."""

from __future__ import annotations

import simba.eval.dataset as ds
import simba.eval.splits as sp


def _case(cid, split=""):
    return ds.EvalCase(id=cid, query="q", relevant_ids=["m"], split=split)


def test_effective_split_is_deterministic() -> None:
    a = sp.effective_split(_case("c1"))
    b = sp.effective_split(_case("c1"))
    assert a == b
    assert a in ("dev", "test")


def test_explicit_split_overrides_hash() -> None:
    assert sp.effective_split(_case("c1", split="test")) == "test"
    assert sp.effective_split(_case("c1", split="dev")) == "dev"


def test_split_is_roughly_balanced() -> None:
    cases = [_case(f"c{i}") for i in range(400)]
    n_test = sum(1 for c in cases if sp.effective_split(c) == "test")
    assert 140 < n_test < 260  # ~50/50, generous bounds


def test_select_filters_by_split() -> None:
    cases = [_case("a", "dev"), _case("b", "test"), _case("c", "test")]
    assert [c.id for c in sp.select(cases, "test")] == ["b", "c"]
    assert [c.id for c in sp.select(cases, "dev")] == ["a"]


def test_select_none_or_empty_returns_all() -> None:
    cases = [_case("a", "dev"), _case("b", "test")]
    assert len(sp.select(cases, None)) == 2
    assert len(sp.select(cases, "")) == 2
