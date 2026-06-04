"""Tests for query decomposition (C4): split a multi-hop query, fuse sub-results."""

from __future__ import annotations

import simba.memory.decompose as dc


class FakeLlm:
    def __init__(self, result: object, *, available: bool = True, raises: bool = False):
        self._result = result
        self._available = available
        self._raises = raises
        self.prompts: list[str] = []

    def available(self) -> bool:
        return self._available

    def complete_json(self, prompt: str) -> object:
        self.prompts.append(prompt)
        if self._raises:
            raise RuntimeError("boom")
        return self._result


def test_prompt_includes_query_and_asks_json() -> None:
    p = dc.build_decompose_prompt("Are A and B the same nationality?")
    assert "Are A and B the same nationality?" in p
    assert "json" in p.lower()


def test_parse_includes_original_first_dedups_caps() -> None:
    out = dc.parse_subqueries(
        ["What is A's nationality?", "what is a's nationality?", "B's nationality?"],
        original="Are A and B the same?",
        max_sub=4,
    )
    assert out[0] == "Are A and B the same?"  # original always first
    assert "What is A's nationality?" in out
    # case-insensitive dedup of the repeated sub-question
    assert sum(s.lower() == "what is a's nationality?" for s in out) == 1


def test_parse_caps_total() -> None:
    subs = [f"q{i}?" for i in range(10)]
    out = dc.parse_subqueries(subs, original="orig?", max_sub=3)
    assert len(out) == 4  # original + at most max_sub


def test_parse_ignores_non_strings_and_blanks() -> None:
    out = dc.parse_subqueries(["good?", "", 5, None, "  "], original="orig?", max_sub=4)
    assert out == ["orig?", "good?"]


def test_decompose_failopen_no_llm() -> None:
    assert dc.decompose("q?", None) == ["q?"]


def test_decompose_failopen_unavailable_or_raises() -> None:
    assert dc.decompose("q?", FakeLlm([], available=False)) == ["q?"]
    assert dc.decompose("q?", FakeLlm(None, raises=True)) == ["q?"]


def test_decompose_happy_path() -> None:
    llm = FakeLlm(["sub one?", "sub two?"])
    out = dc.decompose("orig?", llm, max_sub=4)
    assert out == ["orig?", "sub one?", "sub two?"]


def test_fuse_rankings_rrf_prefers_multi_list_ids() -> None:
    # "x" appears in both rankings, "y"/"z" in one each -> x ranks first.
    fused = dc.fuse_rankings([["y", "x"], ["z", "x"]], k=60)
    assert fused[0] == "x"
    assert set(fused) == {"x", "y", "z"}


def test_fuse_rankings_single_list_is_identity_order() -> None:
    assert dc.fuse_rankings([["a", "b", "c"]]) == ["a", "b", "c"]
