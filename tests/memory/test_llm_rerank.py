"""Tests for the LLM reranker (cross-encoder replacement, fail-open)."""

from __future__ import annotations

import simba.memory.llm_rerank as rr


class FakeClient:
    def __init__(self, response, available=True):
        self._response = response
        self._available = available
        self.prompts: list[str] = []

    def available(self):
        return self._available

    def complete_json(self, prompt):
        self.prompts.append(prompt)
        return self._response


def _cands(*ids):
    return [{"id": i, "content": f"memory {i}"} for i in ids]


def test_reorders_by_llm_order() -> None:
    client = FakeClient(["c", "a", "b"])
    out = rr.rerank("q", _cands("a", "b", "c"), client=client)
    assert [c["id"] for c in out] == ["c", "a", "b"]


def test_query_and_candidates_in_prompt() -> None:
    client = FakeClient(["a"])
    rr.rerank("how to fix gh 401", _cands("a"), client=client)
    p = client.prompts[0]
    assert "how to fix gh 401" in p
    assert "memory a" in p


def test_omitted_ids_appended_in_original_order() -> None:
    # LLM only returns 'b'; a and c must still appear, original order preserved
    client = FakeClient(["b"])
    out = rr.rerank("q", _cands("a", "b", "c"), client=client)
    assert [c["id"] for c in out] == ["b", "a", "c"]


def test_no_candidate_loss() -> None:
    client = FakeClient(["z", "c", "a"])  # 'z' is bogus, not a candidate
    out = rr.rerank("q", _cands("a", "b", "c"), client=client)
    assert {c["id"] for c in out} == {"a", "b", "c"}


def test_unavailable_client_passthrough() -> None:
    client = FakeClient(["c", "b", "a"], available=False)
    out = rr.rerank("q", _cands("a", "b", "c"), client=client)
    assert [c["id"] for c in out] == ["a", "b", "c"]


def test_none_client_passthrough() -> None:
    out = rr.rerank("q", _cands("a", "b"), client=None)
    assert [c["id"] for c in out] == ["a", "b"]


def test_bad_response_passthrough() -> None:
    client = FakeClient(None)  # LLM failed / returned no JSON
    out = rr.rerank("q", _cands("a", "b"), client=client)
    assert [c["id"] for c in out] == ["a", "b"]


def test_only_head_reranked_tail_preserved() -> None:
    # with max_candidates=2, only a,b are sent; c stays at the tail untouched
    client = FakeClient(["b", "a"])
    out = rr.rerank("q", _cands("a", "b", "c"), client=client, max_candidates=2)
    assert [c["id"] for c in out] == ["b", "a", "c"]
    assert "memory c" not in client.prompts[0]
