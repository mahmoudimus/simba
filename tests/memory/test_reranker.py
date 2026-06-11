"""Tests for the config-selectable reranker dispatch (spec 22).

Routing + reorder logic only — no model downloads. The cross-encoder / local-llm
GGUF backends are exercised with an INJECTED fake score fn; the real GGUF load is
covered by the Pillar-0 spike + an opt-in integration test (not CI).
"""

from __future__ import annotations

import simba.memory.config
import simba.memory.llm_rerank as llm_rerank
import simba.memory.reranker as reranker


def _cands(*ids):
    return [{"id": i, "content": f"memory {i}"} for i in ids]


class FakeLlmClient:
    """Mirrors simba.llm client surface used by llm_rerank.rerank."""

    def __init__(self, order, available=True):
        self._order = order
        self._available = available
        self.prompts: list[str] = []

    def available(self):
        return self._available

    def complete_json(self, prompt):
        self.prompts.append(prompt)
        return self._order


# ── mode routing ────────────────────────────────────────────────────────────


def test_mode_none_returns_unchanged() -> None:
    cfg = simba.memory.config.MemoryConfig(reranker_mode="none")
    cands = _cands("a", "b", "c")
    out = reranker.rerank("q", cands, cfg=cfg, llm=FakeLlmClient(["c", "b", "a"]))
    assert [c["id"] for c in out] == ["a", "b", "c"]


def test_mode_llm_delegates_to_llm_rerank() -> None:
    cfg = simba.memory.config.MemoryConfig(reranker_mode="llm")
    client = FakeLlmClient(["c", "a", "b"])
    out = reranker.rerank("q", _cands("a", "b", "c"), cfg=cfg, llm=client)
    assert [c["id"] for c in out] == ["c", "a", "b"]
    assert client.prompts  # the llm path was actually exercised


def test_mode_llm_no_client_fail_open() -> None:
    cfg = simba.memory.config.MemoryConfig(reranker_mode="llm")
    out = reranker.rerank("q", _cands("a", "b"), cfg=cfg, llm=None)
    assert [c["id"] for c in out] == ["a", "b"]


def test_mode_llm_byte_identical_to_llm_rerank() -> None:
    """Regression fence: the "llm" backend == calling llm_rerank.rerank directly.

    Same query, candidates, client, and max_candidates must yield the same order
    (and the same prompt) — the "llm" path is behavior-preserving.
    """
    cfg = simba.memory.config.MemoryConfig(reranker_mode="llm")
    cands = _cands("a", "b", "c", "d")

    via_dispatch = reranker.rerank(
        "fix gh 401",
        list(cands),
        cfg=cfg,
        llm=FakeLlmClient(["d", "b", "c", "a"]),
        max_candidates=3,
    )
    direct = llm_rerank.rerank(
        "fix gh 401",
        list(cands),
        client=FakeLlmClient(["d", "b", "c", "a"]),
        max_candidates=3,
    )
    assert [c["id"] for c in via_dispatch] == [c["id"] for c in direct]
    # and the prompts are identical (same head sent to the model)
    c1, c2 = FakeLlmClient(["a"]), FakeLlmClient(["a"])
    reranker.rerank("Q", list(cands), cfg=cfg, llm=c1, max_candidates=2)
    llm_rerank.rerank("Q", list(cands), client=c2, max_candidates=2)
    assert c1.prompts == c2.prompts


def test_unknown_mode_fail_open_unchanged() -> None:
    cfg = simba.memory.config.MemoryConfig(reranker_mode="banana")
    out = reranker.rerank("q", _cands("a", "b", "c"), cfg=cfg)
    assert [c["id"] for c in out] == ["a", "b", "c"]


def test_empty_candidates_unchanged() -> None:
    cfg = simba.memory.config.MemoryConfig(reranker_mode="cross-encoder")
    assert reranker.rerank("q", [], cfg=cfg) == []


# ── cross-encoder backend with an INJECTED score fn ─────────────────────────


def _inject(monkeypatch, scores: dict[str, float], *, mode="cross-encoder"):
    """Patch the GGUF scorer singleton with a fake (query, doc) -> score fn."""

    class _FakeScorer:
        def __init__(self):
            self.pairs: list[tuple[str, str]] = []

        def score(self, query: str, doc: str) -> float:
            self.pairs.append((query, doc))
            return scores.get(doc, 0.0)

    fake = _FakeScorer()
    target = "_get_cross_encoder" if mode == "cross-encoder" else "_get_local_llm"
    monkeypatch.setattr(reranker, target, lambda cfg: fake)
    return fake


def test_cross_encoder_reorders_by_score(monkeypatch) -> None:
    cfg = simba.memory.config.MemoryConfig(reranker_mode="cross-encoder")
    cands = _cands("a", "b", "c")
    _inject(monkeypatch, {"memory a": 0.1, "memory b": 0.9, "memory c": 0.5})
    out = reranker.rerank("q", cands, cfg=cfg)
    assert [c["id"] for c in out] == ["b", "c", "a"]


def test_cross_encoder_no_drop_no_dup(monkeypatch) -> None:
    cfg = simba.memory.config.MemoryConfig(reranker_mode="cross-encoder")
    cands = _cands("a", "b", "c", "d")
    _inject(monkeypatch, {"memory a": 0.2, "memory b": 0.2})  # ties / missing
    out = reranker.rerank("q", cands, cfg=cfg)
    assert {c["id"] for c in out} == {"a", "b", "c", "d"}
    assert len(out) == 4


def test_cross_encoder_only_head_scored_tail_preserved(monkeypatch) -> None:
    cfg = simba.memory.config.MemoryConfig(reranker_mode="cross-encoder")
    cands = _cands("a", "b", "c")
    fake = _inject(monkeypatch, {"memory a": 0.1, "memory b": 0.9})
    out = reranker.rerank("q", cands, cfg=cfg, max_candidates=2)
    # only a,b are scored+reordered; c stays at the tail untouched
    assert [c["id"] for c in out] == ["b", "a", "c"]
    assert ("q", "memory c") not in fake.pairs


def test_cross_encoder_load_error_fail_open(monkeypatch) -> None:
    cfg = simba.memory.config.MemoryConfig(reranker_mode="cross-encoder")

    def _boom(cfg):
        raise RuntimeError("no model")

    monkeypatch.setattr(reranker, "_get_cross_encoder", _boom)
    out = reranker.rerank("q", _cands("a", "b", "c"), cfg=cfg)
    assert [c["id"] for c in out] == ["a", "b", "c"]  # unchanged on error


def test_cross_encoder_score_error_fail_open(monkeypatch) -> None:
    cfg = simba.memory.config.MemoryConfig(reranker_mode="cross-encoder")

    class _Boom:
        def score(self, query, doc):
            raise RuntimeError("scorer blew up")

    monkeypatch.setattr(reranker, "_get_cross_encoder", lambda cfg: _Boom())
    out = reranker.rerank("q", _cands("a", "b"), cfg=cfg)
    assert [c["id"] for c in out] == ["a", "b"]


# ── local-llm backend (same dispatch + reorder contract) ────────────────────


def test_local_llm_reorders_by_score(monkeypatch) -> None:
    cfg = simba.memory.config.MemoryConfig(reranker_mode="local-llm")
    cands = _cands("a", "b", "c")
    _inject(
        monkeypatch,
        {"memory a": -1.0, "memory b": 5.0, "memory c": 2.0},
        mode="local-llm",
    )
    out = reranker.rerank("q", cands, cfg=cfg)
    assert [c["id"] for c in out] == ["b", "c", "a"]


def test_local_llm_load_error_fail_open(monkeypatch) -> None:
    cfg = simba.memory.config.MemoryConfig(reranker_mode="local-llm")

    def _boom(cfg):
        raise RuntimeError("no model")

    monkeypatch.setattr(reranker, "_get_local_llm", _boom)
    out = reranker.rerank("q", _cands("a", "b"), cfg=cfg)
    assert [c["id"] for c in out] == ["a", "b"]


# ── doc-text extraction (content + context, like llm_rerank) ────────────────


def test_scorer_sees_content_and_context(monkeypatch) -> None:
    cfg = simba.memory.config.MemoryConfig(reranker_mode="cross-encoder")
    cands = [{"id": "a", "content": "the cat sat", "context": "on a mat"}]

    seen: list[str] = []

    class _Capture:
        def score(self, query, doc):
            seen.append(doc)
            return 1.0

    monkeypatch.setattr(reranker, "_get_cross_encoder", lambda cfg: _Capture())
    reranker.rerank("q", cands, cfg=cfg)
    assert "the cat sat" in seen[0]
    assert "on a mat" in seen[0]


# ── intent-gated reranking (spec 22, LME-gate correction) ─────────────────────
# Measured: cross-encoder HURTS multi-evidence temporal (LME complete@5 0.65->0.20)
# and easy single-hop; helps latest/compositional. should_rerank gates the harmful
# query shapes so reranking becomes a router decision, not a global config.


def test_should_rerank_gating_off_is_always_true() -> None:
    cfg = simba.memory.config.MemoryConfig(rerank_intent_gating=False)
    assert reranker.should_rerank("how many days between X and Y", cfg) is True
    assert reranker.should_rerank("anything at all", cfg) is True


def test_should_rerank_skips_multi_endpoint_temporal() -> None:
    cfg = simba.memory.config.MemoryConfig(rerank_intent_gating=True)
    assert reranker.should_rerank(
        "How many days between the wedding and the move?", cfg) is False
    assert reranker.should_rerank(
        "Which happened first, the trip or the promotion?", cfg) is False
    assert reranker.should_rerank(
        "How long after starting the job did I buy the car?", cfg) is False


def test_should_rerank_keeps_latest_and_compositional() -> None:
    cfg = simba.memory.config.MemoryConfig(rerank_intent_gating=True)
    assert reranker.should_rerank("How often do I do yoga now?", cfg) is True
    assert reranker.should_rerank(
        "What did the engineer say about the database schema migration?", cfg) is True
