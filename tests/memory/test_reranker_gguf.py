"""Opt-in real-GGUF reranker integration test (spec 22).

Skipped by default (marked ``gguf``; the suite runs ``-m 'not gguf'``). Run with:

    uv run pytest tests/memory/test_reranker_gguf.py -m gguf

Downloads the configured GGUFs and asserts each local backend gives gold-doc a
higher relevance score than a distractor-doc (the Pillar-0 sane-score property),
exercising the same loader the daemon uses. These call ``.load()`` /``.score()``
directly, bypassing the conftest accessor block.
"""

from __future__ import annotations

import pytest

import simba.memory.config
import simba.memory.reranker as reranker

pytestmark = pytest.mark.gguf

_QUERY = "What pet does Caroline have?"
_GOLD = "Caroline: I adopted a golden retriever puppy named Biscuit."
_DIST = "Caroline: We grilled burgers by the lake on Saturday."


def test_cross_encoder_gguf_gold_beats_distractor() -> None:
    cfg = simba.memory.config.MemoryConfig(reranker_mode="cross-encoder")
    scorer = reranker._CrossEncoderReranker.load(cfg)
    assert scorer.score(_QUERY, _GOLD) > scorer.score(_QUERY, _DIST)


def test_local_llm_gguf_gold_beats_distractor() -> None:
    cfg = simba.memory.config.MemoryConfig(reranker_mode="local-llm")
    scorer = reranker._LocalLlmReranker.load(cfg)
    assert scorer.score(_QUERY, _GOLD) > scorer.score(_QUERY, _DIST)


def test_rerank_dispatch_cross_encoder_end_to_end() -> None:
    """Full rerank() routing through the real GGUF reorders gold above distractor."""
    cfg = simba.memory.config.MemoryConfig(reranker_mode="cross-encoder")
    # bypass the conftest accessor block by installing the real scorer
    scorer = reranker._CrossEncoderReranker.load(cfg)
    cands = [
        {"id": "dist", "content": _DIST, "context": ""},
        {"id": "gold", "content": _GOLD, "context": ""},
    ]

    class _Wrap:
        def score(self, q, d):
            return scorer.score(q, d)

    import unittest.mock

    with unittest.mock.patch.object(
        reranker, "_get_cross_encoder", lambda cfg: _Wrap()
    ):
        out = reranker.rerank(_QUERY, cands, cfg=cfg)
    assert out[0]["id"] == "gold"
