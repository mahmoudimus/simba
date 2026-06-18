"""Measure-first tests for anticipated-query recall."""

from __future__ import annotations

import dataclasses

import simba.eval.recall_adapter
import simba.memory.config
from simba.eval.dataset import Dataset, EvalCase, Memory


def _embed(text: str) -> list[float]:
    # Keep semantic/vector recall out of this fixture so the measured delta is
    # attributable to the anticipated-query lane.
    return [1.0, 0.0] if text.startswith("DOC:") else [0.0, 1.0]


def _dataset() -> Dataset:
    return Dataset(
        name="anticipated-smoke",
        corpus=[
            Memory(
                id="mem_auth",
                content="DOC: rotate the gh token when GitHub returns 401",
                anticipated_queries=["opaque bearer auth failure"],
            ),
            Memory(
                id="mem_restart",
                content="DOC: restart Simba with launchctl kickstart",
            ),
        ],
        cases=[
            EvalCase(
                id="anticipated",
                query="opaque bearer failure",
                relevant_ids=["mem_auth"],
            ),
            EvalCase(
                id="plain",
                query="restart Simba",
                relevant_ids=["mem_restart"],
            ),
        ],
    )


def test_anticipated_query_recall_gate_lifts_without_plain_regression(tmp_path) -> None:
    cfg = simba.memory.config.MemoryConfig(
        min_similarity=0.35,
        max_results=5,
        llm_rerank_enabled=False,
        scoring_enabled=False,
    )
    baseline = simba.eval.recall_adapter.build_retriever(
        _dataset(),
        cfg,
        embed_doc=_embed,
        embed_query=_embed,
        data_dir=tmp_path / "baseline",
    )
    enabled = simba.eval.recall_adapter.build_retriever(
        _dataset(),
        dataclasses.replace(cfg, anticipated_query_recall_enabled=True),
        embed_doc=_embed,
        embed_query=_embed,
        data_dir=tmp_path / "enabled",
    )

    assert "mem_auth" not in baseline("opaque bearer failure")
    assert enabled("opaque bearer failure")[0] == "mem_auth"

    baseline_plain = baseline("restart Simba")
    enabled_plain = enabled("restart Simba")
    assert baseline_plain[0] == "mem_restart"
    assert enabled_plain[0] == "mem_restart"
