"""Tests for the in-process recall adapter (real LanceDB+FTS, fake embedder)."""

from __future__ import annotations

import pathlib
import re

import pytest

import simba.eval.dataset as ds
import simba.eval.recall_adapter as ra
import simba.eval.runner as runner

# A deterministic bag-of-tokens "embedder": no GGUF model needed, so this runs
# in CI. Relevant docs share vocabulary with their query ⇒ high cosine.
_VOCAB = [
    "github_token",
    "gh",
    "auth",
    "peewee",
    "orm",
    "binding",
    "ruff",
    "format",
    "lancedb",
    "vector",
]


def _embed(text: str) -> list[float]:
    toks = set(re.findall(r"[a-z_]+", text.lower()))
    v = [1.0 if w in toks else 0.0 for w in _VOCAB]
    if not any(v):
        v[0] = 1e-6  # avoid a zero vector (cosine undefined)
    return v


_DATASET = ds.Dataset(
    name="adapter-tiny",
    corpus=[
        ds.Memory(
            id="m1", content="use env -u GITHUB_TOKEN gh auth switch", type="GOTCHA"
        ),
        ds.Memory(
            id="m2", content="peewee orm binding at class definition", type="GOTCHA"
        ),
        ds.Memory(id="m3", content="ruff format is black compatible", type="PATTERN"),
        ds.Memory(
            id="m4", content="lancedb vector search cosine distance", type="PATTERN"
        ),
    ],
    cases=[
        ds.EvalCase(id="c1", query="gh auth github_token fails", relevant_ids=["m1"]),
        ds.EvalCase(id="c2", query="peewee orm binding", relevant_ids=["m2"]),
        ds.EvalCase(id="c3", query="lancedb vector cosine", relevant_ids=["m4"]),
    ],
)


@pytest.fixture
def retriever(tmp_path: pathlib.Path):
    return ra.build_retriever(
        _DATASET,
        data_dir=tmp_path,
        embed_doc=_embed,
        embed_query=_embed,
    )


def test_retriever_ranks_relevant_first(retriever) -> None:
    assert retriever("gh auth github_token fails")[0] == "m1"
    assert retriever("peewee orm binding")[0] == "m2"
    assert retriever("lancedb vector cosine")[0] == "m4"


def test_full_eval_run(retriever) -> None:
    rep = runner.run_eval(_DATASET, retriever, ks=(1, 3))
    assert rep.aggregate["recall@1"] == pytest.approx(1.0)
    assert rep.aggregate["mrr"] == pytest.approx(1.0)


def test_empty_corpus_returns_empty(tmp_path: pathlib.Path) -> None:
    empty = ds.Dataset(name="e", corpus=[], cases=[])
    r = ra.build_retriever(
        empty, data_dir=tmp_path, embed_doc=_embed, embed_query=_embed
    )
    assert r("anything") == []


class _HydeFakeLlm:
    def __init__(self, text: str) -> None:
        self._text = text

    def available(self) -> bool:
        return True

    def complete(self, prompt: str) -> str:
        return self._text


def test_retriever_with_llm_hyde_mode_uses_plan_hyde_text(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """The eval retriever embeds the LLM hypothetical answer (not focus terms)."""
    import simba.memory.config
    import simba.memory.recall_plan as recall_plan
    import simba.memory.vector_db as vdb

    cfg = simba.memory.config.MemoryConfig(
        hyde_mode="llm",
        expansion_enabled=True,
        # Keep the eval path purely on the 2nd-arm wiring: no rerank.
        llm_rerank_enabled=False,
    )
    query = "gh auth github_token fails"
    # HyDE text uses a vocab token absent from the query so its embedding is
    # distinct from both the primary arm and the keyword fallback.
    hyde_text = "the peewee orm answer"

    # The string actually fed to the 2nd arm is plan.hyde_text. With the fake LLM
    # returning hyde_text and no cache (eval path), the plan resolves to it.
    plan = recall_plan.plan_recall(
        query, cfg, llm_client=_HydeFakeLlm(hyde_text), hyde_cache=None
    )
    assert plan.hyde_text == hyde_text
    assert plan.hyde_text != plan.expansion_terms  # not the keyword fallback
    assert _embed(hyde_text) != _embed(plan.expansion_terms)

    recorded: list[list[float]] = []
    real_search = vdb.search_memories

    async def _recording_search(table, emb, min_sim, max_res, filters):
        recorded.append(list(emb))
        return await real_search(table, emb, min_sim, max_res, filters)

    monkeypatch.setattr(vdb, "search_memories", _recording_search)

    retriever = ra.build_retriever(
        _DATASET,
        cfg,
        data_dir=tmp_path,
        embed_doc=_embed,
        embed_query=_embed,
        llm_client=_HydeFakeLlm(hyde_text),
    )
    retriever(query)

    # The 2nd arm's embedding must equal embed_query(hyde_text), not the focus
    # terms. First recorded vector = primary arm (the query), second = the HyDE
    # arm. Because _embed(hyde_text) != _embed(expansion_terms), this proves the
    # keyword fallback was NOT used for the 2nd arm.
    assert recorded == [_embed(query), _embed(hyde_text)]
