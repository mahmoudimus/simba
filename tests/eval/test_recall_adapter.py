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
