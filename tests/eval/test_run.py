"""Tests for the eval run orchestrator (model-free glue)."""

from __future__ import annotations

import pathlib
import re

import simba.eval.report as report
import simba.eval.run as run

_VOCAB = ["gh", "github_token", "peewee", "orm", "fts5", "rrf", "episode", "nomic"]


def _embed(text: str) -> list[float]:
    toks = set(re.findall(r"[a-z_0-9]+", text.lower()))
    v = [1.0 if w in toks else 0.0 for w in _VOCAB]
    if not any(v):
        v[0] = 1e-6
    return v


def test_run_dataset_on_seed(tmp_path: pathlib.Path) -> None:
    rep = run.run_dataset(
        report.default_dataset_path(),
        ks=(1, 3, 5),
        data_dir=tmp_path,
        embed_doc=_embed,
        embed_query=_embed,
    )
    assert rep.dataset_name == "simba-seed"
    assert rep.n_cases >= 11
    # With a crude embedder this won't be perfect, but the report is well-formed.
    assert 0.0 <= rep.aggregate["recall@5"] <= 1.0
    assert 0.0 <= rep.aggregate["recall@1"] <= 1.0
