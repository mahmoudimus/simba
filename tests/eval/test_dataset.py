"""Tests for eval dataset loading + validation."""

from __future__ import annotations

import json
import pathlib

import pytest

import simba.eval.dataset as ds

_RAW = {
    "name": "tiny",
    "corpus": [
        {"id": "m1", "content": "use env -u GITHUB_TOKEN gh", "type": "GOTCHA"},
        {"id": "m2", "content": "peewee binds at class definition", "type": "GOTCHA"},
        {"id": "m3", "content": "ruff format is black-compatible"},
    ],
    "cases": [
        {"id": "c1", "query": "gh auth fails", "relevant_ids": ["m1"]},
        {
            "id": "c2",
            "query": "orm binding",
            "relevant_ids": ["m2"],
            "intent": "precise",
        },
    ],
}


def _write(tmp_path: pathlib.Path, raw: dict) -> pathlib.Path:
    p = tmp_path / "d.json"
    p.write_text(json.dumps(raw))
    return p


def test_load_dataset(tmp_path: pathlib.Path) -> None:
    d = ds.load_dataset(_write(tmp_path, _RAW))
    assert d.name == "tiny"
    assert len(d.corpus) == 3
    assert len(d.cases) == 2
    assert d.corpus[0].id == "m1"
    assert d.corpus[0].type == "GOTCHA"
    assert d.cases[0].relevant_ids == ["m1"]
    assert d.cases[1].intent == "precise"


def test_default_type_is_pattern(tmp_path: pathlib.Path) -> None:
    d = ds.load_dataset(_write(tmp_path, _RAW))
    assert d.corpus[2].type == "PATTERN"  # m3 omitted type


def test_dangling_relevant_id_raises(tmp_path: pathlib.Path) -> None:
    raw = json.loads(json.dumps(_RAW))
    raw["cases"][0]["relevant_ids"] = ["nope"]
    with pytest.raises(ValueError, match="unknown memory id"):
        ds.load_dataset(_write(tmp_path, raw))


def test_duplicate_corpus_id_raises(tmp_path: pathlib.Path) -> None:
    raw = json.loads(json.dumps(_RAW))
    raw["corpus"].append({"id": "m1", "content": "dup"})
    with pytest.raises(ValueError, match="duplicate"):
        ds.load_dataset(_write(tmp_path, raw))


def test_corpus_ids(tmp_path: pathlib.Path) -> None:
    d = ds.load_dataset(_write(tmp_path, _RAW))
    assert d.corpus_ids() == {"m1", "m2", "m3"}
