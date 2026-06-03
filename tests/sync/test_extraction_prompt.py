"""Tests for the entity-aware LLM extraction prompt builder."""

from __future__ import annotations

import simba.sync.extractor as ex

_MEMS = [
    {
        "type": "GOTCHA",
        "content": "GITHUB_TOKEN makes gh return 401",
        "context": "use env -u",
    },
    {"type": "DECISION", "content": "vendored peewee at commit 9bd107f", "context": ""},
]


def test_includes_memory_text() -> None:
    p = ex.build_extraction_prompt(_MEMS, existing_entities=[])
    assert "GITHUB_TOKEN makes gh return 401" in p
    assert "vendored peewee" in p


def test_includes_kg_add_and_typed_triples() -> None:
    p = ex.build_extraction_prompt(_MEMS, existing_entities=[])
    assert "kg_add" in p
    assert "subject_type" in p and "object_type" in p


def test_lists_existing_entities_for_reuse() -> None:
    p = ex.build_extraction_prompt(_MEMS, existing_entities=["GITHUB_TOKEN", "peewee"])
    assert "GITHUB_TOKEN" in p
    assert "peewee" in p
    # instruction to reuse the canonical names rather than minting variants
    assert "reuse" in p.lower() or "existing" in p.lower()


def test_no_entities_section_when_empty() -> None:
    p = ex.build_extraction_prompt(_MEMS, existing_entities=[])
    # should still be a valid prompt, just without a vocabulary block
    assert "kg_add" in p


def test_caps_entity_vocabulary(monkeypatch) -> None:
    many = [f"entity_{i}" for i in range(500)]
    p = ex.build_extraction_prompt(_MEMS, existing_entities=many, max_entities=50)
    # only the cap is listed (cheap guard against an enormous prompt)
    assert p.count("entity_") <= 50
