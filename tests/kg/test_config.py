"""Tests for the kg config section."""

from __future__ import annotations

import simba.config
import simba.kg.config


def test_kg_section_registered() -> None:
    assert "kg" in simba.config.list_sections()


def test_kg_defaults() -> None:
    cfg = simba.kg.config.KgConfig()
    assert cfg.min_keyword_len == 2
    assert cfg.inject_max_facts == 3
    assert cfg.fts_tokenize == "trigram"
    assert cfg.default_subject_type == "concept"
    assert cfg.default_object_type == "concept"


def test_entity_resolution_defaults() -> None:
    cfg = simba.kg.config.KgConfig()
    assert cfg.entity_resolution_enabled is False  # opt-in
    assert 0.0 < cfg.entity_similarity_threshold <= 1.0
