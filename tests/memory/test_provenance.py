"""Tests for append-only general-memory provenance sidecar."""

from __future__ import annotations

import pathlib

import simba.db
import simba.memory.provenance as provenance


def test_append_event_defaults_observed_at(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        provenance.append_event(
            memory_id="mem_a",
            occurred_at="2026-06-01",
            source_file="src/a.py",
            source_span="10-12",
            source_url="",
            extraction_agent="simba",
            extraction_version="1",
            source_session="sess",
            trust_source="user_stated",
            capture_origin="cli",
            trust_score=1.23,
            now=1000.0,
        )
        rows = provenance.latest_for(["mem_a"])

    row = rows["mem_a"]
    assert row.occurred_at == "2026-06-01"
    assert row.observed_at == "1970-01-01T00:16:40Z"
    assert row.source_file == "src/a.py"
    assert row.source_span == "10-12"
    assert row.source_session == "sess"
    assert row.trust_source == "user_stated"
    assert row.capture_origin == "cli"
    assert row.trust_score == 1.23


def test_compute_trust_score_orders_sources() -> None:
    user_score = provenance.compute_trust_score(
        trust_source="user_stated",
        capture_origin="cli",
        confidence=0.9,
        memory_type="PREFERENCE",
    )
    extracted_score = provenance.compute_trust_score(
        trust_source="llm_extracted",
        capture_origin="hook",
        confidence=0.95,
        memory_type="PREFERENCE",
    )

    assert user_score > extracted_score
