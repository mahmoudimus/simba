"""Knowledge-graph (kg) subsystem: facts, BM25/FTS retrieval, and injection."""

from __future__ import annotations

from simba.kg.store import kg_add, kg_invalidate, kg_query

__all__ = ["kg_add", "kg_invalidate", "kg_query"]
