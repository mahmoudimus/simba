"""Configuration for simba search / RAG context."""

from __future__ import annotations

import dataclasses

import simba.config


@simba.config.configurable("search")
@dataclasses.dataclass
class SearchConfig:
    max_context_tokens: int = 1500
    max_code_results: int = 3
    min_query_length: int = 15
    memory_token_budget: int = 500
    # Index a git project with qmd in the background the first time a prompt
    # finds no collection for it (BM25 only; see search/qmd_index.py).
    qmd_auto_index: bool = True
