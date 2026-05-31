"""Configuration for the RLM (Recursive Language Model) memory layer."""

from __future__ import annotations

import dataclasses

import simba.config


@simba.config.configurable("rlm")
@dataclasses.dataclass
class RlmConfig:
    # context.py — DocumentSearcher
    max_search_matches: int = 20
    search_context_chars: int = 200
    regex_timeout_seconds: float = 2.0
    max_pattern_length: int = 500
    # transcripts.py — TranscriptProvider
    lru_documents: int = 4
    transcript_source: str = "md"  # "md" | "jsonl"
    # recall.py — router
    default_max_pointers: int = 5
    # inert seam: "claude" (tools-only). Autonomous LLM engine deferred.
    engine: str = "claude"
