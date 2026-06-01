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
    # UserPromptSubmit: passively inject navigable transcript pointers so the
    # agent knows it can rlm_grep/rlm_peek them. Off by default (opt-in).
    inject_pointers: bool = False
    # Autonomous engine (opt-in): "claude" = none (agent-driven default).
    engine: str = "claude"  # claude | claude-cli | api | local-gguf
    engine_model: str = "haiku"  # cheap by default; never opus
    engine_base_url: str = ""  # OpenAI/Anthropic-compatible endpoint or proxy
    engine_api_key_env: str = "ANTHROPIC_API_KEY"  # env var holding the key
    engine_allowed_tools: str = (
        "mcp__neuron__rlm_recall,mcp__neuron__rlm_grep,"
        "mcp__neuron__rlm_peek,mcp__neuron__rlm_window,Bash"
    )
    engine_max_turns: int = 12  # cli cost cap
    engine_max_pointers: int = 5  # transcripts digested per run
    engine_min_new_exchanges: int = 20  # rate-limit: min messages before a digest fires
