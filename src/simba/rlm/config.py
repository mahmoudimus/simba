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
    # context.py — DocumentStore bounded/lazy mode. Post-mortem 2026-07-20:
    # malloc_history on a live daemon (36GB RSS, 50.9GB peak) attributed the
    # heap to a single 2153MB Codex transcript retained whole -- a 2062MB
    # _io_FileIO_readall_impl buffer, 16.5GB in the newline decoder, and
    # 21.3GB across 894k live per-line strings from _Document.text.split("\n").
    # Documents at or under this size keep the fast in-memory (.text/.lines)
    # path; anything larger is never refused, it is served through an
    # offset-indexed lazy mode instead (seek + bounded decode per read --
    # the full document is never materialized as one string).
    max_document_mb: float = 64.0
    # Total retained bytes across the DocumentStore (eager doc text + lazy
    # offset-index arrays). Exceeding it evicts the least-recently-used
    # documents' retained payload -- doc_id and source identity are kept, so
    # a later read transparently rebuilds the offset index. Same 2026-07-20
    # incident: a size cap per document is not enough if enough documents
    # accumulate, so the store also caps its aggregate footprint.
    store_budget_mb: float = 256.0
    # transcripts.py — TranscriptProvider
    lru_documents: int = 4
    transcript_source: str = "md"  # "md" | "jsonl"
    # recall.py — router
    default_max_pointers: int = 5
    # UserPromptSubmit: passively inject navigable transcript pointers so the
    # agent knows it can rlm_grep/rlm_peek them. On by default (experimental).
    inject_pointers: bool = True
    # Autonomous engine: "claude" = none (agent-driven); "claude-cli" spawns a
    # detached cheap digest on PreCompact + enables episodic consolidation. On by
    # default (experimental); set to "claude" to disable the autonomous spend.
    # "llm-cli" routes the digest through `llm -m <engine_model>` (simonw's CLI;
    # e.g. deepseek-v4-flash) as a single JSON-extraction completion instead of an
    # agentic loop — cheaper, and the basis for personal-assistant memory.
    engine: str = "claude-cli"  # claude | claude-cli | llm-cli | api | local-gguf
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
    # Digest prompt template. Empty -> the active engine's built-in default
    # (an agentic, coding-scoped prompt for claude-cli; a JSON-extraction prompt
    # for llm-cli). Override it to repurpose simba as a personal-assistant memory:
    # e.g. extract facts / preferences / events instead of coding learnings.
    # Template fields: {transcript} (conversation text, llm-cli only), {cwd}, {tid}.
    digest_prompt: str = ""
    # Stale-reclaim window for the digest lease: a 'running' digest older than
    # this many seconds is treated as a dead worker and re-acquirable. 0 => no
    # reclaim (preserves the original rlm_jobs behavior: a dead worker locks a
    # transcript from re-digesting). Flip up to recover from crashed digests.
    digest_stale_after_seconds: int = 0
    # Write-time extraction validation gate (Eywa-style, arXiv 2605.30771). When on,
    # each extracted claim is checked for grounding in its source transcript before
    # being stored: hard-value (no hallucinated number/date) + support overlap. An
    # ungrounded claim is dropped, not stored. Off by default; fail-open. (Polarity
    # is left to short-span callers — unreliable over a whole transcript.)
    extraction_validation_enabled: bool = False
    extraction_validation_min_support: float = 0.5
