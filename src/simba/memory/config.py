"""Configuration for the memory daemon.

Ported from claude-memory/config.json.
"""

from __future__ import annotations

import dataclasses
import typing

import simba.config


@simba.config.configurable("memory")
@dataclasses.dataclass
class MemoryConfig:
    port: int = 8741
    db_path: str = ""
    embedding_model: str = "nomic-embed-text"
    embedding_dims: int = 768
    model_repo: str = "nomic-ai/nomic-embed-text-v1.5-GGUF"
    model_file: str = "nomic-embed-text-v1.5.Q4_K_M.gguf"
    model_path: str = ""
    n_gpu_layers: int = -1
    embed_url: str = ""
    min_similarity: float = 0.35
    max_results: int = 3
    duplicate_threshold: float = 0.92
    max_content_length: int = 1000
    auto_start: bool = True
    diagnostics_after: int = 50
    sync_interval: int = 0
    shutdown_timeout: int = 10
    # Hybrid recall (L3): a BM25 keyword arm fused with the vector arm via RRF.
    hybrid_enabled: bool = True
    rrf_k: int = 60
    fts_candidate_pool: int = 20
    fts_tokenize: str = "trigram"
    vector_weight: float = 1.0
    keyword_weight: float = 1.0
    # Read-path query intelligence (Phase 0).
    intent_aware: bool = True  # auto-pick the cosine floor from query intent
    min_similarity_broad: float = 0.28  # recall floor for aggregation/exploration
    fts_max_terms: int = 12  # cap on high-signal terms fed to the keyword arm


def load_config(**overrides: typing.Any) -> MemoryConfig:
    """Load config from TOML files, then apply CLI/keyword overrides."""
    base = simba.config.load("memory")
    valid_keys = {f.name for f in dataclasses.fields(MemoryConfig)}
    filtered = {k: v for k, v in overrides.items() if v is not None and k in valid_keys}
    if not filtered:
        return base
    # Merge overrides on top of TOML-loaded base
    base_dict = dataclasses.asdict(base)
    base_dict.update(filtered)
    return MemoryConfig(**base_dict)
