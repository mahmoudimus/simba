"""Configuration for the memory daemon.

Ported from claude-memory/config.json.
"""

from __future__ import annotations

import dataclasses
import typing


@dataclasses.dataclass
class MemoryConfig:
    port: int = 8741
    db_path: str = ""
    embedding_model: str = "nomic-embed-text"
    embedding_dims: int = 768
    ollama_url: str = "http://localhost:11434"
    min_similarity: float = 0.35
    max_results: int = 3
    duplicate_threshold: float = 0.92
    timeout_ms: int = 10000
    max_content_length: int = 200
    auto_start: bool = True


def load_config(**overrides: typing.Any) -> MemoryConfig:
    """Load config with optional overrides."""
    valid_keys = {f.name for f in dataclasses.fields(MemoryConfig)}
    filtered = {k: v for k, v in overrides.items() if v is not None and k in valid_keys}
    return MemoryConfig(**filtered)
