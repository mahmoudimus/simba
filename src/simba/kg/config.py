"""Configuration for the knowledge-graph (kg) subsystem."""

from __future__ import annotations

import dataclasses
import typing

import simba.config


@simba.config.configurable("kg")
@dataclasses.dataclass
class KgConfig:
    min_keyword_len: int = 2
    inject_max_facts: int = 3
    fts_tokenize: str = "trigram"
    default_subject_type: str = "concept"
    default_object_type: str = "concept"
    # Entity resolution: collapse surface-form variants (case/articles/quotes)
    # of the same entity to one canonical node on kg_add. On by default
    # (experimental); set false to store raw surface forms.
    entity_resolution_enabled: bool = True
    entity_similarity_threshold: float = 0.9
    # Multi-hop traversal: safety bound on edges returned by a single
    # kg_neighbors / expand_hops traversal (guards against runaway subgraphs).
    max_neighbor_edges: int = 200


def load_config(**overrides: typing.Any) -> KgConfig:
    """Load config from TOML files, then apply CLI/keyword overrides."""
    base = simba.config.load("kg")
    valid_keys = {f.name for f in dataclasses.fields(KgConfig)}
    filtered = {k: v for k, v in overrides.items() if v is not None and k in valid_keys}
    if not filtered:
        return base
    # Merge overrides on top of TOML-loaded base
    base_dict = dataclasses.asdict(base)
    base_dict.update(filtered)
    return KgConfig(**base_dict)
