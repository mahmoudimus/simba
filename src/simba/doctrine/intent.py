"""Cheap doctrine intent classifier (spec 28) — embedding match, no LLM.

``UserPromptSubmit`` is the only hook that sees the user's intent before any
action. This module classifies the prompt against stored doctrine *triggers* by
cosine similarity over PRECOMPUTED trigger embeddings — the prompt is embedded
once (the embedder is already loaded; no LLM on the hot path) and matched against
each doctrine's best trigger. Above the floor → prime that doctrine; a matched
risk-tier doctrine additionally arms the preflight mandate.

Pure + fail-open, like ``conflict.py`` / ``pitfall.py``: any embed error returns
no matches (priming is advisory — it must never crash the hook).
"""

from __future__ import annotations

import dataclasses
import math
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import simba.doctrine.store


@dataclasses.dataclass
class DoctrineMatch:
    """A doctrine that matched the prompt, with its best-trigger similarity."""

    doctrine: simba.doctrine.store.Doctrine
    similarity: float


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 on a zero vector."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def match_doctrine(
    prompt_embedding: Sequence[float],
    doctrines: Sequence[simba.doctrine.store.Doctrine],
    *,
    min_similarity: float,
) -> list[DoctrineMatch]:
    """Return doctrines whose BEST trigger clears ``min_similarity``, desc by score.

    A doctrine matches on its strongest trigger (max cosine over its precomputed
    trigger embeddings). Doctrines with no triggers / no embeddings never match.
    """
    matches: list[DoctrineMatch] = []
    for d in doctrines:
        best = 0.0
        for emb in d.trigger_embeddings:
            if not emb:
                continue
            best = max(best, _cosine(prompt_embedding, emb))
        if best >= min_similarity:
            matches.append(DoctrineMatch(doctrine=d, similarity=best))
    matches.sort(key=lambda m: m.similarity, reverse=True)
    return matches


def classify(
    prompt: str,
    doctrines: Sequence[simba.doctrine.store.Doctrine],
    *,
    embed_fn: Callable[[str], list[float]],
    min_similarity: float = 0.55,
) -> list[DoctrineMatch]:
    """Embed ``prompt`` once and match it against ``doctrines``.

    Fail-open: an empty doctrine list never pays the embed; any embed error
    returns ``[]`` (priming is advisory). ``embed_fn`` is injected so the daemon's
    loaded embedder (or a test fake) supplies the single prompt vector — no
    per-trigger embedding happens on the hot path (triggers are embedded at store
    time).
    """
    if not prompt or not doctrines:
        return []
    try:
        prompt_vec = embed_fn(prompt)
    except Exception:
        return []
    if not prompt_vec:
        return []
    return match_doctrine(prompt_vec, doctrines, min_similarity=min_similarity)
