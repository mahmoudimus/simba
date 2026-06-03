"""Entity normalization + resolution for the knowledge graph.

The KG stores raw surface forms, so ``GITHUB_TOKEN`` / ``github_token`` /
``the GITHUB_TOKEN`` fragment into separate nodes. ``normalize_entity`` derives
a conservative canonical *key* (case, articles, quotes, possessives, trailing
punctuation, whitespace — but NOT underscores, so code identifiers stay
distinct), and ``resolve`` maps a surface form to an existing canonical entity
by that key, with an optional embedding-similarity merge for synonyms that don't
share a key (``Bob`` / ``Robert``). Pure + deterministic; the embedder is
injected so this is testable without the model.
"""

from __future__ import annotations

import re
import typing

_ARTICLES = ("the ", "a ", "an ")
_WS = re.compile(r"\s+")
_EDGE_PUNCT = ".,;:!?"
# straight + curly quotes/backtick (curly via escapes to avoid ambiguous-unicode)
_QUOTES = "\"\'`" + "\u201c\u201d\u2018\u2019"


def normalize_entity(name: str) -> str:
    """Return a conservative canonical key for an entity surface form."""
    s = (name or "").strip().strip(_QUOTES).strip()
    s = s.lower()
    for art in _ARTICLES:
        if s.startswith(art):
            s = s[len(art) :]
            break
    if s.endswith("'s") or s.endswith("\u2019s"):
        s = s[:-2]
    s = s.strip(_EDGE_PUNCT).strip()
    return _WS.sub(" ", s)


def _clean_display(name: str) -> str:
    """Trim a surface form for use as a fresh canonical display name."""
    return _WS.sub(" ", (name or "").strip())


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=False))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def resolve(
    name: str,
    existing: typing.Iterable[str],
    *,
    embed: typing.Callable[[str], list[float]] | None = None,
    threshold: float = 0.9,
) -> str:
    """Resolve ``name`` to a canonical entity drawn from ``existing``.

    Returns an existing canonical display form when ``name`` shares its
    normalized key, or (when ``embed`` is given) when their embeddings are at or
    above ``threshold`` cosine. Otherwise returns ``name`` cleaned, as a new
    canonical entity. The first existing form seen for a key wins.
    """
    key = normalize_entity(name)
    by_key: dict[str, str] = {}
    for cand in existing:
        ckey = normalize_entity(cand)
        by_key.setdefault(ckey, cand)

    if key in by_key:
        return by_key[key]

    if embed is not None and by_key:
        target = embed(name)
        best_name, best_sim = None, -1.0
        for cand in by_key.values():
            sim = _cosine(target, embed(cand))
            if sim > best_sim:
                best_name, best_sim = cand, sim
        if best_name is not None and best_sim >= threshold:
            return best_name

    return _clean_display(name)
