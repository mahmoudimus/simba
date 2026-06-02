"""High-signal term selection for the keyword (bm25) recall arm.

The keyword arm of hybrid recall used to receive the *entire* query string —
fine for a short user prompt, but a long thinking block produced a 200-term OR
that bm25 could match against almost anything.  ``focus_terms`` trims that to a
small, entity-biased set: stop words dropped, deduped, and ranked so that
identifiers, paths, version strings, and proper nouns (the things you actually
search code for) come first.  Pure and lexicon-driven — no dependencies.

This module owns the canonical stop-word lexicon; other layers (e.g. the KG
hook's keyword extractor) may import :data:`STOP_WORDS` to avoid duplication.
"""

from __future__ import annotations

import re

# Canonical English stop-word lexicon for keyword extraction across the project.
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "it",
        "in",
        "of",
        "to",
        "and",
        "or",
        "for",
        "on",
        "at",
        "by",
        "be",
        "do",
        "if",
        "as",
        "so",
        "we",
        "he",
        "me",
        "my",
        "no",
        "not",
        "but",
        "are",
        "was",
        "has",
        "had",
        "how",
        "who",
        "what",
        "when",
        "that",
        "this",
        "with",
        "from",
        "have",
        "will",
        "can",
        "all",
        "its",
        "they",
        "them",
        "been",
        "does",
        "did",
        "just",
        "more",
        "also",
        "very",
        "about",
        "would",
        "could",
        "should",
        "which",
        "there",
        "their",
        "than",
        "then",
        "some",
        "into",
        "use",
        "using",
        "used",
        "i",
        "you",
        "your",
    }
)

# Matches identifiers, dotted paths, and version-like tokens (foo_bar, routes.py,
# nomic-embed, v1.5) as single tokens.
_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_.-]*")


def _salience(tok: str) -> int:
    """Cheap proxy for 'entity-ness': identifiers/paths/proper nouns rank first."""
    score = 0
    if any(c in tok for c in "_./-") or any(c.isdigit() for c in tok):
        score += 2  # identifiers, paths, versions
    if tok[:1].isupper():
        score += 1  # proper nouns / CapWords
    if len(tok) > 6:
        score += 1
    return score


def focus_terms(query: str, *, max_terms: int = 12, min_len: int = 3) -> list[str]:
    """Return up to ``max_terms`` high-signal terms from ``query``.

    Stop-word filtered, deduped case-insensitively (first casing kept), and
    salience-ranked (entity-like terms first, ties broken by first appearance).
    """
    seen: set[str] = set()
    candidates: list[tuple[str, int]] = []
    for index, match in enumerate(_TOKEN_RE.finditer(query or "")):
        tok = match.group()
        low = tok.lower()
        if len(tok) < min_len or low in STOP_WORDS or low in seen:
            continue
        seen.add(low)
        candidates.append((tok, index))

    candidates.sort(key=lambda c: (-_salience(c[0]), c[1]))
    return [tok for tok, _ in candidates[:max_terms]]
