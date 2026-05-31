"""KG client for hooks — keyword extraction and knowledge-graph fact lookup.

Queries the temporal ``kg_edges`` store (via :mod:`simba.kg.store`) for facts
relevant to the user's query.  Used by the pre_tool_use hook to inject
currently-valid graph facts alongside recalled memories.  Each emitted fact
carries its source ``transcript_id``/``char_start`` (when present) so the agent
can ``rlm_peek`` the originating transcript span.
"""

from __future__ import annotations

import re

# Common English stop words to filter out during keyword extraction.
_STOP_WORDS = frozenset(
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

_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_.-]*")


def extract_keywords(text: str, max_keywords: int = 3) -> list[str]:
    """Extract meaningful keywords from text.

    Tokenizes the text, removes stop words, deduplicates (case-insensitive),
    and returns up to ``max_keywords`` keywords in order of first appearance.
    """
    seen: set[str] = set()
    keywords: list[str] = []

    for match in _WORD_RE.finditer(text):
        word = match.group()
        lower = word.lower()
        if lower in _STOP_WORDS or len(lower) < 2:
            continue
        if lower in seen:
            continue
        seen.add(lower)
        keywords.append(word)
        if len(keywords) >= max_keywords:
            break

    return keywords


def query_kg(
    query_text: str,
    project_path: str | None = None,
    cwd: str | None = None,
) -> str:
    """Query the knowledge graph for facts relevant to ``query_text``.

    Extracts keywords, joins them into a single FTS query string (the trigram
    FTS index treats the space-separated terms with OR-friendly semantics), and
    asks :func:`simba.kg.store.kg_query` for the top ``inject_max_facts``
    currently-valid edges scoped to ``project_path``.  Returns a ``<kg-facts>``
    XML block — one ``<fact subject=.. predicate=..>object</fact>`` per row,
    carrying ``transcript_id``/``char_start`` attributes when present so the
    agent can ``rlm_peek`` the source.  Returns ``""`` when there are no
    keywords, no rows, or the DB is unavailable (never raises).
    """
    keywords = extract_keywords(query_text, max_keywords=3)
    if not keywords:
        return ""

    fts_query = " ".join(keywords)

    try:
        import simba.config
        import simba.kg.config  # registers the "kg" config section
        import simba.kg.store

        cfg = simba.config.load("kg")
        rows = simba.kg.store.kg_query(
            query=fts_query,
            project_path=project_path,
            limit=cfg.inject_max_facts,
        )
    except Exception:
        return ""

    if not rows:
        return ""

    lines = ["<kg-facts>"]
    for row in rows:
        subject = row.get("subject", "")
        predicate = row.get("predicate", "")
        obj = row.get("object", "")
        attrs = f'subject="{subject}" predicate="{predicate}"'
        transcript_id = row.get("transcript_id")
        if transcript_id is not None:
            attrs += f' transcript_id="{transcript_id}"'
        char_start = row.get("char_start")
        if char_start is not None:
            attrs += f' char_start="{char_start}"'
        lines.append(f"  <fact {attrs}>{obj}</fact>")
    lines.append("</kg-facts>")
    return "\n".join(lines)
