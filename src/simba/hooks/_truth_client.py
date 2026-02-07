"""Truth DB client for hooks — keyword extraction and fact lookup.

Queries the proven_facts table for facts relevant to the user's query.
Used by user_prompt_submit and pre_tool_use hooks to inject proven facts
alongside recalled memories.
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


def query_truth_db(query_text: str) -> str:
    """Query the truth DB for facts relevant to query_text.

    Extracts keywords, searches proven_facts by subject, and returns
    a ``<proven-facts>`` XML block.  Returns ``""`` if no facts found
    or if the DB is unavailable.
    """
    keywords = extract_keywords(query_text, max_keywords=3)
    if not keywords:
        return ""

    try:
        import simba.db

        conn = simba.db.get_connection()
        if conn is None:
            return ""
    except Exception:
        return ""

    try:
        cursor = conn.cursor()
        # Search for facts where subject matches any keyword (case-insensitive).
        placeholders = " OR ".join(["subject LIKE ?"] * len(keywords))
        params = [f"%{kw}%" for kw in keywords]
        sql = (
            "SELECT subject, predicate, object, proof"
            f" FROM proven_facts WHERE {placeholders}"
        )
        rows = cursor.execute(sql, params).fetchall()

        if not rows:
            return ""

        lines = ["<proven-facts>"]
        for row in rows:
            subject, predicate, obj, proof = row
            lines.append(f'  <fact subject="{subject}" predicate="{predicate}">')
            lines.append(f"    {obj}")
            lines.append(f"    <proof>{proof}</proof>")
            lines.append("  </fact>")
        lines.append("</proven-facts>")
        return "\n".join(lines)
    except Exception:
        return ""
    finally:
        conn.close()
