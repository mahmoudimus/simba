"""Wrapper around the ``qmd`` CLI tool for semantic code search.

Ported from claude-turbo-search/scripts/qmd-search.sh.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "i",
        "you",
        "we",
        "they",
        "he",
        "she",
        "what",
        "how",
        "why",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "for",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
        "out",
        "up",
        "down",
        "off",
        "just",
        "only",
        "also",
        "very",
        "really",
        "please",
        "help",
        "me",
        "my",
        "your",
        "want",
        "need",
        "like",
        "make",
        "create",
        "add",
        "fix",
        "update",
        "change",
        "show",
        "tell",
        "explain",
        "find",
        "search",
        "look",
        "get",
        "write",
        "read",
        "use",
        "implement",
        "build",
    }
)
_MIN_TERM_LENGTH = 3
_MAX_TERMS = 8


def is_available() -> bool:
    """Return whether the ``qmd`` CLI tool is installed."""
    return shutil.which("qmd") is not None


def extract_search_terms(prompt: str) -> str:
    """Extract meaningful search terms from a natural-language prompt.

    Lowercases, removes stop words and short tokens, and returns up to
    ``_MAX_TERMS`` terms joined by spaces.
    """
    tokens = re.findall(r"[a-z0-9]+", prompt.lower())
    terms = [t for t in tokens if t not in _STOP_WORDS and len(t) >= _MIN_TERM_LENGTH]
    return " ".join(terms[:_MAX_TERMS])


def search(
    query: str,
    max_results: int = 3,
    *,
    files_only: bool = False,
) -> list[dict[str, str]]:
    """Run a ``qmd search`` and return parsed results.

    Each result dict contains *path*, *snippet*, and *score* keys.
    Returns an empty list on any subprocess or parsing error.
    """
    cmd = ["qmd", "search", query, "--json", "-n", str(max_results)]
    if files_only:
        cmd.append("--files")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        data = json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []

    entries: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", ""))
        path = re.sub(r"^qmd://", "", path)
        entries.append(
            {
                "path": path,
                "snippet": str(item.get("snippet", "")),
                "score": str(item.get("score", "")),
            }
        )
    return entries


def format_file_suggestions(results: list[dict[str, str]]) -> str:
    """Format search results as an XML context block for hook injection.

    Returns an empty string when *results* is empty.
    """
    if not results:
        return ""
    lines = [
        "<qmd-context>",
        "Relevant files found by semantic search (consider reading these first):",
    ]
    for entry in results:
        lines.append(f"  - {entry['path']}")
    lines.append("</qmd-context>")
    return "\n".join(lines)
