"""Wrapper around the ``qmd`` CLI tool for semantic code search.

Ported from claude-turbo-search/scripts/qmd-search.sh.
"""

from __future__ import annotations

import json
import os
import pathlib
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


def _config_file() -> pathlib.Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(pathlib.Path.home() / ".config")
    return pathlib.Path(base) / "qmd" / "index.yml"


def collection_paths(config: pathlib.Path | None = None) -> dict[str, str]:
    """Return collection name -> indexed directory from qmd's ``index.yml``.

    Reads only the ``collections:`` mapping that ``qmd collection add`` writes,
    so simba needs no YAML dependency. Returns ``{}`` when the file is missing
    or unreadable.
    """
    try:
        lines = (config or _config_file()).read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    paths: dict[str, str] = {}
    in_collections = False
    name = ""
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        text = line.strip()
        if indent == 0:
            in_collections = text == "collections:"
            continue
        if not in_collections:
            continue
        if indent == 2 and text.endswith(":"):
            name = text[:-1].strip().strip("'\"")
        elif indent > 2 and name and text.startswith("path:"):
            paths[name] = text[len("path:") :].strip().strip("'\"")
    return paths


def collection_for(cwd: pathlib.Path, config: pathlib.Path | None = None) -> str | None:
    """Return the qmd collection that indexes *cwd*, or ``None``.

    The innermost collection whose directory contains *cwd* wins. A collection
    that merely sits below *cwd*, or one for another project, does not match,
    so a prompt never pulls snippets from unrelated repositories.
    """
    try:
        here = pathlib.Path(cwd).resolve()
    except OSError:
        return None
    best: tuple[int, str] | None = None
    for name, raw in collection_paths(config).items():
        try:
            root = pathlib.Path(raw).expanduser().resolve()
        except OSError:
            continue
        if here == root or root in here.parents:
            depth = len(root.parts)
            if best is None or depth > best[0]:
                best = (depth, name)
    return best[1] if best else None


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
    collection: str | None = None,
) -> list[dict[str, str]]:
    """Run a ``qmd search`` and return parsed results.

    Each result dict contains *path*, *snippet*, and *score* keys. *collection*
    limits the search to one collection. Returns an empty list on any
    subprocess or parsing error.
    """
    cmd = ["qmd", "search", query, "--json", "-n", str(max_results)]
    if collection:
        cmd += ["-c", collection]
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
        # qmd's JSON names the field "file"; older output used "path".
        path = str(item.get("file") or item.get("path") or "")
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
