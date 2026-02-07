"""Orchestrate SQLite project memory and QMD search into formatted context.

Ported from claude-turbo-search/hooks/rag-context-hook.sh.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import simba.db
import simba.search.project_memory
import simba.search.qmd

if TYPE_CHECKING:
    import pathlib

_MAX_CONTEXT_TOKENS = 1500
_MAX_CODE_RESULTS = 3
_MIN_QUERY_LENGTH = 15
_MEMORY_TOKEN_BUDGET = 500
_SKIP_PATTERNS = re.compile(
    r"^(/|yes|no|ok|thanks|hi|hello|hey|commit|push|pull|git )", re.IGNORECASE
)


def build_context(prompt: str, cwd: pathlib.Path) -> str:
    """Build a formatted context string from project memory and QMD search.

    Returns an empty string when the prompt is too short, matches a skip
    pattern, or no relevant context is found.  All external calls are wrapped
    so that failures never crash the hook.
    """
    # -- Guard checks -------------------------------------------------------
    if len(prompt) < _MIN_QUERY_LENGTH:
        return ""
    if _SKIP_PATTERNS.search(prompt):
        return ""

    # -- Extract search terms -----------------------------------------------
    try:
        search_terms = simba.search.qmd.extract_search_terms(prompt)
    except Exception:
        return ""

    if not search_terms:
        return ""

    # -- Phase 1: Query SQLite project memory -------------------------------
    memory_context = ""
    try:
        conn = simba.db.get_connection(cwd)
        if conn is not None:
            try:
                memory_context = simba.search.project_memory.get_context(
                    conn, search_terms, _MEMORY_TOKEN_BUDGET
                )
            finally:
                conn.close()
    except Exception:
        memory_context = ""

    # -- Phase 2: Query QMD -------------------------------------------------
    code_context = ""
    try:
        if simba.search.qmd.is_available():
            results = simba.search.qmd.search(
                search_terms, max_results=_MAX_CODE_RESULTS
            )
            if results:
                lines: list[str] = []
                for entry in results:
                    lines.append(f"### {entry['path']} (relevance: {entry['score']})")
                    lines.append(f"```\n{entry['snippet']}\n```")
                code_context = "\n".join(lines)
    except Exception:
        code_context = ""

    # -- Phase 3: Combine and format ----------------------------------------
    if not memory_context and not code_context:
        return ""

    sections: list[str] = []
    if memory_context:
        sections.append(f"---\n# Memory Context\n{memory_context}")
    if code_context:
        sections.append(f"---\n# Code Context\n{code_context}")

    body = "\n\n".join(sections)
    return (
        '<relevant-context source="project-search">\n'
        "The following context was automatically retrieved based on your prompt.\n"
        "Use this to answer without reading additional files "
        "unless more detail is needed.\n"
        "\n"
        f"**Search terms:** {search_terms}\n"
        "\n"
        f"{body}\n"
        "</relevant-context>"
    )
