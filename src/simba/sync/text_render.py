"""Convert database rows to embedding text and markdown.

Each simba.db table has its own schema. This module provides
per-table rendering functions for both the embedding pipeline
(plain text) and the QMD export pipeline (markdown).
"""

from __future__ import annotations

INDEXABLE_TABLES: list[str] = [
    "reflections",
    "sessions",
    "knowledge",
    "facts",
    "proven_facts",
    "activities",
    "agent_runs",
]


def _get(row: dict, key: str, default: str = "") -> str:
    """Get a string value from a row dict, defaulting to *default*."""
    val = row.get(key)
    return str(val) if val else default


# ---------------------------------------------------------------------------
# Plain-text rendering (for embedding via POST /store)
# ---------------------------------------------------------------------------

_TEXT_RENDERERS: dict[str, object] = {}


def _text_reflections(row: dict) -> str:
    etype = _get(row, "error_type")
    snippet = _get(row, "snippet")
    sig = _get(row, "signature")
    return f"Error [{etype}]: {snippet} (sig: {sig})"


def _text_sessions(row: dict) -> str:
    summary = _get(row, "summary")
    files = _get(row, "files_touched")
    tools = _get(row, "tools_used")
    topics = _get(row, "topics")
    return f"Session: {summary}. Files: {files}. Tools: {tools}. Topics: {topics}"


def _text_knowledge(row: dict) -> str:
    area = _get(row, "area")
    summary = _get(row, "summary")
    patterns = _get(row, "patterns")
    return f"Knowledge [{area}]: {summary}. Patterns: {patterns}"


def _text_facts(row: dict) -> str:
    category = _get(row, "category")
    fact = _get(row, "fact")
    return f"Project fact [{category}]: {fact}"


def _text_proven_facts(row: dict) -> str:
    subj = _get(row, "subject")
    pred = _get(row, "predicate")
    obj = _get(row, "object")
    proof = _get(row, "proof")
    return f"Proven: {subj} {pred} {obj} (proof: {proof})"


def _text_activities(row: dict) -> str:
    tool = _get(row, "tool_name")
    detail = _get(row, "detail")
    return f"Activity [{tool}]: {detail}"


def _text_agent_runs(row: dict) -> str:
    agent = _get(row, "agent")
    ticket = _get(row, "ticket_id")
    result = _get(row, "result")
    return f"Agent [{agent}] {ticket}: {result}"


_TEXT_RENDERERS = {
    "reflections": _text_reflections,
    "sessions": _text_sessions,
    "knowledge": _text_knowledge,
    "facts": _text_facts,
    "proven_facts": _text_proven_facts,
    "activities": _text_activities,
    "agent_runs": _text_agent_runs,
}


def render_row(table_name: str, row: dict) -> str:
    """Convert a DB row to plain text for embedding."""
    renderer = _TEXT_RENDERERS.get(table_name)
    if renderer is None:
        return ""
    return renderer(row)


# ---------------------------------------------------------------------------
# Markdown rendering (for QMD export)
# ---------------------------------------------------------------------------


def _md_reflections(row: dict) -> str:
    etype = _get(row, "error_type")
    snippet = _get(row, "snippet")
    sig = _get(row, "signature")
    return f"## Reflection: {etype}\n\n{snippet}\n\nSignature: {sig}"


def _md_sessions(row: dict) -> str:
    summary = _get(row, "summary")
    files = _get(row, "files_touched")
    tools = _get(row, "tools_used")
    topics = _get(row, "topics")
    return (
        f"## Session\n\n{summary}\n\n"
        f"- Files: {files}\n- Tools: {tools}\n- Topics: {topics}"
    )


def _md_knowledge(row: dict) -> str:
    area = _get(row, "area")
    summary = _get(row, "summary")
    patterns = _get(row, "patterns")
    return f"## Knowledge: {area}\n\n{summary}\n\nPatterns: {patterns}"


def _md_facts(row: dict) -> str:
    category = _get(row, "category")
    fact = _get(row, "fact")
    return f"## Fact ({category})\n\n{fact}"


def _md_proven_facts(row: dict) -> str:
    subj = _get(row, "subject")
    pred = _get(row, "predicate")
    obj = _get(row, "object")
    proof = _get(row, "proof")
    return f"## Proven Fact\n\n{subj} {pred} {obj}\n\nProof: {proof}"


def _md_activities(row: dict) -> str:
    tool = _get(row, "tool_name")
    detail = _get(row, "detail")
    ts = _get(row, "timestamp")
    return f"## Activity: {tool}\n\n{detail}\n\n_{ts}_"


def _md_agent_runs(row: dict) -> str:
    agent = _get(row, "agent")
    ticket = _get(row, "ticket_id")
    result = _get(row, "result")
    status = _get(row, "status")
    return f"## Agent Run: {agent}\n\nTicket: {ticket}\nStatus: {status}\n\n{result}"


_MD_RENDERERS: dict[str, object] = {
    "reflections": _md_reflections,
    "sessions": _md_sessions,
    "knowledge": _md_knowledge,
    "facts": _md_facts,
    "proven_facts": _md_proven_facts,
    "activities": _md_activities,
    "agent_runs": _md_agent_runs,
}


def render_row_markdown(table_name: str, row: dict) -> str:
    """Convert a DB row to markdown for QMD export."""
    renderer = _MD_RENDERERS.get(table_name)
    if renderer is None:
        return ""
    return renderer(row)
