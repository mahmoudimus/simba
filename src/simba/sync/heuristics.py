"""Heuristic fact extraction from memory content.

Fast, dependency-free extraction using regex patterns. Each memory type
has dedicated patterns that map natural-language content to
(subject, predicate, object, proof) tuples.
"""

from __future__ import annotations

import re

Triple = tuple[str, str, str, str]

# ---------------------------------------------------------------------------
# Per-type extractors
# ---------------------------------------------------------------------------

_SOLUTION_PATTERNS = [
    re.compile(r"(?:use|run|try)\s+(.+?)\s+(?:to|for)\s+(.+)", re.IGNORECASE),
    re.compile(r"(.+?)\s+(?:fixes|solves|resolves)\s+(.+)", re.IGNORECASE),
    re.compile(r"(.+?)\s+(?:works|worked)\s+(?:for|with)\s+(.+)", re.IGNORECASE),
]

_GOTCHA_PATTERNS = [
    re.compile(r"(.+?)\s+(?:causes|breaks|blocks)\s+(.+)", re.IGNORECASE),
    re.compile(
        r"(?:watch out|beware|careful)"
        r"(?:\s+(?:for|of|with))?\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:don'?t|avoid|never)\s+(.+)", re.IGNORECASE),
]

_PATTERN_PATTERNS = [
    re.compile(r"(.+?)\s+(?:uses|follows|adopts)\s+(.+)", re.IGNORECASE),
    re.compile(r"(.+?)\s+(?:pattern|convention|approach):\s*(.+)", re.IGNORECASE),
]

_DECISION_PATTERNS = [
    re.compile(
        r"(?:chose|decided|picked|selected)\s+(.+?)(?:\s+over\s+(.+))?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:using|use|prefer)\s+(.+?)(?:\s+(?:for|because)\s+(.+))?$",
        re.IGNORECASE,
    ),
]

_FAILURE_PATTERNS = [
    re.compile(
        r"(.+?)\s+(?:doesn'?t work|fails?|broke)\s*(?:for|with|because)?\s*(.*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(.+?)\s+(?:is broken|is incompatible)\s*(?:with)?\s*(.*)",
        re.IGNORECASE,
    ),
]

_PREFERENCE_PATTERNS = [
    re.compile(
        r"(?:prefer|prefers|likes?)\s+(.+?)(?:\s+over\s+(.+))?$",
        re.IGNORECASE,
    ),
    re.compile(r"(?:always use|always prefer)\s+(.+)", re.IGNORECASE),
]


def _extract_working_solution(
    content: str, context: str, memory_id: str
) -> list[Triple]:
    proof = f"memory:{memory_id}"
    text = f"{content} {context}".strip()
    triples: list[Triple] = []
    for pat in _SOLUTION_PATTERNS:
        m = pat.search(text)
        if m:
            subj = m.group(1).strip()[:80]
            obj = m.group(2).strip()[:80]
            triples.append((subj, "solves", obj, proof))
            break
    return triples


def _extract_gotcha(content: str, context: str, memory_id: str) -> list[Triple]:
    proof = f"memory:{memory_id}"
    text = f"{content} {context}".strip()
    triples: list[Triple] = []
    for pat in _GOTCHA_PATTERNS:
        m = pat.search(text)
        if m:
            groups = [g for g in m.groups() if g]
            if len(groups) >= 2:
                triples.append(
                    (
                        groups[0].strip()[:80],
                        "has_pitfall",
                        groups[1].strip()[:80],
                        proof,
                    )
                )
            elif len(groups) == 1:
                triples.append(
                    (
                        groups[0].strip()[:80],
                        "has_pitfall",
                        "warning",
                        proof,
                    )
                )
            break
    return triples


def _extract_pattern(content: str, context: str, memory_id: str) -> list[Triple]:
    proof = f"memory:{memory_id}"
    text = f"{content} {context}".strip()
    triples: list[Triple] = []
    for pat in _PATTERN_PATTERNS:
        m = pat.search(text)
        if m:
            subj = m.group(1).strip()[:80]
            obj = m.group(2).strip()[:80]
            triples.append((subj, "follows_pattern", obj, proof))
            break
    return triples


def _extract_decision(content: str, context: str, memory_id: str) -> list[Triple]:
    proof = f"memory:{memory_id}"
    text = f"{content} {context}".strip()
    triples: list[Triple] = []
    for pat in _DECISION_PATTERNS:
        m = pat.search(text)
        if m:
            obj = m.group(1).strip()[:80]
            has_reason = m.lastindex and m.lastindex >= 2
            reason = (m.group(2) or "").strip()[:80] if has_reason else ""
            predicate = "chose"
            if reason:
                triples.append(("project", predicate, f"{obj} ({reason})", proof))
            else:
                triples.append(("project", predicate, obj, proof))
            break
    return triples


def _extract_failure(content: str, context: str, memory_id: str) -> list[Triple]:
    proof = f"memory:{memory_id}"
    text = f"{content} {context}".strip()
    triples: list[Triple] = []
    for pat in _FAILURE_PATTERNS:
        m = pat.search(text)
        if m:
            subj = m.group(1).strip()[:80]
            obj = (m.group(2) or "unknown reason").strip()[:80]
            triples.append((subj, "fails_for", obj or "unknown reason", proof))
            break
    return triples


def _extract_preference(content: str, context: str, memory_id: str) -> list[Triple]:
    proof = f"memory:{memory_id}"
    text = f"{content} {context}".strip()
    triples: list[Triple] = []
    for pat in _PREFERENCE_PATTERNS:
        m = pat.search(text)
        if m:
            obj = m.group(1).strip()[:80]
            triples.append(("user", "prefers", obj, proof))
            break
    return triples


def _extract_tool_rule(
    content: str, context: str, memory_id: str
) -> list[Triple]:
    proof = f"memory:{memory_id}"
    triples: list[Triple] = []
    # TOOL_RULE context is JSON with tool, pattern, error_source, correction
    import json as _json

    try:
        ctx = _json.loads(context)
    except (ValueError, TypeError):
        ctx = {}

    tool = ctx.get("tool", "")
    pattern = ctx.get("pattern", "")
    error_source = ctx.get("error_source", "")
    correction = ctx.get("correction", "")

    subj = f"{tool} {pattern}".strip()[:80]
    if error_source:
        triples.append((subj, "fails_for", error_source[:80], proof))
    if correction:
        triples.append((subj, "has_fix", correction[:80], proof))
    return triples


_EXTRACTORS: dict[str, object] = {
    "WORKING_SOLUTION": _extract_working_solution,
    "GOTCHA": _extract_gotcha,
    "PATTERN": _extract_pattern,
    "DECISION": _extract_decision,
    "FAILURE": _extract_failure,
    "PREFERENCE": _extract_preference,
    "TOOL_RULE": _extract_tool_rule,
}


def extract_facts(
    memory_type: str,
    content: str,
    context: str = "",
    memory_id: str = "",
) -> list[Triple]:
    """Extract fact triples from memory content using heuristic patterns.

    Returns a list of ``(subject, predicate, object, proof)`` tuples.
    Returns an empty list when no patterns match or the type is unknown.
    """
    extractor = _EXTRACTORS.get(memory_type)
    if extractor is None:
        return []
    return extractor(content, context, memory_id)
