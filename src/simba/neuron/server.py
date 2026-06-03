"""FastMCP server — registers Neuron verification MCP tools and runs the server."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from pathlib import Path

import simba.kg.store
import simba.neuron.verify

mcp = FastMCP("Neuron")


# --- Knowledge-graph tools ---


@mcp.tool()
def kg_add(
    subject: str,
    predicate: str,
    object: str,
    proof: str,
    subject_type: str = "concept",
    object_type: str = "concept",
    transcript_id: str | None = None,
    char_start: int | None = None,
    occurred_at: str | None = None,
) -> str:
    """Inserts an open temporal edge into the knowledge graph.

    ``occurred_at`` is the *event time* (when the fact was true in the world),
    distinct from the belief time stamped automatically on insert.
    """
    return json.dumps(
        simba.kg.store.kg_add(
            subject,
            predicate,
            object,
            proof,
            subject_type=subject_type,
            object_type=object_type,
            transcript_id=transcript_id,
            char_start=char_start,
            project_path=None,
            occurred_at=occurred_at,
        )
    )


@mcp.tool()
def kg_query(
    query: str | None = None,
    subject: str | None = None,
    predicate: str | None = None,
    as_of: str | None = None,
    include_expired: bool = False,
    occurred_after: str | None = None,
    occurred_before: str | None = None,
    limit: int = 10,
) -> str:
    """Queries the knowledge graph with FTS ranking and bitemporal filters.

    ``as_of`` snapshots belief time; ``occurred_after``/``occurred_before`` bound
    event time (``occurred_at``). Returned facts include their ``occurred_at``.
    """
    return json.dumps(
        simba.kg.store.kg_query(
            query,
            subject=subject,
            predicate=predicate,
            as_of=as_of,
            include_expired=include_expired,
            occurred_after=occurred_after,
            occurred_before=occurred_before,
            limit=limit,
        )
    )


@mcp.tool()
def kg_invalidate(subject: str, predicate: str, object: str) -> str:
    """Closes every matching open edge and reports how many were closed."""
    return json.dumps(
        {"closed": simba.kg.store.kg_invalidate(subject, predicate, object)}
    )


# --- Verification tools ---


@mcp.tool()
def verify_z3(python_script: str) -> str:
    """Executes a Z3 proof script in an isolated process.

    The script MUST print 'PROVEN' or 'COUNTEREXAMPLE' to stdout.
    The environment already has 'from z3 import *'.
    """
    return simba.neuron.verify.verify_z3(python_script)


@mcp.tool()
def analyze_datalog(datalog_code: str, facts_dir: str = ".") -> str:
    """Runs a Souffle Datalog analysis.

    Writes code to a temporary file and executes against the specified
    fact directory.
    """
    return simba.neuron.verify.analyze_datalog(datalog_code, facts_dir)


# --- RLM lossless-recall tools ---


@mcp.tool()
def rlm_recall(query: str, max_pointers: int | None = None) -> str:
    """Find transcripts relevant to a query (project-scoped). Returns pointers
    {snippet, transcript_id, project_path, similarity, available} to navigate
    with rlm_grep/rlm_peek/rlm_window."""
    import simba.rlm.service

    result = simba.rlm.service.get_service().recall(
        query, cwd=os.getcwd(), max_pointers=max_pointers
    )
    return json.dumps(result)


@mcp.tool()
def rlm_grep(transcript_id: str, pattern: str, max_matches: int | None = None) -> str:
    """Regex-search a transcript. Returns matches with line numbers and char
    offsets (use the offsets with rlm_peek/rlm_window)."""
    import simba.rlm.service

    return json.dumps(
        simba.rlm.service.get_service().grep(transcript_id, pattern, max_matches)
    )


@mcp.tool()
def rlm_peek(transcript_id: str, start_char: int, end_char: int) -> str:
    """Return the exact character range [start_char, end_char) of a transcript."""
    import simba.rlm.service

    return json.dumps(
        simba.rlm.service.get_service().peek(transcript_id, start_char, end_char)
    )


@mcp.tool()
def rlm_window(transcript_id: str, around_char: int, radius: int | None = None) -> str:
    """Return transcript text within +/- radius chars of an offset (expand a hit)."""
    import simba.rlm.service

    return json.dumps(
        simba.rlm.service.get_service().window(transcript_id, around_char, radius)
    )


@mcp.tool()
def rlm_head(transcript_id: str, n_lines: int = 20) -> str:
    """Return the first n_lines of a transcript."""
    import simba.rlm.service

    return json.dumps(simba.rlm.service.get_service().head(transcript_id, n_lines))


@mcp.tool()
def rlm_tail(transcript_id: str, n_lines: int = 20) -> str:
    """Return the last n_lines of a transcript."""
    import simba.rlm.service

    return json.dumps(simba.rlm.service.get_service().tail(transcript_id, n_lines))


# --- Server entry point ---


def run_server(root_dir: Path | None = None) -> None:
    """Configure and start the FastMCP server."""
    if root_dir is not None:
        resolved = root_dir.resolve()
        os.chdir(resolved)

    mcp.run()
