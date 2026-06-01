"""FastMCP server — registers Neuron verification MCP tools and runs the server."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from pathlib import Path

import simba.neuron.truth
import simba.neuron.verify

mcp = FastMCP("Neuron")


# --- Truth DB tools ---


@mcp.tool()
def truth_add(subject: str, predicate: str, object: str, proof: str) -> str:
    """Records a proven fact into the Truth DB.

    Use this ONLY when a verifier (Z3/Datalog) has proven a hypothesis.
    """
    return simba.neuron.truth.truth_add(subject, predicate, object, proof)


@mcp.tool()
def truth_query(subject: str | None = None, predicate: str | None = None) -> str:
    """Queries the Truth DB for existing proven facts.

    Use this BEFORE assuming capabilities or behavior about the codebase.
    """
    return simba.neuron.truth.truth_query(subject, predicate)


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
