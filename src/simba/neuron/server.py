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
) -> str:
    """Inserts an open temporal edge into the knowledge graph."""
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
        )
    )


@mcp.tool()
def kg_query(
    query: str | None = None,
    subject: str | None = None,
    predicate: str | None = None,
    as_of: str | None = None,
    include_expired: bool = False,
    limit: int = 10,
) -> str:
    """Queries the knowledge graph with FTS ranking and temporal filters."""
    return json.dumps(
        simba.kg.store.kg_query(
            query,
            subject=subject,
            predicate=predicate,
            as_of=as_of,
            include_expired=include_expired,
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


# --- Server entry point ---


def run_server(root_dir: Path | None = None) -> None:
    """Configure and start the FastMCP server."""
    if root_dir is not None:
        resolved = root_dir.resolve()
        os.chdir(resolved)

    mcp.run()
