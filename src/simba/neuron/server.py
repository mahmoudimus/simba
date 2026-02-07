"""FastMCP server — registers Neuron verification MCP tools and runs the server."""

from __future__ import annotations

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


# --- Server entry point ---


def run_server(root_dir: Path | None = None) -> None:
    """Configure and start the FastMCP server."""
    if root_dir is not None:
        resolved = root_dir.resolve()
        os.chdir(resolved)

    mcp.run()
