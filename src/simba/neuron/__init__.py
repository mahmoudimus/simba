"""Neuron: Neuro-Symbolic Logic Server for Claude Code.

Provides formal verification tools (Z3, Soufffle Datalog) and a truth database
via MCP (Model Context Protocol).
"""

from __future__ import annotations

# Install the Phase 7 schema (kg_derived_edges + kg_rules + dormant flag) on the
# first import of any neuron module.
import simba.neuron.schema  # noqa: F401
