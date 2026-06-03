"""Eval harness: measure recall quality of the simba memory system.

Backend-agnostic IR metrics + a runner that scores any retriever
(``query -> ranked memory ids``) against a curated dataset, so changes to
ranking/extraction can be measured instead of guessed at.
"""
