"""Repeatable embedder bake-off (`simba eval embedders`).

Measures candidate GGUF embedders against Simba's deployed one on LoCoMo and
LongMemEval-S (dense rank-only arm) and through Simba's own recall pipeline
(cold-query production arm), with reference-implementation and in-process checks.
See docs/embedder-bakeoff.md.
"""
