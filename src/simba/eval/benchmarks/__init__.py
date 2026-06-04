"""Adapters for external memory benchmarks (LoCoMo, LongMemEval).

Each loader maps a benchmark into simba's eval ``Dataset`` shape (corpus +
cases-with-gold), so the existing recall harness scores recall@k of the gold
evidence on real data — no LLM calls. An LLM-judge answer-accuracy layer (to
compare with Mem0/Zep headline numbers) sits on top separately.
"""
