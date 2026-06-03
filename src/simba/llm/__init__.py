"""Synchronous LLM client for in-process tasks (reranking, extraction).

Distinct from the RLM *engine* (rlm/engine.py), which is detached/fire-and-forget
for transcript digests. This client makes a blocking, bounded call and returns
text, shelling out to a local CLI (``claude`` or ``llm``) so there is no SDK
dependency and the model/endpoint is fully configurable (incl. a DeepSeek-style
backend via ``base_url``). Fail-open: any error returns empty so callers degrade
to their non-LLM path.
"""
