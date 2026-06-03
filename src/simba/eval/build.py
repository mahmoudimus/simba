"""Build an eval dataset from real memories — LLM-generated queries.

Authored synthetic datasets bias the queries toward what the author imagined and
saturate quickly. This samples the *actual* corpus and asks the LLM to write a
natural question each memory answers; the source memory is the gold answer and
the rest of the sample are real distractors. The result measures recall on real
content with non-author-biased queries. Fail-open: a memory whose query can't be
generated is kept as a distractor but contributes no case.
"""

from __future__ import annotations

import typing

from simba.eval.dataset import Dataset, EvalCase, Memory

_QUERY_PROMPT = (
    "Below is a stored memory from a developer's knowledge base. Write ONE "
    "natural question that a developer would ask whose answer is exactly this "
    "memory. Reply with only the question, nothing else.\n\n"
    "Memory [{mtype}]: {content}{ctx}"
)


def generate_query(memory: dict[str, typing.Any], *, client: typing.Any) -> str:
    """Ask the LLM for a question this memory answers; "" on failure."""
    content = (memory.get("content") or "").strip()
    if not content:
        return ""
    ctx = memory.get("context") or ""
    ctx_s = f"\nContext: {ctx.strip()}" if ctx.strip() else ""
    prompt = _QUERY_PROMPT.format(
        mtype=memory.get("type", "?"), content=content, ctx=ctx_s
    )
    reply = (client.complete(prompt) or "").strip()
    return reply.splitlines()[0].strip() if reply else ""


def build_from_memories(
    memories: list[dict[str, typing.Any]],
    *,
    client: typing.Any,
    name: str = "real-corpus",
    max_cases: int = 50,
) -> Dataset:
    """Assemble a Dataset: full sample as corpus, LLM queries as gold cases."""
    corpus = [
        Memory(
            id=str(m.get("id")),
            content=(m.get("content") or ""),
            type=(m.get("type") or "PATTERN"),
            context=(m.get("context") or ""),
        )
        for m in memories
        if m.get("id") and (m.get("content") or "").strip()
    ]

    cases: list[EvalCase] = []
    if client is not None and client.available():
        for m in memories:
            if len(cases) >= max_cases:
                break
            mid = str(m.get("id") or "")
            if not mid:
                continue
            query = generate_query(m, client=client)
            if query:
                cases.append(
                    EvalCase(id=f"q_{mid}", query=query, relevant_ids=[mid])
                )

    return Dataset(name=name, corpus=corpus, cases=cases)
