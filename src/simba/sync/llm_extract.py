"""Synchronous LLM fact extraction — structured triples via the LLM client.

Distinct from the detached ``researcher`` agent path (sync/extractor.py): this
makes a blocking, bounded call through ``simba.llm.client`` and returns parsed
(subject, predicate, object, proof) triples, so the sync pipeline can extract
relations from memories the regex heuristics miss without spawning an agent.
Fail-open: an unavailable client or a bad reply yields no triples (the caller
keeps whatever the regex path produced).
"""

from __future__ import annotations

import typing

Triple = tuple[str, str, str, str]

_PROMPT = (
    "Extract typed knowledge-graph triples from the text below. Return ONLY a "
    "JSON array of objects with keys: subject, predicate, object, subject_type, "
    "object_type. Use short canonical entity names and a specific predicate "
    "(uses, causes, fixes, depends_on, prefers, ...). Skip generic knowledge."
    "{vocab}\n\nText:\n{text}"
)


def _vocab_block(existing_entities: typing.Iterable[str], cap: int) -> str:
    listed = list(existing_entities)[:cap]
    if not listed:
        return ""
    return (
        "\nREUSE these existing canonical entity names when you mean the same "
        f"thing: {', '.join(listed)}"
    )


def extract_triples(
    text: str,
    *,
    client: typing.Any,
    existing_entities: typing.Iterable[str] = (),
    proof: str = "llm_extracted",
    max_triples: int = 10,
    max_entities: int = 60,
) -> list[Triple]:
    """Extract up to ``max_triples`` triples from ``text`` via the LLM client."""
    if client is None or not client.available() or not (text or "").strip():
        return []

    prompt = _PROMPT.format(
        vocab=_vocab_block(existing_entities, max_entities), text=text
    )
    data = client.complete_json(prompt)
    if not isinstance(data, list):
        return []

    triples: list[Triple] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        subj = str(item.get("subject", "")).strip()
        pred = str(item.get("predicate", "")).strip()
        obj = str(item.get("object", "")).strip()
        if subj and pred and obj:
            triples.append((subj, pred, obj, proof))
        if len(triples) >= max_triples:
            break
    return triples
