"""Extraction-in-the-loop: turn a raw-turn eval corpus into the memories
simba's learn-from-chat path would actually store.

The benchmark loaders embed *raw conversation turns* and never run simba's
digest, so QA numbers measure raw-turn retrieval — not simba's digested
product. ``digest_dataset`` closes that gap: it groups a question's corpus back
into sessions, runs simba's *real* extraction (``rlm.engine._parse_memories``
over a single ``llm -m <model>`` completion) with a configurable prompt, and
returns a Dataset whose corpus is the extracted memories. The cases (query /
answer / intent) are preserved so the QA + judge layer runs unchanged.

Metric note: gold evidence in LongMemEval is turn-level ids, which the digested
memories do not carry — so the digested arm is measured by **QA accuracy**, not
turn-recall@k. Swap ``digest_prompt`` to switch domains (a personal-fact prompt
for LongMemEval/LoCoMo; the coding prompt for coding transcripts) — the harness
is identical either way.
"""

from __future__ import annotations

import typing

import simba.rlm.engine
from simba.eval.dataset import Dataset, EvalCase, Memory

# Personal-assistant default: extract durable personal facts / preferences /
# events / relationships, not coding learnings. Override per run via the
# ``digest_prompt`` argument (e.g. the coding prompt, or an eval'd variant).
_PERSONAL_DIGEST_PROMPT = (
    "Below is a conversation between a user and an assistant. Extract the "
    "durable, specific personal facts worth remembering long-term: the user's "
    "preferences, possessions, plans, events, relationships, and stable "
    "attributes. Resolve pronouns and relative dates to concrete references. "
    'Return ONLY a JSON array; each element an object with keys "type", '
    '"content", "context". "type" is one of PREFERENCE, FACT, DECISION, '
    "PATTERN, GOTCHA (use PREFERENCE for likes/dislikes, PATTERN for recurring "
    'facts). "content" is a self-contained statement of at most 200 '
    'characters; "context" holds supporting detail (e.g. when it was said). '
    "Capture every distinct fact a future question might ask about; skip "
    "small talk. Output nothing but the JSON array.\n\nConversation:\n{transcript}"
)


def _session_id(memory_id: str) -> str:
    return memory_id.split("#", 1)[0]


def _turn_index(memory_id: str) -> int:
    parts = memory_id.split("#", 1)
    if len(parts) < 2:
        return 0
    try:
        return int(parts[1])
    except ValueError:
        return 0


def _group_sessions(corpus: list[Memory]) -> dict[str, list[Memory]]:
    """Group corpus turns back into sessions, preserving first-seen order."""
    sessions: dict[str, list[Memory]] = {}
    for mem in corpus:
        sessions.setdefault(_session_id(mem.id), []).append(mem)
    for turns in sessions.values():
        turns.sort(key=lambda m: _turn_index(m.id))
    return sessions


def digest_dataset(
    dataset: Dataset,
    *,
    client: typing.Any,
    digest_prompt: str = "",
) -> Dataset:
    """Return a copy of ``dataset`` whose corpus is the *digested* memories.

    For each session in the corpus, reconstruct the transcript, run a single
    ``client.complete`` over the (filled) ``digest_prompt``, and parse the reply
    with simba's real ``_parse_memories``. Fail-open per session: a bad/empty
    reply simply contributes no memories. Cases are carried over verbatim."""
    template = digest_prompt or _PERSONAL_DIGEST_PROMPT
    new_corpus: list[Memory] = []
    for sid, turns in _group_sessions(dataset.corpus).items():
        transcript = "\n".join(t.content for t in turns)
        prompt = template.format(transcript=transcript, cwd="", tid=sid)
        reply = client.complete(prompt) or ""
        for j, mem in enumerate(simba.rlm.engine._parse_memories(reply)):
            new_corpus.append(
                Memory(
                    id=f"{sid}::dig{j}",
                    content=mem["content"],
                    type=mem["type"],
                    context=mem["context"],
                )
            )
    cases: list[EvalCase] = list(dataset.cases)
    return Dataset(name=dataset.name, corpus=new_corpus, cases=cases)
