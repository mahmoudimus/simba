"""Tests for building an eval dataset from real memories (LLM-generated queries)."""

from __future__ import annotations

import simba.eval.build as build


class FakeClient:
    def __init__(self, replies):
        self._replies = list(replies)
        self.available_ = True
        self.prompts = []

    def available(self):
        return self.available_

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self._replies.pop(0) if self._replies else ""


_MEMS = [
    {
        "id": "m1",
        "type": "GOTCHA",
        "content": "stale GITHUB_TOKEN makes gh 401",
        "context": "env",
    },
    {
        "id": "m2",
        "type": "PATTERN",
        "content": "RRF fuses vector + bm25",
        "context": "",
    },
    {
        "id": "m3",
        "type": "DECISION",
        "content": "peewee vendored at 9bd107f",
        "context": "",
    },
]


def test_build_makes_corpus_and_gold_cases() -> None:
    client = FakeClient(
        ["why does gh return 401?", "how does hybrid recall fuse?", "where is peewee?"]
    )
    d = build.build_from_memories(_MEMS, client=client, name="rc")
    assert d.name == "rc"
    assert d.corpus_ids() == {"m1", "m2", "m3"}
    # one case per memory, gold = the source memory
    golds = {c.relevant_ids[0] for c in d.cases}
    assert golds == {"m1", "m2", "m3"}
    q_by_gold = {c.relevant_ids[0]: c.query for c in d.cases}
    assert q_by_gold["m1"] == "why does gh return 401?"


def test_memory_content_in_query_prompt() -> None:
    client = FakeClient(["q1"])
    build.build_from_memories(_MEMS[:1], client=client)
    assert "stale GITHUB_TOKEN" in client.prompts[0]


def test_skips_memories_with_empty_generated_query() -> None:
    client = FakeClient(["", "good question", ""])  # only m2 yields a query
    d = build.build_from_memories(_MEMS, client=client)
    assert {c.relevant_ids[0] for c in d.cases} == {"m2"}
    assert d.corpus_ids() == {"m1", "m2", "m3"}  # full corpus kept as distractors


def test_max_cases_caps_queries() -> None:
    client = FakeClient(["a", "b", "c"])
    d = build.build_from_memories(_MEMS, client=client, max_cases=1)
    assert len(d.cases) == 1
    assert len(client.prompts) == 1  # didn't query beyond the cap


def test_unavailable_client_yields_no_cases() -> None:
    client = FakeClient([])
    client.available_ = False
    d = build.build_from_memories(_MEMS, client=client)
    assert d.cases == []
    assert d.corpus_ids() == {"m1", "m2", "m3"}
