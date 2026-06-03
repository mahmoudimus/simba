"""Tests for synchronous LLM triple extraction (client injected, fail-open)."""

from __future__ import annotations

import simba.sync.llm_extract as le


class FakeClient:
    def __init__(self, response, available=True):
        self._response = response
        self._available = available
        self.prompts: list[str] = []

    def available(self):
        return self._available

    def complete_json(self, prompt):
        self.prompts.append(prompt)
        return self._response


def test_extracts_typed_triples() -> None:
    client = FakeClient(
        [
            {
                "subject": "GITHUB_TOKEN",
                "predicate": "causes",
                "object": "401",
                "subject_type": "env_var",
                "object_type": "error",
            }
        ]
    )
    out = le.extract_triples("a stale GITHUB_TOKEN causes a 401", client=client)
    assert out == [("GITHUB_TOKEN", "causes", "401", "llm_extracted")]


def test_prompt_includes_text_and_existing_entities() -> None:
    client = FakeClient([])
    le.extract_triples("some text", client=client, existing_entities=["peewee", "gh"])
    p = client.prompts[0]
    assert "some text" in p
    assert "peewee" in p and "gh" in p
    assert "JSON" in p or "json" in p


def test_skips_incomplete_items() -> None:
    client = FakeClient([{"subject": "a", "predicate": "b"}, {"object": "c"}])
    assert le.extract_triples("t", client=client) == []


def test_caps_triples() -> None:
    many = [{"subject": f"s{i}", "predicate": "p", "object": "o"} for i in range(50)]
    out = le.extract_triples("t", client=FakeClient(many), max_triples=5)
    assert len(out) == 5


def test_unavailable_client_returns_empty() -> None:
    client = FakeClient(
        [{"subject": "a", "predicate": "b", "object": "c"}], available=False
    )
    assert le.extract_triples("t", client=client) == []


def test_none_client_returns_empty() -> None:
    assert le.extract_triples("t", client=None) == []


def test_bad_response_returns_empty() -> None:
    assert le.extract_triples("t", client=FakeClient(None)) == []
