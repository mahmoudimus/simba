"""Tests for extraction-in-the-loop: digesting a raw-turn corpus into the
memories simba's real learn-from-chat path would store, so the eval measures
the digested product instead of raw turns."""

from __future__ import annotations

import simba.eval.digest_corpus as dc
from simba.eval.dataset import Dataset, EvalCase, Memory


class _Client:
    def __init__(self, fn) -> None:
        self._fn = fn

    def complete(self, prompt: str) -> str:
        return self._fn(prompt)


def _ds() -> Dataset:
    corpus = [
        Memory(id="s1#0", content="user: I ran a 5K in 24:10", type="PATTERN"),
        Memory(id="s1#1", content="assistant: nice PB!", type="PATTERN"),
        Memory(id="s2#0", content="user: I tried Mapo Korean", type="PATTERN"),
    ]
    case = EvalCase(
        id="q1",
        query="What's my 5K PB?",
        relevant_ids=["s1#0"],
        intent="single-session-preference",
        answer="24:10",
    )
    return Dataset(name="q1", corpus=corpus, cases=[case])


def test_digest_dataset_replaces_corpus_with_extracted_memories():
    ds = _ds()
    prompts = []

    pb = '[{"type":"PREFERENCE","content":"User 5K PB is 24:10","context":"ran"}]'
    food = '[{"type":"PATTERN","content":"User tried Mapo Korean","context":"x"}]'

    def fake(prompt):
        prompts.append(prompt)
        return pb if "5K" in prompt else food

    out = dc.digest_dataset(
        ds, client=_Client(fake), digest_prompt="DIGEST:\n{transcript}"
    )

    # cases + name preserved so the QA layer runs unchanged
    assert out.cases == ds.cases
    assert out.name == ds.name
    # corpus is now digested memories, not raw turns
    contents = [m.content for m in out.corpus]
    assert "User 5K PB is 24:10" in contents
    assert "User tried Mapo Korean" in contents
    # ids are new + unique, NOT turn-level ids (recall@k is undefined here)
    ids = [m.id for m in out.corpus]
    assert len(ids) == len(set(ids))
    assert all("#" not in i for i in ids)
    # one completion per session, transcript carries the turns in order
    assert len(prompts) == 2
    s1 = next(p for p in prompts if "5K" in p)
    assert "I ran a 5K in 24:10" in s1 and "nice PB!" in s1
    assert s1.index("I ran a 5K") < s1.index("nice PB!")
    # parsed type/context preserved
    pref = next(m for m in out.corpus if m.content == "User 5K PB is 24:10")
    assert pref.type == "PREFERENCE" and pref.context == "ran"


def test_digest_dataset_failopen_on_bad_reply():
    corpus = [
        Memory(id="s1#0", content="user: hi", type="PATTERN"),
        Memory(id="s2#0", content="user: I like deepseek", type="PATTERN"),
    ]
    ds = Dataset(
        name="q",
        corpus=corpus,
        cases=[EvalCase(id="q", query="?", relevant_ids=[], answer="")],
    )

    pref = '[{"type":"PREFERENCE","content":"likes deepseek"}]'

    def fake(prompt):
        return "" if "hi" in prompt else pref

    out = dc.digest_dataset(ds, client=_Client(fake), digest_prompt="{transcript}")
    assert [m.content for m in out.corpus] == ["likes deepseek"]


def test_digest_dataset_default_prompt_when_unset():
    ds = _ds()
    prompts = []

    def fake(prompt):
        prompts.append(prompt)
        return "[]"

    # no digest_prompt -> uses a built-in personal-fact default that still
    # interpolates {transcript} (checked across all per-session prompts)
    dc.digest_dataset(ds, client=_Client(fake))
    assert any("I ran a 5K in 24:10" in p for p in prompts)
    assert any("personal facts" in p.lower() for p in prompts)
