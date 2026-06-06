"""Tests for the LLM HyDE module (hypothetical-answer generation, fail-open)."""

from __future__ import annotations

import simba.memory.hyde as hyde


class FakeLlm:
    def __init__(
        self, text: str = "X", raises: bool = False, avail: bool = True
    ) -> None:
        self._text = text
        self._raises = raises
        self._avail = avail

    def available(self) -> bool:
        return self._avail

    def complete(self, prompt: str) -> str:
        if self._raises:
            raise RuntimeError("boom")
        return self._text


def test_build_hyde_prompt_contains_query() -> None:
    prompt = hyde.build_hyde_prompt("how do I rotate the gh token")
    assert "how do I rotate the gh token" in prompt
    assert "answer" in prompt.lower()


def test_hypothetical_answer_returns_llm_text() -> None:
    assert hyde.hypothetical_answer("q", FakeLlm(text="The answer is X")) == (
        "The answer is X"
    )


def test_hypothetical_answer_fail_open_on_empty() -> None:
    assert hyde.hypothetical_answer("q", FakeLlm(text="   ")) == ""


def test_hypothetical_answer_fail_open_on_exception() -> None:
    assert hyde.hypothetical_answer("q", FakeLlm(raises=True)) == ""


def test_hypothetical_answer_truncates_to_300_chars() -> None:
    assert len(hyde.hypothetical_answer("q", FakeLlm(text="x" * 500))) <= 300


def test_hypothetical_answer_unavailable_client_returns_empty() -> None:
    assert hyde.hypothetical_answer("q", FakeLlm(avail=False)) == ""
