"""Tests for the temporal codegen reader (the 0.823 LME-S temporal lever).

The reader writes Python (datetime / relativedelta / weekRange), execs it in a
restricted sandbox subprocess, then finalizes the answer from the program
output. Any failure mode (bad code, blocked import, timeout, empty output)
falls back to the direct reader so the route never errors a case.

Pure/injectable: a fake LLM client drives the codegen -> exec -> finalize loop
with no live model.
"""

from __future__ import annotations

import simba.eval.benchmarks.codegen_reader as cgr


class _ScriptedLlm:
    """Returns queued replies in order; records prompts for inspection."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._replies.pop(0) if self._replies else ""


# --- sandbox: happy path ----------------------------------------------------


def test_run_code_executes_and_captures_stdout() -> None:
    res = cgr.run_code(
        "d = datetime.date(2023, 4, 7)\n"
        "start, end = weekRange(d)\n"
        "print('week:', start, 'to', end)\n"
        "print('plus one month:', d + relativedelta(months=1))\n"
    )
    assert res["ok"] is True
    assert "week: 2023-04-03 to 2023-04-09" in res["stdout"]
    assert "plus one month: 2023-05-07" in res["stdout"]


def test_run_code_relativedelta_and_weekrange_prebound() -> None:
    # weekRange + relativedelta are injected (TReMu API), not imported.
    res = cgr.run_code(
        "mon, sun = weekRange('2023-04-05')\n"
        "print(mon, sun)\n"
        "d1 = datetime.date(2023, 6, 1)\n"
        "d2 = datetime.date(2023, 4, 1)\n"
        "print(relativedelta(d1, d2).months)\n"
    )
    assert res["ok"] is True
    assert "2023-04-03 2023-04-09" in res["stdout"]
    assert "2" in res["stdout"]


# --- sandbox: blocked import ------------------------------------------------


def test_run_code_blocks_os_import() -> None:
    res = cgr.run_code("import os\nprint(os.listdir('.'))\n")
    assert res["ok"] is False
    assert "not allowed" in res["stderr"]


def test_run_code_blocks_subprocess_import() -> None:
    res = cgr.run_code("import subprocess\nprint('x')\n")
    assert res["ok"] is False
    assert "not allowed" in res["stderr"]


def test_run_code_allows_listed_imports() -> None:
    res = cgr.run_code(
        "import calendar, math\nprint(math.floor(2.7), calendar.isleap(2024))\n"
    )
    assert res["ok"] is True
    assert "2 True" in res["stdout"]


# --- code extraction --------------------------------------------------------


def test_extract_code_pulls_fenced_python_block() -> None:
    reply = "Here you go:\n```python\nprint(1 + 1)\n```\nDone."
    assert cgr.extract_code(reply) == "print(1 + 1)"


def test_extract_code_returns_empty_on_syntax_error() -> None:
    assert cgr.extract_code("```python\ndef (:\n```") == ""


def test_extract_code_returns_empty_on_blank() -> None:
    assert cgr.extract_code("") == ""
    assert cgr.extract_code("   ") == ""


# --- prompts ----------------------------------------------------------------


def test_build_codegen_prompt_has_api_and_memories() -> None:
    p = cgr.build_codegen_prompt(
        "How many days ago?", ["[2023-05-01] bought bike"], "2023-05-30"
    )
    assert "weekRange" in p and "relativedelta" in p
    assert "bought bike" in p
    assert "Current date: 2023-05-30" in p
    assert "How many days ago?" in p


def test_build_finalize_prompt_carries_code_and_output() -> None:
    p = cgr.build_finalize_prompt("Q?", "2023-05-30", "print(29)", "29")
    assert "print(29)" in p
    assert "29" in p
    assert "Q?" in p


# --- full route: answer_via_codegen -----------------------------------------


def test_answer_via_codegen_happy_path_uses_finalize() -> None:
    llm = _ScriptedLlm(
        [
            "```python\nprint('days ago:', 29)\n```",  # codegen
            "29 days",  # finalize
        ]
    )
    pred, route = cgr.answer_via_codegen(
        "How many days ago?", ["[2023-05-01] bought bike"], "2023-05-30", llm
    )
    assert pred == "29 days"
    assert route == "codegen"
    assert len(llm.prompts) == 2


def test_answer_via_codegen_falls_back_on_blocked_import() -> None:
    fallback_calls: list[str] = []

    def fallback() -> str:
        fallback_calls.append("called")
        return "direct answer"

    llm = _ScriptedLlm(["```python\nimport os\nprint(os.getcwd())\n```"])
    pred, route = cgr.answer_via_codegen(
        "Q?", ["ctx"], "2023-05-30", llm, fallback=fallback
    )
    assert pred == "direct answer"
    assert route == "codegen-fallback"
    assert fallback_calls == ["called"]
    # finalize was never called (only the codegen attempt)
    assert len(llm.prompts) == 1


def test_answer_via_codegen_falls_back_on_extraction_failure() -> None:
    llm = _ScriptedLlm(["I cannot write code for this."])
    pred, route = cgr.answer_via_codegen(
        "Q?", ["ctx"], "", llm, fallback=lambda: "direct"
    )
    assert pred == "direct"
    assert route == "codegen-fallback"


def test_answer_via_codegen_falls_back_on_empty_stdout() -> None:
    llm = _ScriptedLlm(["```python\nx = 1 + 1\n```"])  # valid but prints nothing
    pred, route = cgr.answer_via_codegen(
        "Q?", ["ctx"], "", llm, fallback=lambda: "direct"
    )
    assert pred == "direct"
    assert route == "codegen-fallback"


def test_answer_via_codegen_falls_back_on_empty_finalize() -> None:
    llm = _ScriptedLlm(["```python\nprint(42)\n```", "   "])  # finalize blank
    pred, route = cgr.answer_via_codegen(
        "Q?", ["ctx"], "", llm, fallback=lambda: "direct"
    )
    assert pred == "direct"
    assert route == "codegen-fallback"
