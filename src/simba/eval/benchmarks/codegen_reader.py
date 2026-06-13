"""Temporal codegen reader — the temporal lever of the 0.823 LME-S stack.

Instead of asking the LLM to answer a date-arithmetic question directly, this
route asks it to *write Python* (with ``datetime`` / ``relativedelta`` /
``weekRange`` pre-bound — TReMu's API), execs that code in a restricted
subprocess sandbox, then finalizes the answer from the program output. The
explicit arithmetic recovers temporal-reasoning accuracy the free-form reader
loses.

EVAL fidelity, not a daemon path: in production the reader is the host LLM /
Claude, which can run code itself. This makes ``simba eval bench longmemeval
--qa`` measure that shipped reader protocol. It is opt-in via
``bench.temporal_codegen``.

Sandbox: the child harness execs the model's code with an allow-listed import
hook (``datetime`` / ``calendar`` / ``math`` / ``json`` / ``re`` / ``dateutil``
only) and a limited ``__builtins__`` (no ``open``, ``eval``, ``__import__`` of
anything else). The parent enforces a 5s wall-clock timeout. Any failure mode
(bad code, blocked import, timeout, empty output, empty finalize) falls back to
the caller-supplied direct reader, so a case is never errored by this route.

Ported verbatim (prompts + sandbox) from the lme_stacked_final probe scripts.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import typing

# Codegen sandbox wall-clock seconds (TReMu mech-2). Module constant rather than
# a config field: it is an internal safety bound on a sandboxed subprocess, not
# a behavioral knob a bench operator tunes — the whole codegen route is gated by
# the ``bench.temporal_codegen`` config field.
CODE_TIMEOUT = 5


# Child harness: reads the model's code on stdin, execs it with an allow-listed
# import hook + limited builtins. datetime / relativedelta / weekRange are
# pre-bound (TReMu's API). Unhandled exceptions -> traceback on stderr + nonzero
# exit; the parent enforces the 5s wall-clock timeout.
CHILD_HARNESS = r"""
import sys, datetime, calendar, math, json, re
import dateutil, dateutil.parser
from dateutil.relativedelta import relativedelta

def weekRange(d):
    '''Return (monday, sunday) date objects for the week containing d.
    d may be a datetime.date, datetime.datetime, or "YYYY-MM-DD" string.'''
    if isinstance(d, str):
        d = dateutil.parser.parse(d.replace("/", "-")).date()
    if isinstance(d, datetime.datetime):
        d = d.date()
    monday = d - datetime.timedelta(days=d.weekday())
    return monday, monday + datetime.timedelta(days=6)

_ALLOWED_ROOTS = {"datetime", "calendar", "math", "json", "re", "dateutil"}

def _imp(name, *a, **k):
    if name.split(".")[0] not in _ALLOWED_ROOTS:
        raise ImportError("import of %r is not allowed" % name)
    return __import__(name, *a, **k)

import builtins as _b
_SAFE = {n: getattr(_b, n) for n in (
    "print", "range", "len", "sorted", "min", "max", "sum", "abs",
    "enumerate", "zip", "map", "filter", "str", "int", "float", "bool",
    "list", "dict", "set", "tuple", "round", "divmod", "isinstance",
    "repr", "any", "all", "reversed", "next", "iter", "type", "format",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "ZeroDivisionError", "StopIteration", "ArithmeticError",
    "AttributeError", "NameError", "ImportError",
)}
_SAFE["__import__"] = _imp
ns = {"__builtins__": _SAFE, "__name__": "__main__",
      "datetime": datetime, "calendar": calendar, "math": math,
      "json": json, "re": re, "dateutil": dateutil,
      "relativedelta": relativedelta, "weekRange": weekRange}
code = sys.stdin.read()
exec(compile(code, "<temporal-code>", "exec"), ns)
"""


def run_code(code: str) -> dict[str, typing.Any]:
    """Exec model code in the restricted child; return ok/stdout/stderr."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", CHILD_HARNESS],
            input=code,
            capture_output=True,
            text=True,
            timeout=CODE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": f"timeout (> {CODE_TIMEOUT}s)"}
    except Exception as e:  # the route must never crash on a bad exec
        return {"ok": False, "stdout": "", "stderr": f"harness error: {e}"}
    return {
        "ok": proc.returncode == 0,
        "stdout": (proc.stdout or "").strip()[:2000],
        "stderr": (proc.stderr or "").strip()[-1000:],
    }


_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the python block out of the model reply ('' = extraction failure)."""
    text = text or ""
    m = _FENCE.search(text)
    candidate = m.group(1) if m else text
    candidate = candidate.strip()
    if not candidate:
        return ""
    try:
        ast.parse(candidate)
    except SyntaxError:
        return ""
    return candidate


EXEMPLARS = """Example 1:
Memories:
- [2023/05/20 (Sat)] user: I adopted my puppy two weeks ago.
Current date: 2023/05/30
Question: How many days ago did the user adopt their puppy?
```python
mention = datetime.date(2023, 5, 20)
adoption = mention - datetime.timedelta(weeks=2)
print("adoption date:", adoption)
today = datetime.date(2023, 5, 30)
print("days ago:", (today - adoption).days)
print("elapsed:", relativedelta(today, adoption).days, "days,",
      relativedelta(today, adoption).months, "months")
```

Example 2:
Memories:
- [2023/04/03 (Mon)] user: My dentist appointment is this Friday.
Current date: 2023/04/20
Question: During which week (Monday to Sunday) was the user's dentist appointment?
```python
mention = datetime.date(2023, 4, 3)
appointment = mention + datetime.timedelta(days=(4 - mention.weekday()) % 7)
print("appointment:", appointment)
start, end = weekRange(appointment)
print("week:", start, "to", end)
```"""


def build_codegen_prompt(question: str, contexts: list[str], qdate: str) -> str:
    """Ask the model to write date-arithmetic Python over the memories."""
    memories = "\n".join(f"- {c}" for c in contexts) or "(no context)"
    return (
        "You answer questions about a user's chat history by writing Python "
        "code that does the date arithmetic explicitly.\n\n"
        "Already defined for you (do not redefine): the `datetime` module, "
        "`relativedelta` (dateutil.relativedelta.relativedelta), and "
        "`weekRange(d)` which returns (monday, sunday) date objects for the "
        "week containing d; d may be a datetime.date, a datetime.datetime, or "
        "a 'YYYY-MM-DD' string. You may also import datetime, calendar, math, "
        "re, json, or dateutil. No other imports, no file or network access.\n\n"
        "Write ONE short Python script that computes the answer from the "
        "memories below. print() every intermediate result (resolved absolute "
        "dates, deltas) and then the final result. Note each memory is tagged "
        'with the date it was recorded — resolve relative phrases ("last '
        'Saturday", "two weeks ago") against that memory\'s date, and '
        'resolve "now"/"ago" phrasing in the question against the current '
        "date. Reply with ONLY a fenced ```python code block.\n\n"
        f"{EXEMPLARS}\n\n"
        "Now the real task.\n"
        f"Memories:\n{memories}\n"
        f"Current date: {qdate or '(unknown)'}\n"
        f"Question: {question}\n"
    )


def build_finalize_prompt(question: str, qdate: str, code: str, stdout: str) -> str:
    """Ask the model for the final answer given the executed code's output."""
    return (
        "You wrote Python code to answer a question about a user's chat "
        "history; the code was executed and its output is below. Give the "
        "final answer, concise, based primarily on the program output. If the "
        "output is inconclusive, reason from the code's logic. If the answer "
        "is genuinely not derivable, say you don't know.\n\n"
        f"Question: {question}\n"
        f"Current date: {qdate or '(unknown)'}\n"
        f"Code:\n```python\n{code}\n```\n"
        f"Program output:\n{stdout or '(no output)'}\n\n"
        "Final answer:"
    )


def answer_via_codegen(
    question: str,
    contexts: list[str],
    qdate: str,
    answerer: typing.Any,
    *,
    fallback: typing.Callable[[], str] | None = None,
) -> tuple[str, str]:
    """Codegen -> sandbox exec -> finalize. Returns ``(prediction, route)``.

    ``route`` is ``"codegen"`` on the happy path or ``"codegen-fallback"`` when
    any failure mode fires (code extraction failed, exec failed/blocked, empty
    stdout, or an empty finalize answer). On fallback the ``fallback`` callable
    (the caller's direct reader) is invoked for the prediction; if no fallback
    is given the prediction is "".
    """
    reply = str(
        answerer.complete(build_codegen_prompt(question, contexts, qdate)) or ""
    )
    code = extract_code(reply)
    fail = False
    stdout = ""
    if not code:
        fail = True
    else:
        res = run_code(code)
        stdout = res["stdout"]
        if not res["ok"] or not stdout:
            fail = True

    if not fail:
        finalize = str(
            answerer.complete(build_finalize_prompt(question, qdate, code, stdout))
            or ""
        )
        if finalize.strip():
            return finalize, "codegen"
        # empty finalize answer -> fall back

    pred = fallback() if fallback is not None else ""
    return pred, "codegen-fallback"
