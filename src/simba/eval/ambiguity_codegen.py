"""LLM-codegen runner for executable ambiguity cases.

The hand-written ambiguity evaluator is the oracle. This module is the thing we
actually want to measure: ask an LLM to write executable code that preserves an
answer space, then run that generated artifact and compare it to the oracle.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import typing

import simba.eval.ambiguity as ambiguity
from simba.eval.ambiguity_backends import BackendUnavailableError

CODE_TIMEOUT = 5

_LANGUAGES = frozenset({"python", "souffle", "clingo"})
_FENCE = re.compile(
    r"```(?:python|souffle|prolog|asp|clingo|datalog)?\s*\n(.*?)```",
    re.DOTALL,
)


@dataclasses.dataclass(frozen=True)
class GeneratedProgram:
    case_id: str
    language: str
    code: str
    source: str = "llm"
    model: str = ""

    def to_dict(self) -> dict[str, typing.Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class GeneratedRun:
    case_id: str
    language: str
    answer_space: ambiguity.Answer
    ok: bool
    stdout: str = ""
    stderr: str = ""
    provenance: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, typing.Any]:
        return dataclasses.asdict(self)


class CodegenError(RuntimeError):
    """Raised when a generated ambiguity program cannot be produced or run."""


def build_codegen_prompt(case: ambiguity.AmbiguityCase, language: str) -> str:
    """Build the LLM prompt for one ambiguity case and target language."""
    lang = _normalize_language(language)
    payload = _case_payload(case)
    contract = _language_contract(lang)
    return (
        "You are writing executable code for an ambiguity-preserving structured "
        "data question. Do not collapse vague language to one interpretation. "
        "Represent lower/upper bounds, alternative interpretations, and record "
        "provenance explicitly.\n\n"
        f"Target language: {lang}\n"
        f"{contract}\n\n"
        "Input case JSON:\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n\n"
        "Return only one fenced code block for the target language."
    )


def generate_program(
    case: ambiguity.AmbiguityCase,
    *,
    language: str = "python",
    client: typing.Any | None = None,
) -> GeneratedProgram:
    """Ask the configured LLM client to write a runnable ambiguity program."""
    lang = _normalize_language(language)
    if client is None:
        import simba.llm.client

        client = simba.llm.client.get_client()
    reply = str(client.complete(build_codegen_prompt(case, lang)) or "")
    code = extract_code(reply, lang)
    if not code:
        raise CodegenError(f"LLM did not return valid {lang} code for {case.id}")
    model = str(getattr(getattr(client, "_cfg", object()), "model", ""))
    return GeneratedProgram(
        case_id=case.id,
        language=lang,
        code=code,
        model=model,
    )


def extract_code(reply: str, language: str) -> str:
    """Extract a generated code block from an LLM response."""
    lang = _normalize_language(language)
    text = reply or ""
    match = _FENCE.search(text)
    code = (match.group(1) if match else text).strip()
    if not code:
        return ""
    if lang == "python":
        try:
            ast.parse(code)
        except SyntaxError:
            return ""
    return code


def run_generated_program(
    case: ambiguity.AmbiguityCase,
    program: GeneratedProgram,
) -> GeneratedRun:
    """Execute a generated artifact and return its answer space."""
    lang = _normalize_language(program.language)
    if lang == "python":
        return run_generated_python(case, program.code)
    if lang == "souffle":
        return run_generated_souffle(case, program.code)
    if lang == "clingo":
        return run_generated_clingo(case, program.code)
    raise AssertionError(lang)


def generate_and_run(
    case: ambiguity.AmbiguityCase,
    *,
    language: str = "python",
    client: typing.Any | None = None,
) -> tuple[GeneratedProgram, GeneratedRun]:
    program = generate_program(case, language=language, client=client)
    return program, run_generated_program(case, program)


def run_generated_python(case: ambiguity.AmbiguityCase, code: str) -> GeneratedRun:
    """Run generated Python in a restricted child process."""
    envelope = {"case": _case_payload(case), "code": code}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PYTHON_CHILD_HARNESS],
            input=json.dumps(envelope),
            capture_output=True,
            text=True,
            timeout=CODE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return GeneratedRun(
            case_id=case.id,
            language="python",
            answer_space={"lower": 0, "upper": 0},
            ok=False,
            stderr=f"timeout (> {CODE_TIMEOUT}s)",
        )
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return GeneratedRun(
            case_id=case.id,
            language="python",
            answer_space={"lower": 0, "upper": 0},
            ok=False,
            stdout=stdout,
            stderr=stderr,
        )
    try:
        payload = _parse_generated_json(stdout)
        answer = _normalize_answer_space(payload["answer_space"])
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return GeneratedRun(
            case_id=case.id,
            language="python",
            answer_space={"lower": 0, "upper": 0},
            ok=False,
            stdout=stdout,
            stderr=f"invalid generated output: {exc}",
        )
    return GeneratedRun(
        case_id=case.id,
        language="python",
        answer_space=answer,
        ok=True,
        stdout=stdout,
        stderr=stderr,
        provenance=[str(item) for item in payload.get("provenance", [])],
    )


def run_generated_souffle(case: ambiguity.AmbiguityCase, code: str) -> GeneratedRun:
    """Run generated Souffle over a generic field(id,key,value) fact table."""
    executable = shutil.which("souffle") or ""
    if not executable:
        raise BackendUnavailableError("souffle executable not found")
    with tempfile.TemporaryDirectory(prefix="simba-ambiguity-codegen-souffle-") as tmp:
        root = pathlib.Path(tmp)
        facts = root / "facts"
        out = root / "out"
        facts.mkdir()
        out.mkdir()
        _write_field_facts(facts / "field.facts", case.records)
        program = root / "generated.dl"
        program.write_text(code, encoding="utf-8")
        proc = subprocess.run(
            [executable, "-F", str(facts), "-D", str(out), str(program)],
            capture_output=True,
            text=True,
            timeout=CODE_TIMEOUT,
            check=False,
        )
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        if proc.returncode != 0:
            return GeneratedRun(
                case_id=case.id,
                language="souffle",
                answer_space={"lower": 0, "upper": 0},
                ok=False,
                stdout=stdout,
                stderr=stderr,
            )
        try:
            answer = _parse_answer_space_csv(out / "answer_space.csv")
        except (OSError, ValueError) as exc:
            return GeneratedRun(
                case_id=case.id,
                language="souffle",
                answer_space={"lower": 0, "upper": 0},
                ok=False,
                stdout=stdout,
                stderr=f"invalid answer_space output: {exc}",
            )
    return GeneratedRun(
        case_id=case.id,
        language="souffle",
        answer_space=answer,
        ok=True,
        stdout=stdout,
        stderr=stderr,
    )


def run_generated_clingo(case: ambiguity.AmbiguityCase, code: str) -> GeneratedRun:
    """Run generated Clingo over generic field/3 facts."""
    executable = shutil.which("clingo") or ""
    if not executable:
        raise BackendUnavailableError("clingo executable not found")
    source = _clingo_facts(case.records) + "\n" + code
    proc = subprocess.run(
        [executable, "--outf=2", "-"],
        input=source,
        capture_output=True,
        text=True,
        timeout=CODE_TIMEOUT,
        check=False,
    )
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return GeneratedRun(
            case_id=case.id,
            language="clingo",
            answer_space={"lower": 0, "upper": 0},
            ok=False,
            stdout=stdout,
            stderr=stderr,
        )
    try:
        answer = _parse_clingo_answer_space(stdout)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return GeneratedRun(
            case_id=case.id,
            language="clingo",
            answer_space={"lower": 0, "upper": 0},
            ok=False,
            stdout=stdout,
            stderr=f"invalid answer_space output: {exc}",
        )
    return GeneratedRun(
        case_id=case.id,
        language="clingo",
        answer_space=answer,
        ok=True,
        stdout=stdout,
        stderr=stderr,
    )


def save_program(program: GeneratedProgram, root: str | pathlib.Path) -> pathlib.Path:
    """Write a generated program artifact for audit/replay."""
    out = pathlib.Path(root)
    out.mkdir(parents=True, exist_ok=True)
    suffix = {"python": ".py", "souffle": ".dl", "clingo": ".lp"}[program.language]
    path = out / f"{program.case_id}{suffix}"
    path.write_text(program.code, encoding="utf-8")
    meta = out / f"{program.case_id}.json"
    meta.write_text(json.dumps(program.to_dict(), indent=2), encoding="utf-8")
    return path


def _normalize_language(language: str) -> str:
    lang = language.strip().lower()
    if lang not in _LANGUAGES:
        raise ValueError(f"unknown generated ambiguity language: {language!r}")
    return lang


def _case_payload(case: ambiguity.AmbiguityCase) -> dict[str, typing.Any]:
    return {
        "id": case.id,
        "category": case.category,
        "question": case.question,
        "records": case.records,
    }


def _language_contract(language: str) -> str:
    if language == "python":
        return (
            "Python contract: INPUT, QUESTION, and RECORDS are pre-bound. Write a "
            "short script. It must set ANSWER_SPACE to either {'count': N} or "
            "{'lower': L, 'upper': U}. Optionally set PROVENANCE to record ids. "
            "Imports are limited to json, math, datetime, calendar, re, and "
            "statistics. No file or network access."
        )
    if language == "souffle":
        return (
            "Souffle contract: facts are provided as field(id:symbol, key:symbol, "
            "value:symbol). Write a complete .dl program with `.input field` and "
            "an `.output answer_space` relation declared as "
            "answer_space(kind:symbol, value:number), emitting lower/upper or "
            "count rows."
        )
    return (
        "Clingo contract: facts are provided as field(Id, Key, Value). Write an "
        "ASP program that emits answer_space(\"lower\", L) and "
        "answer_space(\"upper\", U), or answer_space(\"count\", N), and includes "
        "#show answer_space/2."
    )


def _normalize_answer_space(raw: typing.Any) -> ambiguity.Answer:
    answer = ambiguity._normalize_answer(raw)
    if "count" in answer:
        count = int(answer["count"])
        return {"lower": count, "upper": count}
    return answer


def _parse_generated_json(stdout: str) -> dict[str, typing.Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            payload = json.loads(line)
            if isinstance(payload, dict):
                return payload
    raise ValueError("no JSON object printed")


def _write_field_facts(
    path: pathlib.Path, records: list[dict[str, typing.Any]]
) -> None:
    lines: list[str] = []
    for record in records:
        rid = _fact_symbol(str(record.get("id", "")))
        for key, value in record.items():
            if key == "id":
                continue
            lines.append(f"{rid}\t{_fact_symbol(str(key))}\t{_fact_symbol(str(value))}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _fact_symbol(value: str) -> str:
    return value.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def _parse_answer_space_csv(path: pathlib.Path) -> ambiguity.Answer:
    rows = [
        line.split("\t")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw: dict[str, int] = {}
    for row in rows:
        if len(row) != 2:
            raise ValueError(f"bad row: {row!r}")
        raw[row[0].strip('"')] = int(row[1])
    if "count" in raw:
        return {"lower": raw["count"], "upper": raw["count"]}
    return ambiguity._normalize_answer(raw)


def _clingo_facts(records: list[dict[str, typing.Any]]) -> str:
    facts: list[str] = []
    for record in records:
        rid = _clingo_string(str(record.get("id", "")))
        for key, value in record.items():
            if key == "id":
                continue
            facts.append(
                f"field({rid},{_clingo_string(str(key))},{_clingo_string(str(value))})."
            )
    return "\n".join(facts) + ("\n" if facts else "")


def _clingo_string(value: str) -> str:
    return json.dumps(value)


def _parse_clingo_answer_space(raw: str) -> ambiguity.Answer:
    payload = json.loads(raw)
    witnesses = payload.get("Call", [{}])[0].get("Witnesses", [])
    atoms = witnesses[0].get("Value", []) if witnesses else []
    parsed: dict[str, int] = {}
    for atom in atoms:
        match = re.fullmatch(r'answer_space\("([^"]+)",(-?\d+)\)', atom)
        if match:
            parsed[match.group(1)] = int(match.group(2))
    if "count" in parsed:
        return {"lower": parsed["count"], "upper": parsed["count"]}
    return ambiguity._normalize_answer(parsed)


_PYTHON_CHILD_HARNESS = r"""
import calendar
import datetime
import json
import math
import re
import statistics
import sys

envelope = json.loads(sys.stdin.read())
code = envelope["code"]
INPUT = envelope["case"]
QUESTION = INPUT["question"]
RECORDS = INPUT["records"]

_ALLOWED_ROOTS = {"calendar", "datetime", "json", "math", "re", "statistics"}

def _imp(name, *a, **k):
    if name.split(".")[0] not in _ALLOWED_ROOTS:
        raise ImportError("import of %r is not allowed" % name)
    return __import__(name, *a, **k)

import builtins as _b
_SAFE = {n: getattr(_b, n) for n in (
    "print", "range", "len", "sorted", "min", "max", "sum", "abs",
    "enumerate", "zip", "map", "filter", "str", "int", "float", "bool",
    "list", "dict", "set", "tuple", "round", "isinstance", "repr",
    "any", "all", "next", "iter", "type", "Exception", "ValueError",
    "TypeError", "KeyError", "IndexError", "ImportError", "AttributeError",
)}
_SAFE["__import__"] = _imp
ns = {
    "__builtins__": _SAFE,
    "__name__": "__main__",
    "INPUT": INPUT,
    "QUESTION": QUESTION,
    "RECORDS": RECORDS,
    "json": json,
    "math": math,
    "datetime": datetime,
    "calendar": calendar,
    "re": re,
    "statistics": statistics,
}
exec(compile(code, "<ambiguity-generated-python>", "exec"), ns)
if "ANSWER_SPACE" in ns:
    print(json.dumps({
        "answer_space": ns["ANSWER_SPACE"],
        "provenance": ns.get("PROVENANCE", []),
    }))
"""
