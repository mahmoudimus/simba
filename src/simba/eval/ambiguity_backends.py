"""Optional external backends for executable ambiguity smoke cases."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib
import shutil
import subprocess
import tempfile
import typing

if typing.TYPE_CHECKING:
    import simba.eval.ambiguity as ambiguity


@dataclasses.dataclass(frozen=True)
class BackendResult:
    answer: dict[str, int]
    evidence_ids: list[str]
    backend: str
    raw_output: str = ""


class BackendUnavailableError(RuntimeError):
    """Raised when an optional external solver backend is not installed."""


class PythonBackend:
    name = "python"

    def evaluate(
        self,
        case: ambiguity.AmbiguityCase,
        interp: ambiguity.Interpretation,
    ) -> BackendResult:
        import simba.eval.ambiguity as ambiguity

        answer, evidence_ids = ambiguity.evaluate_interpretation_python(case, interp)
        return BackendResult(
            answer=answer,
            evidence_ids=evidence_ids,
            backend=self.name,
        )


class SouffleBackend:
    """Run a tiny relation-materialization program through Souffle.

    The Python ambiguity semantics still decide which records are lower/upper
    evidence for an interpretation. Souffle is used as an external relational
    checker over that evidence relation, which is enough for an off-by-default
    backend smoke without making the main eval path depend on Souffle syntax.
    """

    name = "souffle"

    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable or shutil.which("souffle") or ""

    def evaluate(
        self,
        case: ambiguity.AmbiguityCase,
        interp: ambiguity.Interpretation,
    ) -> BackendResult:
        if not self.executable:
            raise BackendUnavailableError("souffle executable not found")

        import simba.eval.ambiguity as ambiguity

        answer, evidence_ids = ambiguity.evaluate_interpretation_python(case, interp)
        lower_ids, upper_ids = _evidence_bounds(answer, evidence_ids)
        with tempfile.TemporaryDirectory(prefix="simba-ambiguity-souffle-") as tmp:
            root = pathlib.Path(tmp)
            facts = root / "facts"
            out = root / "out"
            facts.mkdir()
            out.mkdir()
            _write_facts(facts / "lower.facts", lower_ids)
            _write_facts(facts / "upper.facts", upper_ids)
            program = root / "ambiguity.dl"
            program.write_text(_SOUFFLE_PROGRAM, encoding="utf-8")
            proc = subprocess.run(
                [self.executable, "-F", str(facts), "-D", str(out), str(program)],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "souffle ambiguity backend failed: "
                    f"{proc.stderr.strip() or proc.stdout.strip()}"
                )
            lower_count = _count_output_rows(out / "lower_out.csv")
            upper_count = _count_output_rows(out / "upper_out.csv")
            upper_evidence = _read_output_ids(out / "upper_out.csv")
        return BackendResult(
            answer=_answer_from_bounds(lower_count, upper_count),
            evidence_ids=upper_evidence,
            backend=self.name,
            raw_output=proc.stdout + proc.stderr,
        )


class ClingoBackend:
    """Run a tiny ASP count program through pyclingo or the clingo CLI."""

    name = "clingo"

    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable or shutil.which("clingo") or ""
        self.module_available = importlib.util.find_spec("clingo") is not None

    def evaluate(
        self,
        case: ambiguity.AmbiguityCase,
        interp: ambiguity.Interpretation,
    ) -> BackendResult:
        if not self.module_available and not self.executable:
            raise BackendUnavailableError(
                "clingo Python module or executable not found"
            )

        import simba.eval.ambiguity as ambiguity

        answer, evidence_ids = ambiguity.evaluate_interpretation_python(case, interp)
        lower_ids, upper_ids = _evidence_bounds(answer, evidence_ids)
        source = _clingo_program(lower_ids, upper_ids)
        if self.module_available:
            lower_count, upper_count, raw_output = _run_clingo_module(source)
            return BackendResult(
                answer=_answer_from_bounds(lower_count, upper_count),
                evidence_ids=upper_ids,
                backend=self.name,
                raw_output=raw_output,
            )
        proc = subprocess.run(
            [self.executable, "--outf=2", "-"],
            input=source,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "clingo ambiguity backend failed: "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        lower_count, upper_count = _parse_clingo_counts(proc.stdout)
        return BackendResult(
            answer=_answer_from_bounds(lower_count, upper_count),
            evidence_ids=upper_ids,
            backend=self.name,
            raw_output=proc.stdout,
        )


def resolve_backend(name: str) -> PythonBackend | SouffleBackend | ClingoBackend:
    normalized = name.strip().lower()
    if normalized in {"", "python"}:
        return PythonBackend()
    if normalized == "souffle":
        return SouffleBackend()
    if normalized == "clingo":
        return ClingoBackend()
    raise ValueError(f"unknown ambiguity backend: {name!r}")


def backend_available(name: str) -> bool:
    try:
        backend = resolve_backend(name)
    except ValueError:
        return False
    if isinstance(backend, PythonBackend):
        return True
    return bool(
        getattr(backend, "executable", "")
        or getattr(backend, "module_available", False)
    )


def _evidence_bounds(
    answer: dict[str, int], evidence_ids: list[str]
) -> tuple[list[str], list[str]]:
    if "count" in answer:
        ids = _ids_for_count("evidence", int(answer["count"]), evidence_ids)
        return ids, ids
    lower = int(answer["lower"])
    upper = int(answer["upper"])
    upper_ids = _ids_for_count("upper", upper, evidence_ids)
    return upper_ids[:lower], upper_ids


def _ids_for_count(prefix: str, count: int, ids: list[str]) -> list[str]:
    if len(ids) >= count:
        return ids[:count]
    out = list(ids)
    for idx in range(len(out), count):
        out.append(f"{prefix}_{idx}")
    return out


def _write_facts(path: pathlib.Path, ids: list[str]) -> None:
    text = "".join(f"{_escape_fact(item)}\n" for item in ids)
    path.write_text(text, encoding="utf-8")


def _escape_fact(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _count_output_rows(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line])


def _read_output_ids(path: pathlib.Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _answer_from_bounds(lower: int, upper: int) -> dict[str, int]:
    if lower == upper:
        return {"count": lower}
    return {"lower": lower, "upper": upper}


def _clingo_program(lower_ids: list[str], upper_ids: list[str]) -> str:
    facts = []
    for item in lower_ids:
        facts.append(f'lower("{_escape_fact(item)}").')
    for item in upper_ids:
        facts.append(f'upper("{_escape_fact(item)}").')
    facts.extend(
        [
            "#defined lower/1.",
            "#defined upper/1.",
            "answer_lower(N) :- N = #count { I : lower(I) }.",
            "answer_upper(N) :- N = #count { I : upper(I) }.",
            "#show answer_lower/1.",
            "#show answer_upper/1.",
        ]
    )
    return "\n".join(facts) + "\n"


def _parse_clingo_counts(raw: str) -> tuple[int, int]:
    payload = json.loads(raw)
    witnesses = payload.get("Call", [{}])[0].get("Witnesses", [])
    atoms = witnesses[0].get("Value", []) if witnesses else []
    return _parse_clingo_atoms(atoms)


def _run_clingo_module(source: str) -> tuple[int, int, str]:
    import clingo

    ctl = clingo.Control(["--models=1"])
    ctl.add("base", [], source)
    ctl.ground([("base", [])])
    atoms: list[str] = []

    def _on_model(model: typing.Any) -> None:
        atoms.extend(str(symbol) for symbol in model.symbols(shown=True))

    result = ctl.solve(on_model=_on_model)
    if not result.satisfiable:
        raise RuntimeError("clingo ambiguity backend found no satisfying model")
    lower, upper = _parse_clingo_atoms(atoms)
    return lower, upper, "\n".join(atoms)


def _parse_clingo_atoms(atoms: list[str]) -> tuple[int, int]:
    lower = upper = 0
    for atom in atoms:
        if atom.startswith("answer_lower(") and atom.endswith(")"):
            lower = int(atom.removeprefix("answer_lower(").removesuffix(")"))
        if atom.startswith("answer_upper(") and atom.endswith(")"):
            upper = int(atom.removeprefix("answer_upper(").removesuffix(")"))
    return lower, upper


_SOUFFLE_PROGRAM = """
.decl lower(id:symbol)
.decl upper(id:symbol)
.input lower
.input upper

.decl lower_out(id:symbol)
.decl upper_out(id:symbol)
.output lower_out
.output upper_out

lower_out(id) :- lower(id).
upper_out(id) :- upper(id).
"""
