"""Typed IR + deterministic evaluator for individuated counting / sum.

A counting question is ``individuate ∘ aggregate``: decide *what counts as one*
(the equivalence relation + a qualifying filter), then count distinct individuals.
This module owns the *aggregate* half as a thin, total, pure function over a typed
IR — the deterministic contract and the audit trail. The *individuate* half (which
rows qualify, how they group) is produced upstream (by an LLM relation or by
write-time extraction) and handed in as row statuses + group labels.

For INDEPENDENT unary choices the whole possible-worlds semantics is::

    certain  = # distinct individuals with >=1 In member        (a filtered len)
    possible = # distinct individuals not entirely Excluded      (+ the maybes)

so Python owns the execution path. ``clingo`` is kept only as a differential
*checker* (:func:`clingo_check`) and the declarative spec-of-record — it must agree
with the Python evaluator on independent inputs (disagreement = a bug). It becomes
the *real* backend only when the IR carries ``constraints`` (correlated choices —
uncertain identity, mutual exclusion, cardinality — where a ``set()`` won't do).

``clingo`` is imported lazily inside :func:`clingo_check` so this module adds no
hard dependency; the pure evaluators never need it.

Row status:
    ``"In"``        DefinitelyIn  — passed the filter (unanimously / dated in-window)
    ``"Possible"``  PossiblyIn    — uncertain (undated/boundary date, or split vote)
    ``"Excluded"``  DefinitelyOut — filtered out

Identity is ``group_label`` (the dedup / individuation relation): rows with the
same label are the SAME individual. A group is *certain* if it has any In member,
*possible* if it is not entirely Excluded.
"""
from __future__ import annotations

import dataclasses

#: The three row dispositions an individuation step may assign.
STATUSES = ("In", "Possible", "Excluded")


@dataclasses.dataclass
class Row:
    """One evidence row after individuation.

    ``rid`` is a stable id (provenance). ``group_label`` is the individual the row
    maps to under the dedup relation. ``status`` is one of :data:`STATUSES`.
    ``keep_rate`` (0..1) records the filter's confidence (provenance for an audit
    trail). ``value`` is the per-row quantity for ``sum`` aggregation.
    """

    rid: str
    group_label: str
    status: str = "In"
    keep_rate: float = 1.0
    value: float | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(
                f"status must be one of {STATUSES}, got {self.status!r}")


@dataclasses.dataclass
class IR:
    """The typed intermediate representation a count/sum compiles to.

    ``constraints`` is empty for the independent case (the pure evaluators apply);
    a non-empty ``constraints`` slot signals correlated choices that must route to
    a model counter (clingo) instead.
    """

    rows: list[Row]
    intent: str = "count"
    constraints: list = dataclasses.field(default_factory=list)

    def is_independent(self) -> bool:
        return not self.constraints


def _groups(ir: IR) -> dict[str, list[Row]]:
    """Group non-excluded rows by identity (group_label), first-seen order."""
    g: dict[str, list[Row]] = {}
    for r in ir.rows:
        if r.status == "Excluded":
            continue
        g.setdefault(r.group_label, []).append(r)
    return g


def evaluate_count(ir: IR) -> tuple[int, int]:
    """Return ``(certain, possible)`` distinct-individual counts (independent case)."""
    g = _groups(ir)
    certain = sum(1 for rows in g.values() if any(r.status == "In" for r in rows))
    possible = len(g)  # every surviving group has >=1 In-or-Possible member
    return certain, possible


def _group_value(rows: list[Row]) -> float:
    """A group's representative value (the deduped individual contributes once)."""
    for r in rows:
        if r.status == "In" and r.value is not None:
            return float(r.value)
    for r in rows:
        if r.value is not None:
            return float(r.value)
    return 0.0


def evaluate_sum(ir: IR) -> tuple[float, float]:
    """Return ``(certain_sum, possible_sum)`` over distinct individuals."""
    g = _groups(ir)
    certain = sum(_group_value(rows) for rows in g.values()
                  if any(r.status == "In" for r in rows))
    possible = sum(_group_value(rows) for rows in g.values())
    return certain, possible


def evaluate(ir: IR) -> tuple[float, float]:
    """Dispatch to :func:`evaluate_sum` for ``intent == "sum"`` else count."""
    return evaluate_sum(ir) if ir.intent == "sum" else evaluate_count(ir)


def _ir_to_asp(ir: IR) -> str:
    """One rep per group: In -> ``in_scope`` fact; Possible -> a choice rule."""
    lines = []
    for i, (_label, rows) in enumerate(_groups(ir).items()):
        gid = f"g{i}"
        lines.append(f"instance({gid}).")
        if any(r.status == "In" for r in rows):
            lines.append(f"in_scope({gid}).")        # DefinitelyIn (fact)
        else:
            lines.append(f"{{ in_scope({gid}) }}.")   # PossiblyIn (choice)
    return "\n".join(lines) + "\n"


def clingo_check(ir: IR) -> tuple[bool, tuple[int, int]]:
    """Differential checker: does clingo's brave/cautious count match the Python eval?

    POSSIBLE = brave (largest n true in SOME answer set); CERTAIN = cautious (largest
    k with ``atleast(k)`` in ALL answer sets). For independent unary choices this
    equals :func:`evaluate_count`; that equality is the contract. Returns
    ``(agrees, (certain, possible))``. If ``clingo`` is not installed the check is
    vacuously True (the production path never depends on it).
    """
    try:
        import clingo
    except ImportError:
        return True, evaluate_count(ir)

    base = _ir_to_asp(ir) + "match(X) :- instance(X), in_scope(X).\n"

    def brave_max() -> int:
        prog = base + "n(N) :- N = #count{ X : match(X) }.\n#show n/1.\n"
        ctl = clingo.Control(["--models=0"])
        ctl.add("base", [], prog)
        ctl.ground([("base", [])])
        best = 0
        with ctl.solve(yield_=True) as h:
            for m in h:
                for sym in m.symbols(shown=True):
                    if sym.name == "n" and sym.arguments:
                        best = max(best, sym.arguments[0].number)
        return best

    def cautious_min(cap: int) -> int:
        prog = (base + "cnt(N) :- N = #count{ X : match(X) }.\n"
                + "".join(f"atleast({k}) :- cnt(N), N >= {k}.\n"
                          for k in range(cap + 1))
                + "#show atleast/1.\n")
        ctl = clingo.Control(["--models=0", "--enum-mode=cautious"])
        ctl.add("base", [], prog)
        ctl.ground([("base", [])])
        # In cautious mode only the LAST yielded model is the intersection of all
        # answer sets; intermediate models are shrinking supersets.
        last: list = []
        with ctl.solve(yield_=True) as h:
            for m in h:
                last = list(m.symbols(shown=True))
        best = 0
        for sym in last:
            if sym.name == "atleast" and sym.arguments:
                best = max(best, sym.arguments[0].number)
        return best

    poss = brave_max()
    cert = cautious_min(poss)
    return (cert, poss) == evaluate_count(ir), (cert, poss)
