"""SubtleMemory loader + per-relation-slice aggregation.

SubtleMemory is a *relational / contradiction* memory eval: the hard signal is
not raw recall but whether the system surfaces the right combination of memories
when they **relate** — complementary (combine / any-one), nuanced (a time /
context boundary decides which applies), or **contradictory** (an unresolved
conflict that should be surfaced, not silently collapsed). ``contradictory`` is
the headline slice — the primary differentiator between memory systems.

Data layout (one dir per persona)::

    data/subtlememory/persona_{0..9}/
        bench_instances.json   # cases: relation labels + facts + QA pairs
        history_sessions.json  # the conversation corpus (multi-turn, timestamped)

This loader maps one persona -> one simba ``Dataset`` (mirroring
``halumem.py`` / ``locomo.py``): every dialogue turn becomes a ``Memory`` keyed
``{session_id}_{turn_index}``; every QA pair becomes an ``EvalCase`` whose gold
(``relevant_ids``) is all turns of the case's target ``session_ids`` and whose
``intent`` is the ``relation_type`` (so the existing recall / QA harness groups
by relation slice automatically) and ``note`` the ``relation_subtype``.

``aggregate_by_relation`` mirrors ``halumem.aggregate_qa``: overall + per-slice
accuracy with a dedicated ``contradictory`` block flagged ``is_headline``.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import tempfile
import typing

import simba.eval.metrics
import simba.eval.recall_adapter
import simba.eval.runner
from simba.eval.dataset import Dataset, EvalCase, Memory

# The headline relation slice: an unresolved conflict between memories.
CONTRADICTORY = "contradictory"
COMPLEMENTARY = "complementary"
NUANCED = "nuanced"

_MAX_PERSONAS = 10  # persona_0 .. persona_9


@dataclasses.dataclass
class QAPair:
    query: str
    correct_answers: list[str]
    incorrect_answers: list[str] = dataclasses.field(default_factory=list)

    @property
    def gold(self) -> str:
        """Canonical gold answer for the LLM-judge QA layer (first correct one)."""
        return self.correct_answers[0] if self.correct_answers else ""

    @classmethod
    def from_dict(cls, raw: dict[str, typing.Any]) -> QAPair:
        return cls(
            query=str(raw.get("query", "")),
            correct_answers=[str(a) for a in (raw.get("correct_answers") or [])],
            incorrect_answers=[str(a) for a in (raw.get("incorrect_answers") or [])],
        )


@dataclasses.dataclass
class BenchInstance:
    instance_id: str
    case_id: str
    persona_id: str
    relation_type: str
    relation_subtype: str
    session_ids: list[str]
    qas: list[QAPair]
    topic: str = ""
    source: str = ""
    case: str = ""
    facts: list[str] = dataclasses.field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, typing.Any]) -> BenchInstance:
        return cls(
            instance_id=str(raw.get("instance_id", "")),
            case_id=str(raw.get("case_id", "")),
            persona_id=str(raw.get("persona_id", "")),
            relation_type=str(raw.get("relation_type", "")),
            relation_subtype=str(raw.get("relation_subtype", "")),
            session_ids=[str(s) for s in (raw.get("session_ids") or [])],
            qas=[QAPair.from_dict(q) for q in (raw.get("qas") or [])],
            topic=str(raw.get("topic", "")),
            source=str(raw.get("source", "")),
            case=str(raw.get("case", "")),
            facts=[str(f) for f in (raw.get("facts") or [])],
        )


@dataclasses.dataclass
class Turn:
    role: str
    content: str

    @classmethod
    def from_dict(cls, raw: dict[str, typing.Any]) -> Turn:
        return cls(role=str(raw.get("role", "")), content=str(raw.get("content", "")))


@dataclasses.dataclass
class HistorySession:
    session_id: str
    persona_id: str
    case_id: str
    source: str
    timestamp: str
    history: list[Turn]
    conversation_type: str = ""
    order: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, typing.Any]) -> HistorySession:
        return cls(
            session_id=str(raw.get("session_id", "")),
            persona_id=str(raw.get("persona_id", "")),
            case_id=str(raw.get("case_id", "")),
            source=str(raw.get("source", "")),
            timestamp=str(raw.get("timestamp", "")),
            history=[Turn.from_dict(t) for t in (raw.get("history") or [])],
            conversation_type=str(raw.get("conversation_type", "")),
            order=int(raw.get("order", 0) or 0),
        )


def _persona_corpus(
    sessions: list[HistorySession],
) -> tuple[list[Memory], dict[str, list[str]]]:
    """Build the retrievable corpus + a session_id -> [turn memory ids] index.

    Each turn is one ``Memory`` id'd ``{session_id}_{turn_index}`` (globally
    unique within the persona). The timestamp goes to ``created_at`` (the recency
    signal ``format_memories`` injects) and is also prefixed into the content so
    a bare answerer still sees it; ``session_source`` carries the session id.
    """
    corpus: list[Memory] = []
    by_session: dict[str, list[str]] = {}
    for session in sessions:
        for ti, turn in enumerate(session.history):
            text = turn.content.strip()
            if not text:
                continue
            mid = f"{session.session_id}_{ti}"
            body = f"{turn.role}: {text}" if turn.role else text
            content = f"[{session.timestamp}] {body}" if session.timestamp else body
            corpus.append(
                Memory(
                    id=mid,
                    content=content,
                    type="HISTORY",
                    session_source=session.session_id,
                    created_at=session.timestamp,
                    confidence=0.9,
                )
            )
            by_session.setdefault(session.session_id, []).append(mid)
    return corpus, by_session


_RawJson = dict[str, typing.Any]
_PersonaTriple = tuple[str, list[_RawJson], list[_RawJson]]


def load_subtlememory_data(personas: list[_PersonaTriple]) -> list[Dataset]:
    """Convert ``(persona_id, bench_instances, history_sessions)`` tuples into
    one ``Dataset`` per persona.

    Each QA pair becomes one ``EvalCase``; its gold is every turn of the case's
    target ``session_ids`` (unresolvable session ids are dropped; a case with no
    resolvable gold is dropped — it isn't scoreable). ``intent`` =
    ``relation_type``, ``note`` = ``relation_subtype``.
    """
    datasets: list[Dataset] = []
    for persona_id, raw_bench, raw_history in personas:
        sessions = [HistorySession.from_dict(s) for s in raw_history]
        corpus, by_session = _persona_corpus(sessions)

        cases: list[EvalCase] = []
        for raw_inst in raw_bench:
            inst = BenchInstance.from_dict(raw_inst)
            gold: list[str] = []
            for sid in inst.session_ids:
                gold.extend(by_session.get(sid, []))
            if not gold:  # nothing resolves -> not scoreable
                continue
            for qi, qa in enumerate(inst.qas):
                if not qa.query.strip():
                    continue
                cases.append(
                    EvalCase(
                        id=f"{persona_id}_{inst.instance_id}_{qi}",
                        query=qa.query,
                        relevant_ids=list(gold),
                        intent=inst.relation_type,
                        note=inst.relation_subtype,
                        answer=qa.gold,
                    )
                )
        datasets.append(
            Dataset(
                name=f"subtlememory_persona_{persona_id}",
                corpus=corpus,
                cases=cases,
            )
        )
    return datasets


def load_persona(path: str | pathlib.Path, persona_id: int | str) -> Dataset:
    """Load a single ``persona_{N}/`` dir into one ``Dataset``."""
    pdir = pathlib.Path(path) / f"persona_{persona_id}"
    bench = json.loads((pdir / "bench_instances.json").read_text())
    history = json.loads((pdir / "history_sessions.json").read_text())
    return load_subtlememory_data([(str(persona_id), bench, history)])[0]


def load_subtlememory(
    path: str | pathlib.Path, *, persona_limit: int = 0, persona_start: int = 0
) -> list[Dataset]:
    """Load ``persona_{0..N}/`` dirs under ``path`` into per-persona Datasets.

    ``persona_limit > 0`` loads only N personas starting at ``persona_start``
    (cheap smoke runs; 1 persona ~= 100 cases / ~2.5k turns). ``0`` loads all
    available personas from ``persona_start``.
    """
    root = pathlib.Path(path)
    start = max(0, persona_start)
    stop = (
        min(_MAX_PERSONAS, start + persona_limit)
        if persona_limit > 0
        else _MAX_PERSONAS
    )
    datasets: list[Dataset] = []
    for pid in range(start, stop):
        pdir = root / f"persona_{pid}"
        if not (pdir / "bench_instances.json").exists():
            continue
        datasets.append(load_persona(root, pid))
    return datasets


def _acc(rows: list[tuple[str, bool]]) -> float:
    return (sum(1 for _, c in rows if c) / len(rows)) if rows else 0.0


def aggregate_by_relation(rows: list[tuple[str, bool]]) -> dict[str, typing.Any]:
    """Aggregate per-case ``(relation_type, correct)`` into per-slice accuracy.

    Returns overall + ``by_relation`` (one block per relation_type) + a dedicated
    ``contradictory`` block — the headline differentiator (unresolved conflict
    that should be surfaced, not collapsed). The contradictory block inside
    ``by_relation`` is flagged ``is_headline`` so report consumers can highlight
    it. Mirrors ``halumem.aggregate_qa``.
    """

    def block(slice_rows: list[tuple[str, bool]]) -> dict[str, typing.Any]:
        return {"n": len(slice_rows), "accuracy": _acc(slice_rows)}

    by_relation: dict[str, list[tuple[str, bool]]] = {}
    for relation, correct in rows:
        by_relation.setdefault(relation or "?", []).append((relation, correct))

    by_relation_out: dict[str, typing.Any] = {}
    for relation, slice_rows in sorted(by_relation.items()):
        blk = block(slice_rows)
        if relation == CONTRADICTORY:
            blk["is_headline"] = True
        by_relation_out[relation] = blk

    contradictory = [r for r in rows if r[0] == CONTRADICTORY]
    return {
        "overall": block(rows),
        "by_relation": by_relation_out,
        "contradictory": block(contradictory),
    }


def _recall_metric_names(ks: tuple[int, ...]) -> list[str]:
    names = [f"recall@{k}" for k in ks]
    names += [f"bridge_recall@{k}" for k in ks]
    names += [f"ndcg@{k}" for k in ks]
    names.append("mrr")
    return names


def _mean_metrics(
    rows: list[dict[str, float]], metric_names: list[str]
) -> dict[str, float]:
    n = len(rows)
    return {
        name: (sum(row.get(name, 0.0) for row in rows) / n if n else 0.0)
        for name in metric_names
    }


def _readback_ranked_ids(dataset: Dataset, case: EvalCase) -> list[str]:
    """Return exact transcript/session readback ids for a SubtleMemory case.

    The loader marks every target-session turn as gold. This readback retriever
    therefore ranks all corpus turns from those sessions, preserving transcript
    order. If a caller supplies a generic dataset without ``session_source``
    metadata, fall back to the case's gold ids so the ceiling remains scoreable.
    """
    by_id = {mem.id: mem for mem in dataset.corpus}
    target_sessions = {
        by_id[rid].session_source
        for rid in case.relevant_ids
        if rid in by_id and by_id[rid].session_source
    }
    if not target_sessions:
        return list(case.relevant_ids)
    return [mem.id for mem in dataset.corpus if mem.session_source in target_sessions]


def run_readback_recall(
    datasets: list[Dataset],
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, typing.Any]:
    """Score a session-readback ceiling for SubtleMemory.

    This is intentionally an oracle-style diagnostic, not a competing retriever:
    it uses each case's target sessions to read back the relevant transcript
    turns and reports the same aggregate shape as ``benchmarks.run.run_recall``.
    The ceiling is still bounded by ``k``; when gold spans more than ``k`` turns,
    ``recall@k`` and ``bridge_recall@k`` cannot reach 1.0.
    """
    metric_names = _recall_metric_names(ks)
    overall: list[dict[str, float]] = []
    by_category: dict[str, list[dict[str, float]]] = {}
    gold_widths: list[int] = []
    readback_widths: list[int] = []

    for dataset in datasets:
        for case in dataset.cases:
            ranked = _readback_ranked_ids(dataset, case)
            metrics = simba.eval.runner._case_metrics(
                ranked, set(case.relevant_ids), ks
            )
            overall.append(metrics)
            by_category.setdefault(case.intent or "?", []).append(metrics)
            gold_widths.append(len(set(case.relevant_ids)))
            readback_widths.append(len(ranked))

    def avg(values: list[int]) -> float:
        return (sum(values) / len(values)) if values else 0.0

    return {
        "mode": "session_readback_ceiling",
        "n_conversations": len(datasets),
        "n_cases": len(overall),
        "overall": _mean_metrics(overall, metric_names),
        "by_category": {
            cat: {"n": len(rows), **_mean_metrics(rows, metric_names)}
            for cat, rows in sorted(by_category.items())
        },
        "latency": {"p50_ms": 0.0, "p95_ms": 0.0, "n": len(overall)},
        "diagnostics": {
            "avg_gold_ids": avg(gold_widths),
            "max_gold_ids": max(gold_widths, default=0),
            "avg_readback_ids": avg(readback_widths),
            "max_readback_ids": max(readback_widths, default=0),
            "gold_gt_k": {
                str(k): sum(1 for width in gold_widths if width > k) for k in ks
            },
        },
    }


def _metric_delta(
    high: dict[str, typing.Any], low: dict[str, typing.Any]
) -> dict[str, float]:
    return {
        key: float(high[key]) - float(low[key])
        for key in sorted(high.keys() & low.keys())
        if isinstance(high.get(key), int | float)
        and isinstance(low.get(key), int | float)
        and key != "n"
    }


def compare_readback_ceiling(
    normal_recall: dict[str, typing.Any],
    datasets: list[Dataset],
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, typing.Any]:
    """Return readback ceiling metrics and deltas against normal recall."""
    ceiling = run_readback_recall(datasets, ks=ks)
    normal_by_cat = normal_recall.get("by_category") or {}
    ceiling_by_cat = ceiling.get("by_category") or {}
    cats = sorted(set(normal_by_cat) | set(ceiling_by_cat))
    return {
        "mode": "session_readback_ceiling",
        "ceiling": ceiling,
        "delta_vs_recall": {
            "overall": _metric_delta(
                ceiling.get("overall") or {}, normal_recall.get("overall") or {}
            ),
            "by_category": {
                cat: _metric_delta(
                    ceiling_by_cat.get(cat) or {}, normal_by_cat.get(cat) or {}
                )
                for cat in cats
            },
        },
    }


def run_recall_with_ranked(
    datasets: list[Dataset],
    *,
    embed_doc: typing.Callable[[str], list[float]],
    embed_query: typing.Callable[[str], list[float]],
    cfg: typing.Any,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    keep_top: int = 20,
    llm_client: typing.Any = None,
    kg_extract: typing.Any = None,
) -> tuple[dict[str, typing.Any], dict[str, list[str]]]:
    """Run normal recall and retain each case's ranked ids for diagnostics."""
    metric_names = _recall_metric_names(ks)
    by_cat: dict[str, list[dict[str, float]]] = {}
    overall: list[dict[str, float]] = []
    all_latencies: list[float] = []
    ranked_by_case: dict[str, list[str]] = {}

    for dataset in datasets:
        cat_of = {case.id: (case.intent or "?") for case in dataset.cases}
        with tempfile.TemporaryDirectory(prefix="simba-subtle-driver-") as td:
            retriever = simba.eval.recall_adapter.build_retriever(
                dataset,
                cfg,
                embed_doc=embed_doc,
                embed_query=embed_query,
                data_dir=td,
                llm_client=llm_client,
                kg_extract=kg_extract,
            )
            report = simba.eval.runner.run_eval(
                dataset, retriever, ks=ks, keep_top=keep_top
            )
        for case in report.per_case:
            overall.append(case.metrics)
            by_cat.setdefault(cat_of.get(case.case_id, "?"), []).append(case.metrics)
            all_latencies.append(case.latency_ms)
            ranked_by_case[case.case_id] = list(case.ranked)

    return (
        {
            "n_conversations": len(datasets),
            "n_cases": len(overall),
            "overall": _mean_metrics(overall, metric_names),
            "by_category": {
                cat: {"n": len(rows), **_mean_metrics(rows, metric_names)}
                for cat, rows in sorted(by_cat.items())
            },
            "latency": {
                "p50_ms": simba.eval.runner._percentile(all_latencies, 50),
                "p95_ms": simba.eval.runner._percentile(all_latencies, 95),
                "n": len(all_latencies),
            },
        },
        ranked_by_case,
    )


def _target_sessions(dataset: Dataset, case: EvalCase) -> set[str]:
    by_id = {mem.id: mem for mem in dataset.corpus}
    return {
        by_id[rid].session_source
        for rid in case.relevant_ids
        if rid in by_id and by_id[rid].session_source
    }


def _ranked_sessions(dataset: Dataset, ranked: list[str]) -> list[str]:
    by_id = {mem.id: mem for mem in dataset.corpus}
    sessions: list[str] = []
    seen: set[str] = set()
    for rid in ranked:
        sid = by_id.get(rid).session_source if rid in by_id else ""
        if sid and sid not in seen:
            sessions.append(sid)
            seen.add(sid)
    return sessions


def _gap_label(
    *,
    target_sessions: set[str],
    hit_sessions: set[str],
    normal_recall: float,
    readback_recall: float,
) -> str:
    if not hit_sessions:
        return "no_session_hit"
    if hit_sessions != target_sessions:
        return "partial_session_hit"
    if normal_recall + 1e-9 < readback_recall:
        return "session_content_gap"
    return "matched_readback_at_k"


def _inc(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _driver_recommendation(counts: dict[str, int]) -> str:
    session_expandable = counts.get("partial_session_hit", 0) + counts.get(
        "session_content_gap", 0
    )
    no_hit = counts.get("no_session_hit", 0)
    if session_expandable and session_expandable >= no_hit:
        return "session_expansion"
    if no_hit:
        return "query_session_nomination"
    return "answer_time_or_cutoff"


def build_failure_ledger(
    datasets: list[Dataset],
    ranked_by_case: dict[str, list[str]],
    *,
    analysis_k: int = 10,
) -> dict[str, typing.Any]:
    """Classify SubtleMemory failures so the next lever is benchmark-driven."""
    rows: list[dict[str, typing.Any]] = []
    gap_counts: dict[str, int] = {}
    relation_counts: dict[str, dict[str, int]] = {}

    for dataset in datasets:
        by_id = {mem.id: mem for mem in dataset.corpus}
        for case in dataset.cases:
            ranked = list(ranked_by_case.get(case.id, []))
            ranked_top = ranked[:analysis_k]
            relevant = set(case.relevant_ids)
            target_sessions = _target_sessions(dataset, case)
            ranked_sessions = _ranked_sessions(dataset, ranked_top)
            hit_sessions = set(ranked_sessions) & target_sessions
            readback_ranked = _readback_ranked_ids(dataset, case)
            normal_recall = simba.eval.metrics.recall_at_k(ranked, relevant, analysis_k)
            readback_recall = simba.eval.metrics.recall_at_k(
                readback_ranked, relevant, analysis_k
            )
            label = _gap_label(
                target_sessions=target_sessions,
                hit_sessions=hit_sessions,
                normal_recall=normal_recall,
                readback_recall=readback_recall,
            )
            _inc(gap_counts, label)
            rel = case.intent or "?"
            relation_counts.setdefault(rel, {})
            _inc(relation_counts[rel], label)
            rows.append(
                {
                    "dataset": dataset.name,
                    "case_id": case.id,
                    "query": case.query,
                    "relation_type": case.intent,
                    "relation_subtype": case.note,
                    "target_sessions": sorted(target_sessions),
                    "hit_sessions": sorted(hit_sessions),
                    "ranked_sessions": ranked_sessions,
                    "ranked_ids": ranked_top,
                    "relevant_ids": list(case.relevant_ids),
                    "gold_width": len(relevant),
                    "normal_recall@k": normal_recall,
                    "readback_recall@k": readback_recall,
                    "readback_improves": readback_recall > normal_recall + 1e-9,
                    "gold_width_bound": len(relevant) > analysis_k,
                    "gap_label": label,
                    "top_hit_contents": [
                        by_id[rid].content if rid in by_id else "" for rid in ranked_top
                    ],
                }
            )

    summary = {
        "analysis_k": analysis_k,
        "n_cases": len(rows),
        "gap_counts": dict(sorted(gap_counts.items())),
        "by_relation": {
            rel: dict(sorted(counts.items()))
            for rel, counts in sorted(relation_counts.items())
        },
        "recommendation": _driver_recommendation(gap_counts),
    }
    return {"summary": summary, "cases": rows}


def write_failure_ledger(
    report: dict[str, typing.Any], path: str | pathlib.Path
) -> pathlib.Path:
    """Write the full driver report to JSON; parent dirs are created."""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    return out


def recall_metric_snapshot(report: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Compact metrics used by the driver loop and docs."""
    by_cat = report.get("by_category") or {}
    contradictory = by_cat.get(CONTRADICTORY) or {}
    overall = report.get("overall") or {}
    return {
        "overall": {
            "recall@5": overall.get("recall@5", 0.0),
            "recall@10": overall.get("recall@10", 0.0),
            "mrr": overall.get("mrr", 0.0),
        },
        CONTRADICTORY: {
            "recall@5": contradictory.get("recall@5", 0.0),
            "recall@10": contradictory.get("recall@10", 0.0),
            "mrr": contradictory.get("mrr", 0.0),
        },
    }


def driver_objective(
    report: dict[str, typing.Any],
) -> tuple[float, float, float, float]:
    """Optimization tuple for the loop: headline contradiction first."""
    snap = recall_metric_snapshot(report)
    return (
        float(snap[CONTRADICTORY]["recall@10"]),
        float(snap["overall"]["recall@10"]),
        float(snap[CONTRADICTORY]["mrr"]),
        float(snap["overall"]["mrr"]),
    )


def metric_snapshot_delta(
    current: dict[str, typing.Any], baseline: dict[str, typing.Any]
) -> dict[str, typing.Any]:
    """Delta between two ``recall_metric_snapshot`` outputs."""
    return {
        group: {
            key: float(current[group].get(key, 0.0))
            - float(baseline[group].get(key, 0.0))
            for key in sorted(current[group].keys())
        }
        for group in ("overall", CONTRADICTORY)
    }


def _gate_check(
    *,
    name: str,
    passed: bool,
    actual: float | bool,
    threshold: float | bool,
    rationale: str,
) -> dict[str, typing.Any]:
    return {
        "name": name,
        "passed": passed,
        "actual": actual,
        "threshold": threshold,
        "rationale": rationale,
    }


def driver_promotion_gate(
    *,
    winner_positive: bool,
    winner_delta: dict[str, typing.Any],
    min_contradictory_recall10_delta: float = 0.0,
    min_overall_recall10_delta: float = 0.0,
    max_contradictory_mrr_drop: float = 0.01,
    max_overall_mrr_drop: float = 0.005,
) -> dict[str, typing.Any]:
    """Gate whether a driver-loop winner is safe enough to keep advancing.

    The objective intentionally puts SubtleMemory's contradiction recall first.
    This gate keeps that optimization honest by requiring the winner to beat the
    baseline objective, lift headline and overall recall@10, and avoid material
    MRR regressions. It is an in-benchmark promotion gate; held-out persona and
    cross-benchmark runs are still separate evidence.
    """

    def delta(group: str, metric: str) -> float:
        block = winner_delta.get(group) or {}
        return float(block.get(metric, 0.0))

    contradictory_r10 = delta(CONTRADICTORY, "recall@10")
    overall_r10 = delta("overall", "recall@10")
    contradictory_mrr = delta(CONTRADICTORY, "mrr")
    overall_mrr = delta("overall", "mrr")
    checks = [
        _gate_check(
            name="objective_positive",
            passed=winner_positive,
            actual=winner_positive,
            threshold=True,
            rationale="winner objective must beat the baseline objective",
        ),
        _gate_check(
            name="contradictory_recall@10_lift",
            passed=contradictory_r10 > min_contradictory_recall10_delta,
            actual=contradictory_r10,
            threshold=min_contradictory_recall10_delta,
            rationale="headline contradiction recall must improve",
        ),
        _gate_check(
            name="overall_recall@10_lift",
            passed=overall_r10 > min_overall_recall10_delta,
            actual=overall_r10,
            threshold=min_overall_recall10_delta,
            rationale="overall recall must not be traded away",
        ),
        _gate_check(
            name="contradictory_mrr_guard",
            passed=contradictory_mrr >= -max_contradictory_mrr_drop,
            actual=contradictory_mrr,
            threshold=-max_contradictory_mrr_drop,
            rationale="headline early-rank quality must not materially regress",
        ),
        _gate_check(
            name="overall_mrr_guard",
            passed=overall_mrr >= -max_overall_mrr_drop,
            actual=overall_mrr,
            threshold=-max_overall_mrr_drop,
            rationale="overall early-rank quality must not materially regress",
        ),
    ]
    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "thresholds": {
            "min_contradictory_recall@10_delta": min_contradictory_recall10_delta,
            "min_overall_recall@10_delta": min_overall_recall10_delta,
            "max_contradictory_mrr_drop": max_contradictory_mrr_drop,
            "max_overall_mrr_drop": max_overall_mrr_drop,
        },
        "scope": "in_benchmark",
    }
