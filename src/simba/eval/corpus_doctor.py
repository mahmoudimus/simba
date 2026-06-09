"""Corpus-doctor: inject -> detect -> score harness for contradiction detection.

A Phase-7 eval instrument that synthetically corrupts a KG-edge corpus with
typed contradictions, runs an arbitrary ``detect_fn`` over the corrupted corpus,
and scores precision/recall/F1 of the detector against the known injections.

Pattern (SME inject->detect->score):

    corrupted, truth = inject(corpus, kinds, seed, count)
    detected_ids     = detect_fn(corrupted)          # e.g. z3_verify
    metrics          = score(detected_ids, truth)

The injection vocabulary mirrors ``simba.neuron.z3_verify`` so a Z3 consistency
pass is the natural detector:

    antonym           two edges sharing (subject, object) with opposite
                      predicates (uses/does_not_use, prefers/avoids, fixes/breaks)
    temporal_overlap  two edges sharing (subject, predicate, object) endpoints
                      with overlapping belief-time windows
    duplicate         an identical (subject, predicate, object) triple

Everything here is a pure function and deterministic for a fixed seed. The whole
module is config-gated: ``run_corpus_doctor_eval`` is a no-op (returns a zeroed
``DetectionMetrics``) when ``CorpusDoctorConfig.enabled`` is False, so the harness
costs nothing in production CI until explicitly turned on.
"""

from __future__ import annotations

import copy
import dataclasses
import random
import typing
from typing import Literal

import simba.config

ContradictionType = Literal["antonym", "temporal_overlap", "duplicate"]

# Mirror of simba.neuron.z3_verify._ANTONYMS so injected antonyms are exactly
# what the Z3 detector treats as mutually exclusive.
_ANTONYMS: list[tuple[str, str]] = [
    ("uses", "does_not_use"),
    ("prefers", "avoids"),
    ("fixes", "breaks"),
]

_ALL_KINDS: tuple[ContradictionType, ...] = ("antonym", "temporal_overlap", "duplicate")


def _antonym_lookup() -> dict[str, str]:
    table: dict[str, str] = {}
    for a, b in _ANTONYMS:
        table[a] = b
        table[b] = a
    return table


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@simba.config.configurable("corpus_doctor")
@dataclasses.dataclass
class CorpusDoctorConfig:
    """Gate + parameters for the Phase-7 corpus-doctor eval (default OFF)."""

    enabled: bool = False
    seed: int = 42
    # Comma-separated subset of {antonym, temporal_overlap, duplicate}; split on
    # load so it round-trips through
    # `simba config set corpus_doctor.contradiction_types`.
    contradiction_types: str = "antonym,temporal_overlap"
    num_corruptions_per_corpus: int = 5
    detection_threshold: float = 0.5

    def kinds_tuple(self) -> tuple[ContradictionType, ...]:
        """Parse ``contradiction_types`` into a validated tuple of kinds."""
        out: list[ContradictionType] = []
        for part in self.contradiction_types.split(","):
            part = part.strip()
            if part in _ALL_KINDS:
                out.append(typing.cast("ContradictionType", part))
        return tuple(out) if out else ("antonym",)


# ---------------------------------------------------------------------------
# Ground-truth + metric dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class InjectionResult:
    """Ground truth for a single synthetic contradiction (edge pair a<->b)."""

    corpus_id: str
    edge_id_a: int
    edge_id_b: int
    contradiction_type: ContradictionType
    injected_edge: dict
    original_edge: dict | None


@dataclasses.dataclass
class DetectionMetrics:
    """Scored detection performance over a set of injected contradictions."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


_ZERO_METRICS = DetectionMetrics(0, 0, 0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# inject
# ---------------------------------------------------------------------------


def _next_id(corpus: list[dict]) -> int:
    return (max((int(e["id"]) for e in corpus), default=0)) + 1


def _make_antonym(rng: random.Random, corpus: list[dict], next_id: int) -> dict | None:
    antonyms = _antonym_lookup()
    candidates = [e for e in corpus if e.get("predicate") in antonyms]
    if not candidates:
        return None
    orig = rng.choice(candidates)
    edge = copy.deepcopy(orig)
    edge["id"] = next_id
    edge["predicate"] = antonyms[orig["predicate"]]
    return _result(corpus, orig, edge, "antonym")


def _make_duplicate(
    rng: random.Random, corpus: list[dict], next_id: int
) -> dict | None:
    if not corpus:
        return None
    orig = rng.choice(corpus)
    edge = copy.deepcopy(orig)
    edge["id"] = next_id
    return _result(corpus, orig, edge, "duplicate")


def _make_temporal_overlap(
    rng: random.Random, corpus: list[dict], next_id: int
) -> dict | None:
    if not corpus:
        return None
    orig = rng.choice(corpus)
    edge = copy.deepcopy(orig)
    edge["id"] = next_id
    # Same (subject, predicate, object) + open-ended window guarantees overlap
    # with the original, which the Z3 same-predicate exclusion treats as a clash.
    edge["valid_from"] = orig.get("valid_from") or ""
    edge["valid_to"] = None
    return _result(corpus, orig, edge, "temporal_overlap")


def _result(
    corpus: list[dict], orig: dict, edge: dict, kind: ContradictionType
) -> dict:
    return {
        "edge": edge,
        "truth": InjectionResult(
            corpus_id=str(corpus[0]["id"]) if corpus else "",
            edge_id_a=int(orig["id"]),
            edge_id_b=int(edge["id"]),
            contradiction_type=kind,
            injected_edge=edge,
            original_edge=orig,
        ),
    }


_BUILDERS: dict[ContradictionType, typing.Callable[..., dict | None]] = {
    "antonym": _make_antonym,
    "temporal_overlap": _make_temporal_overlap,
    "duplicate": _make_duplicate,
}


def inject(
    corpus: list[dict],
    kinds: list[ContradictionType] | None = None,
    seed: int = 42,
    count: int = 5,
) -> tuple[list[dict], list[InjectionResult]]:
    """Corrupt *corpus* with ``count`` typed contradictions.

    Returns ``(corrupted_corpus, ground_truth)``. The input corpus is never
    mutated; injected edges get fresh, monotonically increasing ids appended to
    the end. Deterministic for a fixed ``seed``. If a requested kind has no
    eligible source edge in *corpus*, that draw is skipped (so fewer than
    ``count`` contradictions may be returned for a degenerate corpus).
    """
    kinds = list(kinds) if kinds else list(_ALL_KINDS)
    rng = random.Random(seed)
    corrupted = copy.deepcopy(corpus)
    truth: list[InjectionResult] = []
    next_id = _next_id(corrupted)
    for _ in range(max(0, count)):
        kind = rng.choice(kinds)
        built = _BUILDERS[kind](rng, corrupted, next_id)
        if built is None:
            continue
        corrupted.append(built["edge"])
        truth.append(built["truth"])
        next_id += 1
    return corrupted, truth


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def score(
    detected_edge_ids: typing.Iterable[int],
    ground_truth: list[InjectionResult],
) -> DetectionMetrics:
    """Score detector output against injected contradictions.

    A contradiction (edge pair) counts as a true positive if the detector
    flagged *either* edge of the pair; a false negative if it flagged neither.
    Any flagged id that belongs to no injected pair is a false positive.
    """
    detected = set(int(i) for i in detected_edge_ids)
    pair_ids: set[int] = set()
    tp = 0
    for gt in ground_truth:
        a, b = int(gt.edge_id_a), int(gt.edge_id_b)
        pair_ids.update((a, b))
        if a in detected or b in detected:
            tp += 1
    fn = len(ground_truth) - tp
    fp = len(detected - pair_ids)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return DetectionMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


# ---------------------------------------------------------------------------
# run (end-to-end)
# ---------------------------------------------------------------------------


def run_corpus_doctor_eval(
    corpus: list[dict],
    detect_fn: typing.Callable[[list[dict]], typing.Iterable[int]],
    cfg: CorpusDoctorConfig | None = None,
) -> DetectionMetrics:
    """Inject -> ``detect_fn`` -> score, end to end.

    ``detect_fn`` takes the corrupted corpus and returns the edge ids it judges
    contradictory (e.g. ``simba.neuron.z3_verify.run_verify(...).unsat_edge_ids``).
    Returns a zeroed ``DetectionMetrics`` when ``cfg.enabled`` is False.
    """
    cfg = cfg or CorpusDoctorConfig()
    if not cfg.enabled:
        return dataclasses.replace(_ZERO_METRICS)

    corrupted, truth = inject(
        corpus,
        kinds=list(cfg.kinds_tuple()),
        seed=cfg.seed,
        count=cfg.num_corruptions_per_corpus,
    )
    detected = detect_fn(corrupted)
    return score(detected, truth)
