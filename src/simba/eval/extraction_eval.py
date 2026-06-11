"""Extraction-quality evaluator (scaffold) — the gate that lets the Importance rubric
be TUNED instead of vibed.

The "Evaluator" gap: today "store 5-15 high-value, skip generic" is a prompt vibe with
no way to tell if it's working. This is the measurement harness the future
extract->score->keep/drop worker is graded against. Not wired live yet — it provides
the gold-set shape + the two metrics:

- **keep/drop precision/recall/F1** — did the gate keep the right conclusions and drop
  the generic ones? (micro-averaged over normalized conclusion strings).
- **recurrence_hit_rate** — do cached conclusions actually get re-matched on later
  questions? This is the go/no-go for the whole continuous-extraction feature: if
  rediscovery is rare, the loop is overhead.

Pure + injectable (the matcher / predictor are passed in), so it's unit-testable with
synthetic fixtures now and swaps to real recall + a real gold set later.
"""

from __future__ import annotations

import dataclasses
import re
import typing


@dataclasses.dataclass(frozen=True)
class GoldWindow:
    """A transcript window + the conclusions that SHOULD be cached from it."""

    window: str
    expected_keep: list[str]


def _norm(s: typing.Any) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def keep_drop_prf(predicted: typing.Iterable[str], gold: typing.Iterable[str]) -> dict:
    """Precision/recall/F1 of kept vs gold conclusions (set-based, normalized)."""
    pred = {_norm(x) for x in predicted if _norm(x)}
    want = {_norm(x) for x in gold if _norm(x)}
    tp = len(pred & want)
    fp = len(pred - want)
    fn = len(want - pred)
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not want else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def recurrence_hit_rate(
    cached: typing.Sequence[str],
    later_questions: typing.Sequence[str],
    *,
    match: typing.Callable[[str, str], bool],
) -> float:
    """Fraction of later questions answerable by a cached conclusion.

    ``match(question, cached_conclusion) -> bool`` is injected (semantic recall in
    production; a lambda in tests). This is the feature's justification metric — high
    means the agent really does rediscover the same durable things.
    """
    if not later_questions:
        return 0.0
    hits = sum(1 for q in later_questions if any(match(q, c) for c in cached))
    return round(hits / len(later_questions), 3)


def evaluate(
    goldset: typing.Sequence[GoldWindow],
    *,
    predict: typing.Callable[[str], typing.Iterable[str]],
) -> dict:
    """Micro-averaged keep/drop PRF of ``predict`` over the gold set."""
    all_pred: list[str] = []
    all_gold: list[str] = []
    for gw in goldset:
        all_pred.extend(predict(gw.window))
        all_gold.extend(gw.expected_keep)
    return keep_drop_prf(all_pred, all_gold)
