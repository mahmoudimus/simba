"""Recall demand log — the query-side half of the usage ledger (spec 33 v2).

Borrowed from yantrikdb's ``knowledge_gaps()``: the memory-side ledger
(match/inject/use/noise) can only grade memories that EXIST; this O(1)
aggregate per normalized query records what was ASKED and how well the
corpus answered, so "asked often, answered poorly" — the known unknowns —
becomes queryable. Feeds health trends and promotion drafting.

One row per normalized query, mutated in place (an aggregate, not an event
log — the append-only rule covers memory content, and this table is derived
telemetry like the FTS mirror). Writers are fire-and-forget from the recall
tail; internal daemon self-calls and TOOL_RULE gate probes never count.
"""

from __future__ import annotations

import simba._vendor.peewee as pw
import simba.db

_MAX_QUERY_CHARS = 400


class RecallDemand(simba.db.BaseModel):
    query_norm = pw.CharField(max_length=_MAX_QUERY_CHARS, primary_key=True)
    ask_count = pw.IntegerField(default=0)
    zero_count = pw.IntegerField(default=0)
    best_score_max = pw.FloatField(default=0.0)
    best_score_sum = pw.FloatField(default=0.0)
    last_asked = pw.FloatField(default=0.0)

    class Meta:
        table_name = "recall_demand"


simba.db.register_model(RecallDemand)


def normalize(query: str) -> str:
    """Lowercase, whitespace-collapsed, length-capped query key."""
    return " ".join((query or "").lower().split())[:_MAX_QUERY_CHARS]


def record(query: str, result_count: int, best_score: float, *, now: float) -> None:
    """UPSERT one ask into the aggregate. Must run inside ``simba.db.connect``."""
    key = normalize(query)
    if not key:
        return
    RecallDemand.get_or_create(query_norm=key)
    RecallDemand.update(
        ask_count=RecallDemand.ask_count + 1,
        zero_count=RecallDemand.zero_count + (1 if result_count == 0 else 0),
        best_score_sum=RecallDemand.best_score_sum + float(best_score or 0.0),
        best_score_max=pw.fn.MAX(RecallDemand.best_score_max, float(best_score or 0.0)),
        last_asked=now,
    ).where(RecallDemand.query_norm == key).execute()


def gaps(*, min_asks: int = 3, max_best: float = 0.5, limit: int = 20) -> list[dict]:
    """Queries asked ≥ ``min_asks`` times whose best hit never reached
    ``max_best`` — the corpus's known unknowns, most-asked first."""
    rows = (
        RecallDemand.select()
        .where(
            (RecallDemand.ask_count >= min_asks)
            & (RecallDemand.best_score_max < max_best)
        )
        .order_by(RecallDemand.ask_count.desc())
        .limit(max(1, limit))
    )
    return [
        {
            "query": row.query_norm,
            "askCount": int(row.ask_count),
            "zeroCount": int(row.zero_count),
            "bestScoreMax": float(row.best_score_max),
            "avgBestScore": (
                float(row.best_score_sum) / row.ask_count if row.ask_count else 0.0
            ),
            "lastAsked": float(row.last_asked),
        }
        for row in rows
    ]
