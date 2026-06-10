"""simba.workflow — a lean, embedded, pure-Python durable-workflow engine.

The substrate for CQRS-style async work in simba: a durable task queue with
exactly-once enqueue + retries/backoff + stale-reclaim (``queue``), resumable
exactly-once projections that rebuild by replay (``projection``), and an
asset + freshness-policy model that decides "should I enqueue a refresh?"
(``asset``). Execution helpers live in ``runner`` (in-process, detached
fire-and-forget, bounded fan-out, and a draining worker loop).

Single-node only: SQLite (via :mod:`simba.db`) for all mutable status/cursor
state plus ``subprocess`` for detached workers — no broker, no server, no
daemon, no new dependencies. This is the durable plumbing beneath spec 17's
materialized-view refresh; it ships default-off and unwired (nothing in the
live hooks depends on it yet).
"""

from __future__ import annotations
