"""Pipeline 1: Semantic indexing — SQLite rows to LanceDB + QMD.

One cycle:
1. For each indexable table, read rows after the watermark.
2. Render to text, POST to daemon ``/store``.
3. Export rows as markdown for QMD.
4. Advance watermarks.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path

import httpx

import simba.config
import simba.db
from simba.sync import exporter
from simba.sync.text_render import INDEXABLE_TABLES, render_row
from simba.sync.watermarks import get_watermark, set_watermark

logger = logging.getLogger("simba.sync.indexer")


def _sync_cfg():
    import simba.sync.config

    return simba.config.load("sync")


@dataclasses.dataclass
class IndexResult:
    tables_polled: int = 0
    rows_indexed: int = 0
    rows_exported: int = 0
    duplicates: int = 0
    errors: int = 0


def _post_to_daemon(
    client: httpx.Client,
    text: str,
    table_name: str,
) -> str:
    """POST a rendered row to the daemon /store endpoint.

    Returns ``"ok"``, ``"duplicate"``, or ``"error"``.
    """
    payload = {
        "type": "SYSTEM",
        "content": text[:200],
        "context": text,
        "tags": [table_name, "sync"],
        "confidence": 1.0,
    }
    last_exc: Exception | None = None
    cfg = _sync_cfg()
    for attempt in range(cfg.retry_count + 1):
        try:
            resp = client.post("/store", json=payload)
            if resp.status_code == 200:
                body = resp.json()
                if body.get("duplicate"):
                    return "duplicate"
                return "ok"
            if resp.status_code == 503 and attempt < cfg.retry_count:
                time.sleep(cfg.retry_backoff * (attempt + 1))
                continue
            return "error"
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt < cfg.retry_count:
                time.sleep(cfg.retry_backoff * (attempt + 1))
                continue
    logger.debug("POST /store failed: %s", last_exc)
    return "error"


def run_index(
    cwd: str | Path,
    *,
    daemon_url: str | None = None,
    dry_run: bool = False,
) -> IndexResult:
    """Run one cycle of semantic indexing."""
    cfg = _sync_cfg()
    if daemon_url is None:
        daemon_url = cfg.daemon_url

    result = IndexResult()
    export_rows: dict[str, list[dict]] = {}

    conn = simba.db.get_connection(Path(cwd))
    if conn is None:
        logger.warning("Database not found at %s", cwd)
        return result

    client: httpx.Client | None = None
    if not dry_run:
        client = httpx.Client(base_url=daemon_url, timeout=10)

    try:
        for table_name in INDEXABLE_TABLES:
            result.tables_polled += 1
            cursor = get_watermark(conn, table_name, "index")

            try:
                rows = conn.execute(
                    f"SELECT rowid, * FROM {table_name} "
                    "WHERE rowid > ? ORDER BY rowid LIMIT ?",
                    (cursor, cfg.batch_limit),
                ).fetchall()
            except Exception:
                logger.debug("Table %s not found, skipping", table_name)
                continue

            if not rows:
                continue

            table_export: list[dict] = []
            max_rowid = cursor

            for row in rows:
                row_dict = dict(row)
                rowid = str(row_dict.pop("rowid", "0"))
                text = render_row(table_name, row_dict)
                if not text:
                    continue

                if dry_run:
                    logger.info(
                        "[dry-run] %s rowid=%s: %s",
                        table_name,
                        rowid,
                        text[:80],
                    )
                    result.rows_indexed += 1
                else:
                    assert client is not None
                    status = _post_to_daemon(client, text, table_name)
                    if status == "ok":
                        result.rows_indexed += 1
                    elif status == "duplicate":
                        result.duplicates += 1
                    else:
                        result.errors += 1
                    time.sleep(cfg.rate_limit_sec)

                table_export.append(row_dict)
                max_rowid = rowid

            if table_export:
                export_rows[table_name] = table_export
                set_watermark(
                    conn,
                    table_name,
                    "index",
                    max_rowid,
                    rows_processed=len(table_export),
                    errors=0,
                )
    finally:
        if client is not None:
            client.close()
        conn.close()

    if export_rows and not dry_run:
        paths = exporter.export_all_tables(cwd, export_rows)
        result.rows_exported = sum(len(rows) for rows in export_rows.values())
        logger.info("Exported %d rows to %d files", result.rows_exported, len(paths))

    return result
