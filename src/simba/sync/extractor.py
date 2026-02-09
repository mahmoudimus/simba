"""Pipeline 2: Fact extraction — LanceDB memories to proven_facts.

Phase 1 (heuristic): regex-based extraction, always runs.
Phase 2 (Claude): agent-based refinement, opt-in via ``use_claude``.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import httpx

from simba.sync.heuristics import extract_facts
from simba.sync.watermarks import get_watermark, set_watermark

logger = logging.getLogger("simba.sync.extractor")


def _sync_cfg():
    import simba.sync.config

    return simba.config.load("sync")


@dataclasses.dataclass
class ExtractResult:
    memories_processed: int = 0
    facts_extracted: int = 0
    facts_duplicate: int = 0
    agent_dispatched: bool = False
    errors: int = 0


def _fetch_memories(
    client: httpx.Client,
    *,
    offset: int = 0,
    limit: int | None = None,
) -> tuple[list[dict], int]:
    """GET /list from daemon, excluding SYSTEM memories.

    Returns (memories, total).
    """
    if limit is None:
        limit = _sync_cfg().page_size
    try:
        resp = client.get("/list", params={"limit": limit, "offset": offset})
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        return [], 0

    memories = [m for m in data.get("memories", []) if m.get("type") != "SYSTEM"]
    return memories, data.get("total", 0)


def _store_fact(
    subject: str,
    predicate: str,
    obj: str,
    proof: str,
    *,
    cwd: str,
) -> str:
    """Store a fact triple via simba.neuron.truth."""
    import simba.db

    with simba.db.get_db(Path(cwd)) as conn:
        try:
            conn.execute(
                "INSERT INTO proven_facts VALUES (?, ?, ?, ?)",
                (subject, predicate, obj, proof),
            )
            conn.commit()
            return "ok"
        except Exception:
            return "duplicate"


def _dispatch_claude_agent(memories: list[dict], *, cwd: str) -> str | None:
    """Dispatch a researcher agent for deeper fact extraction.

    Returns the ticket_id or None on failure.
    """
    try:
        from simba.orchestration.agents import dispatch_agent
    except ImportError:
        logger.debug("orchestration.agents not available")
        return None

    memory_text = "\n\n".join(
        f"[{m.get('type', '?')}] {m.get('content', '')} "
        f"(context: {m.get('context', '')})"
        for m in memories[:20]
    )
    instructions = (
        "Extract (subject, predicate, object) fact triples from "
        "these memories. For each fact, call "
        "truth_add(subject, predicate, object, "
        'proof="researcher_extracted").\n\n'
        f"Memories:\n{memory_text}"
    )
    ticket_id = f"sync-facts-{len(memories)}"
    try:
        dispatch_agent("researcher", ticket_id, instructions, cwd=cwd)
        return ticket_id
    except Exception:
        logger.debug("Failed to dispatch agent", exc_info=True)
        return None


def run_extract(
    cwd: str | Path,
    *,
    daemon_url: str | None = None,
    use_claude: bool = False,
    dry_run: bool = False,
) -> ExtractResult:
    """Run one cycle of fact extraction."""
    import simba.db

    cfg = _sync_cfg()
    if daemon_url is None:
        daemon_url = cfg.daemon_url

    cwd_path = Path(cwd) if isinstance(cwd, str) else cwd
    cwd_str = str(cwd_path)
    result = ExtractResult()

    # Ensure proven_facts table exists
    with simba.db.get_db(cwd_path) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS proven_facts
               (subject TEXT, predicate TEXT, object TEXT, proof TEXT,
               UNIQUE(subject, predicate, object))"""
        )
        conn.commit()
        watermark = get_watermark(conn, "memories", "facts")

    client = httpx.Client(base_url=daemon_url, timeout=10)
    no_fact_memories: list[dict] = []

    try:
        offset = 0
        while True:
            memories, total = _fetch_memories(
                client, offset=offset, limit=cfg.page_size
            )
            if not memories:
                break

            latest_ts = watermark
            for mem in memories:
                created = mem.get("createdAt", "")
                if created <= watermark:
                    continue

                result.memories_processed += 1
                mem_type = mem.get("type", "")
                content = mem.get("content", "")
                context = mem.get("context", "")
                mem_id = mem.get("id", "")

                triples = extract_facts(mem_type, content, context, mem_id)

                if dry_run:
                    for s, p, o, proof in triples:
                        logger.info(
                            "[dry-run] %s %s %s (proof: %s)",
                            s,
                            p,
                            o,
                            proof,
                        )
                    result.facts_extracted += len(triples)
                else:
                    for s, p, o, proof in triples:
                        status = _store_fact(s, p, o, proof, cwd=cwd_str)
                        if status == "ok":
                            result.facts_extracted += 1
                        else:
                            result.facts_duplicate += 1

                if not triples:
                    no_fact_memories.append(mem)

                if created > latest_ts:
                    latest_ts = created

            offset += cfg.page_size
            if offset >= total:
                break

        # Update watermark
        if result.memories_processed > 0 and not dry_run:
            with simba.db.get_db(cwd_path) as conn:
                set_watermark(
                    conn,
                    "memories",
                    "facts",
                    latest_ts,
                    rows_processed=result.memories_processed,
                )

    except Exception:
        logger.debug("Extraction error", exc_info=True)
        result.errors += 1
    finally:
        client.close()

    # Phase 2: Claude refinement
    if use_claude and no_fact_memories and not dry_run:
        ticket = _dispatch_claude_agent(no_fact_memories, cwd=cwd_str)
        if ticket:
            result.agent_dispatched = True
            logger.info("Dispatched researcher agent: %s", ticket)

    return result
