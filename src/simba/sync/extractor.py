"""Pipeline 2: Fact extraction — LanceDB memories to proven_facts.

Phase 1 (heuristic): regex-based extraction, always runs.
Phase 2 (Claude): agent-based refinement, opt-in via ``use_claude``.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import httpx

from simba.sync.dates import resolve_occurred_at
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
    occurred_at: str | None = None,
) -> str:
    """Store a fact triple into the knowledge graph (kg_edges)."""
    import simba.db
    import simba.kg.store

    result = simba.kg.store.kg_add(
        subject,
        predicate,
        obj,
        proof,
        project_path=simba.db.resolve_project_id(Path(cwd)),
        occurred_at=occurred_at,
    )
    return "ok" if result == "added" else "duplicate"


def build_extraction_prompt(
    memories: list[dict],
    existing_entities: list[str],
    *,
    max_memories: int = 20,
    max_entities: int = 60,
) -> str:
    """Build an entity-aware extraction prompt for the LLM agent.

    The prompt asks for **typed** (subject, predicate, object) triples and, when
    the project already has entities, lists them so the agent **reuses** those
    canonical names instead of minting surface-form variants — entity resolution
    at the source, complementing the normalization-based merge on ``kg_add``.
    """
    memory_text = "\n\n".join(
        f"[{m.get('type', '?')}] {m.get('content', '')} "
        f"(context: {m.get('context', '')})"
        for m in memories[:max_memories]
    )

    vocab_block = ""
    if existing_entities:
        listed = ", ".join(existing_entities[:max_entities])
        vocab_block = (
            "\n\nThis project already has these canonical entities — REUSE the "
            "exact existing name when you mean the same thing (do not invent a "
            f"new variant):\n{listed}\n"
        )

    return (
        "Extract typed (subject, predicate, object) fact triples from these "
        "memories. Use short, canonical entity names; prefer a specific "
        "predicate (uses, causes, fixes, prefers, depends_on, ...). For each "
        "fact call:\n"
        "  kg_add(subject, predicate, object, subject_type=<type>, "
        'object_type=<type>, proof="researcher_extracted")\n'
        "Skip generic knowledge; one triple per real relationship."
        f"{vocab_block}\n"
        f"Memories:\n{memory_text}"
    )


def _project_entity_vocab(cwd: str) -> list[str]:
    """Best-effort: the distinct entity surface forms already in this project."""
    try:
        import simba.db
        import simba.kg.store

        pid = simba.db.resolve_project_id(Path(cwd))
        with simba.db.connect(Path(cwd)):
            names = sorted(simba.kg.store._project_entities(pid))
        return names
    except Exception:
        logger.debug("entity-vocab lookup failed", exc_info=True)
        return []


def _dispatch_claude_agent(memories: list[dict], *, cwd: str) -> str | None:
    """Dispatch a researcher agent for deeper, entity-aware fact extraction.

    Returns the ticket_id or None on failure.
    """
    try:
        from simba.orchestration.agents import dispatch_agent
    except ImportError:
        logger.debug("orchestration.agents not available")
        return None

    instructions = build_extraction_prompt(memories, _project_entity_vocab(cwd))
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

    cfg = _sync_cfg()
    if daemon_url is None:
        daemon_url = cfg.daemon_url

    cwd_path = Path(cwd) if isinstance(cwd, str) else cwd
    cwd_str = str(cwd_path)
    result = ExtractResult()

    watermark = get_watermark("memories", "facts", cwd=cwd_path)

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
                # Event time (occurred_at): resolve a narrative date from the
                # memory text, falling back to None when none is present.
                occurred_at = resolve_occurred_at(
                    f"{content} {context}", created_at=created
                )

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
                        status = _store_fact(
                            s, p, o, proof, cwd=cwd_str, occurred_at=occurred_at
                        )
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
            set_watermark(
                "memories",
                "facts",
                latest_ts,
                rows_processed=result.memories_processed,
                cwd=cwd_path,
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
