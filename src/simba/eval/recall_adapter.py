"""In-process recall adapter: build a retriever over a dataset's corpus.

Builds a throwaway LanceDB table + FTS mirror from the dataset corpus and
returns a ``retriever(query) -> [memory_id, ...]`` that runs the **real**
hybrid recall stack (``plan_recall`` + ``hybrid_search`` + RRF). The embedding
function is injected: the CLI passes the live GGUF embedder; tests pass a
deterministic fake, so the harness runs in CI without the model.

LanceDB's async handles are event-loop bound, so we persist only the paths and
reconnect inside each query's own ``asyncio.run`` — slower, but robust and fine
for benchmark-sized corpora.
"""

from __future__ import annotations

import asyncio
import pathlib
import time
import typing

import simba.memory.config
import simba.memory.entity_bridge
import simba.memory.fts
import simba.memory.hybrid
import simba.memory.recall_plan
import simba.memory.vector_db

if typing.TYPE_CHECKING:
    from simba.eval.dataset import Dataset, Memory

EmbedFn = typing.Callable[[str], list[float]]
Retriever = typing.Callable[[str], list[str]]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _row(
    mem: Memory, vector: list[float], default_created: str
) -> dict[str, typing.Any]:
    # A single stable timestamp for all undated rows (passed in), so a uniform
    # corpus stays uniform — otherwise a slow embed loop would stamp later rows
    # with a later second and leak a spurious recency signal into the eval.
    created = mem.created_at or default_created
    return {
        "id": mem.id,
        "type": mem.type,
        "content": mem.content,
        "context": mem.context,
        "tags": "[]",
        "confidence": float(mem.confidence),
        "sessionSource": mem.session_source,
        "projectPath": mem.project_path,
        "createdAt": created,
        "lastAccessedAt": created,
        "accessCount": 0,
        "vector": vector,
    }


async def _create_table(db_path: pathlib.Path, rows: list[dict]) -> None:
    import lancedb

    db = await lancedb.connect_async(str(db_path))
    await db.create_table("memories", rows)


async def _search(
    db_path: pathlib.Path,
    fts_path: str | None,
    cfg: typing.Any,
    query: str,
    embedding: list[float],
    extra_embedding: list[float] | None,
    plan: simba.memory.recall_plan.RecallPlan,
    llm_client: typing.Any,
    entity_bridge_index: typing.Any = None,
    entity_bridge_lookup: dict[str, dict[str, typing.Any]] | None = None,
) -> list[str]:
    import lancedb

    db = await lancedb.connect_async(str(db_path))
    table = await db.open_table("memories")
    fused = await simba.memory.hybrid.hybrid_search(
        table,
        fts_path,
        embedding,
        query,
        min_similarity=plan.min_similarity,
        max_results=plan.max_results,
        filters={},
        cfg=cfg,
        candidate_pool=plan.candidate_pool,
        extra_embedding=extra_embedding,
        llm_client=llm_client,
        entity_bridge_index=entity_bridge_index,
        entity_bridge_lookup=entity_bridge_lookup,
    )
    return [r["id"] for r in fused]


def build_retriever(
    dataset: Dataset,
    cfg: typing.Any | None = None,
    *,
    embed_doc: EmbedFn,
    embed_query: EmbedFn,
    data_dir: str | pathlib.Path,
    llm_client: typing.Any = None,
) -> Retriever:
    """Build the LanceDB+FTS store from ``dataset.corpus`` and return a retriever."""
    cfg = cfg or simba.memory.config.MemoryConfig()
    data_dir = pathlib.Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "memories.lance"
    fts_path = data_dir / simba.memory.fts.FTS_FILENAME

    build_now = _now()
    rows = [_row(m, embed_doc(m.content), build_now) for m in dataset.corpus]
    if not rows:
        return lambda query: []

    asyncio.run(_create_table(db_path, rows))

    simba.memory.fts.init(fts_path, tokenize=cfg.fts_tokenize)
    with simba.memory.fts.connect(fts_path, cfg.fts_tokenize):
        simba.memory.fts.rebuild(rows)

    # Optional entity-bridge index (spec 09): built from the corpus so the lever is
    # measurable on corpora that ship no KG — cheap (regex NER, no LLM).
    eb_index = None
    eb_lookup: dict[str, dict[str, typing.Any]] = {}
    if getattr(cfg, "entity_bridge_enabled", False):
        _ner = getattr(cfg, "entity_bridge_ner", "regex")
        _extract = (
            simba.memory.entity_bridge.spacy_entities
            if _ner == "spacy"
            else simba.memory.entity_bridge.extract_entities
        )
        eb_index = simba.memory.entity_bridge.build_index(
            ((m.id, f"{m.content} {m.context}".strip()) for m in dataset.corpus),
            extract=_extract,
        )
        eb_lookup = {
            m.id: {
                "id": m.id,
                "type": m.type,
                "content": m.content,
                "context": m.context,
                "similarity": 0.0,
                "confidence": float(m.confidence),
                "createdAt": m.created_at or build_now,
                "projectPath": m.project_path,
            }
            for m in dataset.corpus
        }

    def retriever(query: str) -> list[str]:
        # Eval is always the sync (no-cache) path: hyde_mode=="llm" generates the
        # hypothetical answer inline so the benchmark exercises the same 2nd-arm
        # text the daemon resolves to.
        plan = simba.memory.recall_plan.plan_recall(
            query, cfg, llm_client=llm_client, hyde_cache=None
        )
        embedding = embed_query(query)
        extra = embed_query(plan.hyde_text) if plan.hyde_text else None
        return asyncio.run(
            _search(
                db_path,
                str(fts_path),
                cfg,
                query,
                embedding,
                extra,
                plan,
                llm_client,
                entity_bridge_index=eb_index,
                entity_bridge_lookup=eb_lookup,
            )
        )

    return retriever
