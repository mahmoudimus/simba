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

import simba.db
import simba.eval.kg_corpus
import simba.kg.entities
import simba.memory.anticipated
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
    kg: typing.Any = None,
    kg_record_lookup: dict[str, dict[str, typing.Any]] | None = None,
    kg_seeds: list[str] | None = None,
    cwd: pathlib.Path | None = None,
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
        kg_adjacency=kg.adjacency if kg is not None else None,
        kg_entity_memories=kg.entity_memories if kg is not None else None,
        kg_record_lookup=kg_record_lookup,
        kg_seeds=kg_seeds,
        cwd=cwd,
    )
    return [r["id"] for r in fused]


def _write_anticipated_sidecar(
    dataset: Dataset,
    *,
    data_dir: pathlib.Path,
    limit: int,
) -> None:
    """Populate the eval temp sidecar from corpus anticipated queries."""
    with simba.db.connect(data_dir):
        for mem in dataset.corpus:
            if mem.anticipated_queries:
                simba.memory.anticipated.append_queries(
                    memory_id=mem.id,
                    queries=list(mem.anticipated_queries),
                    source="eval_fixture",
                    now=time.time(),
                    limit=limit,
                )


def build_retriever(
    dataset: Dataset,
    cfg: typing.Any | None = None,
    *,
    embed_doc: EmbedFn,
    embed_query: EmbedFn,
    data_dir: str | pathlib.Path,
    llm_client: typing.Any = None,
    kg_extract: typing.Any = None,
) -> Retriever:
    """Build the LanceDB+FTS store from ``dataset.corpus`` and return a retriever.

    When ``cfg.kg_ppr_enabled`` and ``kg_extract`` is given, also build a throwaway
    corpus KG (via the injected extractor) + a record lookup, so the retriever
    exercises the Track B PPR fold on a corpus that ships no KG.
    """
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
    _write_anticipated_sidecar(
        dataset,
        data_dir=data_dir,
        limit=getattr(cfg, "anticipated_query_max_per_memory", 5),
    )

    # Shared record shape for the optional folds below (id -> materializable
    # record): both entity-bridge and Track-B PPR map a corpus id to the same dict.
    def _record(m: Memory) -> dict[str, typing.Any]:
        return {
            "id": m.id,
            "type": m.type,
            "content": m.content,
            "context": m.context,
            "similarity": 0.0,
            "confidence": float(m.confidence),
            "createdAt": m.created_at or build_now,
            "projectPath": m.project_path,
        }

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
        eb_lookup = {m.id: _record(m) for m in dataset.corpus}

    # Optional Track B throwaway KG + record lookup (id -> materializable record).
    kg = None
    kg_lookup: dict[str, dict[str, typing.Any]] = {}
    if getattr(cfg, "kg_ppr_enabled", False) and kg_extract is not None:
        kg = simba.eval.kg_corpus.build_corpus_kg(dataset.corpus, kg_extract)
        kg_lookup = {m.id: _record(m) for m in dataset.corpus}

    def retriever(query: str) -> list[str]:
        # Eval is always the sync (no-cache) path: hyde_mode=="llm" generates the
        # hypothetical answer inline so the benchmark exercises the same 2nd-arm
        # text the daemon resolves to.
        plan = simba.memory.recall_plan.plan_recall(
            query, cfg, llm_client=llm_client, hyde_cache=None
        )
        embedding = embed_query(query)
        extra = embed_query(plan.hyde_text) if plan.hyde_text else None
        seeds = None
        if kg is not None:
            seeds = [
                simba.kg.entities.normalize_entity(t)
                for t in simba.eval.kg_corpus.entities_of(query)
            ]
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
                kg=kg,
                kg_record_lookup=kg_lookup,
                kg_seeds=seeds,
                cwd=data_dir,
            )
        )

    return retriever
