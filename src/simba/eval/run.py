"""CLI orchestration for the eval harness.

``run_dataset`` is the model-free glue (load → build retriever → score) and is
unit-tested with a fake embedder. ``sync_embedders`` loads the real GGUF model
once and exposes synchronous doc/query embedders for the in-process adapter.
"""

from __future__ import annotations

import typing

import simba.eval.dataset
import simba.eval.recall_adapter
import simba.eval.runner
import simba.memory.config

if typing.TYPE_CHECKING:
    import pathlib

EmbedFn = typing.Callable[[str], list[float]]


def run_dataset(
    dataset_path: str | pathlib.Path,
    *,
    ks: tuple[int, ...],
    data_dir: str | pathlib.Path,
    embed_doc: EmbedFn,
    embed_query: EmbedFn,
    cfg: typing.Any | None = None,
) -> simba.eval.runner.EvalReport:
    """Load a dataset, build the real recall retriever, and score it."""
    cfg = cfg or simba.memory.config.MemoryConfig()
    dataset = simba.eval.dataset.load_dataset(dataset_path)
    retriever = simba.eval.recall_adapter.build_retriever(
        dataset,
        cfg,
        embed_doc=embed_doc,
        embed_query=embed_query,
        data_dir=data_dir,
    )
    return simba.eval.runner.run_eval(dataset, retriever, ks=ks)


def sync_embedders(cfg: typing.Any) -> tuple[EmbedFn, EmbedFn]:
    """Load the GGUF model once; return synchronous (embed_doc, embed_query).

    Reuses ``EmbeddingService``'s model resolution + sync embed path so the eval
    uses exactly the production embeddings (same model, same task prefixes).
    """
    import simba.memory.embeddings as emb

    service = emb.EmbeddingService(cfg)
    model_path = service._resolve_model_path()
    service._model = service._load_model(model_path)

    def embed_doc(text: str) -> list[float]:
        return service._embed_sync(text, emb.TaskType.DOCUMENT)

    def embed_query(text: str) -> list[float]:
        return service._embed_sync(text, emb.TaskType.QUERY)

    return embed_doc, embed_query
