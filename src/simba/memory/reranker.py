"""Config-selectable reranker dispatch (spec 22).

The reranker re-scores the fused recall candidates by query relevance before
truncation (the cross-encoder's role). ``rerank`` routes on ``cfg.reranker_mode``:

    "none"          -> candidates unchanged
    "llm"           -> the cloud LLM client (existing ``llm_rerank.rerank``)
    "cross-encoder" -> bge-reranker-v2-m3 GGUF, llama-cpp RANK pooling (default)
    "local-llm"     -> zerank-2 Qwen3-4B GGUF, llama-cpp "Yes"-token logit

Both local backends run GGUF via the EXISTING llama-cpp stack (no torch),
mirroring ``memory.embeddings.EmbeddingService`` loading + log suppression. The
model auto-downloads from the configured HF repo/file on first use; the loaded
model is a lazily-built per-process singleton (one per backend).

Fail-open everywhere: any load/score/backend error leaves the candidates
unchanged. The reranker never drops or duplicates a candidate.
"""

from __future__ import annotations

import ctypes
import logging
import pathlib
import re
import typing

import simba.memory.llm_rerank as llm_rerank
from simba.memory._llama import LLAMA_LOCK

if typing.TYPE_CHECKING:
    import llama_cpp

    import simba.memory.config

logger = logging.getLogger("simba.memory")

# Lazily-built per-process singletons, one per GGUF backend.
_CROSS_ENCODER: _CrossEncoderReranker | None = None
_LOCAL_LLM: _LocalLlmReranker | None = None
# ALL native llama.cpp access (embedder + reranker, load + score) shares this
# one process-global lock — ggml is not safe under concurrent contexts. See
# simba.memory._llama. Guards the lazy load below AND each score() call.
_LOCK = LLAMA_LOCK

# Intent gate (spec 22, LME-gate correction). The reranker is a POINTWISE relevance
# pass: it helps latest/compositional-multihop recall but HURTS multi-evidence
# temporal — it can promote the single most-relevant turn and demote a co-required
# one, breaking the evidence SET (measured: LME complete@5 0.65 -> 0.20). Skip the
# shapes it measurably harms so reranking is a router decision, not a global flag.
_TEMPORAL_MULTI_ENDPOINT = re.compile(
    r"\b(?:days?|weeks?|months?|years?|hours?)\b.*\b(?:between|after|before|since|apart)\b"
    r"|\bbetween\b[^?]*\band\b"
    r"|\bhow long\b.*\b(?:after|before|since|between)\b"
    r"|\bwhich\b[^?]*\b(?:first|earlier|later|come|came|happened)\b[^?]*\bor\b"
    r"|\b(?:first|last)\b[^?]*\bor\b[^?]*\?",
    re.IGNORECASE,
)


def should_rerank(query: str, cfg: typing.Any) -> bool:
    """Whether reranking should fire for ``query`` (intent gate, spec 22).

    When ``cfg.rerank_intent_gating`` is on, skip query shapes the pointwise
    reranker measurably harms (multi-endpoint temporal — needs ALL endpoints
    ranked, which pointwise scoring disrupts). Off -> always True (prior
    behavior). Fail-open: any error -> True (rerank).
    """
    if not getattr(cfg, "rerank_intent_gating", False):
        return True
    try:
        return _TEMPORAL_MULTI_ENDPOINT.search(query or "") is None
    except Exception:
        return True

# Shared C-level llama.cpp log-suppression callback (kept alive at module scope
# to prevent GC of the C callback), mirroring EmbeddingService._llama_log_cb.
_LLAMA_LOG_CB: typing.Any = None


def _silence_llama_logs(llama_cpp: typing.Any) -> None:
    """Install a no-op llama.cpp log callback once (suppresses ggml noise)."""
    global _LLAMA_LOG_CB
    if _LLAMA_LOG_CB is None:
        _LLAMA_LOG_CB = llama_cpp.llama_log_callback(lambda *_args: None)
        llama_cpp.llama_log_set(_LLAMA_LOG_CB, ctypes.c_void_p(0))


def _resolve_gguf(repo: str, filename: str) -> pathlib.Path:
    """Download (or hit the HF cache for) a GGUF file, like the embedder."""
    import huggingface_hub

    return pathlib.Path(
        huggingface_hub.hf_hub_download(repo_id=repo, filename=filename)
    )


def _doc_text(candidate: dict[str, typing.Any]) -> str:
    """Render a candidate as scorer doc text (content + trimmed context)."""
    content = (candidate.get("content") or "").strip()
    ctx = (candidate.get("context") or "").strip()
    return f"{content} — {ctx[:120]}" if ctx else content


def _score_and_reorder(
    query: str,
    candidates: list[dict[str, typing.Any]],
    scorer: typing.Any,
    max_candidates: int,
) -> list[dict[str, typing.Any]]:
    """Score the head with ``scorer.score(query, doc)`` and sort desc; tail kept.

    Stable: candidates with equal score keep their original relative order. Never
    drops or duplicates a candidate.
    """
    head = candidates[:max_candidates]
    tail = candidates[max_candidates:]
    scored = [(scorer.score(query, _doc_text(c)), idx, c) for idx, c in enumerate(head)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, _, c in scored] + tail


# ── GGUF backend: cross-encoder (bge-reranker-v2-m3, RANK pooling) ───────────


class _CrossEncoderReranker:
    """bge-reranker-v2-m3 GGUF scorer via llama-cpp RANK pooling.

    With ``pooling_type=LLAMA_POOLING_TYPE_RANK`` llama-cpp emits a single
    relevance logit per encoded sequence (the rest of the returned vector is
    zero); the pair is joined as ``"query</s>doc"`` (XLM-RoBERTa SEP) and the
    score is element 0 of the embedding. (Measured: gold-doc >> distractor-doc.)
    """

    def __init__(self, model: llama_cpp.Llama) -> None:
        self._model = model

    @classmethod
    def load(cls, cfg: simba.memory.config.MemoryConfig) -> _CrossEncoderReranker:
        import llama_cpp

        _silence_llama_logs(llama_cpp)
        path = _resolve_gguf(cfg.reranker_model_repo, cfg.reranker_model_file)
        model = llama_cpp.Llama(
            model_path=str(path),
            embedding=True,
            n_ctx=0,  # use the model's training context
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_RANK,
            n_gpu_layers=cfg.n_gpu_layers,
            verbose=False,
        )
        logger.info("[rerank] cross-encoder loaded: %s", path.name)
        return cls(model)

    def score(self, query: str, doc: str) -> float:
        with _LOCK:
            res = self._model.create_embedding(f"{query}</s>{doc}")
        emb = res["data"][0]["embedding"]
        return float(emb[0]) if isinstance(emb, list) else float(emb)


# ── GGUF backend: local-llm (zerank-2 Qwen3-4B, "Yes"-token logit) ───────────


class _LocalLlmReranker:
    """zerank-2 Qwen3-4B GGUF scorer via llama-cpp generative logits.

    Per zeroentropy/zerank-2 (a CrossEncoder with a LogitScore head): the pair is
    formatted with the model's chat template (query as system, document as user)
    and the relevance score is the logit of the "Yes"/true token at the final
    (assistant) position. ``logits_all=True`` retains per-position logits so the
    last row is readable. (Measured: gold-doc >> distractor-doc.)
    """

    def __init__(self, model: llama_cpp.Llama, true_token: int) -> None:
        self._model = model
        self._true_token = true_token

    @classmethod
    def load(cls, cfg: simba.memory.config.MemoryConfig) -> _LocalLlmReranker:
        import llama_cpp

        _silence_llama_logs(llama_cpp)
        path = _resolve_gguf(cfg.reranker_local_llm_repo, cfg.reranker_local_llm_file)
        model = llama_cpp.Llama(
            model_path=str(path),
            n_ctx=cfg.reranker_n_ctx,
            logits_all=True,
            n_gpu_layers=cfg.n_gpu_layers,
            verbose=False,
        )
        logger.info("[rerank] local-llm loaded: %s", path.name)
        return cls(model, cfg.reranker_local_llm_true_token)

    @staticmethod
    def _prompt(query: str, doc: str) -> str:
        # zeroentropy/zerank-2 chat_template.jinja (query/document roles).
        return (
            f"<|im_start|>system\n{query}<|im_end|>\n"
            f"<|im_start|>user\n{doc}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def score(self, query: str, doc: str) -> float:
        with _LOCK:
            toks = self._model.tokenize(
                self._prompt(query, doc).encode("utf-8"), add_bos=True, special=True
            )
            self._model.reset()
            self._model.eval(toks)
            logits = self._model.eval_logits[-1]  # final (assistant) position
            return float(logits[self._true_token])


# ── lazy singleton accessors (patched in unit tests) ─────────────────────────


def _get_cross_encoder(
    cfg: simba.memory.config.MemoryConfig,
) -> _CrossEncoderReranker:
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        with _LOCK:
            if _CROSS_ENCODER is None:
                _CROSS_ENCODER = _CrossEncoderReranker.load(cfg)
    return _CROSS_ENCODER


def _get_local_llm(
    cfg: simba.memory.config.MemoryConfig,
) -> _LocalLlmReranker:
    global _LOCAL_LLM
    if _LOCAL_LLM is None:
        with _LOCK:
            if _LOCAL_LLM is None:
                _LOCAL_LLM = _LocalLlmReranker.load(cfg)
    return _LOCAL_LLM


# ── public dispatch ──────────────────────────────────────────────────────────


def rerank(
    query: str,
    candidates: list[dict[str, typing.Any]],
    *,
    cfg: typing.Any,
    llm: typing.Any = None,
    max_candidates: int = 20,
) -> list[dict[str, typing.Any]]:
    """Reorder ``candidates`` by query relevance, routed on ``cfg.reranker_mode``.

    Fail-open: any backend/load/score error returns the candidates unchanged.
    Never drops or duplicates a candidate. ``llm`` is the cloud client used only
    by the "llm" backend (ignored otherwise).
    """
    if not candidates:
        return candidates

    mode = getattr(cfg, "reranker_mode", "none")

    if mode == "none":
        return candidates

    if mode == "llm":
        return llm_rerank.rerank(
            query, candidates, client=llm, max_candidates=max_candidates
        )

    if mode == "cross-encoder":
        accessor: typing.Callable[[typing.Any], typing.Any] = _get_cross_encoder
    elif mode == "local-llm":
        accessor = _get_local_llm
    else:
        # Unknown mode -> fail-open, unchanged.
        logger.debug("[rerank] unknown reranker_mode %r (fail-open)", mode)
        return candidates

    try:
        scorer = accessor(cfg)
        return _score_and_reorder(query, candidates, scorer, max_candidates)
    except Exception:
        logger.debug("[rerank] %s backend failed (fail-open)", mode, exc_info=True)
        return candidates
