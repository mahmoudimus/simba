# 22 — Config-selectable reranker backends (retire the 23s LLM rerank)

> **STATUS (2026-06-10): READY TO IMPLEMENT (Pillar 0 spike first).** Measured
> ([[cross-encoder-vs-llm-rerank]]): a real cross-encoder REPLACES simba's cloud-LLM
> reranker. LoCoMo n=60 recall@5 — NONE 0.581 / bge-base 0.635 / **bge-v2-m3 0.688
> @240ms** / **zerank-2 0.718 @1.3s** / LLM 0.721 @**23s**. Make the reranker a
> config-selectable backend; default the lean local cross-encoder; keep the cloud LLM
> as an option. Branch off `main`.

## Why

simba's only reranker today is `memory/llm_rerank.py` — a **cloud-LLM** relevance pass
(~23s/query, the latency trap). A local cross-encoder (`bge-reranker-v2-m3`) gets 95%
of its recall at **240ms** (~96× faster); a local 4B reranker (`zerank-2`) matches it
at 1.3s. Both run via **GGUF on simba's existing llama-cpp stack — no torch**. So:
make `reranker_mode` selectable, default the lean cross-encoder, retire the cloud
dependency on the recall hot path.

## The backends (`memory.reranker_mode`)

| mode | model | mechanism | deps | latency |
|---|---|---|---|---|
| **`cross-encoder`** (default) | `bge-reranker-v2-m3` GGUF | llama-cpp `LLAMA_POOLING_TYPE_RANK` (1 score/pair) | llama-cpp (existing) | ~240ms |
| `local-llm` | `zerank-2` Qwen3-4B GGUF (`godkingleto/zerank-2-Q4_K_M-GGUF`) | llama-cpp **generative/logit** scoring (per the model's protocol) | llama-cpp (existing) | ~1.3s |
| `llm` | the cloud LLM client | existing `llm_rerank.rerank` (unchanged) | none new | ~23s |
| `none` | — | skip reranking | — | 0 |

Both local backends are **GGUF via llama-cpp** (mirror `memory/embeddings.py`
`EmbeddingService` loading: `Llama(embedding=True, pooling_type=...)`). No
`sentence-transformers`/torch in the default path. (`llama-cpp-python` 0.3.16 has
`LLAMA_POOLING_TYPE_RANK=4` + `pooling_type` on `Llama.__init__`, but **no high-level
`rerank()` method** — score via the RANK-pooled embedding output for the encoder, and
via generation/logits for zerank-2.)

## Pillar 0 — SPIKE the two GGUF scoring paths FIRST (decision-grade)

Before wiring, prove each local backend produces *sane relevance scores* (the probe
ranking is the oracle). On ~10 LoCoMo (query, gold-doc) vs (query, distractor-doc) pairs:
- **cross-encoder**: load a `bge-reranker-v2-m3` GGUF with `pooling_type=LLAMA_POOLING_TYPE_RANK`;
  confirm gold-doc scores > distractor-doc and the top-5 ordering matches the probe's
  `bge-v2-m3` arm (recall@5 ≈ 0.688).
- **local-llm**: load `godkingleto/zerank-2-Q4_K_M-GGUF`; replicate zerank-2's scoring
  protocol (read its model card / `generate.py` for the prompt format + which
  token/logit is the score), confirm gold > distractor.

**Decision:** a path that yields sane scores → that backend ships. A path that's
intractable (e.g. zerank-2's generative protocol won't replicate cleanly via raw
llama-cpp) → ship that backend **behind an optional `sentence-transformers` extra**
(the probe's working path, `CrossEncoder(model, trust_remote_code=True)`), torch-gated,
not in the default. The **cross-encoder GGUF path is the must-have**; local-llm GGUF is
best-effort with the ST fallback. Commit the spike result here.

### Pillar-0 SPIKE RESULT (2026-06-10) — BOTH PATHS SANE via raw llama-cpp (no torch)

Spike `.simba/rerank_spike.py` (gitignored), 10 LoCoMo-style (query, gold, distractor)
pairs, llama-cpp-python 0.3.16, M-series. **No `sentence-transformers` fallback needed
— both backends ship as GGUF.**

| path | model / repo | mechanism | gold>dist | latency |
|---|---|---|---|---|
| **cross-encoder** | `gpustack/bge-reranker-v2-m3-GGUF` / `bge-reranker-v2-m3-Q4_K_M.gguf` | `Llama(embedding=True, pooling_type=LLAMA_POOLING_TYPE_RANK)`; pair joined `"query</s>doc"` (XLM-RoBERTa SEP); score = `embedding[0]` (RANK head emits one logit, rest 0.0) | **10/10** | ~32ms/pair |
| **local-llm** | `godkingleto/zerank-2-Q4_K_M-GGUF` / `zerank-2-q4_k_m.gguf` | `Llama(logits_all=True)`; prompt from zerank-2 `chat_template.jinja` (query=system, doc=user, assistant gen-prompt); score = logit of `true_token_id=9454` ("Yes", from `1_LogitScore/config.json`) at the final position via `eval_logits[-1]` | **10/10** | ~336ms/pair |

Score separation is large and consistent (cross-encoder gold ≈ +0.6…+5.5 vs dist
≈ −5.5…−10.5; local-llm gold ≈ +9…+14 vs dist ≈ −9…−13). Two raw-llama-cpp
footguns found & fixed: (1) bge RANK pooling returns a full-width vector with only
`emb[0]` meaningful and the pair MUST be joined with `</s>`, not a tab/newline;
(2) zerank-2 needs `logits_all=True` + reading `eval_logits[-1]` (default
`Llama` does not retain per-position logits → all-zero scores). Both shipped as
GGUF backends; the cloud `llm` backend remains the fallback.

## Build (after the spike)

### 1. Reranker dispatch (`memory/reranker.py`)
```python
def rerank(query, candidates, *, cfg, llm=None, max_candidates=20) -> list[dict]:
    # route on cfg.reranker_mode:
    #   "none"          -> candidates unchanged
    #   "llm"           -> llm_rerank.rerank(query, candidates, client=llm, ...)  (existing)
    #   "cross-encoder" -> _CrossEncoderReranker (GGUF, RANK pooling) score+sort
    #   "local-llm"     -> _LocalLlmReranker (zerank-2 GGUF generative) score+sort
    # Fail-open: on any backend error, return candidates unchanged (never drop/dup).
```
- `hybrid.py` (lines ~328/353) currently calls `simba.memory.llm_rerank.rerank` directly
  → route through `reranker.rerank(..., cfg=...)` instead. Behavior-identical when
  `reranker_mode="llm"`.
- The GGUF reranker is a lazily-loaded singleton (like `EmbeddingService`), model
  auto-downloaded from `reranker_model_repo`/`_file` (HF, like the embedder). Reuses the
  embedder's llama-cpp loading + log-suppression pattern.

### 2. Config (`@configurable("memory")`, all defaults lean + safe)
```
reranker_mode: str = "cross-encoder"            # cross-encoder | local-llm | llm | none
reranker_model_repo: str = "<bge-reranker-v2-m3 GGUF repo>"
reranker_model_file: str = "<...Q?_K_M.gguf>"
reranker_local_llm_repo: str = "godkingleto/zerank-2-Q4_K_M-GGUF"
reranker_local_llm_file: str = "<...Q4_K_M.gguf>"
```
Keep the existing `llm_rerank_*` fields (the `llm` backend reads them). No hidden constants.

### 3. Default-flip discipline
Default `cross-encoder` **per the measured win**, BUT the flip's acceptance gate is an
**LME A/B** (the bake-off was LoCoMo only): `simba eval bench longmemeval --qa` with
`reranker_mode=cross-encoder` vs `llm` — cross-encoder must be ≥ llm on QA (and the
multi-session recall) at lower latency. If LME contradicts LoCoMo, default stays `llm`
and cross-encoder is opt-in. Record the LME numbers here.

> **LME A/B: PENDING (orchestrator).** The code defaults `cross-encoder` per the
> LoCoMo measurement; the LME A/B acceptance gate has NOT been run in this
> implementation pass. If the LME A/B contradicts LoCoMo, set the default back to
> `llm` via `reranker_mode` (config-only, no code change). Numbers to be recorded
> here once run.

## TDD (RED first)

- `reranker.rerank` routing: `none` → unchanged; `llm` → delegates to `llm_rerank`
  (mock); unknown mode → fail-open unchanged.
- cross-encoder backend: with a fake scorer (inject the score fn), reorders by score,
  never drops/dups, fail-open on load error. (The real GGUF load is exercised by the
  Pillar-0 spike + an opt-in integration test, not unit CI — no model download in CI.)
- config: `reranker_mode == "cross-encoder"`; fields present.
- `hybrid.py` regression fence: `reranker_mode="llm"` path is byte-identical to today.

## Constraints / non-goals

- Pure Python under `src/simba/`; **no new *required* deps** (GGUF via existing
  llama-cpp; `sentence-transformers` only as an *optional extra* if a local backend
  needs the fallback). All config via `@configurable`. ruff-clean. TDD. Fail-open.
- No CI model downloads (gate real-model tests behind an opt-in marker).
- Async/sync rerank wiring (`llm_rerank_mode`) + the rerank cache stay as-is — this swaps
  the *scorer*, not the scheduling.

## Acceptance

- Pillar 0 spike result (both paths, sane-score verdict) committed here.
- `uv run pytest` green; ruff clean; `simba config get memory.reranker_mode` → `cross-encoder`.
- `hybrid.py` `llm`-mode regression fence green (behavior-preserving).
- LME A/B recorded; default-flip justified by it.
- Commit on `feat/reranker-backends` (no Claude attribution; no push; don't touch `uv.lock`).
