# Embedder bake-off (`simba eval embedders`)

A repeatable way to decide whether a new embedding model should replace Simba's
deployed one. Re-run it when a new embedder ships, when llama.cpp gains a new
architecture, or before changing `memory.model_*`.

## What it measures

| Arm | Question it answers | Benchmarks |
|---|---|---|
| `dense` | Does the representation alone rank the evidence higher? Cosine only: no floors, caps, BM25 or reranker. | LoCoMo (1,977 q, full), LongMemEval-S (stratified fixed-seed subsample, default 150 of 470) |
| `pipeline` | Does it improve what a user actually gets? Simba's own recall path (`recall_adapter.build_retriever`) with the effective deployed `memory` config, reranker **off**: on a rerank-cache miss the daemon serves the un-reranked fused order (`memory/hybrid.py`, async rerank branch), so this is the cold-query result. | LoCoMo |

Metrics per case: R@{1,3,5,10,20,50}, complete-evidence (`All@k`: every gold
item in the top k), MRR, nDCG@10. Every delta against the baseline
(`embedders.baseline`, default `bge_mean` = the deployed recipe) is a paired
bootstrap mean with a 95% CI; `*` marks CIs that exclude 0.

Two checks gate a candidate before its numbers are trusted:

- `refcheck` compares the GGUF (through llama-server) with the authors'
  Sentence-Transformers model on identical inputs. A conversion or pooling bug
  otherwise looks like "this model is worse". Pass = min cosine ≥ 0.99 and top-5
  agreement ≥ 0.9.
- `inprocess` loads the GGUF the way the daemon does (in-process
  llama-cpp-python, `EmbeddingService`'s forced MEAN pooling) and reports
  whether it loads, its output dim, whether its vectors match the correctly
  pooled server vectors, plus query latency, doc throughput and peak RSS.

## One-time setup

```bash
# where GGUFs and benchmark data go (default ~/.cache/simba/models)
simba config set --global embedders.models_dir /Volumes/code/llms
# a llama.cpp build new enough for every candidate architecture
# (prebuilt: github.com/ggml-org/llama.cpp/releases, llama-bNNNNN-bin-macos-arm64)
simba config set --global embedders.llama_server /path/to/llama-server
```

LoCoMo and LongMemEval-S are fetched on first use into
`<models_dir>/datasets/`. The sha256 of each benchmark file is recorded in
every run.

## Running

```bash
simba eval embedders list                       # registry; ✓ = downloaded
simba eval embedders run                        # default set, both arms, both benchmarks
simba eval embedders run --models bge_mean,mynew --bench locomo
simba eval embedders score                      # re-score saved runs -> report.md
simba eval embedders inprocess --models mynew
uv run --with sentence-transformers --with torch --with torchvision --with pillow \
  simba eval embedders refcheck --models mynew
```

`run` downloads missing GGUFs and skips any (arm, benchmark, model) whose saved
result has the same **vector fingerprint**: a hash of the GGUF, pooling,
prefixes, input budget, EOS and the llama.cpp build. Changing any of those
re-runs that model automatically, and the embedding cache
(`<out_dir>/cache/<fingerprint>/`) can never serve vectors from a different
recipe. `--force` re-runs regardless.

Outputs, under `embedders.out_dir` (default `.simba/eval/embedders`):

- `runs/<arm>/<bench>/<key>.json`: per-case rankings plus provenance (spec,
  fingerprint, llama.cpp build, simba git sha, benchmark sha, sample).
- `report.md` and `summary.json`: scored tables, deltas, R@10 by intent, and the
  latest refcheck / inprocess rows.
- `refcheck.jsonl` and `inprocess.jsonl` (append-only).

Approximate cost on an M-series Mac: LoCoMo is about 0.5 to 3 minutes per
model. LongMemEval-S at 150 questions (about 74k turns) is about 12 to 60
minutes per model.

## Adding a new model

No code change is needed. Add a `[[candidate]]` table to
`embedders.candidates_file` (default `.simba/eval/embedders/candidates.toml`):

```toml
[[candidate]]
key = "newembed_06"
repo = "someorg/NewEmbed-0.6B-GGUF"   # HF repo holding the GGUF
file = "NewEmbed-0.6B-Q8_0.gguf"      # use Q8_0 to match the baseline
pooling = "last"                      # mean | cls | last | model (GGUF metadata)
query_prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
doc_prefix = ""
ctx = 4096                            # max input tokens (prefix included)
eos = ""                              # only if the GGUF has add_eos_token=false
dims = 0                              # >0 = Matryoshka slice, renormalized
ref_repo = "someorg/NewEmbed-0.6B"    # for refcheck
```

Use the model card's **official** prompts and pooling. On 2026-11-23 a custom
memory-flavoured instruction cost Qwen3-0.6B 0.067 R@10 on LoCoMo compared with
its official prompt. A table with an existing `key` overrides the built-in spec.

Then run `run --models bge_mean,newembed_06`, `refcheck`, and `inprocess`.

## Reading the result

Promote a model only when all of these hold:

1. `refcheck` passes, so the numbers reflect the model rather than a broken
   conversion.
2. `dense` improves on both benchmarks with a CI that excludes 0, and there is
   no material per-intent regression.
3. `pipeline` R@3 (what the user actually receives) improves.
4. `inprocess` shows it loads in the pinned llama-cpp-python with
   `matches_server = true`. If it doesn't, the daemon needs a newer binding or a
   pooling change first.
5. The RSS, latency and re-embed time are acceptable.

Adopting a model means re-embedding the corpus. Re-embed into a new table: the
current `reembed` keeps a row's old vector when re-embedding fails, which would
mix vector spaces of the same width. Also check the duplicate (0.92) and
supersession thresholds under the new model's cosine scale.

## Results, 2026-11-23 (llama.cpp b12651, Q8_0 everywhere)

All eight GGUFs passed `refcheck` (min cosine ≥ 0.9976).

| Model | LoCoMo dense ΔR@10 | Pipeline ΔR@3 | LME-S dense ΔR@10 | LME-S ΔMRR |
|---|---|---|---|---|
| nemotron3_1b | **+0.0433*** | **+0.0306*** | **+0.1031*** | +0.0545* |
| nemotron3_1b_1024 | +0.0218* | – | +0.0939* | +0.0250 |
| egemma300 | +0.0452* | +0.0277* | +0.0486* | −0.0897* |
| arctic_l2 | +0.0129 | +0.0125 | −0.0233 | −0.0058 |
| bge_cls | −0.0348* | −0.0039 | +0.0188* | −0.0336* |
| harrier_06 (official prompt) | −0.0184* | – | – | – |
| egemma2 | −0.0229* | – | – | – |
| qwen3_06 (official prompt) | −0.0921* | – | – | – |

Findings:

- BGE's reference CLS pooling is *worse* than Simba's MEAN override on LoCoMo.
  Do not "fix" BGE's pooling.
- BGE truncated 18.52% of LongMemEval turns at its 512-token limit.
- llama-cpp-python 0.3.16 cannot load gemma-embedding, gemma-embedding2 or
  mistral3. 0.3.38 can, and it leaves BGE's vectors unchanged (cosine
  0.999999). Building it on macOS needs `CC=/usr/bin/clang
  CXX=/usr/bin/clang++`, because CMake otherwise picks Homebrew `llvm@18`, which
  fails on the SDK's visionOS annotations.
- 0.3.38 truncates EmbeddingGemma 2's output to 512 of its 768 dims.
- Nemotron 3 in-process: 1,548 MB peak RSS (BGE 528 MB), 36.5 docs/s (BGE
  67.5), 13.2 ms median query latency (BGE 10.7).
