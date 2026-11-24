"""Candidate embedder specs: the built-in registry plus user TOML additions.

A spec pins everything that changes the vector a model produces — the exact GGUF,
pooling, query/document prefixes, input budget — so a result is reproducible and
the embedding cache can never mix recipes. To evaluate a new model, add a
``[[candidate]]`` table to ``embedders.candidates_file`` (see docs/embedder-bakeoff.md);
no code change is needed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
import typing

if typing.TYPE_CHECKING:
    import pathlib

POOLINGS = ("mean", "cls", "last", "model")

# Bump when the embedding procedure itself changes (truncation algorithm, input
# assembly): old cached vectors then stop matching any fingerprint.
RECIPE_VERSION = 1

# Default instruction for instruct-style (last-token) embedders, in the official
# Qwen3/Harrier `Instruct: …\nQuery:` format. Measured 2026-11-23: a custom
# memory-flavoured instruction scored worse than this one for both models.
WEB_INSTRUCT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer "
    "the query\nQuery:"
)


@dataclasses.dataclass(frozen=True)
class EmbedderSpec:
    key: str
    repo: str  # HF repo holding the GGUF
    file: str  # GGUF filename inside the repo
    pooling: str = "model"  # mean | cls | last | model (GGUF metadata default)
    query_prefix: str = ""
    doc_prefix: str = ""
    ctx: int = 4096  # max input tokens per text (prefix included)
    eos: str = ""  # literal appended to every input (only if GGUF add_eos=false)
    dims: int = 0  # >0 ⇒ Matryoshka slice to the first N dims, renormalized
    ref_repo: str = ""  # Sentence-Transformers reference for `refcheck`
    notes: str = ""

    def __post_init__(self) -> None:
        if self.pooling not in POOLINGS:
            msg = f"{self.key}: pooling must be one of {POOLINGS}, got {self.pooling!r}"
            raise ValueError(msg)
        if not self.file.endswith(".gguf"):
            raise ValueError(f"{self.key}: file must be a .gguf, got {self.file!r}")

    def vector_fingerprint(
        self, runtime: str, artifact_sha: str = "", headroom: int = 8
    ) -> str:
        """Identity of the raw vectors (excludes ``dims``: slicing is post-hoc).

        Everything that changes the vector — the GGUF bytes (``artifact_sha``),
        pooling, prefixes, input budget and headroom, EOS, the llama.cpp build and
        the embedding procedure version — is in the key, so editing any of them
        can never serve vectors produced by a different recipe.
        """
        ident = {
            "recipe": RECIPE_VERSION,
            "artifact_sha": artifact_sha,
            "headroom": headroom,
            "repo": self.repo,
            "file": self.file,
            "pooling": self.pooling,
            "qp": self.query_prefix,
            "dp": self.doc_prefix,
            "ctx": self.ctx,
            "eos": self.eos,
            "runtime": runtime,
        }
        blob = json.dumps(ident, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def local_path(self, models_dir: pathlib.Path) -> pathlib.Path:
        return models_dir / self.repo.split("/")[-1] / self.file


_BGE = dict(
    repo="CompendiumLabs/bge-large-en-v1.5-gguf",
    file="bge-large-en-v1.5-q8_0.gguf",
    query_prefix="Represent this sentence for searching relevant passages: ",
    ctx=512,
    ref_repo="BAAI/bge-large-en-v1.5",
)
_QWEN3 = dict(
    repo="Qwen/Qwen3-Embedding-0.6B-GGUF",
    file="Qwen3-Embedding-0.6B-Q8_0.gguf",
    pooling="last",
    ref_repo="Qwen/Qwen3-Embedding-0.6B",
)
_HARRIER = dict(
    repo="mradermacher/harrier-oss-v1-0.6b-GGUF",
    file="harrier-oss-v1-0.6b.Q8_0.gguf",
    pooling="last",
    ref_repo="microsoft/harrier-oss-v1-0.6b",
)
_NEMOTRON = dict(
    repo="mradermacher/Nemotron-3-Embed-1B-BF16-GGUF",
    file="Nemotron-3-Embed-1B-BF16.Q8_0.gguf",
    pooling="mean",
    query_prefix="query: ",
    doc_prefix="passage: ",
    ref_repo="nvidia/Nemotron-3-Embed-1B-BF16",
)

BUILTIN: tuple[EmbedderSpec, ...] = (
    EmbedderSpec(
        "bge_mean",
        pooling="mean",
        notes="deployed Simba recipe (MEAN forced over GGUF's CLS)",
        **_BGE,
    ),
    EmbedderSpec(
        "bge_cls", pooling="cls", notes="BGE reference pooling (BAAI)", **_BGE
    ),
    EmbedderSpec(
        "arctic_l2",
        repo="Casual-Autopsy/snowflake-arctic-embed-l-v2.0-gguf",
        file="snowflake-arctic-embed-l-v2.0-q8_0.gguf",
        pooling="cls",
        query_prefix="query: ",
        ref_repo="Snowflake/snowflake-arctic-embed-l-v2.0",
    ),
    EmbedderSpec(
        "egemma300",
        repo="ggml-org/embeddinggemma-300M-GGUF",
        file="embeddinggemma-300M-Q8_0.gguf",
        query_prefix="task: search result | query: ",
        doc_prefix="title: none | text: ",
        ctx=2048,
        ref_repo="google/embeddinggemma-300m",
    ),
    EmbedderSpec(
        "egemma2",
        repo="ggml-org/embeddinggemma-2-GGUF",
        file="embeddinggemma-2-Q8_0.gguf",
        query_prefix="task: search result | query: ",
        doc_prefix="title: none | text: ",
        ref_repo="google/embeddinggemma-2",
        notes="llama-cpp-python 0.3.38 truncates output to 512 of 768 dims",
    ),
    EmbedderSpec("qwen3_06", query_prefix=WEB_INSTRUCT, **_QWEN3),
    EmbedderSpec("harrier_06", query_prefix=WEB_INSTRUCT + " ", **_HARRIER),
    EmbedderSpec("nemotron3_1b", **_NEMOTRON),
    EmbedderSpec("nemotron3_1b_1024", dims=1024, notes="Matryoshka slice", **_NEMOTRON),
)

# Default run set: baseline + every candidate that has not been ruled out.
DEFAULT_SET = ("bge_mean", "bge_cls", "arctic_l2", "egemma300", "nemotron3_1b")


def _from_table(t: dict) -> EmbedderSpec:
    fields = {f.name for f in dataclasses.fields(EmbedderSpec)}
    unknown = set(t) - fields
    if unknown:
        raise ValueError(
            f"candidate {t.get('key')!r}: unknown fields {sorted(unknown)}"
        )
    return EmbedderSpec(**t)


def load_candidates(path: pathlib.Path | None) -> dict[str, EmbedderSpec]:
    """Built-ins, then ``[[candidate]]`` tables from ``path`` (same key overrides)."""
    out = {s.key: s for s in BUILTIN}
    if path is not None and path.exists():
        data = tomllib.loads(path.read_text())
        for t in data.get("candidate", []):
            spec = _from_table(t)
            out[spec.key] = spec
    return out


def select(cands: dict[str, EmbedderSpec], keys: list[str]) -> list[EmbedderSpec]:
    missing = [k for k in keys if k not in cands]
    if missing:
        raise KeyError(f"unknown candidate(s): {', '.join(missing)}")
    return [cands[k] for k in keys]
