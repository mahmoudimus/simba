"""Configuration for the embedder bake-off (`simba eval embedders`)."""

from __future__ import annotations

import dataclasses
import pathlib

import simba.config


@simba.config.configurable("embedders")
@dataclasses.dataclass
class EmbedderBakeoffConfig:
    # Where candidate GGUFs (and fetched benchmark data) are downloaded. Empty ⇒
    # ~/.cache/simba/models. Each repo lands in <models_dir>/<repo-name>/.
    models_dir: str = ""
    # llama.cpp server binary used for the dense + pipeline arms. Must be new
    # enough for every candidate architecture (gemma-embedding2 needs >= b12651).
    llama_server: str = "llama-server"
    # Results, rankings, and the embedding cache (relative to project root).
    out_dir: str = ".simba/eval/embedders"
    # Extra / overriding candidate specs (TOML, `[[candidate]]` tables). Relative
    # to project root. Missing file ⇒ built-in candidates only.
    candidates_file: str = ".simba/eval/embedders/candidates.toml"
    # Benchmark inputs. Empty ⇒ <models_dir>/datasets/{locomo,longmemeval}/, fetched
    # on first use (LoCoMo from snap-research/locomo, LongMemEval-S from
    # xiaowu0162/longmemeval-cleaned). The file's sha256 is recorded per run.
    locomo_path: str = ""
    longmemeval_path: str = ""
    # LongMemEval-S is ~231k turns; score a stratified, fixed-seed subsample.
    lme_questions: int = 150
    sample_seed: int = 20261123
    # Baseline candidate key every delta / bootstrap CI is reported against.
    baseline: str = "bge_mean"
    # Paired-bootstrap resamples for the 95% CIs, and the resampling seed (same
    # seed ⇒ every model is resampled on identical case draws).
    bootstrap: int = 2000
    bootstrap_seed: int = 7
    # Ranked ids kept per case in the dense arm (max k any metric can use).
    top_k: int = 50
    # Tokens reserved for BOS/CLS/SEP/EOS specials when fitting a text into a
    # candidate's ctx. Part of the vector fingerprint.
    token_headroom: int = 8
    # llama-server process layout. Port is the first of a small range (dense,
    # pipeline, refcheck use port, port+1, port+2).
    port: int = 18741
    slots: int = 4
    batch: int = 16
    gpu_layers: int = 99
    server_start_timeout_s: int = 300
    request_timeout_s: int = 600

    def models_path(self) -> pathlib.Path:
        if self.models_dir:
            return pathlib.Path(self.models_dir).expanduser()
        return pathlib.Path.home() / ".cache" / "simba" / "models"

    def resolve(self, rel: str, root: pathlib.Path) -> pathlib.Path:
        p = pathlib.Path(rel).expanduser()
        return p if p.is_absolute() else root / p

    def locomo_file(self, root: pathlib.Path) -> pathlib.Path:
        if self.locomo_path:
            return self.resolve(self.locomo_path, root)
        return self.models_path() / "datasets" / "locomo" / "locomo10.json"

    def lme_path(self, root: pathlib.Path) -> pathlib.Path:
        if self.longmemeval_path:
            return self.resolve(self.longmemeval_path, root)
        return (
            self.models_path()
            / "datasets"
            / "longmemeval"
            / "longmemeval_s_cleaned.json"
        )


def load_config() -> EmbedderBakeoffConfig:
    return simba.config.load("embedders")
