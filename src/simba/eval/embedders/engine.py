"""llama-server lifecycle + a cached, context-aware HTTP embedder.

The bake-off drives every candidate through one llama.cpp build (``llama-server
--embeddings``) so architectures newer than the in-process llama-cpp-python can
still be measured, and every candidate is compared on the same runtime.
"""

from __future__ import annotations

import contextlib
import hashlib
import pathlib
import subprocess
import time
import typing

import numpy as np

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from .candidates import EmbedderSpec

TokenCount = typing.Callable[[str], int]


def runtime_version(binary: str) -> str:
    """``build N, commit X`` of the llama-server binary (part of the cache key)."""
    try:
        out = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot run llama-server at {binary!r}: {exc}") from exc
    for line in (out.stdout + out.stderr).splitlines():
        if line.startswith("version:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(f"unrecognised `{binary} --version` output")


def artifact_sha(path: pathlib.Path) -> str:
    """sha256[:16] of a GGUF (part of the vector fingerprint)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()[:16]


def port_free(port: int) -> bool:
    """True unless something is accepting connections on 127.0.0.1:port.

    A connect probe, not a bind test: a just-stopped server leaves the port in
    TIME_WAIT, which fails a plain bind() but cannot serve requests.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) != 0


@contextlib.contextmanager
def llama_server(
    binary: str,
    gguf: pathlib.Path,
    *,
    pooling: str,
    ctx: int,
    port: int,
    slots: int,
    log_path: pathlib.Path,
    gpu_layers: int = 99,
    start_timeout_s: float = 300,
) -> Iterator[str]:
    """Run an embedding server for one GGUF; yield its base URL.

    Refuses a busy port and verifies the healthy server reports our model path —
    otherwise a leftover server on the port would silently supply (and poison the
    cache with) another model's vectors.
    """
    import httpx

    if not port_free(port):
        msg = f"port {port} is in use; stop that server or set embedders.port"
        raise RuntimeError(msg)
    cmd = [
        binary, "-m", str(gguf), "--embeddings", "--port", str(port),
        "-c", str(ctx * slots), "-np", str(slots), "-b", str(ctx), "-ub", str(ctx),
        "-ngl", str(gpu_layers),
    ]  # fmt: skip
    if pooling != "model":
        cmd += ["--pooling", pooling]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + start_timeout_s
            while True:
                if proc.poll() is not None:
                    msg = f"llama-server exited loading {gguf.name}; see {log_path}"
                    raise RuntimeError(msg)
                with contextlib.suppress(httpx.HTTPError):
                    if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                        break
                if time.monotonic() > deadline:
                    raise RuntimeError(f"llama-server not healthy; see {log_path}")
                time.sleep(0.5)
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited; see {log_path}")
            props = httpx.get(f"{url}/props", timeout=10).json()
            served = props.get("model_path", "")
            if served and pathlib.Path(served).resolve() != gguf.resolve():
                raise RuntimeError(f"port {port} serves {served}, not {gguf}")
            yield url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


def fit_to_budget(
    text: str,
    budget: int,
    count: TokenCount,
    pieces: Callable[[str], list[str]],
) -> tuple[str, bool]:
    """Truncate ``text`` to at most ``budget`` tokens. Returns (text, truncated).

    Every token covers >= 1 character, so text no longer than ``budget`` chars
    cannot overflow and skips the tokenizer round-trip entirely.
    """
    if budget <= 0:
        raise ValueError("prefix alone exceeds the context budget")
    if len(text) <= budget:
        return text, False
    toks = pieces(text)
    if len(toks) <= budget:
        return text, False
    keep = budget
    while keep > 0:
        out = "".join(toks[:keep])
        if count(out) <= budget:
            return out, True
        keep -= max(1, keep // 32)
    return "", True


class HttpEmbedder:
    """Embeds via a running llama-server; caches unit vectors on disk.

    Cache layout: ``<cache_root>/<vector_fingerprint>/<sha1(kind, text)>.npy``.
    The fingerprint covers model artifact, pooling, prefixes, budget, EOS and the
    llama.cpp build, so a changed recipe never reads stale vectors.
    """

    def __init__(
        self,
        spec: EmbedderSpec,
        url: str,
        cache_root: pathlib.Path,
        runtime: str,
        *,
        artifact_sha: str = "",
        batch: int = 16,
        headroom: int = 8,
        timeout_s: float = 600,
    ) -> None:
        import httpx

        self.spec = spec
        self.client = httpx.Client(base_url=url, timeout=timeout_s)
        self.fingerprint = spec.vector_fingerprint(runtime, artifact_sha, headroom)
        self.cache = cache_root / self.fingerprint
        self.cache.mkdir(parents=True, exist_ok=True)
        self.batch = batch
        self.headroom = headroom
        # distinct texts whose input was cut to fit ctx (cache hits included, via
        # the persistent .trunc markers)
        self.truncated_keys: set[str] = set()
        self._prefix_tokens: dict[str, int] = {}

    @property
    def truncated(self) -> int:
        return len(self.truncated_keys)

    # -- tokenizer helpers ----------------------------------------------------
    def _count(self, text: str) -> int:
        r = self.client.post("/tokenize", json={"content": text, "add_special": False})
        r.raise_for_status()
        return len(r.json()["tokens"])

    def _pieces(self, text: str) -> list[str]:
        r = self.client.post(
            "/tokenize",
            json={"content": text, "add_special": False, "with_pieces": True},
        )
        r.raise_for_status()
        out = []
        for t in r.json()["tokens"]:
            p = t["piece"]
            out.append(p if isinstance(p, str) else bytes(p).decode("utf-8", "ignore"))
        return out

    def _fit(self, prefix: str, text: str) -> tuple[str, bool]:
        if prefix not in self._prefix_tokens:
            self._prefix_tokens[prefix] = self._count(prefix) if prefix else 0
        # headroom for BOS/CLS/SEP/EOS specials the server adds
        budget = self.spec.ctx - self.headroom - self._prefix_tokens[prefix]
        return fit_to_budget(text, budget, self._count, self._pieces)

    # -- embedding ------------------------------------------------------------
    @staticmethod
    def cache_key(kind: str, text: str) -> str:
        return hashlib.sha1(f"{kind}\x00{text}".encode()).hexdigest()

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        """Unit vectors for ``texts``; ``kind`` is ``"q"`` (query) or ``"d"`` (doc)."""
        if kind not in ("q", "d"):
            raise ValueError(kind)
        prefix = self.spec.query_prefix if kind == "q" else self.spec.doc_prefix
        keys = [self.cache_key(kind, t) for t in texts]
        got: dict[str, np.ndarray] = {}
        todo: list[tuple[str, str]] = []
        for k, t in zip(keys, texts, strict=True):
            f = self.cache / f"{k}.npy"
            if k in got:
                continue
            if f.exists():
                got[k] = np.load(f)
                if (self.cache / f"{k}.trunc").exists():
                    self.truncated_keys.add(k)
            else:
                todo.append((k, t))
                got[k] = None  # type: ignore[assignment]  # dedup marker
        for i in range(0, len(todo), self.batch):
            chunk = todo[i : i + self.batch]
            fitted = [self._fit(prefix, t) for _, t in chunk]
            inputs = [prefix + t + self.spec.eos for t, _ in fitted]
            r = self.client.post("/v1/embeddings", json={"input": inputs})
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            if len(data) != len(chunk):
                raise RuntimeError(f"server returned {len(data)} of {len(chunk)}")
            for (k, _), (_, cut), d in zip(chunk, fitted, data, strict=True):
                v = np.asarray(d["embedding"], dtype=np.float32)
                norm = float(np.linalg.norm(v))
                if not np.isfinite(v).all() or norm == 0.0:
                    raise RuntimeError(f"non-finite or zero vector for cache key {k}")
                v /= norm
                if cut:
                    (self.cache / f"{k}.trunc").touch()
                    self.truncated_keys.add(k)
                tmp = self.cache / f"{k}.tmp.npy"
                np.save(tmp, v)
                tmp.replace(self.cache / f"{k}.npy")
                got[k] = v
        return slice_dims(np.stack([got[k] for k in keys]), self.spec.dims)


def slice_dims(m: np.ndarray, dims: int) -> np.ndarray:
    """Matryoshka slice to the first ``dims`` columns, renormalized (0 ⇒ no-op)."""
    if not dims or dims >= m.shape[1]:
        return m
    s = m[:, :dims]
    return s / np.linalg.norm(s, axis=1, keepdims=True)
