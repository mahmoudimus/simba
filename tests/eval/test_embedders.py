"""Unit tests for the embedder bake-off package (no models, no network)."""

from __future__ import annotations

import json
import math
import pathlib
from fractions import Fraction

import httpx
import numpy as np
import pytest

from simba.eval.dataset import Dataset, EvalCase, Memory
from simba.eval.embedders import benchmarks, candidates, cli, engine, scoring

# ── scoring ──────────────────────────────────────────────────────────────────


def test_case_metrics_exact_values() -> None:
    m = scoring.case_metrics(["a", "b"], ["x", "a", "y", "b"])
    assert m["R@1"] == 0
    assert m["R@3"] == Fraction(1, 2)
    assert m["R@5"] == 1
    assert m["All@10"] == 1
    assert m["MRR"] == Fraction(1, 2)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert m["nDCG@10"] == pytest.approx(dcg / idcg)


def test_case_metrics_miss_and_empty_gold() -> None:
    m = scoring.case_metrics(["z"], ["a", "b"])
    assert m["R@50"] == 0 and m["MRR"] == 0 and m["All@50"] == 0
    with pytest.raises(ValueError):
        scoring.case_metrics([], ["a"])


def test_paired_bootstrap_deterministic_and_brackets_mean() -> None:
    d = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.25]
    a = scoring.paired_bootstrap(d, 500)
    b = scoring.paired_bootstrap(d, 500)
    assert a == b
    mean, lo, hi = a
    assert mean == pytest.approx(sum(d) / len(d))
    assert lo <= mean <= hi
    assert scoring.paired_bootstrap([0.0] * 10, 200) == (0.0, 0.0, 0.0)
    assert scoring.paired_bootstrap([], 200) == (0.0, 0.0, 0.0)


def _run(key: str, ranked: dict[str, list[str]], gold: dict[str, list[str]]) -> dict:
    return {
        "vector_fingerprint": key,
        "runtime": "b1",
        "truncated": 0,
        "cases": [
            {"id": i, "intent": "x" if i < "c" else "y", "gold": gold[i], "ranked": r}
            for i, r in ranked.items()
        ],
    }


def test_score_group_common_cases_and_deltas() -> None:
    gold = {"a": ["1"], "b": ["2"], "c": ["3"]}
    base = _run("base", {"a": ["9", "1"], "b": ["2"], "c": ["9"]}, gold)
    cand = _run("cand", {"a": ["1"], "b": ["2"]}, gold)  # missing case "c"
    g = scoring.score_group({"base": base, "cand": cand}, "base", 200)
    assert g["n"] == 2  # scored on the intersection only
    assert g["dropped"] == {"base": 1, "cand": 0}
    assert g["summary"]["base"]["MRR"] == pytest.approx(0.75)
    assert g["summary"]["cand"]["MRR"] == pytest.approx(1.0)
    assert g["deltas"]["cand"]["MRR"][0] == pytest.approx(0.25)
    assert set(g["by_intent"]) == {"x"}
    with pytest.raises(KeyError):
        scoring.score_group({"cand": cand}, "base", 10)


def test_score_all_and_render(tmp_path: pathlib.Path) -> None:
    gold = {"a": ["1"], "b": ["2"]}
    d = tmp_path / "dense" / "locomo"
    d.mkdir(parents=True)
    (d / "base.json").write_text(json.dumps(_run("f1", {"a": ["1"], "b": ["9"]}, gold)))
    (d / "cand.json").write_text(json.dumps(_run("f2", {"a": ["1"], "b": ["2"]}, gold)))
    groups = scoring.score_all(tmp_path, "base", 100)
    assert list(groups) == ["dense/locomo"]
    text = scoring.render(groups, "base")
    assert "## dense/locomo (n=2 cases)" in text
    assert "| cand |" in text and "Δ vs `base`" in text


# ── benchmarks ───────────────────────────────────────────────────────────────


def _ds(name: str, intent: str) -> Dataset:
    return Dataset(
        name=name,
        corpus=[Memory(id=f"{name}#0", content="x")],
        cases=[EvalCase(id=name, query="q", relevant_ids=[f"{name}#0"], intent=intent)],
    )


def test_stratified_sample_proportional_deterministic() -> None:
    pool = [_ds(f"a{i}", "A") for i in range(60)] + [
        _ds(f"b{i}", "B") for i in range(30)
    ]
    pool += [_ds("c0", "C")]
    s1 = benchmarks.stratified_sample(pool, 30, seed=1)
    s2 = benchmarks.stratified_sample(list(reversed(pool)), 30, seed=1)
    assert [d.name for d in s1] == [d.name for d in s2]  # input-order independent
    counts = {k: sum(d.cases[0].intent == k for d in s1) for k in "ABC"}
    assert counts == {"A": 20, "B": 10, "C": 1}  # min 1 per type
    assert benchmarks.stratified_sample(pool, 0, seed=1) == pool
    assert benchmarks.stratified_sample(pool, 10_000, seed=1) == pool


# ── candidates ───────────────────────────────────────────────────────────────


def test_builtin_registry_sane() -> None:
    keys = [s.key for s in candidates.BUILTIN]
    assert len(keys) == len(set(keys))
    assert set(candidates.DEFAULT_SET) <= set(keys)
    with pytest.raises(ValueError):
        candidates.EmbedderSpec("x", repo="r", file="f.gguf", pooling="max")
    with pytest.raises(ValueError):
        candidates.EmbedderSpec("x", repo="r", file="f.bin")


def test_fingerprint_tracks_recipe_not_slice() -> None:
    s = candidates.EmbedderSpec("x", repo="r", file="f.gguf", query_prefix="q: ")
    fp = s.vector_fingerprint("b1")
    import dataclasses as dc

    assert dc.replace(s, dims=256).vector_fingerprint("b1") == fp
    assert dc.replace(s, query_prefix="query: ").vector_fingerprint("b1") != fp
    assert dc.replace(s, pooling="cls").vector_fingerprint("b1") != fp
    assert s.vector_fingerprint("b2") != fp


def test_load_candidates_toml_adds_and_overrides(tmp_path: pathlib.Path) -> None:
    f = tmp_path / "c.toml"
    f.write_text(
        '[[candidate]]\nkey = "newmodel"\nrepo = "org/new-GGUF"\n'
        'file = "new-Q8_0.gguf"\n'
        'pooling = "last"\nquery_prefix = "Q: "\n\n'
        '[[candidate]]\nkey = "bge_mean"\nrepo = "org/bge"\nfile = "bge.gguf"\n'
    )
    c = candidates.load_candidates(f)
    assert c["newmodel"].pooling == "last"
    assert c["bge_mean"].repo == "org/bge"
    assert "nemotron3_1b" in c
    f.write_text('[[candidate]]\nkey = "z"\nrepo = "r"\nfile = "f.gguf"\nbogus = 1\n')
    with pytest.raises(ValueError, match="bogus"):
        candidates.load_candidates(f)
    assert candidates.load_candidates(tmp_path / "missing.toml").keys() == {
        s.key for s in candidates.BUILTIN
    }
    with pytest.raises(KeyError):
        candidates.select(c, ["nope"])


# ── engine ───────────────────────────────────────────────────────────────────


def _ws_count(t: str) -> int:
    return len(t.split())


def _ws_pieces(t: str) -> list[str]:
    return [w + " " for w in t.split()]


def test_fit_to_budget_short_text_skips_tokenizer() -> None:
    def boom(_: str):
        raise AssertionError("tokenizer must not be called")

    assert engine.fit_to_budget("abc", 10, boom, boom) == ("abc", False)


def test_fit_to_budget_truncates_within_budget() -> None:
    text = " ".join(f"w{i}" for i in range(100))
    out, cut = engine.fit_to_budget(text, 20, _ws_count, _ws_pieces)
    assert cut and _ws_count(out) <= 20 and out.startswith("w0 w1")
    fits = " ".join(["w"] * 15)  # longer than 20 chars but only 15 tokens
    assert engine.fit_to_budget(fits, 20, _ws_count, _ws_pieces) == (fits, False)
    with pytest.raises(ValueError):
        engine.fit_to_budget("x", 0, _ws_count, _ws_pieces)


def test_slice_dims_renormalizes() -> None:
    m = np.array([[3.0, 4.0, 12.0]])
    s = engine.slice_dims(m, 2)
    assert s.shape == (1, 2)
    assert np.allclose(np.linalg.norm(s, axis=1), 1.0)
    assert engine.slice_dims(m, 0) is m


def test_http_embedder_caches_dedups_and_prefixes(tmp_path: pathlib.Path) -> None:
    seen: list[list[str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        if req.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": body["content"].split()})
        seen.append(body["input"])
        data = [
            {"index": i, "embedding": [float(len(t)), 1.0]}
            for i, t in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": data})

    spec = candidates.EmbedderSpec(
        "t", repo="r", file="f.gguf", query_prefix="Q: ", doc_prefix="D: ", ctx=64
    )
    emb = engine.HttpEmbedder(spec, "http://x", tmp_path, "b1", batch=2)
    emb.client = httpx.Client(
        base_url="http://x", transport=httpx.MockTransport(handler)
    )
    v = emb.embed(["aa", "bbb", "aa"], "d")
    assert v.shape == (3, 2)
    assert np.allclose(v[0], v[2])
    assert seen == [["D: aa", "D: bbb"]]  # deduped, doc prefix applied
    emb.embed(["aa", "bbb"], "d")
    assert len(seen) == 1  # served from disk cache
    emb.embed(["aa"], "q")
    assert seen[-1] == ["Q: aa"]  # query cache is separate from doc cache
    assert (tmp_path / spec.vector_fingerprint("b1")).is_dir()
    with pytest.raises(ValueError):
        emb.embed(["x"], "z")


# ── cli ──────────────────────────────────────────────────────────────────────


def test_cli_parse_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    opts, flags = cli._parse(["--models", "a,b", "--force"])
    assert opts == {"models": "a,b"} and flags == {"force"}
    with pytest.raises(SystemExit):
        cli._parse(["--nope"])
    assert cli.main(["--help"]) == 0
    assert "simba eval embedders run" in capsys.readouterr().out
    assert cli._csv(" a, ,b ", ()) == ["a", "b"]
    assert cli._csv(None, ("x",)) == ["x"]


# ── review regressions ───────────────────────────────────────────────────────


def _mock_embedder(tmp_path, handler, **spec_kw):
    spec = candidates.EmbedderSpec("t", repo="r", file="f.gguf", **spec_kw)
    emb = engine.HttpEmbedder(spec, "http://x", tmp_path, "b1", batch=4, headroom=2)
    emb.client = httpx.Client(
        base_url="http://x", transport=httpx.MockTransport(handler)
    )
    return emb


def _tok_handler(vec=lambda t: [1.0, float(len(t))]):
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        if req.url.path == "/tokenize":
            toks = [
                {"id": i, "piece": w + " "}
                for i, w in enumerate(body["content"].split())
            ]
            return httpx.Response(200, json={"tokens": toks})
        data = [{"index": i, "embedding": vec(t)} for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"data": data})

    return handler


def test_truncation_count_survives_cache_hits(tmp_path: pathlib.Path) -> None:
    long = " ".join(f"w{i}" for i in range(50))
    emb = _mock_embedder(tmp_path, _tok_handler(), ctx=12)
    emb.embed([long, "short"], "d")
    assert emb.truncated == 1
    warm = _mock_embedder(tmp_path, _tok_handler(), ctx=12)
    warm.embed([long, "short"], "d")  # all cache hits
    assert warm.truncated == 1


def test_non_finite_vector_is_rejected_not_cached(tmp_path: pathlib.Path) -> None:
    emb = _mock_embedder(tmp_path, _tok_handler(vec=lambda t: [0.0, 0.0]))
    with pytest.raises(RuntimeError, match="non-finite or zero"):
        emb.embed(["x"], "d")
    assert not list(emb.cache.glob("*.npy"))


def test_fingerprint_includes_artifact_and_headroom() -> None:
    s = candidates.EmbedderSpec("x", repo="r", file="f.gguf")
    base = s.vector_fingerprint("b1", "aaaa", 8)
    assert s.vector_fingerprint("b1", "bbbb", 8) != base
    assert s.vector_fingerprint("b1", "aaaa", 9) != base


def test_busy_port_refused_before_spawn(tmp_path: pathlib.Path) -> None:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        assert not engine.port_free(port)
        with pytest.raises(RuntimeError, match="in use"), engine.llama_server(
            "/nonexistent/llama-server", tmp_path / "m.gguf", pooling="mean",
            ctx=64, port=port, slots=1, log_path=tmp_path / "log",
        ):  # fmt: skip
            pass


def test_identity_tracks_sample_dims_and_memory() -> None:
    import dataclasses as dc

    from simba.eval.embedders import arms
    from simba.eval.embedders.config import EmbedderBakeoffConfig

    cfg = EmbedderBakeoffConfig()
    s = candidates.EmbedderSpec("x", repo="r", file="f.gguf")
    bp = {"sha256_16": "f", "sample": {"n": 150}, "case_ids_sha": "c", "n_docs": 1}
    base = arms.identity("dense", s, cfg, "b1", "a", bp)
    assert arms.identity("dense", s, cfg, "b1", "a", dict(bp)) == base
    assert arms.identity("dense", s, cfg, "b1", "a", bp | {"case_ids_sha": "d"}) != base
    assert arms.identity("dense", dc.replace(s, dims=64), cfg, "b1", "a", bp) != base
    p1 = arms.identity("pipeline", s, cfg, "b1", "a", bp, {"min_similarity": 0.35})
    p2 = arms.identity("pipeline", s, cfg, "b1", "a", bp, {"min_similarity": 0.3})
    assert p1 != p2


def test_render_warns_on_mixed_samples() -> None:
    gold = {"a": ["1"], "b": ["2"]}
    base = _run("f1", {"a": ["1"], "b": ["2"]}, gold)
    cand = _run("f2", {"a": ["1"]}, gold)
    base["bench"] = {"case_ids_sha": "x"}
    cand["bench"] = {"case_ids_sha": "y"}
    g = scoring.score_group({"base": base, "cand": cand}, "base", 50)
    assert g["bench_variants"] == 2
    assert "Warning" in scoring.render({"dense/locomo": g}, "base")
