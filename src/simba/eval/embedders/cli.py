"""`simba eval embedders …` — repeatable embedder bake-off.

    simba eval embedders list
    simba eval embedders fetch   [--models a,b]
    simba eval embedders run     [--models a,b] [--bench locomo,lme]
                                 [--arms dense,pipeline] [--force]
    simba eval embedders score
    simba eval embedders refcheck  [--models a,b]  (needs sentence-transformers)
    simba eval embedders inprocess [--models a,b]

`run` skips a (arm, bench, model) whose saved result has the same vector
fingerprint (spec + llama.cpp build) unless --force, then re-scores everything.
"""

from __future__ import annotations

import json
import pathlib
import sys
import typing

from . import arms, candidates, checks, engine, scoring
from .config import EmbedderBakeoffConfig, load_config

USAGE = __doc__ or ""


def _root() -> pathlib.Path:
    import simba.db

    found = simba.db.find_repo_root(pathlib.Path.cwd())
    return found if found is not None else pathlib.Path.cwd()


def _parse(args: list[str]) -> tuple[dict[str, str], set[str]]:
    opts: dict[str, str] = {}
    flags: set[str] = set()
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--models", "--bench", "--arms") and i + 1 < len(args):
            opts[a[2:]] = args[i + 1]
            i += 2
        elif a in ("--force", "--json"):
            flags.add(a[2:])
            i += 1
        else:
            raise SystemExit(f"unknown option {a!r}\n\n{USAGE}")
    return opts, flags


def _csv(v: str | None, default: typing.Iterable[str]) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()] if v else list(default)


def _specs(cfg: EmbedderBakeoffConfig, root: pathlib.Path, opt: str | None):
    cands = candidates.load_candidates(cfg.resolve(cfg.candidates_file, root))
    keys = _csv(opt, candidates.DEFAULT_SET)
    if cfg.baseline not in keys:
        keys.insert(0, cfg.baseline)
    return cands, candidates.select(cands, keys)


def _append_jsonl(path: pathlib.Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _latest(path: pathlib.Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["key"]] = row
    return out


def cmd_list(cfg, root) -> int:
    cands = candidates.load_candidates(cfg.resolve(cfg.candidates_file, root))
    for s in cands.values():
        have = "✓" if s.local_path(cfg.models_path()).exists() else " "
        dims = f" dims={s.dims}" if s.dims else ""
        star = "*" if s.key in candidates.DEFAULT_SET else " "
        print(f"{have}{star} {s.key:20} {s.repo}/{s.file} pooling={s.pooling}{dims}")
    print(f"\n✓ = downloaded under {cfg.models_path()}   * = default run set")
    return 0


def cmd_run(cfg, root, opts, flags) -> int:
    from . import benchmarks

    _, specs = _specs(cfg, root, opts.get("models"))
    benches = _csv(opts.get("bench"), ("locomo", "lme"))
    arm_names = _csv(opts.get("arms"), ("dense", "pipeline"))
    out_dir = cfg.resolve(cfg.out_dir, root)
    runtime = engine.runtime_version(cfg.llama_server)
    print(f"llama-server: {cfg.llama_server} ({runtime})")
    todo = [("dense", b) for b in benches if "dense" in arm_names]
    if "pipeline" in arm_names and "locomo" in benches:
        todo.append(("pipeline", "locomo"))
    bench_prov = {b: benchmarks.load(b, cfg, root)[1] for b in {b for _, b in todo}}
    memory = arms.memory_snapshot(arms.pipeline_memory_config())
    for spec in specs:
        gguf = checks.fetch(spec, cfg)
        artifact = engine.artifact_sha(gguf)
        for arm, bench in todo:
            want = arms.identity(
                arm, spec, cfg, runtime, artifact, bench_prov[bench], memory
            )
            path = arms.result_path(out_dir, arm, bench, spec.key)
            if (
                path.exists()
                and "force" not in flags
                and json.loads(path.read_text()).get("identity") == want
            ):
                print(f"skip {spec.key} {arm}/{bench} (up to date)")
                continue
            print(f"run  {spec.key} {arm}/{bench}")
            kw = {"runtime": runtime, "artifact": artifact}
            if arm == "dense":
                arms.run_dense(spec, bench, cfg, root, gguf, **kw)
            else:
                arms.run_pipeline(spec, cfg, root, gguf, **kw)
    return cmd_score(cfg, root, flags)


def cmd_score(cfg, root, flags) -> int:
    out_dir = cfg.resolve(cfg.out_dir, root)
    groups = scoring.score_all(
        out_dir / "runs", cfg.baseline, cfg.bootstrap, cfg.bootstrap_seed
    )
    if not groups:
        print(f"no runs with baseline {cfg.baseline!r} under {out_dir / 'runs'}")
        return 1
    text = scoring.render(groups, cfg.baseline)
    for name, fname in (
        ("reference check", "refcheck.jsonl"),
        ("in-process", "inprocess.jsonl"),
    ):
        rows = _latest(out_dir / fname)
        if rows:
            text += f"\n## {name} (latest per model)\n\n```\n"
            text += "\n".join(json.dumps(r) for r in rows.values()) + "\n```\n"
    (out_dir / "report.md").write_text(text)
    (out_dir / "summary.json").write_text(json.dumps(groups, indent=1, default=list))
    print(json.dumps(groups, default=list) if "json" in flags else text)
    print(f"\nwrote {out_dir / 'report.md'}", file=sys.stderr)
    return 0


def cmd_check(cfg, root, opts, which: str) -> int:
    _, specs = _specs(cfg, root, opts.get("models"))
    out_dir = cfg.resolve(cfg.out_dir, root)
    fn = checks.refcheck if which == "refcheck" else checks.inprocess
    rc = 0
    for spec in specs:
        if which == "refcheck" and not spec.ref_repo:
            continue
        row = fn(spec, cfg, checks.fetch(spec, cfg), out_dir / "logs")
        _append_jsonl(out_dir / f"{which}.jsonl", row)
        print(json.dumps(row))
        ok = row.get("pass", True) if which == "refcheck" else row.get("loads", False)
        rc |= 0 if ok else 2
    return rc


def main(args: list[str]) -> int:
    if not args or args[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    sub, rest = args[0], args[1:]
    cfg = load_config()
    root = _root()
    opts, flags = _parse(rest)
    if sub == "list":
        return cmd_list(cfg, root)
    if sub == "fetch":
        for spec in _specs(cfg, root, opts.get("models"))[1]:
            print(checks.fetch(spec, cfg))
        return 0
    if sub == "run":
        return cmd_run(cfg, root, opts, flags)
    if sub == "score":
        return cmd_score(cfg, root, flags)
    if sub in ("refcheck", "inprocess"):
        return cmd_check(cfg, root, opts, sub)
    print(f"unknown subcommand {sub!r}\n\n{USAGE}", file=sys.stderr)
    return 1
