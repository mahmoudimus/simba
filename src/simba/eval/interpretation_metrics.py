"""Baseline and metric artifacts for ambiguous NLIDB evals."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import typing

import simba.eval.ambiguity_fail18 as ambiguity_fail18
import simba.eval.candidate_unit_ir as candidate_unit_ir


def build_fail18_baseline(
    *,
    manifest_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_MANIFEST,
    corpus_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_CORPUS,
    candidate_fixture_path: str | pathlib.Path = candidate_unit_ir.DEFAULT_FIXTURE_PATH,
    live_candidate_path: str | pathlib.Path = (
        "_gitless/fail9_candidate_unit_claude_live.json"
    ),
    include_repair: bool = True,
) -> dict[str, typing.Any]:
    """Freeze the current fail18 baselines before testing new interpretation work."""
    old_range = ambiguity_fail18.summarize(manifest_path, backend="python")
    range_payload = old_range.to_dict()
    range_payload.update(
        {
            "source": "clingo_manifest_range",
            "executor": "python_ambiguity_backend",
        }
    )
    modes: dict[str, typing.Any] = {
        "clingo_manifest_range": range_payload,
    }
    if include_repair:
        repaired = ambiguity_fail18.summarize(
            manifest_path,
            backend="python",
            repair=True,
            corpus_path=corpus_path,
        )
        repair_payload = repaired.to_dict()
        repair_payload.update(
            {
                "source": "fail18_repair_answer_space",
                "executor": "python_ambiguity_backend_plus_repair",
            }
        )
        modes["old_fail18_repair"] = repair_payload

    saved_fixture = _saved_candidate_fixture(candidate_fixture_path)
    live_prompt = _live_candidate_prompt(live_candidate_path)
    return {
        "name": "fail18-ambiguous-nlidb-baseline",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_manifest": str(manifest_path),
        "source_corpus": str(corpus_path),
        "commands": [
            (
                "uv run python -m simba.eval.interpretation_metrics "
                "--fail18-baseline "
                "--output _gitless/fail18_ambiguous_nlidb_baseline.json"
            ),
            "uv run python -m simba.eval.candidate_unit_ir --json",
        ],
        "modes": modes,
        "summary": _summaries(modes),
        "saved_candidate_unit_fixture": saved_fixture,
        "live_candidate_unit_prompt": live_prompt,
        "saved_fixture_succeeds_live_prompt_fails": _saved_wins_live_fails(
            saved_fixture, live_prompt
        ),
    }


def _summaries(modes: dict[str, typing.Any]) -> dict[str, dict[str, int]]:
    return {
        name: {
            "total": int(payload["total"]),
            "gold_known": int(payload["gold_known"]),
            "contains_gold": int(payload["contains_gold"]),
            "misses_gold": int(payload["misses_gold"]),
        }
        for name, payload in modes.items()
    }


def _saved_candidate_fixture(path: str | pathlib.Path) -> dict[str, typing.Any]:
    candidate_path = pathlib.Path(path)
    if not candidate_path.exists():
        return {"exists": False, "path": str(candidate_path)}
    fixture = candidate_unit_ir.load_fixture(candidate_path)
    score = candidate_unit_ir.score_fixture(fixture)
    return {
        "exists": True,
        "path": str(candidate_path),
        "name": fixture.name,
        "prompt_version": fixture.prompt_version,
        "tool": fixture.tool,
        "matches": score.matches,
        "total": score.total,
        "mismatches": list(score.mismatches),
        "matched_case_ids": [
            case.id for case in fixture.cases if case.recomputed_match
        ],
    }


def _live_candidate_prompt(path: str | pathlib.Path) -> dict[str, typing.Any]:
    live_path = pathlib.Path(path)
    if not live_path.exists():
        return {"exists": False, "path": str(live_path)}
    raw = json.loads(live_path.read_text(encoding="utf-8"))
    results = list(raw.get("results", []))
    return {
        "exists": True,
        "path": str(live_path),
        "prompt_version": str(raw.get("prompt_version", "")),
        "tool": str(raw.get("tool", "")),
        "matches": int(raw.get("matches", 0)),
        "total": int(raw.get("total", len(results))),
        "mismatches": [
            str(item.get("id"))
            for item in results
            if not bool(item.get("match", False))
        ],
        "results": [
            {
                "id": str(item.get("id", "")),
                "gold": item.get("gold"),
                "pred": item.get("pred"),
                "match": bool(item.get("match", False)),
            }
            for item in results
        ],
    }


def _saved_wins_live_fails(
    saved_fixture: dict[str, typing.Any],
    live_prompt: dict[str, typing.Any],
) -> list[str]:
    if not saved_fixture.get("exists") or not live_prompt.get("exists"):
        return []
    saved_matches = set(saved_fixture.get("matched_case_ids", []))
    live_misses = set(live_prompt.get("mismatches", []))
    return sorted(saved_matches & live_misses)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail18-baseline",
        action="store_true",
        help="Build the fail18 baseline artifact.",
    )
    parser.add_argument("--path", default="", help="fail18 manifest path.")
    parser.add_argument("--corpus", default="", help="fail18 corpus path.")
    parser.add_argument("--output", type=pathlib.Path, help="Write JSON artifact.")
    parser.add_argument(
        "--no-repair",
        action="store_true",
        help="Skip the old repair baseline.",
    )
    args = parser.parse_args(argv)

    if not args.fail18_baseline:
        raise SystemExit("pass --fail18-baseline")
    manifest_path = (
        pathlib.Path(args.path) if args.path else ambiguity_fail18.DEFAULT_MANIFEST
    )
    corpus_path = (
        pathlib.Path(args.corpus)
        if args.corpus
        else ambiguity_fail18.DEFAULT_CORPUS
    )
    artifact = build_fail18_baseline(
        manifest_path=manifest_path,
        corpus_path=corpus_path,
        include_repair=not args.no_repair,
    )
    encoded = json.dumps(artifact, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(f"{encoded}\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
