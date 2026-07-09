"""Bootstrap local lexical resources for world-lexicon experiments."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import typing

DEFAULT_ROOT = pathlib.Path(".simba/lexicon")


@dataclasses.dataclass(frozen=True)
class NltkResource:
    name: str
    lookup_path: str
    description: str


@dataclasses.dataclass(frozen=True)
class ResourceStatus:
    name: str
    available: bool
    path: str = ""
    description: str = ""


@dataclasses.dataclass(frozen=True)
class LexiconRecord:
    id: str
    provider: str
    kind: str
    provider_ref: str
    label: str
    aliases_json: str = "[]"
    parents_json: str = "[]"
    relations_json: str = "[]"
    definition: str = ""
    search_text: str = ""
    provenance: str = ""
    trust_tier: str = "raw"


NLTK_RESOURCES = (
    NltkResource(
        name="wordnet",
        lookup_path="corpora/wordnet",
        description="WordNet synsets, lemmas, morphology, and lexical relations",
    ),
    NltkResource(
        name="omw-1.4",
        lookup_path="corpora/omw-1.4",
        description="Open Multilingual WordNet lemma aliases",
    ),
    NltkResource(
        name="framenet_v17",
        lookup_path="corpora/framenet_v17",
        description="FrameNet 1.7 frames, frame elements, and lexical units",
    ),
)


def nltk_data_dir(root: pathlib.Path = DEFAULT_ROOT) -> pathlib.Path:
    return pathlib.Path(root) / "nltk_data"


def check_nltk_resources(root: pathlib.Path = DEFAULT_ROOT) -> list[ResourceStatus]:
    nltk = _import_nltk()
    data_dir = nltk_data_dir(root)
    search_paths = [str(data_dir), *nltk.data.path]
    statuses: list[ResourceStatus] = []
    for resource in NLTK_RESOURCES:
        found = _find_nltk_resource(nltk, resource.lookup_path, search_paths)
        statuses.append(
            ResourceStatus(
                name=resource.name,
                available=found is not None,
                path=str(found or ""),
                description=resource.description,
            )
        )
    return statuses


def download_nltk_resources(root: pathlib.Path = DEFAULT_ROOT) -> list[ResourceStatus]:
    nltk = _import_nltk()
    data_dir = nltk_data_dir(root)
    data_dir.mkdir(parents=True, exist_ok=True)
    for resource in NLTK_RESOURCES:
        nltk.download(
            resource.name,
            download_dir=str(data_dir),
            quiet=False,
            raise_on_error=True,
        )
    statuses = check_nltk_resources(root)
    write_manifest(root, statuses)
    return statuses


def write_manifest(root: pathlib.Path, statuses: list[ResourceStatus]) -> pathlib.Path:
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "simba.lexicon.bootstrap",
        "nltk_data_dir": str(nltk_data_dir(root)),
        "resources": [dataclasses.asdict(status) for status in statuses],
        "wikidata": {
            "strategy": "stream-filtered-subset",
            "notes": (
                "Full latest-all Wikidata dumps are too large for default local "
                "bootstrap; stream labels, aliases, instance-of, and subclass-of "
                "into a bounded JSONL/LanceDB subset instead."
            ),
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def export_nltk_records_jsonl(
    root: pathlib.Path = DEFAULT_ROOT,
    *,
    limit: int = 0,
) -> pathlib.Path:
    path = pathlib.Path(root) / "nltk_lexicon.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in iter_nltk_records(root, limit=limit):
            handle.write(json.dumps(dataclasses.asdict(record)) + "\n")
    return path


def build_lancedb_table(
    root: pathlib.Path = DEFAULT_ROOT,
    *,
    limit: int = 0,
    table_name: str = "lexicon",
) -> tuple[pathlib.Path, int]:
    try:
        import lancedb
    except ImportError as exc:
        raise RuntimeError(
            "LanceDB is required for lexicon indexing. Install the embed/dev "
            "dependencies, e.g. `uv sync --extra embed --group dev`."
        ) from exc

    db_path = pathlib.Path(root) / "lexicon.lance"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        dataclasses.asdict(record) for record in iter_nltk_records(root, limit=limit)
    ]
    db = lancedb.connect(str(db_path))
    if rows:
        db.create_table(table_name, data=rows, mode="overwrite")
    return db_path, len(rows)


def iter_nltk_records(
    root: pathlib.Path = DEFAULT_ROOT,
    *,
    limit: int = 0,
) -> typing.Iterator[LexiconRecord]:
    nltk = _import_nltk()
    data_dir = nltk_data_dir(root)
    if str(data_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(data_dir))
    count = 0
    for record in _iter_wordnet_records():
        yield record
        count += 1
        if limit and count >= limit:
            return
    for record in _iter_framenet_records():
        yield record
        count += 1
        if limit and count >= limit:
            return


def _iter_wordnet_records() -> typing.Iterator[LexiconRecord]:
    from nltk.corpus import wordnet as wn

    for synset in wn.all_synsets():
        aliases = sorted({lemma.replace("_", " ") for lemma in synset.lemma_names()})
        parents = sorted({parent.name() for parent in synset.hypernyms()})
        label = aliases[0] if aliases else synset.name()
        definition = synset.definition()
        yield LexiconRecord(
            id=f"wordnet:{synset.name()}",
            provider="wordnet",
            kind="concept",
            provider_ref=synset.name(),
            label=label,
            aliases_json=json.dumps(aliases),
            parents_json=json.dumps(parents),
            definition=definition,
            search_text=" ".join([label, *aliases, definition]),
            provenance="nltk.wordnet",
            trust_tier="raw",
        )


def _iter_framenet_records() -> typing.Iterator[LexiconRecord]:
    from nltk.corpus import framenet as fn

    for frame_stub in fn.frames():
        frame = fn.frame(frame_stub.ID)
        aliases = sorted(str(name).rsplit(".", 1)[0] for name in frame.lexUnit)
        relations = sorted(
            f"{rel.type.name}:{rel.superFrameName or rel.subFrameName}"
            for rel in frame.frameRelations
        )
        fes = sorted(str(name) for name in frame.FE)
        definition = str(frame.definition or "")
        label = str(frame.name)
        yield LexiconRecord(
            id=f"framenet:frame:{label}",
            provider="framenet",
            kind="frame",
            provider_ref=str(frame.ID),
            label=label,
            aliases_json=json.dumps(aliases),
            parents_json=json.dumps(fes),
            relations_json=json.dumps(relations),
            definition=definition,
            search_text=" ".join([label, *aliases, *fes, definition]),
            provenance="nltk.framenet_v17",
            trust_tier="raw",
        )


def _find_nltk_resource(
    nltk: typing.Any, lookup_path: str, search_paths: list[str]
) -> pathlib.Path | None:
    candidates = (lookup_path, f"{lookup_path}.zip")
    for candidate in candidates:
        try:
            return pathlib.Path(str(nltk.data.find(candidate, paths=search_paths)))
        except LookupError:
            continue
    return None


def _import_nltk() -> typing.Any:
    try:
        import nltk
    except ImportError as exc:
        raise RuntimeError(
            "NLTK is required for lexicon bootstrap. Run with "
            "`uv run --with nltk python -m simba.eval.lexicon_bootstrap --download`."
        ) from exc
    return nltk


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/check offline lexical resources for Simba."
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Lexicon root directory (default: .simba/lexicon)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download NLTK WordNet, OMW, and FrameNet resources.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit status as JSON.",
    )
    parser.add_argument(
        "--export-jsonl",
        action="store_true",
        help="Export normalized NLTK lexicon records to nltk_lexicon.jsonl.",
    )
    parser.add_argument(
        "--build-lancedb",
        action="store_true",
        help="Build .simba/lexicon/lexicon.lance from normalized records.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit exported/indexed records; 0 means all.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = pathlib.Path(args.root)
    statuses = (
        download_nltk_resources(root) if args.download else check_nltk_resources(root)
    )
    manifest = write_manifest(root, statuses)
    jsonl_path = None
    lancedb_path = None
    lancedb_count = 0
    if args.export_jsonl:
        jsonl_path = export_nltk_records_jsonl(root, limit=args.limit)
    if args.build_lancedb:
        lancedb_path, lancedb_count = build_lancedb_table(root, limit=args.limit)
    if args.json:
        print(
            json.dumps(
                {
                    "root": str(root),
                    "manifest": str(manifest),
                    "jsonl": str(jsonl_path or ""),
                    "lancedb": str(lancedb_path or ""),
                    "lancedb_count": lancedb_count,
                    "resources": [dataclasses.asdict(status) for status in statuses],
                },
                indent=2,
            )
        )
    else:
        for status in statuses:
            mark = "ok" if status.available else "missing"
            print(f"{mark:<7} {status.name:<13} {status.path}")
        print(f"manifest: {manifest}")
        if jsonl_path:
            print(f"jsonl: {jsonl_path}")
        if lancedb_path:
            print(f"lancedb: {lancedb_path} rows={lancedb_count}")
    return 0 if all(status.available for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
