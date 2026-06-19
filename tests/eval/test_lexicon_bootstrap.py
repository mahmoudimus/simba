from __future__ import annotations

import json
import pathlib

import simba.eval.lexicon_bootstrap as lexicon_bootstrap


def test_nltk_data_dir_is_under_lexicon_root(tmp_path: pathlib.Path) -> None:
    root = tmp_path / ".simba" / "lexicon"

    assert lexicon_bootstrap.nltk_data_dir(root) == root / "nltk_data"


def test_nltk_resource_specs_cover_initial_world_sources() -> None:
    names = {resource.name for resource in lexicon_bootstrap.NLTK_RESOURCES}

    assert names == {"wordnet", "omw-1.4", "framenet_v17"}


def test_write_manifest_records_wikidata_subset_strategy(
    tmp_path: pathlib.Path,
) -> None:
    statuses = [
        lexicon_bootstrap.ResourceStatus(
            name="wordnet",
            available=True,
            path=str(tmp_path / "wordnet.zip"),
            description="WordNet",
        )
    ]

    manifest = lexicon_bootstrap.write_manifest(tmp_path, statuses)

    body = json.loads(manifest.read_text(encoding="utf-8"))
    assert body["kind"] == "simba.lexicon.bootstrap"
    assert body["resources"][0]["name"] == "wordnet"
    assert body["wikidata"]["strategy"] == "stream-filtered-subset"
