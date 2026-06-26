from __future__ import annotations

import json
import pathlib

from simba.eval import type_ontology


def test_ratifies_boots_as_clothing_through_wordnet_sense_bridge(
    tmp_path: pathlib.Path,
) -> None:
    path = _write_test_lexicon(tmp_path)

    result = type_ontology.ratify_type_subsumption(
        "boots",
        "clothing",
        lexicon_path=path,
    )

    assert result.ratified is True
    assert result.reason == "ontology_subsumption"
    assert result.path == (
        "wordnet:boot.n.01",
        "wordnet:footwear.n.02",
        "wordnet:footwear.n.01",
        "wordnet:clothing.n.01",
    )


def test_rejects_unrelated_type_without_subsumption_path(
    tmp_path: pathlib.Path,
) -> None:
    path = _write_test_lexicon(tmp_path)

    result = type_ontology.ratify_type_subsumption(
        "book",
        "clothing",
        lexicon_path=path,
    )

    assert result.ratified is False
    assert result.reason == "no_subsumption_path"


def _write_test_lexicon(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "nltk_lexicon.jsonl"
    records = [
        _record(
            "wordnet:boot.n.01",
            "boot.n.01",
            "boot",
            aliases=("boot",),
            parents=("footwear.n.02",),
            definition="footwear that covers the whole foot and lower leg",
        ),
        _record(
            "wordnet:footwear.n.02",
            "footwear.n.02",
            "footgear",
            aliases=("footgear", "footwear"),
            parents=("covering.n.02",),
            definition="covering for a person's feet",
        ),
        _record(
            "wordnet:footwear.n.01",
            "footwear.n.01",
            "footwear",
            aliases=("footwear",),
            parents=("clothing.n.01",),
            definition="clothing worn on a person's feet",
        ),
        _record(
            "wordnet:clothing.n.01",
            "clothing.n.01",
            "article of clothing",
            aliases=("article of clothing", "clothing"),
            parents=(),
            definition="a covering designed to be worn on a person's body",
        ),
        _record(
            "wordnet:book.n.01",
            "book.n.01",
            "book",
            aliases=("book",),
            parents=(),
            definition="a written work or composition",
        ),
        _record(
            "wordnet:bible.n.01",
            "bible.n.01",
            "Bible",
            aliases=("Bible", "Book"),
            parents=("sacred_text.n.01",),
            definition="the sacred writings of the Christian religions",
        ),
        _record(
            "wordnet:quarrel.n.01",
            "quarrel.n.01",
            "dustup",
            aliases=("dustup", "quarrel", "row"),
            parents=(),
            definition="an angry dispute",
        ),
        _record(
            "wordnet:row.n.05",
            "row.n.05",
            "row",
            aliases=("row",),
            parents=("array.n.01",),
            definition="a linear array of numbers, letters, or symbols",
        ),
        _record(
            "wordnet:array.n.01",
            "array.n.01",
            "array",
            aliases=("array",),
            parents=(),
            definition="an orderly arrangement",
        ),
        _record(
            "wordnet:array.n.03",
            "array.n.03",
            "array",
            aliases=("array", "raiment", "regalia"),
            parents=("clothing.n.01",),
            definition="especially fine or decorative clothing",
        ),
    ]
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _record(
    concept_id: str,
    provider_ref: str,
    label: str,
    *,
    aliases: tuple[str, ...],
    parents: tuple[str, ...],
    definition: str,
) -> dict[str, object]:
    return {
        "id": concept_id,
        "provider": "wordnet",
        "kind": "concept",
        "provider_ref": provider_ref,
        "label": label,
        "aliases_json": json.dumps(list(aliases)),
        "parents_json": json.dumps(list(parents)),
        "definition": definition,
    }
