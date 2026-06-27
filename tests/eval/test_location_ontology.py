from __future__ import annotations

import json
import pathlib

from simba.eval import location_ontology


def test_ratifies_location_containment_through_parent_graph(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "locations.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "loc:united_states",
                    "label": "United States",
                    "aliases": ["USA"],
                    "parents": [],
                    "provenance": ["test:country"],
                },
                {
                    "id": "loc:california",
                    "label": "California",
                    "aliases": [],
                    "parents": ["loc:united_states"],
                    "provenance": ["test:admin1"],
                },
                {
                    "id": "loc:big_sur",
                    "label": "Big Sur",
                    "aliases": [],
                    "parents": ["loc:california"],
                    "provenance": ["test:place"],
                },
            ]
        ),
        encoding="utf-8",
    )

    result = location_ontology.ratify_location_containment(
        "Big Sur",
        "United States",
        location_path=path,
    )

    assert result.ratified is True
    assert result.reason == "location_containment"
    assert result.path == ("Big Sur", "California", "United States")


def test_rejects_uncontained_location(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "locations.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "loc:united_states",
                    "label": "United States",
                    "aliases": ["USA"],
                    "parents": [],
                    "provenance": ["test:country"],
                },
                {
                    "id": "loc:new_zealand",
                    "label": "New Zealand",
                    "aliases": ["NZ"],
                    "parents": [],
                    "provenance": ["test:country"],
                },
            ]
        ),
        encoding="utf-8",
    )

    result = location_ontology.ratify_location_containment(
        "New Zealand",
        "United States",
        location_path=path,
    )

    assert result.ratified is False
    assert result.reason == "no_containment_path"
