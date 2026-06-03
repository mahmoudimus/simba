"""Tests for the temporal knowledge-graph store (src/simba/kg/store.py)."""

from __future__ import annotations

import pathlib
import sqlite3

import pytest

import simba.db
from simba.kg import kg_add, kg_invalidate, kg_query
from simba.kg.store import backup_and_drop_proven_facts


@pytest.fixture(autouse=True)
def _patch_db_path(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point simba.db.get_db_path at a temp directory for every test."""
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


class TestKgAdd:
    def test_add_then_query_round_trip(self) -> None:
        result = kg_add(
            "simba",
            "uses",
            "sqlite",
            "from db.py",
            subject_type="module",
            object_type="library",
            transcript_id="t-1",
            char_start=42,
            project_path="proj-1",
        )
        assert result == "added"

        rows = kg_query(subject="simba", project_path="proj-1")
        assert len(rows) == 1
        row = rows[0]
        assert row["subject"] == "simba"
        assert row["predicate"] == "uses"
        assert row["object"] == "sqlite"
        assert row["subject_type"] == "module"
        assert row["object_type"] == "library"
        assert row["proof"] == "from db.py"
        assert row["transcript_id"] == "t-1"
        assert row["char_start"] == 42
        assert row["valid_from"]
        assert row["valid_to"] is None

    def test_default_types_are_concept(self) -> None:
        kg_add("a", "rel", "b", "proof", project_path="proj-1")
        rows = kg_query(subject="a", project_path="proj-1")
        assert rows[0]["subject_type"] == "concept"
        assert rows[0]["object_type"] == "concept"

    def test_duplicate_returns_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Dedup keys on UNIQUE(..., valid_from); freeze the clock so both adds
        # share a valid_from (otherwise a second-boundary makes the 2nd a new
        # open edge rather than a collision).
        monkeypatch.setattr("simba.kg.store._now", lambda: "2026-01-01T00:00:00Z")
        assert kg_add("a", "rel", "b", "p1", project_path="proj-1") == "added"
        assert kg_add("a", "rel", "b", "p2", project_path="proj-1") == "exists"
        rows = kg_query(subject="a", project_path="proj-1")
        assert len(rows) == 1


class TestKgInvalidate:
    def test_invalidate_closes_edge(self) -> None:
        kg_add("a", "rel", "b", "p", project_path="proj-1")
        closed = kg_invalidate("a", "rel", "b", project_path="proj-1")
        assert closed == 1

        # Default query hides the now-expired edge.
        assert kg_query(subject="a", project_path="proj-1") == []

        # include_expired surfaces it again, with valid_to populated.
        rows = kg_query(subject="a", project_path="proj-1", include_expired=True)
        assert len(rows) == 1
        assert rows[0]["valid_to"] is not None

    def test_invalidate_returns_zero_when_nothing_open(self) -> None:
        assert kg_invalidate("missing", "rel", "obj", project_path="proj-1") == 0


class TestKgTemporal:
    def test_as_of_filters_by_validity_window(self) -> None:
        kg_add("a", "rel", "b", "p", project_path="proj-1")
        rows = kg_query(subject="a", project_path="proj-1")
        valid_from = rows[0]["valid_from"]

        # Before the edge existed → not returned.
        before = kg_query(
            subject="a", project_path="proj-1", as_of="2000-01-01T00:00:00Z"
        )
        assert before == []

        # At/after valid_from while still open → returned.
        after = kg_query(subject="a", project_path="proj-1", as_of=valid_from)
        assert len(after) == 1

        # Close the edge, then ask "as_of" a far-future time → excluded.
        kg_invalidate("a", "rel", "b", project_path="proj-1")
        closed_rows = kg_query(subject="a", project_path="proj-1", include_expired=True)
        valid_to = closed_rows[0]["valid_to"]
        assert kg_query(subject="a", project_path="proj-1", as_of=valid_to) == []


class TestKgQueryFts:
    def test_bm25_substring_match_via_trigram(self) -> None:
        kg_add(
            "blm.ledger_transactions",
            "writes_to",
            "postgres",
            "proof",
            project_path="proj-1",
        )
        kg_add("unrelated", "rel", "thing", "proof", project_path="proj-1")

        rows = kg_query(query="ledg", project_path="proj-1")
        subjects = [r["subject"] for r in rows]
        assert "blm.ledger_transactions" in subjects
        assert "unrelated" not in subjects

    def test_bad_fts_match_returns_empty(self) -> None:
        kg_add("a", "rel", "b", "proof", project_path="proj-1")
        # A malformed FTS expression should be swallowed → [].
        assert kg_query(query='"', project_path="proj-1") == []


class TestKgTriggerSync:
    def test_fts_synced_after_add(self) -> None:
        kg_add(
            "blm.ledger_transactions",
            "writes_to",
            "postgres",
            "proof",
            project_path="proj-1",
        )
        db_path = simba.db.get_db_path()
        conn = sqlite3.connect(str(db_path))
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM kg_edges_fts WHERE kg_edges_fts MATCH 'ledg'"
            ).fetchone()[0]
        finally:
            conn.close()
        assert count == 1

    def test_fts_purged_after_delete(self) -> None:
        kg_add("deletable_subject", "rel", "obj", "proof", project_path="proj-1")
        db_path = simba.db.get_db_path()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("DELETE FROM kg_edges WHERE subject='deletable_subject'")
            conn.commit()
            count = conn.execute(
                "SELECT COUNT(*) FROM kg_edges_fts WHERE kg_edges_fts MATCH 'deletable'"
            ).fetchone()[0]
        finally:
            conn.close()
        assert count == 0  # the _ad trigger purged the stale FTS row

    def test_fts_swapped_after_update(self) -> None:
        kg_add("oldsubject_xyz", "rel", "obj", "proof", project_path="proj-1")
        db_path = simba.db.get_db_path()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "UPDATE kg_edges SET subject='newsubject_xyz' "
                "WHERE subject='oldsubject_xyz'"
            )
            conn.commit()
            old = conn.execute(
                "SELECT COUNT(*) FROM kg_edges_fts "
                "WHERE kg_edges_fts MATCH 'oldsubject'"
            ).fetchone()[0]
            new = conn.execute(
                "SELECT COUNT(*) FROM kg_edges_fts "
                "WHERE kg_edges_fts MATCH 'newsubject'"
            ).fetchone()[0]
        finally:
            conn.close()
        assert old == 0  # _au trigger removed the old content
        assert new == 1  # …and indexed the new content


class TestProvenFactsMigration:
    def test_proven_facts_backed_up_and_dropped_on_connect(self) -> None:
        # Pre-create a legacy 4-col proven_facts table with a row.
        db_path = simba.db.get_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE proven_facts (subject TEXT, predicate TEXT, "
            "object TEXT, proof TEXT)"
        )
        conn.execute(
            "INSERT INTO proven_facts VALUES (?, ?, ?, ?)",
            ("old", "fact", "value", "legacy"),
        )
        conn.commit()
        conn.close()

        # Opening via get_db runs the kg schema initializer → migration.
        with simba.db.get_db():
            pass

        conn = sqlite3.connect(str(db_path))
        try:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "proven_facts" not in tables
            assert "proven_facts_bak" in tables
            rows = conn.execute("SELECT * FROM proven_facts_bak").fetchall()
            assert [tuple(r) for r in rows] == [("old", "fact", "value", "legacy")]
        finally:
            conn.close()

    def test_migration_is_idempotent_when_already_gone(self) -> None:
        # No proven_facts table exists → migration is a no-op.
        with simba.db.get_db() as conn:
            backup_and_drop_proven_facts(conn)
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "proven_facts" not in tables


class TestKgProjectScoping:
    def test_other_projects_edge_never_returned(self) -> None:
        kg_add("a", "rel", "b", "p", project_path="proj-1")
        kg_add("a", "rel", "b", "p", project_path="proj-2")

        rows = kg_query(subject="a", project_path="proj-1")
        assert len(rows) == 1

        fts_rows = kg_query(query="rel", subject=None, project_path="proj-1")
        assert all(r["subject"] == "a" for r in fts_rows)
        # proj-2's edge has a distinct rowid; ensure only one is in scope.
        assert len(fts_rows) == 1


class TestKgBitemporal:
    """occurred_at = event time, distinct from valid_from = belief time."""

    def test_occurred_at_stored_and_returned(self) -> None:
        kg_add(
            "a",
            "moved_to",
            "nyc",
            "p",
            project_path="proj-1",
            occurred_at="2025-03-01T00:00:00Z",
        )
        rows = kg_query(subject="a", project_path="proj-1")
        assert rows[0]["occurred_at"] == "2025-03-01T00:00:00Z"
        # event time is independent of belief time (valid_from = insert time)
        assert rows[0]["valid_from"] != "2025-03-01T00:00:00Z"

    def test_occurred_at_defaults_to_none(self) -> None:
        kg_add("a", "rel", "b", "p", project_path="proj-1")
        assert kg_query(subject="a", project_path="proj-1")[0]["occurred_at"] is None

    def test_occurred_after_is_inclusive_and_excludes_earlier(self) -> None:
        kg_add("e2024", "r", "o", "p", project_path="p1", occurred_at="2024-06-01")
        kg_add("e2025", "r", "o", "p", project_path="p1", occurred_at="2025-06-01")
        kg_add("e2026", "r", "o", "p", project_path="p1", occurred_at="2026-06-01")
        rows = kg_query(project_path="p1", occurred_after="2025-06-01")
        subjects = {r["subject"] for r in rows}
        assert subjects == {"e2025", "e2026"}

    def test_occurred_before_is_inclusive_and_excludes_later(self) -> None:
        kg_add("e2024", "r", "o", "p", project_path="p1", occurred_at="2024-06-01")
        kg_add("e2025", "r", "o", "p", project_path="p1", occurred_at="2025-06-01")
        kg_add("e2026", "r", "o", "p", project_path="p1", occurred_at="2026-06-01")
        rows = kg_query(project_path="p1", occurred_before="2025-06-01")
        subjects = {r["subject"] for r in rows}
        assert subjects == {"e2024", "e2025"}

    def test_occurred_filter_excludes_unknown_event_time(self) -> None:
        kg_add("known", "r", "o", "p", project_path="p1", occurred_at="2025-06-01")
        kg_add("unknown", "r", "o", "p", project_path="p1")  # occurred_at = None
        # With no event-time filter, both show.
        assert len(kg_query(project_path="p1")) == 2
        # With an event-time window, the unknown-event-time edge is excluded.
        rows = kg_query(project_path="p1", occurred_after="2024-01-01")
        assert {r["subject"] for r in rows} == {"known"}

    def test_migration_adds_occurred_at_to_legacy_table(self) -> None:
        # Pre-create a legacy kg_edges WITHOUT the occurred_at column.
        db_path = simba.db.get_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE kg_edges (id INTEGER PRIMARY KEY, subject TEXT, "
            "predicate TEXT, object TEXT, subject_type TEXT, object_type TEXT, "
            "proof TEXT, transcript_id TEXT, char_start INTEGER, valid_from TEXT, "
            "valid_to TEXT, project_path TEXT NOT NULL, created_at TEXT)"
        )
        conn.commit()
        conn.close()

        # connect() runs the schema initializers → migration adds the column.
        with simba.db.connect():
            pass

        conn = sqlite3.connect(str(db_path))
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(kg_edges)")}
        finally:
            conn.close()
        assert "occurred_at" in cols


class TestEntityResolution:
    def _set(self, monkeypatch, *, enabled: bool) -> None:
        import simba.config
        import simba.kg.config as kc

        real = simba.config.load

        def fake(section, *a, **k):
            if section == "kg":
                return kc.KgConfig(entity_resolution_enabled=enabled)
            return real(section, *a, **k)

        monkeypatch.setattr(simba.config, "load", fake)

    def _enable(self, monkeypatch) -> None:
        self._set(monkeypatch, enabled=True)

    def test_disabled_keeps_surface_forms(self, monkeypatch) -> None:
        # Resolution ships on by default now, so disable it explicitly here.
        self._set(monkeypatch, enabled=False)
        kg_add("GITHUB_TOKEN", "causes", "401", "p", project_path="proj-1")
        kg_add("the GITHUB_TOKEN", "causes", "errors", "p", project_path="proj-1")
        subjects = {r["subject"] for r in kg_query(project_path="proj-1")}
        assert subjects == {"GITHUB_TOKEN", "the GITHUB_TOKEN"}  # not merged

    def test_enabled_canonicalizes_subject(self, monkeypatch) -> None:
        self._enable(monkeypatch)
        kg_add("GITHUB_TOKEN", "causes", "401", "p", project_path="proj-1")
        # variant resolves to the existing canonical surface form
        kg_add("the GITHUB_TOKEN", "blocks", "gh", "p", project_path="proj-1")
        subjects = {r["subject"] for r in kg_query(project_path="proj-1")}
        assert subjects == {"GITHUB_TOKEN"}  # collapsed to one node

    def test_enabled_canonicalizes_object(self, monkeypatch) -> None:
        self._enable(monkeypatch)
        kg_add("gh", "uses", "GITHUB_TOKEN", "p", project_path="proj-1")
        kg_add("daemon", "reads", "the github_token", "p", project_path="proj-1")
        objects = {r["object"] for r in kg_query(project_path="proj-1")}
        assert objects == {"GITHUB_TOKEN"}

    def test_resolution_is_project_scoped(self, monkeypatch) -> None:
        self._enable(monkeypatch)
        kg_add("GITHUB_TOKEN", "causes", "401", "p", project_path="proj-1")
        # different project: must NOT borrow proj-1's canonical form
        kg_add("the github_token", "blocks", "gh", "p", project_path="proj-2")
        subj2 = {r["subject"] for r in kg_query(project_path="proj-2")}
        assert subj2 == {"the github_token"}
