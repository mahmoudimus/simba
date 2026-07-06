"""Tests for the cross-store drift audit (src/simba/memory/reconcile.py).

Three stores can drift: LanceDB (`memories.lance`, source of truth), the FTS5
keyword mirror (`memory_fts.db`, derived, SYSTEM-excluded by design), and the
usage sidecar (`memory_usage` in the shared `simba.db`). ``reconcile()`` audits
all three id sets and, only with ``apply=True``, repairs the one safe
direction: re-upserting Lance rows missing from FTS. Everything else (ghost
FTS rows, orphaned usage rows) is always report-only.
"""

from __future__ import annotations

import pathlib

import pytest

import simba.db
import simba.memory.fts as fts
import simba.memory.reconcile as reconcile
import simba.memory.usage as usage


def _mem(mid: str, content: str, **over) -> dict:
    base = {
        "id": mid,
        "type": "GOTCHA",
        "content": content,
        "context": "",
        "tags": "[]",
        "confidence": 0.85,
        "sessionSource": "",
        "projectPath": "proj-1",
        "createdAt": "2026-01-01T00:00:00Z",
        "lastAccessedAt": "2026-01-01T00:00:00Z",
        "accessCount": 0,
        "vector": [0.1] * 768,
    }
    base.update(over)
    return base


async def _seed_three_lance_rows(lance_table) -> None:
    # lance_table (conftest fixture) already carries one SYSTEM init row,
    # which is excluded from FTS by design (fts.py never indexes SYSTEM).
    await lance_table.add(
        [
            _mem("m1", "ruff lints the python code"),
            _mem("m2", "pytest runs the suite"),
            _mem("m3", "uv manages the venv"),
        ]
    )


class TestAuditDryRun:
    @pytest.mark.asyncio
    async def test_reports_drift_in_all_three_directions(
        self, lance_table, tmp_path: pathlib.Path
    ) -> None:
        await _seed_three_lance_rows(lance_table)
        fts_path = tmp_path / fts.FTS_FILENAME
        with fts.connect(fts_path):
            fts.upsert(_mem("m1", "ruff lints the python code"))
            fts.upsert(_mem("m2", "pytest runs the suite"))
        with simba.db.connect(tmp_path):
            usage.get_or_create("orphan1", now=0.0)

        report = await reconcile.reconcile(lance_table, fts_path, tmp_path, apply=False)

        assert report.missing_fts_ids == ["m3"]
        assert report.ghost_fts_ids == []
        assert report.orphan_usage_ids == ["orphan1"]
        assert report.repaired_ids == []
        assert report.lance_total == 4  # SYSTEM + m1 + m2 + m3
        assert report.lance_non_system == 3
        assert report.fts_total == 2
        assert report.usage_total == 1

    @pytest.mark.asyncio
    async def test_dry_run_changes_nothing(
        self, lance_table, tmp_path: pathlib.Path
    ) -> None:
        await _seed_three_lance_rows(lance_table)
        fts_path = tmp_path / fts.FTS_FILENAME
        with fts.connect(fts_path):
            fts.upsert(_mem("m1", "ruff lints the python code"))
            fts.upsert(_mem("m2", "pytest runs the suite"))
        with simba.db.connect(tmp_path):
            usage.get_or_create("orphan1", now=0.0)

        await reconcile.reconcile(lance_table, fts_path, tmp_path, apply=False)

        assert await lance_table.count_rows() == 4
        with fts.connect(fts_path):
            assert fts.count() == 2
        with simba.db.connect(tmp_path):
            assert usage.MemoryUsage.select().count() == 1


class TestApplyRepairsOnlyMissingFts:
    @pytest.mark.asyncio
    async def test_run_upserts_missing_fts_row(
        self, lance_table, tmp_path: pathlib.Path
    ) -> None:
        await _seed_three_lance_rows(lance_table)
        fts_path = tmp_path / fts.FTS_FILENAME
        with fts.connect(fts_path):
            fts.upsert(_mem("m1", "ruff lints the python code"))
            fts.upsert(_mem("m2", "pytest runs the suite"))
        with simba.db.connect(tmp_path):
            usage.get_or_create("orphan1", now=0.0)

        report = await reconcile.reconcile(lance_table, fts_path, tmp_path, apply=True)

        assert report.repaired_ids == ["m3"]
        with fts.connect(fts_path):
            assert fts.count() == 3
            hits = fts.search("manages", project_path="proj-1")
            assert [h["memory_id"] for h in hits] == ["m3"]

    @pytest.mark.asyncio
    async def test_run_leaves_lance_and_usage_untouched(
        self, lance_table, tmp_path: pathlib.Path
    ) -> None:
        await _seed_three_lance_rows(lance_table)
        fts_path = tmp_path / fts.FTS_FILENAME
        with fts.connect(fts_path):
            fts.upsert(_mem("m1", "ruff lints the python code"))
            fts.upsert(_mem("m2", "pytest runs the suite"))
        with simba.db.connect(tmp_path):
            usage.get_or_create("orphan1", now=0.0)

        await reconcile.reconcile(lance_table, fts_path, tmp_path, apply=True)

        assert await lance_table.count_rows() == 4
        with simba.db.connect(tmp_path):
            assert usage.MemoryUsage.select().count() == 1
            assert {r.memory_id for r in usage.MemoryUsage.select()} == {"orphan1"}

    @pytest.mark.asyncio
    async def test_rerun_after_apply_reports_zero_missing(
        self, lance_table, tmp_path: pathlib.Path
    ) -> None:
        await _seed_three_lance_rows(lance_table)
        fts_path = tmp_path / fts.FTS_FILENAME
        with fts.connect(fts_path):
            fts.upsert(_mem("m1", "ruff lints the python code"))
            fts.upsert(_mem("m2", "pytest runs the suite"))
        with simba.db.connect(tmp_path):
            usage.get_or_create("orphan1", now=0.0)

        await reconcile.reconcile(lance_table, fts_path, tmp_path, apply=True)
        second = await reconcile.reconcile(lance_table, fts_path, tmp_path, apply=False)

        assert second.missing_fts_ids == []
        assert second.repaired_ids == []
        # ghost/orphan buckets are unaffected by the FTS-only repair.
        assert second.orphan_usage_ids == ["orphan1"]

    @pytest.mark.asyncio
    async def test_ghost_fts_rows_are_never_deleted(
        self, lance_table, tmp_path: pathlib.Path
    ) -> None:
        await lance_table.add([_mem("m1", "kept memory")])
        fts_path = tmp_path / fts.FTS_FILENAME
        with fts.connect(fts_path):
            fts.upsert(_mem("m1", "kept memory"))
            fts.upsert(_mem("ghost1", "no longer backed by a lance row"))

        report = await reconcile.reconcile(lance_table, fts_path, tmp_path, apply=True)

        assert report.ghost_fts_ids == ["ghost1"]
        assert report.missing_fts_ids == []
        with fts.connect(fts_path):
            assert fts.count() == 2  # ghost row survives -- report-only

    @pytest.mark.asyncio
    async def test_orphan_usage_rows_are_never_deleted(
        self, lance_table, tmp_path: pathlib.Path
    ) -> None:
        await lance_table.add([_mem("m1", "kept memory")])
        fts_path = tmp_path / fts.FTS_FILENAME
        with simba.db.connect(tmp_path):
            usage.get_or_create("orphan1", now=0.0)

        report = await reconcile.reconcile(lance_table, fts_path, tmp_path, apply=True)

        assert report.orphan_usage_ids == ["orphan1"]
        with simba.db.connect(tmp_path):
            assert usage.MemoryUsage.select().count() == 1

    @pytest.mark.asyncio
    async def test_apply_with_no_drift_is_a_noop(
        self, lance_table, tmp_path: pathlib.Path
    ) -> None:
        await lance_table.add([_mem("m1", "kept memory")])
        fts_path = tmp_path / fts.FTS_FILENAME
        with fts.connect(fts_path):
            fts.upsert(_mem("m1", "kept memory"))

        report = await reconcile.reconcile(lance_table, fts_path, tmp_path, apply=True)

        assert report.repaired_ids == []
        assert report.clean is True


class TestResolveDataDir:
    def test_default_is_dot_simba_memory_under_cwd(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        import simba.memory.config

        monkeypatch.setattr(
            simba.memory.config,
            "load_config",
            lambda: simba.memory.config.MemoryConfig(),
        )
        assert reconcile.resolve_data_dir(tmp_path) == tmp_path / ".simba" / "memory"

    def test_honors_db_path_override(self, tmp_path: pathlib.Path, monkeypatch) -> None:
        import simba.memory.config

        override = tmp_path / "custom-store"
        monkeypatch.setattr(
            simba.memory.config,
            "load_config",
            lambda: simba.memory.config.MemoryConfig(db_path=str(override)),
        )
        assert reconcile.resolve_data_dir(tmp_path) == override


class TestFormatReport:
    def test_includes_counts_and_examples(self) -> None:
        report = reconcile.ReconcileReport(
            lance_total=4,
            lance_non_system=3,
            fts_total=2,
            usage_total=1,
            missing_fts_ids=["m3"],
            ghost_fts_ids=[],
            orphan_usage_ids=["orphan1"],
        )
        text = reconcile.format_report(report)
        assert "m3" in text
        assert "orphan1" in text
        assert "missing-fts" in text
        assert "ghost-fts" in text
        assert "orphan-usage" in text

    def test_truncates_examples_to_limit(self) -> None:
        ids = [f"m{i}" for i in range(15)]
        report = reconcile.ReconcileReport(
            lance_total=15,
            lance_non_system=15,
            fts_total=0,
            usage_total=0,
            missing_fts_ids=ids,
            ghost_fts_ids=[],
            orphan_usage_ids=[],
        )
        text = reconcile.format_report(report, example_limit=10)
        assert "m0" in text
        assert "m9" in text
        assert "m14" not in text
        assert "+5 more" in text

    def test_notes_repaired_count(self) -> None:
        report = reconcile.ReconcileReport(
            lance_total=1,
            lance_non_system=1,
            fts_total=1,
            usage_total=0,
            missing_fts_ids=[],
            ghost_fts_ids=[],
            orphan_usage_ids=[],
            repaired_ids=["m3"],
        )
        text = reconcile.format_report(report)
        assert "repaired" in text
        assert "1" in text
