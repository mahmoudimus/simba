"""Tests for simba.markers_cli -- marker discovery, audit, and update CLI."""

from __future__ import annotations

import pathlib

import pytest

import simba.markers
import simba.markers_cli
import simba.orchestration.templates


class TestScanMarkers:
    def test_scan_finds_markers(self, tmp_path: pathlib.Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text(
            "# Title\n<!-- BEGIN SIMBA:core -->\nrule one\n<!-- END SIMBA:core -->\n"
        )
        hits = simba.markers_cli.scan_markers(tmp_path)
        assert len(hits) == 1
        assert hits[0].name == "core"
        assert hits[0].line_no == 2
        assert hits[0].content_len > 0

    def test_scan_multiple_markers(self, tmp_path: pathlib.Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text(
            "<!-- BEGIN SIMBA:alpha -->\nA\n<!-- END SIMBA:alpha -->\n"
            "<!-- BEGIN SIMBA:beta -->\nBB\n<!-- END SIMBA:beta -->\n"
        )
        hits = simba.markers_cli.scan_markers(tmp_path)
        assert len(hits) == 2
        names = {h.name for h in hits}
        assert names == {"alpha", "beta"}

    def test_scan_excludes_gitless(self, tmp_path: pathlib.Path) -> None:
        excluded = tmp_path / "_gitless"
        excluded.mkdir()
        md = excluded / "hidden.md"
        md.write_text(
            "<!-- BEGIN SIMBA:secret -->\nhidden\n<!-- END SIMBA:secret -->\n"
        )
        hits = simba.markers_cli.scan_markers(tmp_path)
        assert len(hits) == 0

    def test_scan_excludes_node_modules(self, tmp_path: pathlib.Path) -> None:
        excluded = tmp_path / "node_modules"
        excluded.mkdir()
        md = excluded / "pkg.md"
        md.write_text("<!-- BEGIN SIMBA:pkg -->\npkg stuff\n<!-- END SIMBA:pkg -->\n")
        hits = simba.markers_cli.scan_markers(tmp_path)
        assert len(hits) == 0

    def test_scan_excludes_git_dir(self, tmp_path: pathlib.Path) -> None:
        excluded = tmp_path / ".git"
        excluded.mkdir()
        md = excluded / "info.md"
        md.write_text("<!-- BEGIN SIMBA:git -->\ngit stuff\n<!-- END SIMBA:git -->\n")
        hits = simba.markers_cli.scan_markers(tmp_path)
        assert len(hits) == 0


class TestCmdAuditUnused:
    def test_reports_unused_section(
        self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # No .md files at all -> all MANAGED_SECTIONS are unused.
        simba.markers_cli.cmd_audit(tmp_path)
        captured = capsys.readouterr()
        assert "Unused sections" in captured.out


class TestCmdAuditOrphaned:
    def test_reports_orphaned_marker(
        self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        md = tmp_path / "doc.md"
        # Use a marker name that is NOT in MANAGED_SECTIONS.
        md.write_text(
            "<!-- BEGIN SIMBA:not_a_real_section -->\n"
            "orphan\n"
            "<!-- END SIMBA:not_a_real_section -->\n"
        )
        simba.markers_cli.cmd_audit(tmp_path)
        captured = capsys.readouterr()
        assert "Orphaned markers" in captured.out
        assert "not_a_real_section" in captured.out


class TestCmdAuditStale:
    def test_reports_stale_marker(
        self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Pick a real section from MANAGED_SECTIONS and put wrong content.
        section = next(iter(simba.orchestration.templates.MANAGED_SECTIONS))
        md = tmp_path / "doc.md"
        md.write_text(
            f"<!-- BEGIN SIMBA:{section} -->\n"
            "this is definitely not the real template content\n"
            f"<!-- END SIMBA:{section} -->\n"
        )
        simba.markers_cli.cmd_audit(tmp_path)
        captured = capsys.readouterr()
        assert "Stale markers" in captured.out
        assert section in captured.out


class TestCmdUpdateWrites:
    def test_update_writes_content(self, tmp_path: pathlib.Path) -> None:
        section = next(iter(simba.orchestration.templates.MANAGED_SECTIONS))
        template = simba.orchestration.templates.MANAGED_SECTIONS[section]
        md = tmp_path / "doc.md"
        md.write_text(
            f"# Title\n"
            f"<!-- BEGIN SIMBA:{section} -->\n"
            f"<!-- END SIMBA:{section} -->\n"
            f"footer\n"
        )
        simba.markers_cli.cmd_update(tmp_path)
        result = md.read_text()
        # The template content should now appear between the markers.
        assert template.strip() in result
        # Surrounding content preserved.
        assert "# Title" in result
        assert "footer" in result


class TestCmdShow:
    def test_show_prints_content(self, capsys: pytest.CaptureFixture[str]) -> None:
        section = next(iter(simba.orchestration.templates.MANAGED_SECTIONS))
        template = simba.orchestration.templates.MANAGED_SECTIONS[section]
        simba.markers_cli.cmd_show(section)
        captured = capsys.readouterr()
        assert template.strip() in captured.out

    def test_show_unknown_section(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = simba.markers_cli.cmd_show("nonexistent_section_xyz")
        assert rc == 1
        captured = capsys.readouterr()
        assert "Unknown section" in captured.err


class TestScanForeignMarkers:
    def test_finds_neuron_markers(self, tmp_path: pathlib.Path) -> None:
        md = tmp_path / "agent.md"
        md.write_text(
            "# Agent\n"
            "<!-- BEGIN NEURON:completion_protocol -->\nold\n"
            "<!-- END NEURON:completion_protocol -->\n"
        )
        hits = simba.markers_cli.scan_foreign_markers(tmp_path)
        assert len(hits) == 1
        assert hits[0].tag == "NEURON:completion_protocol"

    def test_finds_core_markers(self, tmp_path: pathlib.Path) -> None:
        md = tmp_path / "CLAUDE.md"
        md.write_text("<!-- CORE -->\nrule one\n<!-- /CORE -->\n")
        hits = simba.markers_cli.scan_foreign_markers(tmp_path)
        assert any(h.tag == "CORE" for h in hits)

    def test_ignores_simba_markers(self, tmp_path: pathlib.Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text(
            "<!-- BEGIN SIMBA:core -->\nsimba\n<!-- END SIMBA:core -->\n"
        )
        hits = simba.markers_cli.scan_foreign_markers(tmp_path)
        assert len(hits) == 0


class TestMigrateContent:
    def test_neuron_to_simba(self) -> None:
        content = (
            "<!-- BEGIN NEURON:search_tools -->\n"
            "use rg\n"
            "<!-- END NEURON:search_tools -->\n"
        )
        result, changes = simba.markers_cli._migrate_content(content)
        assert "<!-- BEGIN SIMBA:search_tools -->" in result
        assert "<!-- END SIMBA:search_tools -->" in result
        assert "use rg" in result
        assert changes == [("NEURON:search_tools", "search_tools")]

    def test_core_to_simba(self) -> None:
        content = "<!-- CORE -->\nrule\n<!-- /CORE -->\n"
        result, changes = simba.markers_cli._migrate_content(content)
        assert "<!-- BEGIN SIMBA:core -->" in result
        assert "<!-- END SIMBA:core -->" in result
        assert "rule" in result
        assert changes == [("CORE", "core")]

    def test_preserves_body(self) -> None:
        body = "line 1\nline 2\nline 3\n"
        content = (
            f"<!-- BEGIN NEURON:nav_tools -->\n"
            f"{body}<!-- END NEURON:nav_tools -->\n"
        )
        result, _ = simba.markers_cli._migrate_content(content)
        assert body in result


class TestCmdMigrate:
    def test_dry_run(
        self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        md = tmp_path / "agent.md"
        original = (
            "<!-- BEGIN NEURON:foo -->\nbar\n<!-- END NEURON:foo -->\n"
        )
        md.write_text(original)
        simba.markers_cli.cmd_migrate(tmp_path, dry_run=True)
        # File should NOT be modified.
        assert md.read_text() == original
        captured = capsys.readouterr()
        assert "dry run" in captured.out

    def test_migrate_writes(self, tmp_path: pathlib.Path) -> None:
        md = tmp_path / "agent.md"
        md.write_text(
            "<!-- BEGIN NEURON:completion_protocol -->\n"
            "old content\n"
            "<!-- END NEURON:completion_protocol -->\n"
        )
        simba.markers_cli.cmd_migrate(tmp_path)
        result = md.read_text()
        assert "<!-- BEGIN SIMBA:completion_protocol -->" in result
        assert "<!-- END SIMBA:completion_protocol -->" in result
        assert "old content" in result

    def test_no_foreign_markers(
        self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        md = tmp_path / "clean.md"
        md.write_text("<!-- BEGIN SIMBA:core -->\nok\n<!-- END SIMBA:core -->\n")
        simba.markers_cli.cmd_migrate(tmp_path)
        captured = capsys.readouterr()
        assert "No non-SIMBA markers found" in captured.out
