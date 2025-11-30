"""Coverage ingest incremental harness tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.config import ConfigBuilder
from codeintel.config.models import ToolsConfig
from codeintel.ingestion.change_tracker import ChangeTracker, IncrementalIngestPolicy
from codeintel.ingestion.common import ChangeRequest, ChangeSet
from codeintel.ingestion.coverage_ingest import ingest_coverage_lines
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.ingestion.tool_service import CoverageFileReport, ToolService
from tests._helpers.gateway import open_ingestion_gateway


class _FakeCoverageService(ToolService):
    """Provide synthetic coverage reports without invoking external tools."""

    def __init__(self, report: CoverageFileReport, repo_root: Path) -> None:
        tools_cfg = ToolsConfig.default()
        runner = ToolRunner(cache_dir=repo_root / "build" / ".tool_cache", tools_config=tools_cfg)
        super().__init__(runner, tools_cfg)
        self._report = report

    async def run_coverage_json(
        self,
        repo_root: Path,
        *,
        coverage_file: Path | None = None,
        output_path: Path | None = None,
    ) -> list[CoverageFileReport]:
        del repo_root, coverage_file, output_path
        return [self._report]


def test_coverage_ingest_runs_full_rebuild_with_tracker(tmp_path: Path) -> None:
    """Ensure coverage ingest performs a full rebuild when invoked via tracker."""
    gateway = open_ingestion_gateway()
    try:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        coverage_file = repo_root / ".coverage"
        coverage_file.touch()

        builder = ConfigBuilder.from_snapshot(
            repo="demo/repo",
            commit="deadbeef",
            repo_root=repo_root,
            build_dir=repo_root / "build",
        )
        cfg = builder.coverage_ingest(coverage_file=coverage_file)
        report = CoverageFileReport(
            rel_path="pkg/mod.py",
            executed_lines={1, 2},
            missing_lines={3},
        )
        fake_service = _FakeCoverageService(report, repo_root)
        tracker = ChangeTracker(
            gateway=gateway,
            change_request=ChangeRequest(
                repo=cfg.repo,
                commit=cfg.commit,
                repo_root=repo_root,
                modules=(),
            ),
            modules=(),
            change_set=ChangeSet(added=[], modified=[], deleted=[]),
            policy=IncrementalIngestPolicy(),
        )

        ingest_coverage_lines(
            gateway=gateway,
            cfg=cfg,
            tools=ToolsConfig.model_validate({}),
            tool_service=fake_service,
            tracker=tracker,
        )

        count = gateway.con.execute(
            """
            SELECT COUNT(*) FROM analytics.coverage_lines
            WHERE repo = ? AND commit = ?
            """,
            [cfg.repo, cfg.commit],
        ).fetchone()
        if count is None:
            pytest.fail("Expected coverage rows but query returned no result")
        expected = len(report.executed_lines | report.missing_lines)
        count_value = count[0]
        if count_value != expected:
            pytest.fail(f"Expected {expected} coverage rows, found {count_value}")
    finally:
        gateway.close()
