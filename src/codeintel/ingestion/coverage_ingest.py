"""Ingest coverage.py results into `analytics.coverage_lines`."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from coverage import Coverage, CoverageData
from coverage.exceptions import CoverageException

from codeintel.config import CoverageIngestStepConfig
from codeintel.config.dataset_contract import CoverageLineRow, coverage_line_to_tuple
from codeintel.config.models import ToolsConfig
from codeintel.ingestion.change_tracker import (
    ChangeTracker,
    IncrementalIngestOps,
    SupportsFullRebuild,
    run_incremental_ingest,
)
from codeintel.ingestion.common import ModuleRecord, run_batch, should_skip_missing_file
from codeintel.ingestion.paths import normalize_rel_path
from codeintel.ingestion.tool_runner import ToolExecutionError, ToolNotFoundError, ToolRunner
from codeintel.ingestion.tool_service import CoverageFileReport, ToolService
from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class CoverageInsertContext:
    """Shared context for inserting coverage rows."""

    repo: str
    commit: str
    now: datetime
    coverage_file: Path


@dataclass(frozen=True)
class CoverageFileInfo:
    """Resolved coverage file paths used during ingestion."""

    measured_path: Path
    rel_path: str


def _rows_from_reports(
    cfg: CoverageIngestStepConfig,
    reports: list[CoverageFileReport],
    now: datetime,
) -> list[CoverageLineRow]:
    rows: list[CoverageLineRow] = []
    for report in reports:
        all_lines = sorted(report.executed_lines | report.missing_lines)
        for line in all_lines:
            is_covered = line in report.executed_lines
            rows.append(
                CoverageLineRow(
                    repo=cfg.repo,
                    commit=cfg.commit,
                    rel_path=report.rel_path,
                    line=line,
                    is_executable=True,
                    is_covered=is_covered,
                    hits=1 if is_covered else 0,
                    context_count=0,
                    created_at=now,
                )
            )
    return rows


log = logging.getLogger(__name__)


def _collect_coverage_rows(
    _gateway: StorageGateway,
    cfg: CoverageIngestStepConfig,
    *,
    tools: ToolsConfig | None,
    tool_service: ToolService | None,
    json_output_path: Path | None,
    now: datetime,
) -> tuple[list[CoverageLineRow], str]:
    """
    Resolve coverage input into rows and source label.

    Parameters
    ----------
    _gateway :
        Gateway passed for API parity; not used directly.
    cfg :
        Coverage ingest configuration.
    tools :
        Optional tool configuration for resolving binaries.
    tool_service :
        Optional tool service used to execute coverage commands.
    json_output_path :
        Optional explicit path for coverage JSON output.
    now :
        Timestamp applied to emitted rows.

    Returns
    -------
    tuple[list[CoverageLineRow], str]
        Rows ready for ingestion and a source label ("cli" or "api").
    """
    repo_root = cfg.repo_root
    coverage_file = cfg.coverage_file

    if coverage_file is None or should_skip_missing_file(
        coverage_file,
        logger=log,
        label="coverage file",
    ):
        return [], "missing"

    active_tools = tools or ToolsConfig.model_validate({})
    service = tool_service
    if service is None:
        shared_runner = ToolRunner(
            tools_config=active_tools,
            cache_dir=cfg.repo_root / "build" / ".tool_cache",
        )
        service = ToolService(shared_runner, active_tools)

    json_path = json_output_path or (service.runner.cache_dir / "coverage.json")

    reports: list[CoverageFileReport] | None = None
    try:
        reports = asyncio.run(
            service.run_coverage_json(
                repo_root,
                coverage_file=coverage_file,
                output_path=json_path,
            )
        )
    except (ToolExecutionError, ToolNotFoundError) as exc:
        log.warning("coverage CLI failed; falling back to API parsing: %s", exc)

    if reports:
        return _rows_from_reports(cfg, reports, now), "cli"

    return _collect_via_api(repo_root, coverage_file, cfg, now), "api"


@dataclass
class CoverageIngestOps(IncrementalIngestOps[CoverageLineRow], SupportsFullRebuild):
    """
    Wrap coverage ingestion in the incremental harness with full rebuild semantics.

    Coverage is derived from test runs, so this always performs a full rebuild.
    """

    cfg: CoverageIngestStepConfig
    tools: ToolsConfig | None
    tool_service: ToolService | None
    json_output_path: Path | None
    now: datetime
    dataset_name: str = field(init=False, default="analytics.coverage_lines")

    @staticmethod
    def module_filter(module: ModuleRecord) -> bool:
        """
        Indicate that all modules are considered for coverage.

        Returns
        -------
        bool
            Always True to force full rebuild semantics.
        """
        del module
        return True

    def delete_rows(self, gateway: StorageGateway, rel_paths: Sequence[str]) -> None:
        """No-op: full rebuild handles deletion."""

    @staticmethod
    def process_module(module: ModuleRecord) -> Iterable[CoverageLineRow]:
        """
        Return rows for a single module.

        Returns
        -------
        Iterable[CoverageLineRow]
            Always empty because coverage runs as a full rebuild.
        """
        del module
        return []

    def insert_rows(self, gateway: StorageGateway, rows: Sequence[CoverageLineRow]) -> None:
        """No-op: run_full_rebuild performs insertion."""

    def run_full_rebuild(self, tracker: ChangeTracker) -> bool:
        """
        Perform a full rebuild of analytics.coverage_lines for the configured snapshot.

        Returns
        -------
        bool
            True once ingestion completes to short-circuit further processing.
        """
        rows, source = _collect_coverage_rows(
            tracker.gateway,
            self.cfg,
            tools=self.tools,
            tool_service=self.tool_service,
            json_output_path=self.json_output_path,
            now=self.now,
        )
        if not rows:
            log.info(
                "coverage_lines ingestion skipped (no rows) for %s@%s",
                self.cfg.repo,
                self.cfg.commit,
            )
            return True

        run_batch(
            tracker.gateway,
            "analytics.coverage_lines",
            [coverage_line_to_tuple(r) for r in rows],
            delete_params=[self.cfg.repo, self.cfg.commit],
        )
        log.info(
            "coverage_lines ingested for %s@%s rows=%d source=%s",
            self.cfg.repo,
            self.cfg.commit,
            len(rows),
            source,
        )
        return True


def ingest_coverage_lines(
    gateway: StorageGateway,
    cfg: CoverageIngestStepConfig,
    *,
    tools: ToolsConfig | None = None,
    tool_service: ToolService | None = None,
    tracker: ChangeTracker | None = None,
) -> None:
    """
    Read a `.coverage` database and populate `analytics.coverage_lines`.

    Behaviour
    ---------
    When ``tracker`` is provided, coverage ingestion runs as a full rebuild
    through the incremental harness for consistency. When ``tracker`` is None,
    the legacy full rebuild path is used directly.
    """
    now = datetime.now(UTC)
    output_path = cfg.paths.coverage_json

    if tracker is None:
        rows, source = _collect_coverage_rows(
            gateway,
            cfg,
            tools=tools,
            tool_service=tool_service,
            json_output_path=output_path,
            now=now,
        )
        if not rows:
            log.info(
                "coverage_lines ingestion skipped (no rows) for %s@%s",
                cfg.repo,
                cfg.commit,
            )
            return

        run_batch(
            gateway,
            "analytics.coverage_lines",
            [coverage_line_to_tuple(r) for r in rows],
            delete_params=[cfg.repo, cfg.commit],
        )
        log.info(
            "coverage_lines ingested for %s@%s rows=%d source=%s",
            cfg.repo,
            cfg.commit,
            len(rows),
            source,
        )
        return

    ops = CoverageIngestOps(
        cfg=cfg,
        tools=tools,
        tool_service=tool_service,
        json_output_path=output_path,
        now=now,
    )
    tracker_for_coverage = replace(
        tracker,
        change_request=replace(tracker.change_request, full_rebuild=True),
    )
    run_incremental_ingest(tracker_for_coverage, ops)


def _collect_via_api(
    repo_root: Path,
    coverage_file: Path,
    cfg: CoverageIngestStepConfig,
    now: datetime,
) -> list[CoverageLineRow]:
    cov = Coverage(data_file=str(coverage_file))
    cov.load()
    data = cov.get_data()

    insert_ctx = CoverageInsertContext(
        repo=cfg.repo, commit=cfg.commit, now=now, coverage_file=coverage_file
    )

    rows: list[CoverageLineRow] = []
    for measured in data.measured_files():
        measured_path = Path(measured).resolve()
        try:
            rel_path = normalize_rel_path(measured_path.relative_to(repo_root))
        except ValueError:
            continue
        file_info = CoverageFileInfo(measured_path=measured_path, rel_path=rel_path)
        rows.extend(_collect_file_coverage(cov=cov, data=data, file_info=file_info, ctx=insert_ctx))
    return rows


def _collect_file_coverage(
    *,
    cov: Coverage,
    data: CoverageData,
    file_info: CoverageFileInfo,
    ctx: CoverageInsertContext,
) -> list[CoverageLineRow]:
    rows: list[CoverageLineRow] = []
    try:
        _, statements, _, _missing, executed = cov.analysis2(str(file_info.measured_path))
    except CoverageException as exc:
        log.warning("coverage.analysis2 failed for %s: %s", file_info.measured_path, exc)
        return rows

    statements_set = set(statements)
    executed_set = set(executed)
    try:
        contexts_raw = data.contexts_by_lineno(str(file_info.measured_path)) or {}
        contexts_by_lineno: dict[int, set[str]] = {
            line: set(ctxs) for line, ctxs in contexts_raw.items()
        }
    except CoverageException:
        contexts_by_lineno = {}

    for line in sorted(statements_set):
        is_covered = line in executed_set
        hits = 1 if is_covered else 0
        contexts = contexts_by_lineno.get(line)
        context_count = len(contexts) if contexts else 0

        rows.append(
            CoverageLineRow(
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=file_info.rel_path,
                line=line,
                is_executable=True,
                is_covered=is_covered,
                hits=hits,
                context_count=context_count,
                created_at=ctx.now,
            )
        )
    return rows
