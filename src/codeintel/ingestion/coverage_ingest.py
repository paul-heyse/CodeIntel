"""Ingest coverage.py results into analytics.coverage_lines."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from coverage import Coverage, CoverageData
from coverage.exceptions import CoverageException

from codeintel.config import CoverageIngestStepConfig
from codeintel.config.datasets import CoverageLineRow, coverage_line_to_tuple
from codeintel.config.models import ToolsConfig
from codeintel.ingestion.common import ModuleRecord, run_batch, should_skip_missing_file
from codeintel.ingestion.paths import normalize_rel_path
from codeintel.ingestion.pipeline import (
    IngestPipeline,
    PipelineConfig,
    PipelineResult,
    SupportsFullRebuild,
)
from codeintel.ingestion.tool_runner import ToolExecutionError, ToolNotFoundError, ToolRunner
from codeintel.ingestion.tool_service import CoverageFileReport, ToolService

if TYPE_CHECKING:
    from codeintel.ingestion.change_tracker import ChangeTracker
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


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
    repo: str,
    commit: str,
    reports: list[CoverageFileReport],
    now: datetime,
) -> list[CoverageLineRow]:
    """
    Convert CoverageFileReport list to CoverageLineRow list.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit identifier.
    reports
        Coverage reports from tool service.
    now
        Timestamp for row creation.

    Returns
    -------
    list[CoverageLineRow]
        Rows ready for database insertion.
    """
    rows: list[CoverageLineRow] = []
    for report in reports:
        all_lines = sorted(report.executed_lines | report.missing_lines)
        for line in all_lines:
            is_covered = line in report.executed_lines
            rows.append(
                CoverageLineRow(
                    repo=repo,
                    commit=commit,
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


def _collect_via_api(
    repo_root: Path,
    coverage_file: Path,
    repo: str,
    commit: str,
    now: datetime,
) -> list[CoverageLineRow]:
    """
    Collect coverage rows using the coverage.py API.

    Parameters
    ----------
    repo_root
        Repository root for path resolution.
    coverage_file
        Path to .coverage database.
    repo
        Repository identifier.
    commit
        Commit identifier.
    now
        Timestamp for row creation.

    Returns
    -------
    list[CoverageLineRow]
        Rows collected from coverage API.
    """
    cov = Coverage(data_file=str(coverage_file))
    cov.load()
    data = cov.get_data()

    insert_ctx = CoverageInsertContext(
        repo=repo, commit=commit, now=now, coverage_file=coverage_file
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
    """
    Collect coverage rows for a single file.

    Parameters
    ----------
    cov
        Coverage instance.
    data
        Coverage data object.
    file_info
        File path information.
    ctx
        Insert context with identifiers.

    Returns
    -------
    list[CoverageLineRow]
        Rows for the file.
    """
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


class CoveragePipeline(SupportsFullRebuild):
    """Pipeline implementation for coverage ingestion.

    Coverage is derived from test runs, so this always performs a full rebuild.
    The module processing methods are no-ops; all work happens in run_full_rebuild.
    """

    def __init__(
        self,
        *,
        repo: str,
        commit: str,
        repo_root: Path,
        coverage_file: Path | None,
        tool_service: ToolService | None,
        json_output_path: Path | None,
    ) -> None:
        self._repo = repo
        self._commit = commit
        self._repo_root = repo_root
        self._coverage_file = coverage_file
        self._tool_service = tool_service
        self._json_output_path = json_output_path

    @property
    def dataset_name(self) -> str:
        """Return the dataset name for this pipeline."""
        return "analytics.coverage_lines"

    def module_filter(self, module: ModuleRecord) -> bool:
        """
        All modules are considered for coverage.

        Parameters
        ----------
        module
            Module metadata (unused).

        Returns
        -------
        bool
            Always True to force full rebuild semantics.
        """
        del module
        return True

    def process_module(self, module: ModuleRecord) -> Iterable[CoverageLineRow]:
        """
        Return rows for a single module.

        Coverage runs as a full rebuild, so this returns no rows.

        Parameters
        ----------
        module
            Module metadata (unused).

        Returns
        -------
        Iterable[CoverageLineRow]
            Always empty.
        """
        del module
        return []

    def persist_rows(self, gateway: StorageGateway, rows: Sequence[CoverageLineRow]) -> int:
        """
        No-op for normal flow; run_full_rebuild handles persistence.

        Parameters
        ----------
        gateway
            Storage gateway (unused).
        rows
            Rows to persist (unused).

        Returns
        -------
        int
            Always 0.
        """
        del gateway, rows
        return 0

    def delete_rows(self, gateway: StorageGateway, rel_paths: Sequence[str]) -> None:
        """
        No-op: full rebuild handles deletion.

        Parameters
        ----------
        gateway
            Storage gateway (unused).
        rel_paths
            Paths to delete (unused).
        """
        del gateway, rel_paths

    def run_full_rebuild(
        self,
        gateway: StorageGateway,
        modules: Sequence[ModuleRecord],
        config: PipelineConfig,
    ) -> PipelineResult:
        """
        Perform a full rebuild of analytics.coverage_lines.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        modules
            Available modules (unused for coverage).
        config
            Pipeline configuration (unused).

        Returns
        -------
        PipelineResult
            Result with row counts.
        """
        del modules, config
        now = datetime.now(UTC)

        rows, source = self._collect_rows(gateway, now)
        if not rows:
            log.info(
                "coverage_lines ingestion skipped (no rows) for %s@%s",
                self._repo,
                self._commit,
            )
            return PipelineResult(
                dataset_name=self.dataset_name,
                modules_processed=0,
                rows_persisted=0,
                used_full_rebuild=True,
            )

        run_batch(
            gateway,
            "analytics.coverage_lines",
            [coverage_line_to_tuple(r) for r in rows],
            delete_params=[self._repo, self._commit],
        )
        log.info(
            "coverage_lines ingested for %s@%s rows=%d source=%s",
            self._repo,
            self._commit,
            len(rows),
            source,
        )

        return PipelineResult(
            dataset_name=self.dataset_name,
            modules_processed=1,
            rows_persisted=len(rows),
            used_full_rebuild=True,
        )

    def _collect_rows(
        self,
        gateway: StorageGateway,
        now: datetime,
    ) -> tuple[list[CoverageLineRow], str]:
        """
        Resolve coverage input into rows and source label.

        Returns
        -------
        tuple[list[CoverageLineRow], str]
            Rows and source label ("cli", "api", or "missing").
        """
        del gateway
        coverage_file = self._coverage_file

        if coverage_file is None or should_skip_missing_file(
            coverage_file,
            logger=log,
            label="coverage file",
        ):
            return [], "missing"

        service = self._tool_service
        if service is None:
            tools_config = ToolsConfig.model_validate({})
            shared_runner = ToolRunner(
                tools_config=tools_config,
                cache_dir=self._repo_root / "build" / ".tool_cache",
            )
            service = ToolService(shared_runner, tools_config)

        json_path = self._json_output_path or (service.runner.cache_dir / "coverage.json")

        reports: list[CoverageFileReport] | None = None
        try:
            reports = asyncio.run(
                service.run_coverage_json(
                    self._repo_root,
                    coverage_file=coverage_file,
                    output_path=json_path,
                )
            )
        except (ToolExecutionError, ToolNotFoundError) as exc:
            log.warning("coverage CLI failed; falling back to API parsing: %s", exc)

        if reports:
            return _rows_from_reports(self._repo, self._commit, reports, now), "cli"

        return _collect_via_api(
            self._repo_root, coverage_file, self._repo, self._commit, now
        ), "api"


# Type assertion that CoveragePipeline implements IngestPipeline
_: type[IngestPipeline[CoverageLineRow]] = CoveragePipeline


def ingest_coverage_lines(
    gateway: StorageGateway,
    modules_or_cfg: Sequence[ModuleRecord] | object = (),
    *,
    cfg: object = None,
    repo: str | None = None,
    commit: str | None = None,
    repo_root: Path | None = None,
    coverage_file: Path | None = None,
    tool_service: ToolService | None = None,
    json_output_path: Path | None = None,
    tools: ToolsConfig | None = None,
    tracker: ChangeTracker | None = None,
    modules: Sequence[ModuleRecord] | None = None,
) -> PipelineResult:
    """
    Read a .coverage database and populate analytics.coverage_lines.

    Supports both new and legacy calling conventions for backward compatibility.
    Coverage always runs as a full rebuild since it's derived from test
    execution rather than per-file analysis.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    modules_or_cfg
        Either modules sequence (new API) or first positional arg.
    cfg
        Legacy CoverageIngestStepConfig parameter.
    repo
        Repository identifier (new API).
    commit
        Commit identifier (new API).
    repo_root
        Repository root path (new API).
    coverage_file
        Path to .coverage database file.
    tool_service
        Optional tool service for coverage CLI.
    json_output_path
        Optional path for coverage JSON output.
    tools
        Optional tools configuration (legacy API).
    tracker
        Optional change tracker (triggers full rebuild mode).
    modules
        Alternative modules parameter (new API).

    Returns
    -------
    PipelineResult
        Execution result with row counts.
    """
    from codeintel.ingestion.common import iter_modules
    from codeintel.storage.module_index import load_module_map

    # Handle legacy API: ingest_coverage_lines(gateway, cfg=cfg, ...)
    actual_cfg: CoverageIngestStepConfig | None = None
    if isinstance(cfg, CoverageIngestStepConfig):
        actual_cfg = cfg
    elif isinstance(modules_or_cfg, CoverageIngestStepConfig):
        actual_cfg = modules_or_cfg

    if actual_cfg is not None:
        module_map = load_module_map(
            gateway,
            actual_cfg.repo,
            actual_cfg.commit,
            language="python",
            logger=log,
        )
        actual_modules = list(
            iter_modules(
                module_map,
                actual_cfg.repo_root,
                logger=log,
                scan_profile=None,
            )
        )
        actual_repo = actual_cfg.repo
        actual_commit = actual_cfg.commit
        actual_repo_root = actual_cfg.repo_root
        actual_coverage_file = coverage_file if coverage_file else actual_cfg.coverage_file
        actual_json_path = json_output_path or actual_cfg.paths.coverage_json

        # Build tool service if not provided
        if tool_service is None and tools is not None:
            shared_runner = ToolRunner(
                tools_config=tools,
                cache_dir=actual_cfg.repo_root / "build" / ".tool_cache",
            )
            tool_service = ToolService(shared_runner, tools)
    else:
        # New API
        if modules is not None:
            actual_modules = list(modules)
        elif isinstance(modules_or_cfg, Sequence):
            actual_modules = list(modules_or_cfg)
        else:
            actual_modules = []
        actual_repo = repo
        actual_commit = commit
        actual_repo_root = repo_root
        actual_coverage_file = coverage_file
        actual_json_path = json_output_path

    if actual_repo is None or actual_commit is None or actual_repo_root is None:
        message = "repo, commit, and repo_root are required"
        raise ValueError(message)

    pipeline = CoveragePipeline(
        repo=actual_repo,
        commit=actual_commit,
        repo_root=actual_repo_root,
        coverage_file=actual_coverage_file,
        tool_service=tool_service,
        json_output_path=actual_json_path,
    )

    # Coverage always does full rebuild
    return pipeline.run_full_rebuild(gateway, actual_modules, PipelineConfig())


# Backward compatibility: keep old function signature
def ingest_coverage_lines_legacy(
    gateway: StorageGateway,
    cfg: CoverageIngestStepConfig,
    *,
    tools: ToolsConfig | None = None,
    tool_service: ToolService | None = None,
    tracker: ChangeTracker | None = None,
) -> None:
    """
    Legacy entry point for coverage ingestion.

    Deprecated: Use ingest_coverage_lines() with explicit parameters instead.
    """
    from codeintel.ingestion.common import iter_modules
    from codeintel.storage.module_index import load_module_map

    module_map = load_module_map(
        gateway,
        cfg.repo,
        cfg.commit,
        language="python",
        logger=log,
    )

    modules = list(
        iter_modules(
            module_map,
            cfg.repo_root,
            logger=log,
            scan_profile=None,
        )
    )

    # Build tool service if not provided
    service = tool_service
    if service is None and tools is not None:
        shared_runner = ToolRunner(
            tools_config=tools,
            cache_dir=cfg.repo_root / "build" / ".tool_cache",
        )
        service = ToolService(shared_runner, tools)

    ingest_coverage_lines(
        gateway,
        modules,
        repo=cfg.repo,
        commit=cfg.commit,
        repo_root=cfg.repo_root,
        coverage_file=cfg.coverage_file,
        tool_service=service,
        json_output_path=cfg.paths.coverage_json,
        tracker=tracker,
    )
