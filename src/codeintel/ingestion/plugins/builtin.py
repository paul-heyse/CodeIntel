"""Built-in ingestion plugins wrapping existing step implementations.

This module provides plugin wrappers around the existing ingestion
step implementations for backward compatibility while transitioning
to the new plugin architecture.

NOTE: Imports inside functions are intentional to avoid circular dependencies.
"""
# ruff: noqa: PLC0415

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.ingestion.plugins.decorators import ingest_plugin
from codeintel.ingestion.plugins.protocol import (
    IngestPluginContext,
    IngestPluginResult,
    IngestResourceHints,
)

if TYPE_CHECKING:
    from codeintel.ingestion.change_tracker import ChangeTracker

log = logging.getLogger(__name__)


def _build_legacy_context(ctx: IngestPluginContext) -> object:
    """Build a legacy IngestionContext-compatible object from plugin context.

    This helper allows existing step implementations to work with
    the new plugin context.

    Parameters
    ----------
    ctx
        Plugin execution context.

    Returns
    -------
    object
        Legacy-compatible context object.
    """
    # Import here to avoid circular imports
    from codeintel.ingestion.runner import IngestionContext

    return IngestionContext(
        snapshot=ctx.snapshot,
        paths=ctx.paths,
        gateway=ctx.gateway,
        tools=ctx.tools,
        code_profile_cfg=ctx.code_profile,
        config_profile_cfg=ctx.config_profile,
        tool_runner=ctx.tool_runner,
        tool_service=ctx.tool_service,
        change_tracker=ctx.change_tracker,
        ingest_run_sink=ctx.ingest_run_sink,
        current_ingest_run=ctx.current_ingest_run,
    )


@ingest_plugin(
    name="repo_scan",
    description="Scan repository modules and build change-tracker state.",
    stage="scan",
    produces_tables=(
        "core.file_state",
        "core.modules",
        "core.repo_map",
        "analytics.tags_index",
    ),
    provides=("modules", "change_tracker"),
    supports_incremental=False,
    resource_hints=IngestResourceHints(cpu_intensive=False, io_intensive=True),
    register=True,
)
def repo_scan_plugin(ctx: IngestPluginContext) -> IngestPluginResult:
    """Scan repository tree into core tables and change-tracker state.

    Parameters
    ----------
    ctx
        Plugin execution context.

    Returns
    -------
    IngestPluginResult
        Result with change tracker stored in scratch.
    """
    from codeintel.config import RepoScanStepConfig
    from codeintel.ingestion import repo_scan

    cfg = RepoScanStepConfig(
        snapshot=ctx.snapshot,
        paths=ctx.paths,
        tool_runner=ctx.tool_runner,
    )

    try:
        tracker = repo_scan.ingest_repo(
            ctx.gateway,
            cfg=cfg,
            code_profile=ctx.code_profile,
        )

        # Store tracker in scratch for downstream plugins
        ctx.scratch.declare("change_tracker", tracker)

        row_counts = {
            "core.modules": _safe_count(ctx, "core.modules"),
            "core.repo_map": _safe_count(ctx, "core.repo_map"),
        }

        return IngestPluginResult.ok(row_counts=row_counts)

    except Exception as exc:
        log.exception("repo_scan failed")
        return IngestPluginResult.fail(str(exc), error_kind=type(exc).__name__)


@ingest_plugin(
    name="scip_ingest",
    description="Run scip-python and persist symbols and GOID crosswalk.",
    stage="index",
    produces_tables=(
        "index.scip",
        "core.scip_symbols",
        "core.goid_crosswalk",
    ),
    depends_on=("repo_scan",),
    requires=("change_tracker",),
    tool_dependencies=("scip",),
    supports_incremental=True,
    resource_hints=IngestResourceHints(
        cpu_intensive=True,
        io_intensive=True,
        max_runtime_ms=300000,
    ),
    register=True,
)
def scip_ingest_plugin(ctx: IngestPluginContext) -> IngestPluginResult:
    """Run scip-python indexing and register outputs.

    Parameters
    ----------
    ctx
        Plugin execution context.

    Returns
    -------
    IngestPluginResult
        Result with SCIP artifacts.
    """
    from codeintel.config import ScipIngestStepConfig, ToolBinaries
    from codeintel.ingestion import scip_ingest
    from codeintel.ingestion.tool_runner import ToolRunner
    from codeintel.ingestion.tool_service import ToolService

    tracker = _get_change_tracker(ctx)

    binaries = ToolBinaries(
        scip_python_bin=ctx.tools.scip_python_bin,
        scip_bin=ctx.tools.scip_bin,
        pyright_bin=ctx.tools.pyright_bin,
        pyrefly_bin=ctx.tools.pyrefly_bin,
        ruff_bin=ctx.tools.ruff_bin,
        coverage_bin=ctx.tools.coverage_bin,
        pytest_bin=ctx.tools.pytest_bin,
        git_bin=ctx.tools.git_bin,
        default_timeout_s=ctx.tools.default_timeout_s,
    )

    cfg = ScipIngestStepConfig(
        snapshot=ctx.snapshot,
        paths=ctx.paths,
        binaries=binaries,
    )

    runner = ctx.tool_runner or ToolRunner(
        cache_dir=ctx.paths.tool_cache,
        tools_config=ctx.tools,
    )
    service = ctx.tool_service or ToolService(runner, ctx.tools)

    try:
        result = scip_ingest.ingest_scip(
            ctx.gateway,
            cfg=cfg,
            tracker=tracker,
            tool_service=service,
        )

        if result.status == "unavailable":
            return IngestPluginResult.skip(result.reason or "SCIP tools unavailable")

        if result.status == "failed":
            return IngestPluginResult.fail(result.reason or "SCIP ingest failed")

        artifacts = {}
        if result.index_scip:
            artifacts["index_scip"] = result.index_scip
        if result.index_json:
            artifacts["index_json"] = result.index_json

        return IngestPluginResult.ok(artifacts=artifacts)

    except Exception as exc:
        log.exception("scip_ingest failed")
        return IngestPluginResult.fail(str(exc), error_kind=type(exc).__name__)


@ingest_plugin(
    name="cst_extract",
    description="Parse CST via LibCST and write rows into core.cst_nodes.",
    stage="parse",
    produces_tables=("core.cst_nodes",),
    depends_on=("repo_scan",),
    requires=("change_tracker",),
    supports_incremental=True,
    isolation_kind="process",
    resource_hints=IngestResourceHints(cpu_intensive=True, io_intensive=False),
    register=True,
)
def cst_extract_plugin(ctx: IngestPluginContext) -> IngestPluginResult:
    """Parse CST and persist rows.

    Parameters
    ----------
    ctx
        Plugin execution context.

    Returns
    -------
    IngestPluginResult
        Result with row counts.
    """
    import os

    from codeintel.ingestion import cst_extract

    tracker = _get_change_tracker(ctx)
    if tracker is None:
        return IngestPluginResult.fail(
            "No change tracker available; run repo_scan first",
            error_kind="MissingDependency",
        )

    executor_kind = os.getenv("CODEINTEL_CST_EXECUTOR", "process")

    try:
        cst_extract.ingest_cst(tracker, executor_kind=executor_kind)

        row_counts = {"core.cst_nodes": _safe_count(ctx, "core.cst_nodes")}
        return IngestPluginResult.ok(row_counts=row_counts)

    except Exception as exc:
        log.exception("cst_extract failed")
        return IngestPluginResult.fail(str(exc), error_kind=type(exc).__name__)


@ingest_plugin(
    name="ast_extract",
    description="Parse Python AST and persist rows + metrics into core.ast_* tables.",
    stage="parse",
    produces_tables=("core.ast_nodes", "core.ast_metrics"),
    depends_on=("repo_scan",),
    requires=("change_tracker",),
    supports_incremental=True,
    isolation_kind="process",
    resource_hints=IngestResourceHints(cpu_intensive=True, io_intensive=False),
    register=True,
)
def ast_extract_plugin(ctx: IngestPluginContext) -> IngestPluginResult:
    """Parse stdlib AST and persist rows/metrics.

    Parameters
    ----------
    ctx
        Plugin execution context.

    Returns
    -------
    IngestPluginResult
        Result with row counts.
    """
    from codeintel.ingestion import py_ast_extract

    tracker = _get_change_tracker(ctx)
    if tracker is None:
        return IngestPluginResult.fail(
            "No change tracker available; run repo_scan first",
            error_kind="MissingDependency",
        )

    try:
        py_ast_extract.ingest_python_ast(tracker)

        row_counts = {
            "core.ast_nodes": _safe_count(ctx, "core.ast_nodes"),
            "core.ast_metrics": _safe_count(ctx, "core.ast_metrics"),
        }
        return IngestPluginResult.ok(row_counts=row_counts)

    except Exception as exc:
        log.exception("ast_extract failed")
        return IngestPluginResult.fail(str(exc), error_kind=type(exc).__name__)


@ingest_plugin(
    name="typing_ingest",
    description="Populate analytics.typedness and analytics.static_diagnostics.",
    stage="enrich",
    produces_tables=("analytics.typedness", "analytics.static_diagnostics"),
    depends_on=("repo_scan",),
    requires=("change_tracker",),
    tool_dependencies=("pyright", "pyrefly", "ruff"),
    supports_incremental=True,
    resource_hints=IngestResourceHints(
        cpu_intensive=False,
        io_intensive=True,
        max_runtime_ms=180000,
    ),
    register=True,
)
def typing_ingest_plugin(ctx: IngestPluginContext) -> IngestPluginResult:
    """Compute typedness and static diagnostics.

    Parameters
    ----------
    ctx
        Plugin execution context.

    Returns
    -------
    IngestPluginResult
        Result with row counts.
    """
    from codeintel.config.builder import TypingIngestStepConfig
    from codeintel.ingestion import typing_ingest
    from codeintel.ingestion.tool_runner import ToolRunner
    from codeintel.ingestion.tool_service import ToolService

    tracker = _get_change_tracker(ctx)

    cfg = TypingIngestStepConfig(
        snapshot=ctx.snapshot,
        paths=ctx.paths,
        tool_runner=ctx.tool_runner,
    )

    runner = ctx.tool_runner or ToolRunner(
        cache_dir=ctx.paths.tool_cache,
        tools_config=ctx.tools,
    )
    service = ctx.tool_service or ToolService(runner, ctx.tools)

    try:
        typing_ingest.ingest_typing_signals(
            gateway=ctx.gateway,
            cfg=cfg,
            code_profile=ctx.code_profile,
            tool_service=service,
            tracker=tracker,
        )

        row_counts = {
            "analytics.typedness": _safe_count(ctx, "analytics.typedness"),
            "analytics.static_diagnostics": _safe_count(ctx, "analytics.static_diagnostics"),
        }
        return IngestPluginResult.ok(row_counts=row_counts)

    except Exception as exc:
        log.exception("typing_ingest failed")
        return IngestPluginResult.fail(str(exc), error_kind=type(exc).__name__)


@ingest_plugin(
    name="coverage_ingest",
    description="Load coverage.py data into analytics.coverage_lines.",
    stage="enrich",
    produces_tables=("analytics.coverage_lines",),
    depends_on=("repo_scan",),
    requires=("change_tracker",),
    tool_dependencies=("coverage",),
    supports_incremental=True,
    resource_hints=IngestResourceHints(cpu_intensive=False, io_intensive=True),
    register=True,
)
def coverage_ingest_plugin(ctx: IngestPluginContext) -> IngestPluginResult:
    """Load coverage lines from coverage.json or coverage.py data.

    Parameters
    ----------
    ctx
        Plugin execution context.

    Returns
    -------
    IngestPluginResult
        Result with row counts.
    """
    from codeintel.config.builder import CoverageIngestStepConfig
    from codeintel.ingestion import coverage_ingest
    from codeintel.ingestion.steps import resolve_coverage_file
    from codeintel.ingestion.tool_runner import ToolRunner
    from codeintel.ingestion.tool_service import ToolService

    tracker = _get_change_tracker(ctx)

    # Build a minimal context for resolve_coverage_file
    legacy_ctx = _build_legacy_context(ctx)
    coverage_path = resolve_coverage_file(legacy_ctx)  # type: ignore[arg-type]

    cfg = CoverageIngestStepConfig(
        snapshot=ctx.snapshot,
        paths=ctx.paths,
        coverage_file=coverage_path,
        tool_runner=ctx.tool_runner,
    )

    runner = ctx.tool_runner or ToolRunner(
        cache_dir=ctx.paths.tool_cache,
        tools_config=ctx.tools,
    )
    service = ctx.tool_service or ToolService(runner, ctx.tools)

    try:
        coverage_ingest.ingest_coverage_lines(
            gateway=ctx.gateway,
            cfg=cfg,
            tools=ctx.tools,
            tool_service=service,
            tracker=tracker,
        )

        row_counts = {
            "analytics.coverage_lines": _safe_count(ctx, "analytics.coverage_lines"),
        }
        return IngestPluginResult.ok(row_counts=row_counts)

    except Exception as exc:
        log.exception("coverage_ingest failed")
        return IngestPluginResult.fail(str(exc), error_kind=type(exc).__name__)


@ingest_plugin(
    name="tests_ingest",
    description="Ingest pytest JSON report into analytics.test_catalog.",
    stage="enrich",
    produces_tables=("analytics.test_catalog",),
    depends_on=("repo_scan",),
    requires=("change_tracker",),
    tool_dependencies=("pytest",),
    supports_incremental=True,
    resource_hints=IngestResourceHints(cpu_intensive=False, io_intensive=True),
    register=True,
)
def tests_ingest_plugin(ctx: IngestPluginContext) -> IngestPluginResult:
    """Load pytest results into analytics.test_catalog.

    Parameters
    ----------
    ctx
        Plugin execution context.

    Returns
    -------
    IngestPluginResult
        Result with row counts.
    """
    from codeintel.config.builder import TestsIngestStepConfig
    from codeintel.ingestion import tests_ingest
    from codeintel.ingestion.tool_runner import ToolRunner
    from codeintel.ingestion.tool_service import ToolService

    tracker = _get_change_tracker(ctx)

    cfg = TestsIngestStepConfig(
        snapshot=ctx.snapshot,
        paths=ctx.paths,
        pytest_report_path=ctx.paths.pytest_report,
    )

    runner = ctx.tool_runner or ToolRunner(
        cache_dir=ctx.paths.tool_cache,
        tools_config=ctx.tools,
    )
    service = ctx.tool_service or ToolService(runner, ctx.tools)

    try:
        tests_ingest.ingest_tests(
            gateway=ctx.gateway,
            cfg=cfg,
            report_path=ctx.paths.pytest_report,
            tool_service=service,
            tracker=tracker,
        )

        row_counts = {
            "analytics.test_catalog": _safe_count(ctx, "analytics.test_catalog"),
        }
        return IngestPluginResult.ok(row_counts=row_counts)

    except Exception as exc:
        log.exception("tests_ingest failed")
        return IngestPluginResult.fail(str(exc), error_kind=type(exc).__name__)


@ingest_plugin(
    name="docstrings_ingest",
    description="Extract docstrings and persist structured rows into core.docstrings.",
    stage="enrich",
    produces_tables=("core.docstrings",),
    depends_on=("repo_scan",),
    requires=("change_tracker",),
    supports_incremental=True,
    resource_hints=IngestResourceHints(cpu_intensive=True, io_intensive=False),
    register=True,
)
def docstrings_ingest_plugin(ctx: IngestPluginContext) -> IngestPluginResult:
    """Extract and persist docstrings.

    Parameters
    ----------
    ctx
        Plugin execution context.

    Returns
    -------
    IngestPluginResult
        Result with row counts.
    """
    from codeintel.config import DocstringStepConfig
    from codeintel.ingestion import docstrings_ingest

    tracker = _get_change_tracker(ctx)

    cfg = DocstringStepConfig(snapshot=ctx.snapshot)

    try:
        docstrings_ingest.ingest_docstrings(
            ctx.gateway,
            cfg,
            code_profile=ctx.code_profile,
            tracker=tracker,
        )

        row_counts = {
            "core.docstrings": _safe_count(ctx, "core.docstrings"),
        }
        return IngestPluginResult.ok(row_counts=row_counts)

    except Exception as exc:
        log.exception("docstrings_ingest failed")
        return IngestPluginResult.fail(str(exc), error_kind=type(exc).__name__)


@ingest_plugin(
    name="config_ingest",
    description="Flatten config files into analytics.config_values.",
    stage="enrich",
    produces_tables=("analytics.config_values",),
    depends_on=("repo_scan",),
    requires=("change_tracker",),
    supports_incremental=True,
    resource_hints=IngestResourceHints(cpu_intensive=False, io_intensive=True),
    register=True,
)
def config_ingest_plugin(ctx: IngestPluginContext) -> IngestPluginResult:
    """Flatten configuration files into analytics.config_values.

    Parameters
    ----------
    ctx
        Plugin execution context.

    Returns
    -------
    IngestPluginResult
        Result with row counts.
    """
    from codeintel.config.builder import ConfigIngestStepConfig
    from codeintel.ingestion import config_ingest

    tracker = _get_change_tracker(ctx)

    cfg = ConfigIngestStepConfig(snapshot=ctx.snapshot)

    try:
        config_ingest.ingest_config_values(
            ctx.gateway,
            cfg=cfg,
            config_profile=ctx.config_profile,
            tracker=tracker,
        )

        row_counts = {
            "analytics.config_values": _safe_count(ctx, "analytics.config_values"),
        }
        return IngestPluginResult.ok(row_counts=row_counts)

    except Exception as exc:
        log.exception("config_ingest failed")
        return IngestPluginResult.fail(str(exc), error_kind=type(exc).__name__)


def _get_change_tracker(ctx: IngestPluginContext) -> ChangeTracker | None:
    """Get change tracker from context or scratch.

    Parameters
    ----------
    ctx
        Plugin context.

    Returns
    -------
    ChangeTracker | None
        Change tracker if available.
    """
    if ctx.change_tracker is not None:
        return ctx.change_tracker

    # Try to get from scratch (populated by repo_scan)
    tracker = ctx.scratch.consume("change_tracker")
    if tracker is not None:
        from codeintel.ingestion.change_tracker import ChangeTracker

        if isinstance(tracker, ChangeTracker):
            return tracker

    return None


def _safe_count(ctx: IngestPluginContext, table_key: str) -> int:
    """Safely count rows in a table.

    Parameters
    ----------
    ctx
        Plugin context.
    table_key
        Table to count.

    Returns
    -------
    int
        Row count or 0 on error.
    """
    try:
        result = ctx.gateway.con.execute(
            f"SELECT COUNT(*) FROM {table_key}",  # noqa: S608
        ).fetchone()
        return int(result[0]) if result else 0
    except Exception:  # noqa: BLE001
        return 0


def register_all_builtin_plugins() -> None:
    """Explicitly register all built-in plugins.

    This function is called automatically when the module is imported
    due to the register=True flag on each plugin decorator. It can
    also be called manually to ensure registration.
    """
    # Plugins are already registered via decorators with register=True


__all__ = [
    "ast_extract_plugin",
    "config_ingest_plugin",
    "coverage_ingest_plugin",
    "cst_extract_plugin",
    "docstrings_ingest_plugin",
    "register_all_builtin_plugins",
    "repo_scan_plugin",
    "scip_ingest_plugin",
    "tests_ingest_plugin",
    "typing_ingest_plugin",
]
