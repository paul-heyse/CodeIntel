"""Built-in ingestion plugins using port-adapter architecture.

This module provides plugin wrappers that delegate to pure step implementations
via the port-adapter pattern. Each plugin:
1. Creates adapters from the plugin context
2. Instantiates the appropriate step with injected ports
3. Executes the step and returns the result

NOTE: Imports inside functions are intentional to avoid circular dependencies.
"""

# ruff: noqa: PLC0415

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.ingestion.plugins.decorators import ingest_plugin
from codeintel.ingestion.plugins.harness import HarnessConfig
from codeintel.ingestion.plugins.protocol import (
    IngestPluginContext,
    IngestPluginResult,
    IngestResourceHints,
)

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.ingestion.adapters import (
        DuckDBStorageAdapter,
        FilesystemDiscoveryAdapter,
        HashChangeDetectionAdapter,
        ToolRunnerAdapter,
    )
    from codeintel.ingestion.change_tracker import ChangeTracker
    from codeintel.ingestion.common import ModuleRecord

log = logging.getLogger(__name__)


def _create_storage_adapter(ctx: IngestPluginContext) -> DuckDBStorageAdapter:
    """Create a DuckDB storage adapter from context.

    Parameters
    ----------
    ctx
        Plugin execution context.

    Returns
    -------
    DuckDBStorageAdapter
        Storage adapter for the context's gateway.
    """
    from codeintel.ingestion.adapters import DuckDBStorageAdapter

    return DuckDBStorageAdapter(ctx.gateway)


def _create_discovery_adapter(ctx: IngestPluginContext) -> FilesystemDiscoveryAdapter:
    """Create a filesystem discovery adapter from context.

    Parameters
    ----------
    ctx
        Plugin execution context.

    Returns
    -------
    FilesystemDiscoveryAdapter
        Discovery adapter for the context's repo root.
    """
    from codeintel.ingestion.adapters import FilesystemDiscoveryAdapter

    return FilesystemDiscoveryAdapter(ctx.snapshot.repo_root)


def _create_tool_adapter(ctx: IngestPluginContext) -> ToolRunnerAdapter:
    """Create a tool runner adapter from context.

    Parameters
    ----------
    ctx
        Plugin execution context.

    Returns
    -------
    ToolRunnerAdapter
        Tool adapter for the context's tool service.
    """
    from codeintel.ingestion.adapters import ToolRunnerAdapter

    service = ctx.tool_service_or_default()
    return ToolRunnerAdapter(service)


def _create_change_adapter(ctx: IngestPluginContext) -> HashChangeDetectionAdapter:
    """Create a change detection adapter from context.

    Parameters
    ----------
    ctx
        Plugin execution context.

    Returns
    -------
    HashChangeDetectionAdapter
        Change detection adapter using the storage adapter.
    """
    from codeintel.ingestion.adapters import HashChangeDetectionAdapter

    storage = _create_storage_adapter(ctx)
    return HashChangeDetectionAdapter(storage)


def _resolve_coverage_file(ctx: IngestPluginContext) -> Path | None:
    """Resolve the coverage file path from tools config or defaults.

    Parameters
    ----------
    ctx
        Plugin execution context.

    Returns
    -------
    Path | None
        Absolute path to the coverage file when it exists; None when missing.
    """
    candidate = ctx.tools.coverage_file or ctx.paths.coverage_json
    resolved = candidate.expanduser()
    if not resolved.is_absolute():
        resolved = (ctx.snapshot.repo_root / resolved).resolve()
    if not resolved.exists():
        log.warning(
            "Coverage file missing; skipping coverage_ingest repo=%s commit=%s path=%s",
            ctx.snapshot.repo,
            ctx.snapshot.commit,
            resolved,
        )
        return None
    return resolved


def _get_modules_from_tracker(ctx: IngestPluginContext) -> list[ModuleRecord]:
    """Retrieve modules from change tracker for pipeline execution.

    Parameters
    ----------
    ctx
        Plugin execution context.

    Returns
    -------
    list
        List of ModuleRecord instances.
    """
    from codeintel.ingestion.common import iter_modules
    from codeintel.storage.module_index import load_module_map

    module_map = load_module_map(
        ctx.gateway,
        ctx.snapshot.repo,
        ctx.snapshot.commit,
        language="python",
        logger=log,
    )

    return list(
        iter_modules(
            module_map,
            ctx.snapshot.repo_root,
            logger=log,
            scan_profile=ctx.code_profile,
        )
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
    harness=HarnessConfig(auto_row_counts=True),
    register=True,
)
def repo_scan_plugin(ctx: IngestPluginContext) -> IngestPluginResult:
    """Scan repository tree into core tables and change-tracker state.

    Uses port-adapter architecture with RepoScanStep for pure logic.

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

    tracker = repo_scan.ingest_repo(
        ctx.gateway,
        cfg=cfg,
        code_profile=ctx.code_profile,
    )

    # Store tracker in scratch for downstream plugins
    ctx.scratch.declare("change_tracker", tracker)

    return IngestPluginResult.ok()


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
    harness=HarnessConfig(auto_tracker=True, auto_tool_service=True, require_tracker=False),
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

    tracker = _try_get_tracker(ctx)
    service = ctx.tool_service_or_default()

    binaries = ToolBinaries(
        scip_python_bin=ctx.tools.scip_python_bin,
        scip_bin=ctx.tools.scip_bin,
    )
    cfg = ScipIngestStepConfig(
        snapshot=ctx.snapshot,
        paths=ctx.paths,
        binaries=binaries,
        scip_runner=None,
        artifact_writer=None,
    )

    result = scip_ingest.ingest_scip(
        gateway=ctx.gateway,
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
    harness=HarnessConfig(auto_tracker=True, auto_row_counts=True),
    register=True,
)
def cst_extract_plugin(ctx: IngestPluginContext) -> IngestPluginResult:
    """Parse CST and persist rows using CstExtractStep.

    Parameters
    ----------
    ctx
        Plugin execution context.

    Returns
    -------
    IngestPluginResult
        Result with row counts.
    """
    from codeintel.ingestion.steps import CstExtractStep

    # Create adapters
    storage = _create_storage_adapter(ctx)
    discovery = _create_discovery_adapter(ctx)

    # Get modules
    modules = _get_modules_from_tracker(ctx)

    # Execute step
    step = CstExtractStep(storage=storage, discovery=discovery)
    result = step.execute(
        modules,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
    )

    if result.errors:
        for error in result.errors:
            log.warning("CST extraction error: %s", error)

    return IngestPluginResult.ok(row_counts=result.table_counts)


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
    harness=HarnessConfig(auto_tracker=True, auto_row_counts=True),
    register=True,
)
def ast_extract_plugin(ctx: IngestPluginContext) -> IngestPluginResult:
    """Parse stdlib AST and persist rows/metrics using AstExtractStep.

    Parameters
    ----------
    ctx
        Plugin execution context.

    Returns
    -------
    IngestPluginResult
        Result with row counts.
    """
    from codeintel.ingestion.steps import AstExtractStep

    # Create adapters
    storage = _create_storage_adapter(ctx)
    discovery = _create_discovery_adapter(ctx)

    # Get modules
    modules = _get_modules_from_tracker(ctx)

    # Execute step
    step = AstExtractStep(storage=storage, discovery=discovery)
    result = step.execute(
        modules,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
    )

    if result.errors:
        for error in result.errors:
            log.warning("AST extraction error: %s", error)

    return IngestPluginResult.ok(row_counts=result.table_counts)


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
    harness=HarnessConfig(auto_tracker=True, auto_tool_service=True, auto_row_counts=True),
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
    from codeintel.ingestion import typing_ingest

    tracker = ctx.require_tracker()
    service = ctx.tool_service_or_default()
    modules = _get_modules_from_tracker(ctx)

    typing_ingest.ingest_typing_signals(
        gateway=ctx.gateway,
        modules=modules,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
        repo_root=ctx.snapshot.repo_root,
        code_profile=ctx.code_profile,
        tool_service=service,
        tracker=tracker,
    )

    return IngestPluginResult.ok()


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
    harness=HarnessConfig(
        auto_tracker=True, auto_tool_service=True, auto_row_counts=True, require_tracker=False
    ),
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
    from codeintel.ingestion import coverage_ingest

    coverage_path = _resolve_coverage_file(ctx)
    if coverage_path is None:
        return IngestPluginResult.skip("missing_coverage_file")

    tracker = _try_get_tracker(ctx)
    service = ctx.tool_service_or_default()
    modules = _get_modules_from_tracker(ctx)

    coverage_ingest.ingest_coverage_lines(
        ctx.gateway,
        modules,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
        repo_root=ctx.snapshot.repo_root,
        coverage_file=coverage_path,
        tool_service=service,
        json_output_path=ctx.paths.coverage_json,
        tracker=tracker,
    )

    return IngestPluginResult.ok()


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
    harness=HarnessConfig(
        auto_tracker=True, auto_tool_service=True, auto_row_counts=True, require_tracker=False
    ),
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
    from codeintel.ingestion import tests_ingest

    tracker = _try_get_tracker(ctx)
    service = ctx.tool_service_or_default()
    modules = _get_modules_from_tracker(ctx)

    tests_ingest.ingest_tests(
        ctx.gateway,
        modules,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
        repo_root=ctx.snapshot.repo_root,
        report_path=ctx.paths.pytest_report,
        tool_service=service,
        tracker=tracker,
    )

    return IngestPluginResult.ok()


@ingest_plugin(
    name="docstrings_ingest",
    description="Extract docstrings and persist structured rows into core.docstrings.",
    stage="enrich",
    produces_tables=("core.docstrings",),
    depends_on=("repo_scan",),
    requires=("change_tracker",),
    supports_incremental=True,
    resource_hints=IngestResourceHints(cpu_intensive=True, io_intensive=False),
    harness=HarnessConfig(auto_tracker=True, auto_row_counts=True),
    register=True,
)
def docstrings_ingest_plugin(ctx: IngestPluginContext) -> IngestPluginResult:
    """Extract and persist docstrings using DocstringsExtractStep.

    Parameters
    ----------
    ctx
        Plugin execution context.

    Returns
    -------
    IngestPluginResult
        Result with row counts.
    """
    from codeintel.ingestion.steps import DocstringsExtractStep

    # Create adapters
    storage = _create_storage_adapter(ctx)
    discovery = _create_discovery_adapter(ctx)

    # Get modules
    modules = _get_modules_from_tracker(ctx)

    # Execute step
    step = DocstringsExtractStep(storage=storage, discovery=discovery)
    result = step.execute(
        modules,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
    )

    if result.errors:
        for error in result.errors:
            log.warning("Docstring extraction error: %s", error)

    return IngestPluginResult.ok(row_counts=result.table_counts)


@ingest_plugin(
    name="config_ingest",
    description="Flatten config files into analytics.config_values.",
    stage="enrich",
    produces_tables=("analytics.config_values",),
    depends_on=("repo_scan",),
    requires=("change_tracker",),
    supports_incremental=True,
    resource_hints=IngestResourceHints(cpu_intensive=False, io_intensive=True),
    harness=HarnessConfig(auto_tracker=True, auto_row_counts=True, require_tracker=False),
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

    tracker = _try_get_tracker(ctx)

    cfg = ConfigIngestStepConfig(snapshot=ctx.snapshot)

    config_ingest.ingest_config_values(
        ctx.gateway,
        cfg=cfg,
        config_profile=ctx.config_profile,
        tracker=tracker,
    )

    return IngestPluginResult.ok()


def _try_get_tracker(ctx: IngestPluginContext) -> ChangeTracker | None:
    """Try to get change tracker from context or scratch.

    Use this for plugins that can operate with or without a tracker.
    For plugins that require a tracker, use ctx.require_tracker() instead.

    Parameters
    ----------
    ctx
        Plugin context.

    Returns
    -------
    ChangeTracker | None
        Change tracker if available, None otherwise.
    """
    if ctx.change_tracker is not None:
        return ctx.change_tracker

    tracker = ctx.scratch.consume("change_tracker")
    if tracker is not None:
        from codeintel.ingestion.change_tracker import ChangeTracker

        if isinstance(tracker, ChangeTracker):
            return tracker

    return None


def register_all_builtin_plugins() -> None:
    """Explicitly register all built-in plugins.

    This function is called automatically when the module is imported
    due to the register=True flag on each plugin decorator.
    """


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
