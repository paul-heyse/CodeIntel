"""Native Hamilton ingestion targets.

This module consolidates ingestion-domain targets that share similar execution
patterns (tool invocations + table writes) and are frequently evolved together:

- ``modules``: Scan repository modules and write ``core.repo_map``.
- ``config_ingest``: Discover and ingest configuration files.
- ``coverage_ingest``: Ingest coverage results into analytics tables.
- ``tests_ingest``: Ingest pytest report data into analytics tables.
- ``typing``: Run typing diagnostics and persist typedness + diagnostics.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from hamilton.function_modifiers import cache, tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.helpers import (
    build_scan_profile,
    filter_modules,
    get_module_paths_from_env,
    paths_to_modules,
)
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.options.ingestion import ModuleIngestOptions
from codeintel.build.hamilton.templates import executor_materialize
from codeintel.build.targets import TargetGraph
from codeintel.ingestion.adapters import (
    BuildToolAdapter,
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
)
from codeintel.ingestion.adapters.tool_runner import ToolRunnerAdapter
from codeintel.ingestion.compute import (
    ConfigIngestStep,
    CoverageIngestStep,
    TestsIngestStep,
    TypingIngestStep,
)
from codeintel.ingestion.compute.repo_scan import RepoScanStep
from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.ingestion.engine.service import ToolService
from codeintel.ingestion.infrastructure.scanning import default_config_profile
from codeintel.ingestion.ports.discovery import ModuleRecord
from codeintel.storage.warehouse import MaterializeOptions, Warehouse

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from codeintel.ingestion.ports.change_detection import ChangeSet

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ModuleRecord)


@dataclass(frozen=True)
class ModuleScanResult:
    """Result from repository module scanning.

    Attributes
    ----------
    success
        Whether the scan completed successfully.
    modules
        Discovered module records.
    change_set
        Computed change set for incremental processing.
    table_counts
        Row counts per produced table.
    error
        Error message if scan failed.
    """

    success: bool
    modules: Sequence[ModuleRecord] = field(default_factory=tuple)
    change_set: ChangeSet | None = None
    table_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class RepoMapWriteResult:
    """Result from writing repo_map entry.

    Attributes
    ----------
    success
        Whether the write completed successfully.
    row_count
        Number of rows written (typically 1 for repo_map).
    error
        Error message if write failed.
    """

    success: bool
    row_count: int = 0
    error: str | None = None


@cache(format="memory")
@tag(node_type="helper")
def module_paths(env: BuildEnv, t__modules: TargetRunRecord) -> tuple[str, ...]:
    """Load module paths for the current snapshot from storage.

    This node is shared across ingestion targets to avoid repeated queries to
    ``core.modules`` during a single run.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__modules
        Upstream modules target result (ensures module scan has run).

    Returns
    -------
    tuple[str, ...]
        Tuple of module paths for the current snapshot.
    """
    if t__modules.status != "succeeded":
        return ()
    return tuple(get_module_paths_from_env(env))


@cache(format="memory")
@tag(node_type="helper")
def module_records(env: BuildEnv, module_paths: tuple[str, ...]) -> tuple[ModuleRecord, ...]:
    """Convert module paths into ModuleRecord objects.

    Parameters
    ----------
    env
        Build environment with snapshot metadata.
    module_paths
        Module paths loaded from storage.

    Returns
    -------
    tuple[ModuleRecord, ...]
        Module records for downstream ingestion steps.
    """
    if not module_paths:
        return ()
    return tuple(paths_to_modules(module_paths, env.snapshot.repo_root))


@cache(format="memory")
@tag(domain="ingestion", target="modules", node_type="tool")
def t__modules__scan(env: BuildEnv) -> ModuleScanResult:
    """Execute repository scan to discover modules.

    This is the primary compute node for the modules target. It scans
    the repository tree, discovers Python modules, computes file hashes
    for change detection, and persists the modules table.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and configuration.

    Returns
    -------
    ModuleScanResult
        Result containing discovered modules and row counts.

    Notes
    -----
    The modules target is unique in that it has no upstream dependencies.
    It is the root of the ingestion domain dependency tree.
    """
    try:
        storage = DuckDBStorageAdapter(env.gateway)
        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
        change_detection = HashChangeDetectionAdapter(storage)

        # Build scan profile from config
        opts = ModuleIngestOptions()
        profile = build_scan_profile(env.snapshot.repo_root, opts)

        step = RepoScanStep(
            storage=storage,
            discovery=discovery,
            change_detection=change_detection,
            module_filter=lambda discovered: filter_modules(discovered, opts),
        )

        result, modules, change_set = step.execute(
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            repo_root=env.snapshot.repo_root,
            profile=profile,
            full_rebuild=False,
        )

        return ModuleScanResult(
            success=True,
            modules=tuple(modules),
            change_set=change_set,
            table_counts=result.table_counts or {},
        )

    except Exception as exc:
        log.exception("Module scan failed")
        return ModuleScanResult(
            success=False,
            error=str(exc),
        )


@tag(domain="ingestion", target="modules", node_type="tool")
def t__modules__write_repo_map(
    env: BuildEnv,
    t__modules__scan: ModuleScanResult,
) -> RepoMapWriteResult:
    """Write repo_map entry for the repository snapshot.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__modules__scan
        Upstream scan result containing discovered modules.

    Returns
    -------
    RepoMapWriteResult
        Result indicating success/failure of repo_map write.
    """
    if not t__modules__scan.success:
        return RepoMapWriteResult(
            success=False,
            error=f"Upstream scan failed: {t__modules__scan.error}",
        )

    try:
        modules = t__modules__scan.modules
        generated_at = datetime.now(tz=UTC).isoformat()

        # Build module entries JSON
        module_entries: dict[str, str] = {}
        for module in modules:
            name = getattr(module, "module_name", None) or getattr(module, "name", None)
            rel_path = getattr(module, "rel_path", None) or getattr(module, "path", None)
            if name is None:
                name = str(module)
            module_entries[str(name)] = str(rel_path) if rel_path is not None else ""

        modules_json = json.dumps(module_entries)
        overlays_json = json.dumps({})

        warehouse = Warehouse(env.gateway)
        warehouse.materialize_mappings(
            "core.repo_map",
            [
                {
                    "repo": env.snapshot.repo,
                    "commit": env.snapshot.commit,
                    "modules": modules_json,
                    "overlays": overlays_json,
                    "generated_at": generated_at,
                }
            ],
            options=MaterializeOptions(snapshot=env.snapshot, mode="replace"),
        )

        return RepoMapWriteResult(
            success=True,
            row_count=1,
        )

    except Exception as exc:
        log.exception("Repo map write failed")
        return RepoMapWriteResult(
            success=False,
            error=str(exc),
        )


@tag(domain="ingestion", target="modules", node_type="materialize")
def t__modules(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules__scan: ModuleScanResult,
    t__modules__write_repo_map: RepoMapWriteResult,
) -> TargetRunRecord:
    """Materialize modules target with validation.

    This is the entry point for the modules target. It orchestrates
    the scan and repo_map write, then returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    graph
        Target graph for metadata lookup.
    t__modules__scan
        Scan result from upstream compute node.
    t__modules__write_repo_map
        Repo map write result from upstream compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    executor = NativeTargetExecutor.for_target(env, graph, "modules")

    if executor.should_skip():
        return executor.skip()

    # Check for upstream failures
    if not t__modules__scan.success:
        return executor.fail(RuntimeError(t__modules__scan.error or "Module scan failed"))

    if not t__modules__write_repo_map.success:
        return executor.fail(
            RuntimeError(t__modules__write_repo_map.error or "Repo map write failed")
        )

    # Compute final row counts
    def compute() -> dict[str, int]:
        row_counts = dict(t__modules__scan.table_counts)
        row_counts["core.repo_map"] = t__modules__write_repo_map.row_count
        return row_counts

    return executor.execute(compute)


# ---------------------------------------------------------------------------
# config_ingest target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigScanResult:
    """Result from config file discovery.

    Attributes
    ----------
    success
        Whether discovery completed successfully.
    config_files
        List of discovered config files.
    error
        Error message if discovery failed.
    """

    success: bool
    config_files: list[ModuleRecord] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class ConfigIngestResult:
    """Result from config ingestion.

    Attributes
    ----------
    success
        Whether ingestion completed successfully.
    table_counts
        Row counts per produced table.
    errors
        List of parse errors (non-fatal).
    error
        Fatal error message if ingestion failed.
    """

    success: bool
    table_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    error: str | None = None


@tag(domain="ingestion", target="config_ingest", node_type="tool")
def t__config_ingest__scan(env: BuildEnv) -> ConfigScanResult:
    """Discover config files in repository.

    This compute node scans the repository for configuration files
    (YAML, JSON, TOML, INI) using the default config profile.

    Returns
    -------
    ConfigScanResult
        Discovery status and discovered config file records.
    """
    try:
        profile = default_config_profile(env.snapshot.repo_root)
        config_files = list(
            FilesystemDiscoveryAdapter.discover_modules(env.snapshot.repo_root, profile)
        )

        if not config_files:
            log.info("No config files found matching profile")

        return ConfigScanResult(success=True, config_files=config_files)
    except Exception:
        log.exception("Config scan failed")
        return ConfigScanResult(success=False, error="Config file discovery failed with exception")


@tag(domain="ingestion", target="config_ingest", node_type="tool")
def t__config_ingest__ingest(
    env: BuildEnv,
    t__config_ingest__scan: ConfigScanResult,
) -> ConfigIngestResult:
    """Ingest discovered config files into structured tables.

    Returns
    -------
    ConfigIngestResult
        Ingestion status and per-table row counts.
    """
    if not t__config_ingest__scan.success:
        return ConfigIngestResult(
            success=False,
            error=f"Config scan failed: {t__config_ingest__scan.error}",
        )

    config_files = t__config_ingest__scan.config_files
    if not config_files:
        return ConfigIngestResult(success=True, table_counts={})

    try:
        storage = DuckDBStorageAdapter(env.gateway)
        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)

        step = ConfigIngestStep(storage=storage, discovery=discovery)
        result = step.execute(
            config_files,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )

        if result.errors and result.rows_written == 0:
            errors = "; ".join(result.errors)
            return ConfigIngestResult(
                success=False,
                error=f"Config ingest failed: {errors}",
            )

        return ConfigIngestResult(
            success=True,
            table_counts=result.table_counts or {},
            errors=list(result.errors) if result.errors else [],
        )
    except Exception:
        log.exception("Config ingestion failed")
        return ConfigIngestResult(
            success=False,
            error="Config ingestion failed with exception",
        )


@tag(domain="ingestion", target="config_ingest", node_type="materialize")
def t__config_ingest(
    env: BuildEnv,
    graph: TargetGraph,
    t__config_ingest__ingest: ConfigIngestResult,
) -> TargetRunRecord:
    """Finalize config_ingest execution and persist manifest.

    Returns
    -------
    TargetRunRecord
        Record describing the execution outcome.
    """
    # Log parse warnings before materialization
    for error in t__config_ingest__ingest.errors:
        log.warning("Config parse warning: %s", error)

    return executor_materialize(env, graph, "config_ingest", t__config_ingest__ingest)


# ---------------------------------------------------------------------------
# coverage_ingest target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageIngestResult:
    """Result from coverage ingestion."""

    success: bool
    table_counts: dict[str, int] = field(default_factory=dict)
    skipped: bool = False
    error: str | None = None


def _resolve_coverage_file(env: BuildEnv) -> Path | None:
    repo_root = env.snapshot.repo_root
    build_dir = env.paths.build_dir

    candidates = [
        repo_root / ".coverage",
        repo_root / "coverage.json",
        build_dir / "coverage.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@tag(domain="ingestion", target="coverage_ingest", node_type="tool")
async def t__coverage_ingest__ingest(
    env: BuildEnv,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> CoverageIngestResult:
    """Execute coverage data ingestion from coverage.py output.

    Returns
    -------
    CoverageIngestResult
        Ingestion status and per-table row counts.
    """
    if t__modules.status != "succeeded":
        return CoverageIngestResult(
            success=False,
            error=f"Upstream modules target failed: {t__modules.error}",
        )

    coverage_path = _resolve_coverage_file(env)
    if coverage_path is None:
        log.info("No coverage file found, skipping coverage ingestion")
        return CoverageIngestResult(success=True, skipped=True, table_counts={})

    try:
        storage = DuckDBStorageAdapter(env.gateway)
        tool = BuildToolAdapter(
            coverage_collector=None,  # Coverage collector from resources if available
        )

        step = CoverageIngestStep(storage=storage, tools=tool)
        result = await step.execute_async(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            repo_root=env.snapshot.repo_root,
            coverage_file=coverage_path,
        )

        if not result.success:
            errors = "; ".join(result.errors) if result.errors else "Unknown error"
            return CoverageIngestResult(
                success=False,
                error=f"Coverage ingest failed: {errors}",
            )

        return CoverageIngestResult(
            success=True,
            table_counts=result.table_counts or {},
        )
    except Exception:
        log.exception("Coverage ingestion failed")
        return CoverageIngestResult(
            success=False,
            error="Coverage ingestion failed with exception",
        )


@tag(domain="ingestion", target="coverage_ingest", node_type="materialize")
def t__coverage_ingest(
    env: BuildEnv,
    graph: TargetGraph,
    t__coverage_ingest__ingest: CoverageIngestResult,
) -> TargetRunRecord:
    """Finalize coverage_ingest execution and persist manifest.

    Returns
    -------
    TargetRunRecord
        Record describing the execution outcome.
    """
    return executor_materialize(env, graph, "coverage_ingest", t__coverage_ingest__ingest)


# ---------------------------------------------------------------------------
# tests_ingest target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestsIngestResult:
    """Result from test results ingestion."""

    success: bool
    table_counts: dict[str, int] = field(default_factory=dict)
    skipped: bool = False
    error: str | None = None


def _resolve_report_file(env: BuildEnv) -> Path | None:
    build_dir = env.paths.build_dir
    repo_root = env.snapshot.repo_root

    candidates = [
        build_dir / "test-results" / "pytest-report.json",
        build_dir / "test-results" / "pytest_report.json",
        build_dir / "pytest-report.json",
        build_dir / "pytest_report.json",
        build_dir / "report.json",
        repo_root / "pytest-report.json",
        repo_root / "pytest_report.json",
        repo_root / "report.json",
        repo_root / "test-results" / "pytest-report.json",
        repo_root / ".pytest_cache" / "pytest_report.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@tag(domain="ingestion", target="tests_ingest", node_type="tool")
def t__tests_ingest__ingest(
    env: BuildEnv,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> TestsIngestResult:
    """Execute pytest report ingestion into analytics tables.

    Returns
    -------
    TestsIngestResult
        Ingestion status and per-table row counts.
    """
    if t__modules.status != "succeeded":
        return TestsIngestResult(
            success=False,
            error=f"Upstream modules target failed: {t__modules.error}",
        )

    report_path = _resolve_report_file(env)
    if report_path is None:
        log.info("No pytest report found, skipping tests ingestion")
        return TestsIngestResult(success=True, skipped=True, table_counts={})

    try:
        storage = DuckDBStorageAdapter(env.gateway)
        step = TestsIngestStep(storage=storage)
        result = step.execute(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            json_report_path=report_path,
        )

        if not result.success:
            errors = "; ".join(result.errors) if result.errors else "Unknown error"
            return TestsIngestResult(
                success=False,
                error=f"Tests ingest failed: {errors}",
            )

        return TestsIngestResult(
            success=True,
            table_counts=result.table_counts or {},
        )
    except Exception:
        log.exception("Tests ingestion failed")
        return TestsIngestResult(
            success=False,
            error="Tests ingestion failed with exception",
        )


@tag(domain="ingestion", target="tests_ingest", node_type="materialize")
def t__tests_ingest(
    env: BuildEnv,
    graph: TargetGraph,
    t__tests_ingest__ingest: TestsIngestResult,
) -> TargetRunRecord:
    """Finalize tests_ingest execution and persist manifest.

    Returns
    -------
    TargetRunRecord
        Record describing the execution outcome.
    """
    return executor_materialize(env, graph, "tests_ingest", t__tests_ingest__ingest)


# ---------------------------------------------------------------------------
# typing target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypingIngestResult:
    """Result from typing ingestion."""

    success: bool
    table_counts: dict[str, int] = field(default_factory=dict)
    skipped: bool = False
    error: str | None = None


@tag(domain="ingestion", target="typing", node_type="tool")
def t__typing__ingest(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> TypingIngestResult:
    """Execute typing analysis and persist typedness + diagnostics tables.

    Returns
    -------
    TypingIngestResult
        Ingestion status and per-table row counts.
    """
    if t__modules.status != "succeeded":
        return TypingIngestResult(
            success=False,
            error=f"Upstream modules target failed: {t__modules.error}",
        )

    executor = NativeTargetExecutor.for_target(env, graph, "typing")
    if executor.should_skip():
        return TypingIngestResult(success=True, skipped=True)

    if not module_records:
        return TypingIngestResult(success=True, table_counts={})

    storage = DuckDBStorageAdapter(env.gateway)
    discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)

    runner = ToolRunner(
        tools_config=env.providers.tool_runner.tools_config,
        cache_dir=env.paths.build_dir / ".tool_cache",
    )
    service = ToolService(runner)
    tools = ToolRunnerAdapter(service)

    step = TypingIngestStep(storage=storage, discovery=discovery, tools=tools)
    result = asyncio.run(
        step.execute_async(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            repo_root=str(env.snapshot.repo_root),
            run_diagnostics=True,
        )
    )

    if result.skipped:
        return TypingIngestResult(
            success=True,
            skipped=True,
            table_counts=dict(result.table_counts),
            error=result.skip_reason,
        )

    if not result.success:
        return TypingIngestResult(
            success=False,
            table_counts=dict(result.table_counts),
            error="; ".join(result.errors) if result.errors else "Typing ingestion failed",
        )

    return TypingIngestResult(success=True, table_counts=dict(result.table_counts))


@tag(domain="ingestion", target="typing", node_type="materialize")
def t__typing(
    env: BuildEnv,
    graph: TargetGraph,
    t__typing__ingest: TypingIngestResult,
) -> TargetRunRecord:
    """Finalize typing target execution and persist manifest.

    Returns
    -------
    TargetRunRecord
        Record describing the execution outcome.
    """
    return executor_materialize(env, graph, "typing", t__typing__ingest)
