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
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from hamilton.function_modifiers import source, value

from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.helpers import (
    build_scan_profile,
    filter_modules,
    get_module_paths_from_env,
    paths_to_modules,
)
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materializations,
)
from codeintel.build.hamilton.native.options.ingestion import ModuleIngestOptions
from codeintel.build.hamilton.native.target_override_tables import (
    CONFIG_INGEST_OVERRIDE_TABLES,
    COVERAGE_INGEST_OVERRIDE_TABLES,
    MODULES_OVERRIDE_TABLES,
    TESTS_INGEST_OVERRIDE_TABLES,
    TYPING_OVERRIDE_TABLES,
)
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
    register_output_targets,
)
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.hamilton.run_records import (
    TargetRunRecord,
    options_hash_for_target,
    should_skip_native_target,
)
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_helper, tag_materialize, tag_tool
from codeintel.build.hashing import InputHashOptions, compute_input_hash
from codeintel.build.resources import TOOL_EXECUTION, TargetResources
from codeintel.build.schemas import deferred_columns_for_table_key
from codeintel.build.targets import TargetGraph
from codeintel.core.paths import normalize_path
from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
)
from codeintel.ingestion.adapters.tool_runner import ToolRunnerAdapter
from codeintel.ingestion.compute.config_ingest import ConfigIngestResult, ConfigIngestStep
from codeintel.ingestion.compute.coverage_ingest import CoverageIngestResult, CoverageIngestStep
from codeintel.ingestion.compute.repo_scan import RepoScanResult, RepoScanStep
from codeintel.ingestion.compute.tests_ingest import TestsIngestResult, TestsIngestStep
from codeintel.ingestion.compute.typing_ingest import TypingIngestResult, TypingIngestStep
from codeintel.ingestion.infrastructure.scanning import default_config_profile
from codeintel.ingestion.ports.discovery import ModuleRecord

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from codeintel.ingestion.ports.change_detection import ChangeSet, FileDigest

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (
    BuildEnv,
    MaterializationMetadata,
    TargetGraph,
    TargetRunRecord,
    ModuleRecord,
)

MODULES_TARGET_NAME = "modules"
CONFIG_INGEST_TARGET_NAME = "config_ingest"
COVERAGE_INGEST_TARGET_NAME = "coverage_ingest"
TESTS_INGEST_TARGET_NAME = "tests_ingest"
TYPING_TARGET_NAME = "typing"

MODULES_TABLE_KEY = "core.modules"
FILE_STATE_TABLE_KEY = "core.file_state"
REPO_MAP_TABLE_KEY = "core.repo_map"
MODULES_TABLE_KEYS = (MODULES_TABLE_KEY, FILE_STATE_TABLE_KEY, REPO_MAP_TABLE_KEY)

CONFIG_VALUES_TABLE_KEY = "analytics.config_values"
COVERAGE_LINES_TABLE_KEY = "analytics.coverage_lines"
TEST_CATALOG_TABLE_KEY = "analytics.test_catalog"
TYPEDNESS_TABLE_KEY = "analytics.typedness"
STATIC_DIAGNOSTICS_TABLE_KEY = "analytics.static_diagnostics"
TYPING_TABLE_KEYS = (TYPEDNESS_TABLE_KEY, STATIC_DIAGNOSTICS_TABLE_KEY)

register_output_targets(
    make_output_target(
        name=MODULES_TARGET_NAME,
        module="ingestion",
        description="Repository module and file index from scanning.",
        options=TargetSpecOptions(
            table_keys=MODULES_TABLE_KEYS,
            override_tables=MODULES_OVERRIDE_TABLES,
        ),
    ),
    make_output_target(
        name=CONFIG_INGEST_TARGET_NAME,
        module="ingestion",
        description="Configuration file parsing and reference tracking.",
        options=TargetSpecOptions(
            table_keys=(CONFIG_VALUES_TABLE_KEY,),
            override_tables=CONFIG_INGEST_OVERRIDE_TABLES,
        ),
    ),
    make_output_target(
        name=COVERAGE_INGEST_TARGET_NAME,
        module="ingestion",
        description="Line-level test coverage ingestion.",
        options=TargetSpecOptions(
            table_keys=(COVERAGE_LINES_TABLE_KEY,),
            override_tables=COVERAGE_INGEST_OVERRIDE_TABLES,
        ),
    ),
    make_output_target(
        name=TESTS_INGEST_TARGET_NAME,
        module="ingestion",
        description="Test catalog ingestion from pytest.",
        options=TargetSpecOptions(
            table_keys=(TEST_CATALOG_TABLE_KEY,),
            override_tables=TESTS_INGEST_OVERRIDE_TABLES,
        ),
    ),
    make_output_target(
        name=TYPING_TARGET_NAME,
        module="ingestion",
        description="Type annotation analysis and static diagnostics.",
        options=TargetSpecOptions(
            table_keys=TYPING_TABLE_KEYS,
            override_tables=TYPING_OVERRIDE_TABLES,
            resources=TargetResources(
                tracker=True,
                modules=True,
                tools=(
                    "pyright",
                    "pyrefly",
                    "ruff",
                ),
            ),
            execution=TOOL_EXECUTION,
        ),
    ),
)


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
    file_state_hash
        Stable hash of the current file state.
    module_rows
        Row tuples for core.modules.
    file_state_rows
        Row tuples for core.file_state.
    repo_map_rows
        Row tuples for core.repo_map.
    error
        Error message if scan failed.
    """

    success: bool
    modules: tuple[ModuleRecord, ...] = field(default_factory=tuple)
    change_set: ChangeSet | None = None
    file_state_hash: str | None = None
    module_rows: tuple[tuple[object, ...], ...] = field(default_factory=tuple)
    file_state_rows: tuple[tuple[object, ...], ...] = field(default_factory=tuple)
    repo_map_rows: tuple[tuple[object, ...], ...] = field(default_factory=tuple)
    error: str | None = None


@dataclass(frozen=True)
class ConfigScanResult:
    """Result from config file discovery.

    Attributes
    ----------
    success
        Whether discovery completed successfully.
    config_files
        List of discovered config files.
    file_state_hash
        Stable hash of the config file state for input hashing.
    error
        Error message if discovery failed.
    """

    success: bool
    config_files: list[ModuleRecord] = field(default_factory=list)
    file_state_hash: str | None = None
    error: str | None = None


@tag_helper()
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


@tag_helper()
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


@tag_tool(domain="ingestion", target=MODULES_TARGET_NAME)
def t__modules__scan(env: BuildEnv) -> ModuleScanResult:
    """Execute repository scan to discover modules.

    This is the primary compute node for the modules target. It scans
    the repository tree, discovers Python modules, and computes file hashes
    for change detection.

    Returns
    -------
    ModuleScanResult
        Result containing discovered modules and row tuples.
    """
    try:
        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
        storage = DuckDBStorageAdapter(env.gateway)
        change_detection = HashChangeDetectionAdapter(storage)

        opts = load_target_options(
            env,
            target_name=MODULES_TARGET_NAME,
            options_type=ModuleIngestOptions,
        )
        profile = build_scan_profile(env.snapshot.repo_root, opts)

        step = RepoScanStep(
            discovery=discovery,
            change_detection=change_detection,
            module_filter=lambda discovered: filter_modules(discovered, opts),
        )

        scan_result = step.execute(
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            repo_root=env.snapshot.repo_root,
            profile=profile,
            full_rebuild=False,
        )

        return ModuleScanResult(
            success=True,
            modules=scan_result.modules,
            change_set=scan_result.change_set,
            file_state_hash=scan_result.change_set.state_hash,
            module_rows=scan_result.module_rows,
            file_state_rows=scan_result.file_state_rows,
            repo_map_rows=scan_result.repo_map_rows,
        )

    except Exception as exc:
        log.exception("Module scan failed")
        return ModuleScanResult(success=False, error=str(exc))


@tag_helper(domain="ingestion", target=MODULES_TARGET_NAME)
def modules__hash_options(env: BuildEnv, t__modules__scan: ModuleScanResult) -> InputHashOptions:
    """Build input hash options for modules target materialization.

    Returns
    -------
    InputHashOptions
        Hash inputs used to gate target materialization.
    """
    options_hash = options_hash_for_target(env, MODULES_TARGET_NAME)
    return InputHashOptions(
        options_hash=options_hash,
        manifests=env.manifest_index,
        file_state_hash=t__modules__scan.file_state_hash,
    )


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(MODULES_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(MODULES_TARGET_NAME),
    table_key=value(MODULES_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(MODULES_TABLE_KEY)),
    hash_options=source("modules__hash_options"),
)
@tag_compute(domain="ingestion", target=MODULES_TARGET_NAME, target_="modules__module_rows")
def modules__module_rows(
    t__modules__scan: ModuleScanResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for core.modules.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the modules table, or None when scanning failed.
    """
    if not t__modules__scan.success:
        return None
    return t__modules__scan.module_rows


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(FILE_STATE_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(MODULES_TARGET_NAME),
    table_key=value(FILE_STATE_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(FILE_STATE_TABLE_KEY)),
    hash_options=source("modules__hash_options"),
)
@tag_compute(domain="ingestion", target=MODULES_TARGET_NAME, target_="modules__file_state_rows")
def modules__file_state_rows(
    t__modules__scan: ModuleScanResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for core.file_state.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the file_state table, or None when scanning failed.
    """
    if not t__modules__scan.success:
        return None
    return t__modules__scan.file_state_rows


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(REPO_MAP_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(MODULES_TARGET_NAME),
    table_key=value(REPO_MAP_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(REPO_MAP_TABLE_KEY)),
    hash_options=source("modules__hash_options"),
)
@tag_compute(domain="ingestion", target=MODULES_TARGET_NAME, target_="modules__repo_map_rows")
def modules__repo_map_rows(
    t__modules__scan: ModuleScanResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for core.repo_map.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the repo_map table, or None when scanning failed.
    """
    if not t__modules__scan.success:
        return None
    return t__modules__scan.repo_map_rows


@tag_helper(domain="ingestion", target=MODULES_TARGET_NAME)
def modules__materializations(
    m__core__modules: MaterializationMetadata,
    m__core__file_state: MaterializationMetadata,
    m__core__repo_map: MaterializationMetadata,
) -> dict[str, MaterializationMetadata]:
    """Collect materialization metadata for modules target tables.

    Returns
    -------
    dict[str, MaterializationMetadata]
        Mapping from table key to saver metadata.
    """
    return {
        MODULES_TABLE_KEY: m__core__modules,
        FILE_STATE_TABLE_KEY: m__core__file_state,
        REPO_MAP_TABLE_KEY: m__core__repo_map,
    }


@tag_materialize(domain="ingestion", target=MODULES_TARGET_NAME)
def t__modules(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules__scan: ModuleScanResult,
    modules__hash_options: InputHashOptions,
    modules__materializations: dict[str, MaterializationMetadata],
) -> TargetRunRecord:
    """Materialize modules target with validation.

    Returns
    -------
    TargetRunRecord
        Record describing the execution outcome.
    """
    if not t__modules__scan.success:
        executor = NativeTargetExecutor.for_target(
            env,
            graph,
            MODULES_TARGET_NAME,
            hash_options=modules__hash_options,
        )
        return executor.fail(RuntimeError(t__modules__scan.error or "Module scan failed"))

    change_delta: dict[str, object] | None = None
    change_set = t__modules__scan.change_set
    if change_set is not None:
        change_delta = cast(
            "dict[str, object]",
            {
                "state_hash": change_set.state_hash,
                "added": [module.rel_path for module in change_set.added],
                "modified": [module.rel_path for module in change_set.modified],
                "deleted": [module.rel_path for module in change_set.deleted],
                "counts": {
                    "added": len(change_set.added),
                    "modified": len(change_set.modified),
                    "deleted": len(change_set.deleted),
                },
            },
        )

    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name=MODULES_TARGET_NAME,
        materializations=modules__materializations,
        change_delta=change_delta,
    )


# ---------------------------------------------------------------------------
# config_ingest target
# ---------------------------------------------------------------------------


def _compute_file_state_hash(config_files: Sequence[ModuleRecord]) -> str:
    state: dict[str, FileDigest] = {}
    for record in config_files:
        digest = HashChangeDetectionAdapter.compute_file_digest(record.file_path)
        if digest is None:
            continue
        state[normalize_path(record.rel_path)] = digest
    return HashChangeDetectionAdapter.compute_state_hash(state)


def _should_skip_target(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    *,
    file_state_hash: str | None = None,
) -> bool:
    target = graph.get(target_name)
    if target is None:
        return False
    options_hash = options_hash_for_target(env, target_name)
    hash_options = InputHashOptions(
        options_hash=options_hash,
        manifests=env.manifest_index,
        file_state_hash=file_state_hash,
    )
    input_hash = compute_input_hash(
        target=target,
        snapshot=env.snapshot,
        gateway=env.gateway,
        settings=env.settings,
        options=hash_options,
    )
    return should_skip_native_target(env, target, input_hash, options_hash=options_hash)


@tag_tool(domain="ingestion", target=CONFIG_INGEST_TARGET_NAME)
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

        file_state_hash = _compute_file_state_hash(config_files)
        return ConfigScanResult(
            success=True,
            config_files=config_files,
            file_state_hash=file_state_hash,
        )
    except Exception:
        log.exception("Config scan failed")
        return ConfigScanResult(success=False, error="Config file discovery failed with exception")


@tag_tool(domain="ingestion", target=CONFIG_INGEST_TARGET_NAME)
def t__config_ingest__ingest(
    env: BuildEnv,
    graph: TargetGraph,
    t__config_ingest__scan: ConfigScanResult,
) -> ConfigIngestResult:
    """Ingest discovered config files into structured tables.

    Returns
    -------
    ConfigIngestResult
        Ingestion status and row tuples.
    """
    if not t__config_ingest__scan.success:
        return ConfigIngestResult(
            result=ExecutionResult.failed(f"Config scan failed: {t__config_ingest__scan.error}")
        )

    if _should_skip_target(
        env,
        graph,
        CONFIG_INGEST_TARGET_NAME,
        file_state_hash=t__config_ingest__scan.file_state_hash,
    ):
        return ConfigIngestResult(result=ExecutionResult.skip("Config ingest skipped"))

    config_files = t__config_ingest__scan.config_files
    discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
    step = ConfigIngestStep(discovery=discovery)
    return step.execute(
        config_files,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
    )


@tag_helper(domain="ingestion", target=CONFIG_INGEST_TARGET_NAME)
def config_ingest__hash_options(
    env: BuildEnv,
    t__config_ingest__scan: ConfigScanResult,
) -> InputHashOptions:
    options_hash = options_hash_for_target(env, CONFIG_INGEST_TARGET_NAME)
    return InputHashOptions(
        options_hash=options_hash,
        manifests=env.manifest_index,
        file_state_hash=t__config_ingest__scan.file_state_hash,
    )


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(CONFIG_VALUES_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(CONFIG_INGEST_TARGET_NAME),
    table_key=value(CONFIG_VALUES_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(CONFIG_VALUES_TABLE_KEY)),
    hash_options=source("config_ingest__hash_options"),
)
@tag_compute(domain="ingestion", target=CONFIG_INGEST_TARGET_NAME, target_="config_ingest__rows")
def config_ingest__rows(
    t__config_ingest__ingest: ConfigIngestResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.config_values.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the config_values table, or None when ingestion is skipped or failed.
    """
    if t__config_ingest__ingest.result.skipped or not t__config_ingest__ingest.result.success:
        return None
    return t__config_ingest__ingest.rows


@tag_materialize(domain="ingestion", target=CONFIG_INGEST_TARGET_NAME)
def t__config_ingest(
    env: BuildEnv,
    graph: TargetGraph,
    t__config_ingest__ingest: ConfigIngestResult,
    config_ingest__hash_options: InputHashOptions,
    m__analytics__config_values: MaterializationMetadata,
) -> TargetRunRecord:
    """Finalize config_ingest execution and persist manifest.

    Returns
    -------
    TargetRunRecord
        Record describing the execution outcome.
    """
    if not t__config_ingest__ingest.result.success and not t__config_ingest__ingest.result.skipped:
        executor = NativeTargetExecutor.for_target(
            env,
            graph,
            CONFIG_INGEST_TARGET_NAME,
            hash_options=config_ingest__hash_options,
        )
        return executor.fail(
            RuntimeError(t__config_ingest__ingest.result.error or "Config ingest failed")
        )

    for warning in t__config_ingest__ingest.result.warnings:
        log.warning("Config parse warning: %s", warning)

    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name=CONFIG_INGEST_TARGET_NAME,
        materializations={
            CONFIG_VALUES_TABLE_KEY: m__analytics__config_values,
        },
    )


# ---------------------------------------------------------------------------
# coverage_ingest target
# ---------------------------------------------------------------------------


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


@tag_tool(domain="ingestion", target=COVERAGE_INGEST_TARGET_NAME)
def t__coverage_ingest__ingest(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> CoverageIngestResult:
    """Execute coverage data ingestion from coverage.py output.

    Returns
    -------
    CoverageIngestResult
        Ingestion status and row tuples.
    """
    if t__modules.status != "succeeded":
        return CoverageIngestResult(
            result=ExecutionResult.failed(f"Upstream modules target failed: {t__modules.error}")
        )

    if _should_skip_target(env, graph, COVERAGE_INGEST_TARGET_NAME):
        return CoverageIngestResult(result=ExecutionResult.skip("Coverage ingest skipped"))

    coverage_path = _resolve_coverage_file(env)
    if coverage_path is None:
        log.info("No coverage file found, writing empty coverage rows")
        return CoverageIngestResult(result=ExecutionResult.ok(), rows=())

    tools = ToolRunnerAdapter(env.providers.tool_service)
    step = CoverageIngestStep(tools=tools)
    return asyncio.run(
        step.execute_async(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            repo_root=env.snapshot.repo_root,
            coverage_file=coverage_path,
        )
    )


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(COVERAGE_LINES_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(COVERAGE_INGEST_TARGET_NAME),
    table_key=value(COVERAGE_LINES_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(COVERAGE_LINES_TABLE_KEY)),
)
@tag_compute(domain="ingestion", target=COVERAGE_INGEST_TARGET_NAME, target_="coverage__rows")
def coverage__rows(
    t__coverage_ingest__ingest: CoverageIngestResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.coverage_lines.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the coverage_lines table, or None when ingestion is skipped or failed.
    """
    if t__coverage_ingest__ingest.result.skipped or not t__coverage_ingest__ingest.result.success:
        return None
    return t__coverage_ingest__ingest.rows


@tag_materialize(domain="ingestion", target=COVERAGE_INGEST_TARGET_NAME)
def t__coverage_ingest(
    env: BuildEnv,
    graph: TargetGraph,
    t__coverage_ingest__ingest: CoverageIngestResult,
    m__analytics__coverage_lines: MaterializationMetadata,
) -> TargetRunRecord:
    """Finalize coverage_ingest execution and persist manifest.

    Returns
    -------
    TargetRunRecord
        Record describing the execution outcome.
    """
    if (
        not t__coverage_ingest__ingest.result.success
        and not t__coverage_ingest__ingest.result.skipped
    ):
        executor = NativeTargetExecutor.for_target(env, graph, COVERAGE_INGEST_TARGET_NAME)
        return executor.fail(
            RuntimeError(t__coverage_ingest__ingest.result.error or "Coverage ingest failed")
        )

    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name=COVERAGE_INGEST_TARGET_NAME,
        materializations={
            COVERAGE_LINES_TABLE_KEY: m__analytics__coverage_lines,
        },
    )


# ---------------------------------------------------------------------------
# tests_ingest target
# ---------------------------------------------------------------------------


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


@tag_tool(domain="ingestion", target=TESTS_INGEST_TARGET_NAME)
def t__tests_ingest__ingest(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> TestsIngestResult:
    """Execute pytest report ingestion into analytics tables.

    Returns
    -------
    TestsIngestResult
        Ingestion status and row tuples.
    """
    if t__modules.status != "succeeded":
        return TestsIngestResult(
            result=ExecutionResult.failed(f"Upstream modules target failed: {t__modules.error}")
        )

    if _should_skip_target(env, graph, TESTS_INGEST_TARGET_NAME):
        return TestsIngestResult(result=ExecutionResult.skip("Tests ingest skipped"))

    report_path = _resolve_report_file(env)
    if report_path is None:
        log.info("No pytest report found, writing empty test rows")
        return TestsIngestResult(result=ExecutionResult.ok(), rows=())

    step = TestsIngestStep()
    return step.execute(
        module_records,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        json_report_path=report_path,
    )


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(TEST_CATALOG_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(TESTS_INGEST_TARGET_NAME),
    table_key=value(TEST_CATALOG_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(TEST_CATALOG_TABLE_KEY)),
)
@tag_compute(domain="ingestion", target=TESTS_INGEST_TARGET_NAME, target_="tests__rows")
def tests__rows(
    t__tests_ingest__ingest: TestsIngestResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.test_catalog.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the test_catalog table, or None when ingestion is skipped or failed.
    """
    if t__tests_ingest__ingest.result.skipped or not t__tests_ingest__ingest.result.success:
        return None
    return t__tests_ingest__ingest.rows


@tag_materialize(domain="ingestion", target=TESTS_INGEST_TARGET_NAME)
def t__tests_ingest(
    env: BuildEnv,
    graph: TargetGraph,
    t__tests_ingest__ingest: TestsIngestResult,
    m__analytics__test_catalog: MaterializationMetadata,
) -> TargetRunRecord:
    """Finalize tests_ingest execution and persist manifest.

    Returns
    -------
    TargetRunRecord
        Record describing the execution outcome.
    """
    if not t__tests_ingest__ingest.result.success and not t__tests_ingest__ingest.result.skipped:
        executor = NativeTargetExecutor.for_target(env, graph, TESTS_INGEST_TARGET_NAME)
        return executor.fail(
            RuntimeError(t__tests_ingest__ingest.result.error or "Tests ingest failed")
        )

    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name=TESTS_INGEST_TARGET_NAME,
        materializations={
            TEST_CATALOG_TABLE_KEY: m__analytics__test_catalog,
        },
    )


# ---------------------------------------------------------------------------
# typing target
# ---------------------------------------------------------------------------


@tag_tool(domain="ingestion", target=TYPING_TARGET_NAME)
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
        Ingestion status and row tuples.
    """
    if t__modules.status != "succeeded":
        return TypingIngestResult(
            result=ExecutionResult.failed(f"Upstream modules target failed: {t__modules.error}")
        )

    if _should_skip_target(env, graph, TYPING_TARGET_NAME):
        return TypingIngestResult(result=ExecutionResult.skip("Typing ingest skipped"))

    if not module_records:
        return TypingIngestResult(
            result=ExecutionResult.ok(), typedness_rows=(), diagnostic_rows=()
        )

    discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
    tools = ToolRunnerAdapter(env.providers.tool_service)

    step = TypingIngestStep(discovery=discovery, tools=tools)
    return asyncio.run(
        step.execute_async(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            repo_root=str(env.snapshot.repo_root),
            run_diagnostics=True,
        )
    )


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(TYPEDNESS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(TYPING_TARGET_NAME),
    table_key=value(TYPEDNESS_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(TYPEDNESS_TABLE_KEY)),
)
@tag_compute(domain="ingestion", target=TYPING_TARGET_NAME, target_="typing__typedness_rows")
def typing__typedness_rows(
    t__typing__ingest: TypingIngestResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.typedness.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the typedness table, or None when ingestion is skipped or failed.
    """
    if t__typing__ingest.result.skipped or not t__typing__ingest.result.success:
        return None
    return t__typing__ingest.typedness_rows


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(STATIC_DIAGNOSTICS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(TYPING_TARGET_NAME),
    table_key=value(STATIC_DIAGNOSTICS_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(STATIC_DIAGNOSTICS_TABLE_KEY)),
)
@tag_compute(domain="ingestion", target=TYPING_TARGET_NAME, target_="typing__diagnostic_rows")
def typing__diagnostic_rows(
    t__typing__ingest: TypingIngestResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.static_diagnostics.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the static_diagnostics table, or None when ingestion is skipped or failed.
    """
    if t__typing__ingest.result.skipped or not t__typing__ingest.result.success:
        return None
    return t__typing__ingest.diagnostic_rows


@tag_materialize(domain="ingestion", target=TYPING_TARGET_NAME)
def t__typing(
    env: BuildEnv,
    graph: TargetGraph,
    t__typing__ingest: TypingIngestResult,
    m__analytics__typedness: MaterializationMetadata,
    m__analytics__static_diagnostics: MaterializationMetadata,
) -> TargetRunRecord:
    """Finalize typing target execution and persist manifest.

    Returns
    -------
    TargetRunRecord
        Record describing the execution outcome.
    """
    if not t__typing__ingest.result.success and not t__typing__ingest.result.skipped:
        executor = NativeTargetExecutor.for_target(env, graph, TYPING_TARGET_NAME)
        return executor.fail(
            RuntimeError(t__typing__ingest.result.error or "Typing ingestion failed")
        )

    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name=TYPING_TARGET_NAME,
        materializations={
            TYPEDNESS_TABLE_KEY: m__analytics__typedness,
            STATIC_DIAGNOSTICS_TABLE_KEY: m__analytics__static_diagnostics,
        },
    )


__all__: list[str] = [
    "ConfigIngestResult",
    "ConfigScanResult",
    "CoverageIngestResult",
    "ModuleScanResult",
    "RepoScanResult",
    "TestsIngestResult",
    "TypingIngestResult",
    "t__config_ingest",
    "t__config_ingest__ingest",
    "t__config_ingest__scan",
    "t__coverage_ingest",
    "t__coverage_ingest__ingest",
    "t__modules",
    "t__modules__scan",
    "t__tests_ingest",
    "t__tests_ingest__ingest",
    "t__typing",
    "t__typing__ingest",
]
