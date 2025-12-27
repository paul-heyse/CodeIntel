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
from pathlib import Path
from typing import TYPE_CHECKING, cast

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.helpers import (
    build_scan_profile,
    filter_modules,
    get_module_paths_from_env,
    paths_to_modules,
)
from codeintel.build.hamilton.native.ingestion.pipelines import pipe_ingest_rows
from codeintel.build.hamilton.native.options.ingestion import ModuleIngestOptions
from codeintel.build.hamilton.native.patterns import (
    IngestStep,
    SaverContext,
    TableSaveSpec,
    ToolFinalizeContext,
    ToolRunContext,
    finalize_target_from_materializations,
    run_tool_step,
    save_rows,
)
from codeintel.build.hamilton.native.target_decorators import (
    TargetSpecDescriptor,
    codeintel_target,
)
from codeintel.build.hamilton.native.tool_results import ToolStepOutput
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_compute, tag_helper, tag_tool
from codeintel.build.hashing import compute_options_hash
from codeintel.build.resources import TOOL_EXECUTION, TargetResources
from codeintel.build.schemas.service import get_schema_service
from codeintel.core.paths import normalize_path
from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
)
from codeintel.ingestion.adapters.tool_runner import ToolRunnerAdapter
from codeintel.ingestion.compute.config_ingest import ConfigIngestStep
from codeintel.ingestion.compute.coverage_ingest import CoverageIngestStep
from codeintel.ingestion.compute.repo_scan import RepoScanStep
from codeintel.ingestion.compute.tests_ingest import TestsIngestStep
from codeintel.ingestion.compute.typing_ingest import TypingIngestStep
from codeintel.ingestion.infrastructure.scanning import default_config_profile
from codeintel.ingestion.ports.discovery import ModuleRecord

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.ingestion.ports.change_detection import ChangeSet, FileDigest

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (
    BuildEnv,
    MaterializationResult,
    DagCatalog,
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

_DUPLICATE_SAMPLE_LIMIT = 5

MODULES_SAVE_CONTEXT = SaverContext(
    domain="ingestion",
    target=MODULES_TARGET_NAME,
)
CONFIG_INGEST_SAVE_CONTEXT = SaverContext(
    domain="ingestion",
    target=CONFIG_INGEST_TARGET_NAME,
)
COVERAGE_INGEST_SAVE_CONTEXT = SaverContext(
    domain="ingestion",
    target=COVERAGE_INGEST_TARGET_NAME,
)
TESTS_INGEST_SAVE_CONTEXT = SaverContext(
    domain="ingestion",
    target=TESTS_INGEST_TARGET_NAME,
)
TYPING_SAVE_CONTEXT = SaverContext(
    domain="ingestion",
    target=TYPING_TARGET_NAME,
)


@dataclass(frozen=True)
class ModuleToolOutput(ToolStepOutput):
    """Tool step output for repository module scanning."""

    modules: tuple[ModuleRecord, ...] = field(default_factory=tuple)
    change_set: ChangeSet | None = None
    file_state_hash: str | None = None
    module_rows: tuple[tuple[object, ...], ...] = field(default_factory=tuple)
    file_state_rows: tuple[tuple[object, ...], ...] = field(default_factory=tuple)
    repo_map_rows: tuple[tuple[object, ...], ...] = field(default_factory=tuple)


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


@dataclass(frozen=True)
class ConfigToolOutput(ToolStepOutput):
    """Tool step output for config ingestion."""

    rows: tuple[tuple[object, ...], ...] = ()


@dataclass(frozen=True)
class CoverageToolOutput(ToolStepOutput):
    """Tool step output for coverage ingestion."""

    rows: tuple[tuple[object, ...], ...] = ()


@dataclass(frozen=True)
class TestsToolOutput(ToolStepOutput):
    """Tool step output for tests ingestion."""

    rows: tuple[tuple[object, ...], ...] = ()


@dataclass(frozen=True)
class TypingToolOutput(ToolStepOutput):
    """Tool step output for typing ingestion."""

    typedness_rows: tuple[tuple[object, ...], ...] = ()
    diagnostic_rows: tuple[tuple[object, ...], ...] = ()


@tag_helper(domain="ingestion")
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
    if t__modules.status == "failed":
        return ()
    return tuple(get_module_paths_from_env(env))


@tag_helper(domain="ingestion")
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
def t__modules__run(env: BuildEnv) -> ModuleToolOutput:
    """Execute repository scan to discover modules.

    Returns
    -------
    ModuleToolOutput
        Tool output containing discovered modules and row tuples.
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

        return ModuleToolOutput(
            result=ExecutionResult.ok(),
            modules=scan_result.modules,
            change_set=scan_result.change_set,
            file_state_hash=scan_result.change_set.state_hash,
            module_rows=scan_result.module_rows,
            file_state_rows=scan_result.file_state_rows,
            repo_map_rows=scan_result.repo_map_rows,
        )

    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        log.exception("Module scan failed")
        return ModuleToolOutput(result=ExecutionResult.failed(str(exc)))


@tag_compute(domain="ingestion", target=MODULES_TARGET_NAME)
def t__modules__ingest(
    t__modules__run: ModuleToolOutput,
) -> IngestStep[dict[str, tuple[tuple[object, ...], ...]]]:
    """Package module scan rows for table materialization.

    Returns
    -------
    IngestStep[dict[str, tuple[tuple[object, ...], ...]]]
        Ingest result with table row payloads.
    """
    result = t__modules__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "Modules ingest skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "Module scan failed",
                warnings=result.warnings,
            )
        )

    module_rows = _dedupe_rows_for_table(
        t__modules__run.module_rows,
        table_key=MODULES_TABLE_KEY,
    )
    file_state_rows = _dedupe_rows_for_table(
        t__modules__run.file_state_rows,
        table_key=FILE_STATE_TABLE_KEY,
        prefer_columns=("mtime_ns", "content_hash"),
    )
    repo_map_rows = t__modules__run.repo_map_rows
    payload = {
        MODULES_TABLE_KEY: module_rows,
        FILE_STATE_TABLE_KEY: file_state_rows,
        REPO_MAP_TABLE_KEY: repo_map_rows,
    }
    table_counts = {
        MODULES_TABLE_KEY: len(module_rows),
        FILE_STATE_TABLE_KEY: len(file_state_rows),
        REPO_MAP_TABLE_KEY: len(repo_map_rows),
    }
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


@save_rows(
    context=MODULES_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=MODULES_TABLE_KEY),
)
@tag_compute(domain="ingestion", target=MODULES_TARGET_NAME, target_="modules__module_rows")
def modules__module_rows(
    t__modules__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for core.modules.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the modules table, or None when ingestion skipped or failed.

    Raises
    ------
    ValueError
        If the ingest payload is missing expected row data.
    """
    if t__modules__ingest.result.skipped or not t__modules__ingest.result.success:
        return None

    payload = t__modules__ingest.payload
    if payload is None:
        msg = "Missing modules ingest payload"
        raise ValueError(msg)
    rows = payload.get(MODULES_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {MODULES_TABLE_KEY}"
        raise ValueError(msg)
    return rows


@save_rows(
    context=MODULES_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=FILE_STATE_TABLE_KEY),
)
@tag_compute(domain="ingestion", target=MODULES_TARGET_NAME, target_="modules__file_state_rows")
def modules__file_state_rows(
    t__modules__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for core.file_state.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the file_state table, or None when ingestion skipped or failed.

    Raises
    ------
    ValueError
        If the ingest payload is missing expected row data.
    """
    if t__modules__ingest.result.skipped or not t__modules__ingest.result.success:
        return None

    payload = t__modules__ingest.payload
    if payload is None:
        msg = "Missing modules ingest payload"
        raise ValueError(msg)
    rows = payload.get(FILE_STATE_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {FILE_STATE_TABLE_KEY}"
        raise ValueError(msg)
    return rows


@save_rows(
    context=MODULES_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=REPO_MAP_TABLE_KEY),
)
@tag_compute(domain="ingestion", target=MODULES_TARGET_NAME, target_="modules__repo_map_rows")
def modules__repo_map_rows(
    t__modules__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for core.repo_map.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the repo_map table, or None when ingestion skipped or failed.

    Raises
    ------
    ValueError
        If the ingest payload is missing expected row data.
    """
    if t__modules__ingest.result.skipped or not t__modules__ingest.result.success:
        return None

    payload = t__modules__ingest.payload
    if payload is None:
        msg = "Missing modules ingest payload"
        raise ValueError(msg)
    rows = payload.get(REPO_MAP_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {REPO_MAP_TABLE_KEY}"
        raise ValueError(msg)
    return rows


def _dedupe_rows_for_table(
    rows: tuple[tuple[object, ...], ...],
    *,
    table_key: str,
    prefer_columns: tuple[str, ...] | None = None,
) -> tuple[tuple[object, ...], ...]:
    if not rows:
        return rows
    schema = get_schema_service().get_table_schema(table_key)
    if schema is None or not schema.primary_key:
        return rows
    columns = tuple(schema.column_names())
    return _dedupe_rows_by_keys(
        rows,
        table_key=table_key,
        columns=columns,
        key_columns=tuple(schema.primary_key),
        prefer_columns=prefer_columns,
    )


def _dedupe_rows_by_keys(
    rows: tuple[tuple[object, ...], ...],
    *,
    table_key: str,
    columns: tuple[str, ...],
    key_columns: tuple[str, ...],
    prefer_columns: tuple[str, ...] | None = None,
) -> tuple[tuple[object, ...], ...]:
    try:
        key_indexes = tuple(columns.index(col) for col in key_columns)
    except ValueError as exc:
        log.warning("Unable to dedupe %s rows: missing key column (%s)", table_key, exc)
        return rows
    prefer_indexes = ()
    if prefer_columns:
        columns_set = set(columns)
        prefer_indexes = tuple(columns.index(col) for col in prefer_columns if col in columns_set)
    deduped: dict[tuple[object, ...], tuple[object, ...]] = {}
    duplicate_count = 0
    sample_keys: list[tuple[object, ...]] = []
    for row in rows:
        key = tuple(row[idx] for idx in key_indexes)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = row
            continue
        duplicate_count += 1
        if len(sample_keys) < _DUPLICATE_SAMPLE_LIMIT and key not in sample_keys:
            sample_keys.append(key)
        if prefer_indexes and _is_preferred_row(row, existing, prefer_indexes):
            deduped[key] = row
    if duplicate_count:
        log.warning(
            "Duplicate rows detected for %s (duplicates=%d, sample_keys=%s)",
            table_key,
            duplicate_count,
            sample_keys,
        )
    return tuple(deduped.values())


def _is_preferred_row(
    candidate: tuple[object, ...],
    existing: tuple[object, ...],
    prefer_indexes: tuple[int, ...],
) -> bool:
    for idx in prefer_indexes:
        candidate_value = candidate[idx]
        existing_value = existing[idx]
        if candidate_value == existing_value:
            continue
        if existing_value is None and candidate_value is not None:
            return True
        if candidate_value is None:
            return False
        if isinstance(candidate_value, (int, float)) and isinstance(existing_value, (int, float)):
            return candidate_value > existing_value
        if isinstance(candidate_value, str) and isinstance(existing_value, str):
            return candidate_value > existing_value
        return str(candidate_value) > str(existing_value)
    return False


@tag_helper(domain="ingestion", target=MODULES_TARGET_NAME)
def modules__table_materializations(
    m__core__modules: MaterializationResult,
    m__core__file_state: MaterializationResult,
    m__core__repo_map: MaterializationResult,
) -> dict[str, MaterializationResult]:
    """Collect materialization results for modules target tables.

    Returns
    -------
    dict[str, MaterializationResult]
        Mapping from table key to saver metadata.
    """
    return {
        MODULES_TABLE_KEY: m__core__modules,
        FILE_STATE_TABLE_KEY: m__core__file_state,
        REPO_MAP_TABLE_KEY: m__core__repo_map,
    }


@tag_helper(domain="ingestion", target=MODULES_TARGET_NAME)
def modules__finalize_context(
    env: BuildEnv,
    catalog: DagCatalog,
    t__modules__run: ModuleToolOutput,
) -> ToolFinalizeContext:
    """Build finalization context for the modules target.

    Returns
    -------
    ToolFinalizeContext
        Finalization context for modules.
    """
    change_delta: dict[str, object] | None = None
    change_set = t__modules__run.change_set
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
    return ToolFinalizeContext(
        env=env,
        catalog=catalog,
        target_name=MODULES_TARGET_NAME,
        change_delta=change_delta,
    )


@codeintel_target(domain="ingestion", target=MODULES_TARGET_NAME)
def t__modules(
    modules__finalize_context: ToolFinalizeContext,
    t__modules__run: ModuleToolOutput,
    t__modules__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
    modules__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Scan repository modules and file index.

    Returns
    -------
    TargetRunRecord
        Record describing the execution outcome.
    """
    return finalize_target_from_materializations(
        context=modules__finalize_context,
        tool_step=t__modules__run,
        ingest_step=t__modules__ingest,
        artifact_materializations=None,
        table_materializations=modules__table_materializations,
    )


# ---------------------------------------------------------------------------
# config_ingest target
# ---------------------------------------------------------------------------


def _merge_result_warnings(result: ExecutionResult, warnings: tuple[str, ...]) -> ExecutionResult:
    if not warnings:
        return result

    merged = (*result.warnings, *warnings)
    if result.skipped:
        return ExecutionResult.skip(
            result.skip_reason or "Ingestion skipped",
            table_counts=result.table_counts,
            warnings=merged,
        )
    if result.success:
        return ExecutionResult.ok(table_counts=result.table_counts, warnings=merged)
    return ExecutionResult.failed(
        result.error or "Ingestion failed",
        table_counts=result.table_counts,
        warnings=merged,
    )


def _state_hash_for_records(records: Sequence[ModuleRecord]) -> str | None:
    state: dict[str, FileDigest] = {}
    for record in records:
        digest = HashChangeDetectionAdapter.compute_file_digest(record.file_path)
        if digest is None:
            continue
        state[normalize_path(record.rel_path)] = digest
    if not state:
        return None
    return HashChangeDetectionAdapter.compute_state_hash(state)


def _state_hash_for_paths(paths: Sequence[Path], *, root: Path | None = None) -> str | None:
    state: dict[str, FileDigest] = {}
    for path in paths:
        digest = HashChangeDetectionAdapter.compute_file_digest(path)
        if digest is None:
            continue
        if root is not None and path.is_relative_to(root):
            key = normalize_path(str(path.relative_to(root)))
        else:
            key = normalize_path(str(path))
        state[key] = digest
    if not state:
        return None
    return HashChangeDetectionAdapter.compute_state_hash(state)


def _combine_state_hashes(*hashes: str | None) -> str | None:
    values = [value for value in hashes if value]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return compute_options_hash(values)


def _modules_precheck(
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> tuple[ExecutionResult | None, tuple[str, ...]]:
    if t__modules.status == "succeeded":
        return None, ()
    if not module_records:
        message = t__modules.error or "No module inventory available"
        return (
            ExecutionResult.failed(f"Upstream modules target {t__modules.status}: {message}"),
            (),
        )
    warnings = (f"Upstream modules target {t__modules.status}; using stored module inventory.",)
    return None, warnings


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

        file_state_hash = _state_hash_for_records(config_files)
        return ConfigScanResult(
            success=True,
            config_files=config_files,
            file_state_hash=file_state_hash,
        )
    except Exception:
        log.exception("Config scan failed")
        return ConfigScanResult(success=False, error="Config file discovery failed with exception")


@tag_tool(domain="ingestion", target=CONFIG_INGEST_TARGET_NAME)
def t__config_ingest__run(
    env: BuildEnv,
    catalog: DagCatalog,
    t__config_ingest__scan: ConfigScanResult,
) -> ConfigToolOutput:
    """Discover and ingest config files into structured tables.

    Returns
    -------
    ConfigToolOutput
        Tool output containing config ingestion rows.
    """
    if not t__config_ingest__scan.success:
        return ConfigToolOutput(
            result=ExecutionResult.failed(f"Config scan failed: {t__config_ingest__scan.error}")
        )

    context = ToolRunContext(
        env=env,
        catalog=catalog,
        target_name=CONFIG_INGEST_TARGET_NAME,
    )

    def _execute() -> ConfigToolOutput:
        config_files = t__config_ingest__scan.config_files
        if not config_files:
            return ConfigToolOutput(result=ExecutionResult.ok(), rows=())
        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
        step = ConfigIngestStep(discovery=discovery)
        ingest_result = step.execute(
            config_files,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
        return ConfigToolOutput(result=ingest_result.result, rows=ingest_result.rows)

    output = run_tool_step(context=context, run=_execute)
    if isinstance(output, ConfigToolOutput):
        return output
    return ConfigToolOutput(result=output.result, rows=())


@tag_compute(domain="ingestion", target=CONFIG_INGEST_TARGET_NAME)
def t__config_ingest__ingest(
    t__config_ingest__run: ConfigToolOutput,
) -> IngestStep[dict[str, tuple[tuple[object, ...], ...]]]:
    """Package config ingestion rows for table materialization.

    Returns
    -------
    IngestStep[dict[str, tuple[tuple[object, ...], ...]]]
        Ingest result with table row payloads.
    """
    result = t__config_ingest__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "Config ingest skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "Config ingest failed",
                warnings=result.warnings,
            )
        )

    payload = {CONFIG_VALUES_TABLE_KEY: t__config_ingest__run.rows}
    table_counts = {CONFIG_VALUES_TABLE_KEY: len(t__config_ingest__run.rows)}
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


@tag_compute(
    domain="ingestion", target=CONFIG_INGEST_TARGET_NAME, target_="config_ingest__raw_rows"
)
def config_ingest__raw_rows(
    t__config_ingest__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract raw rows for analytics.config_values.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Raw rows for the config values table, or None when skipped/failed.

    Raises
    ------
    ValueError
        If the ingest payload or rows are missing.
    """
    if t__config_ingest__ingest.result.skipped or not t__config_ingest__ingest.result.success:
        return None

    payload = t__config_ingest__ingest.payload
    if payload is None:
        msg = "Missing config ingest payload"
        raise ValueError(msg)
    rows = payload.get(CONFIG_VALUES_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {CONFIG_VALUES_TABLE_KEY}"
        raise ValueError(msg)
    return rows


@save_rows(
    context=CONFIG_INGEST_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=CONFIG_VALUES_TABLE_KEY),
)
@pipe_ingest_rows(required_indices=(0, 1, 2, 3, 4), input_name="config_ingest__raw_rows")
@tag_compute(domain="ingestion", target=CONFIG_INGEST_TARGET_NAME, target_="config_ingest__rows")
def config_ingest__rows(
    config_ingest__raw_rows: tuple[tuple[object, ...], ...] | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Return cleaned rows for analytics.config_values.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Cleaned rows for the config values table.
    """
    return config_ingest__raw_rows


@tag_helper(domain="ingestion", target=CONFIG_INGEST_TARGET_NAME)
def config_ingest__table_materializations(
    m__analytics__config_values: MaterializationResult,
) -> dict[str, MaterializationResult]:
    """Collect config ingest table materializations.

    Returns
    -------
    dict[str, MaterializationResult]
        Mapping of table keys to materialization results.
    """
    return {CONFIG_VALUES_TABLE_KEY: m__analytics__config_values}


@tag_helper(domain="ingestion", target=CONFIG_INGEST_TARGET_NAME)
def config_ingest__finalize_context(
    env: BuildEnv,
    catalog: DagCatalog,
) -> ToolFinalizeContext:
    """Build finalization context for config ingest.

    Returns
    -------
    ToolFinalizeContext
        Finalization context for config ingest.
    """
    return ToolFinalizeContext(
        env=env,
        catalog=catalog,
        target_name=CONFIG_INGEST_TARGET_NAME,
    )


@codeintel_target(domain="ingestion", target=CONFIG_INGEST_TARGET_NAME)
def t__config_ingest(
    config_ingest__finalize_context: ToolFinalizeContext,
    t__config_ingest__run: ConfigToolOutput,
    t__config_ingest__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
    config_ingest__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Parse configuration files and track references.

    Returns
    -------
    TargetRunRecord
        Record describing the execution outcome.
    """
    for warning in t__config_ingest__run.result.warnings:
        log.warning("Config parse warning: %s", warning)

    return finalize_target_from_materializations(
        context=config_ingest__finalize_context,
        tool_step=t__config_ingest__run,
        ingest_step=t__config_ingest__ingest,
        artifact_materializations=None,
        table_materializations=config_ingest__table_materializations,
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


@tag_helper(domain="ingestion", target=COVERAGE_INGEST_TARGET_NAME)
def coverage_ingest__file_state_hash(env: BuildEnv) -> str | None:
    """Return a stable hash for the coverage report file.

    Returns
    -------
    str | None
        Hash string for the coverage file, or None when missing.
    """
    coverage_path = _resolve_coverage_file(env)
    if coverage_path is None:
        return None
    return _state_hash_for_paths((coverage_path,), root=env.snapshot.repo_root)


def _coerce_coverage_output(
    output: ToolStepOutput,
    warnings: tuple[str, ...],
) -> CoverageToolOutput:
    if isinstance(output, CoverageToolOutput):
        if warnings:
            return CoverageToolOutput(
                result=_merge_result_warnings(output.result, warnings),
                rows=output.rows,
            )
        return output
    merged = _merge_result_warnings(output.result, warnings)
    return CoverageToolOutput(result=merged, rows=())


@tag_tool(domain="ingestion", target=COVERAGE_INGEST_TARGET_NAME)
def t__coverage_ingest__run(
    env: BuildEnv,
    catalog: DagCatalog,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> CoverageToolOutput:
    """Execute coverage data ingestion from coverage.py output.

    Returns
    -------
    CoverageToolOutput
        Tool output containing coverage rows.
    """
    failure, warnings = _modules_precheck(t__modules, module_records)
    if failure is not None:
        return CoverageToolOutput(result=failure)

    context = ToolRunContext(
        env=env,
        catalog=catalog,
        target_name=COVERAGE_INGEST_TARGET_NAME,
    )

    def _execute() -> CoverageToolOutput:
        coverage_path = _resolve_coverage_file(env)
        if coverage_path is None:
            log.info("No coverage file found, writing empty coverage rows")
            result = ExecutionResult.ok(warnings=warnings)
            return CoverageToolOutput(result=result, rows=())

        tools = ToolRunnerAdapter(env.providers.tool_service)
        step = CoverageIngestStep(tools=tools)
        ingest_result = asyncio.run(
            step.execute_async(
                module_records,
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
                repo_root=env.snapshot.repo_root,
                coverage_file=coverage_path,
            )
        )
        return CoverageToolOutput(
            result=_merge_result_warnings(ingest_result.result, warnings),
            rows=ingest_result.rows,
        )

    output = run_tool_step(context=context, run=_execute)
    return _coerce_coverage_output(output, warnings)


@tag_compute(domain="ingestion", target=COVERAGE_INGEST_TARGET_NAME)
def t__coverage_ingest__ingest(
    t__coverage_ingest__run: CoverageToolOutput,
) -> IngestStep[dict[str, tuple[tuple[object, ...], ...]]]:
    """Package coverage ingestion rows for table materialization.

    Returns
    -------
    IngestStep[dict[str, tuple[tuple[object, ...], ...]]]
        Ingest result with table row payloads.
    """
    result = t__coverage_ingest__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "Coverage ingest skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "Coverage ingest failed",
                warnings=result.warnings,
            )
        )

    payload = {COVERAGE_LINES_TABLE_KEY: t__coverage_ingest__run.rows}
    table_counts = {COVERAGE_LINES_TABLE_KEY: len(t__coverage_ingest__run.rows)}
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


@save_rows(
    context=COVERAGE_INGEST_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=COVERAGE_LINES_TABLE_KEY),
)
@tag_compute(domain="ingestion", target=COVERAGE_INGEST_TARGET_NAME, target_="coverage__rows")
def coverage__rows(
    t__coverage_ingest__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.coverage_lines.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the coverage_lines table, or None when ingestion is skipped or failed.

    Raises
    ------
    ValueError
        If the ingest payload is missing expected row data.
    """
    if t__coverage_ingest__ingest.result.skipped or not t__coverage_ingest__ingest.result.success:
        return None

    payload = t__coverage_ingest__ingest.payload
    if payload is None:
        msg = "Missing coverage ingest payload"
        raise ValueError(msg)
    rows = payload.get(COVERAGE_LINES_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {COVERAGE_LINES_TABLE_KEY}"
        raise ValueError(msg)
    return rows


@tag_helper(domain="ingestion", target=COVERAGE_INGEST_TARGET_NAME)
def coverage_ingest__table_materializations(
    m__analytics__coverage_lines: MaterializationResult,
) -> dict[str, MaterializationResult]:
    """Collect coverage ingest table materializations.

    Returns
    -------
    dict[str, MaterializationResult]
        Mapping of table keys to materialization results.
    """
    return {COVERAGE_LINES_TABLE_KEY: m__analytics__coverage_lines}


@tag_helper(domain="ingestion", target=COVERAGE_INGEST_TARGET_NAME)
def coverage_ingest__finalize_context(
    env: BuildEnv,
    catalog: DagCatalog,
) -> ToolFinalizeContext:
    """Build finalization context for coverage ingest.

    Returns
    -------
    ToolFinalizeContext
        Finalization context for coverage ingest.
    """
    return ToolFinalizeContext(
        env=env,
        catalog=catalog,
        target_name=COVERAGE_INGEST_TARGET_NAME,
    )


@codeintel_target(domain="ingestion", target=COVERAGE_INGEST_TARGET_NAME)
def t__coverage_ingest(
    coverage_ingest__finalize_context: ToolFinalizeContext,
    t__coverage_ingest__run: CoverageToolOutput,
    t__coverage_ingest__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
    coverage_ingest__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Ingest line-level test coverage.

    Returns
    -------
    TargetRunRecord
        Record describing the execution outcome.
    """
    return finalize_target_from_materializations(
        context=coverage_ingest__finalize_context,
        tool_step=t__coverage_ingest__run,
        ingest_step=t__coverage_ingest__ingest,
        artifact_materializations=None,
        table_materializations=coverage_ingest__table_materializations,
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
@tag_helper(domain="ingestion", target=TESTS_INGEST_TARGET_NAME)
def tests_ingest__file_state_hash(env: BuildEnv) -> str | None:
    """Return a stable hash for the pytest report file.

    Returns
    -------
    str | None
        Hash string for the report file, or None when missing.
    """
    report_path = _resolve_report_file(env)
    if report_path is None:
        return None
    return _state_hash_for_paths((report_path,), root=env.snapshot.repo_root)


def _coerce_tests_output(
    output: ToolStepOutput,
    warnings: tuple[str, ...],
) -> TestsToolOutput:
    if isinstance(output, TestsToolOutput):
        if warnings:
            return TestsToolOutput(
                result=_merge_result_warnings(output.result, warnings),
                rows=output.rows,
            )
        return output
    merged = _merge_result_warnings(output.result, warnings)
    return TestsToolOutput(result=merged, rows=())


@tag_tool(domain="ingestion", target=TESTS_INGEST_TARGET_NAME)
def t__tests_ingest__run(
    env: BuildEnv,
    catalog: DagCatalog,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> TestsToolOutput:
    """Execute pytest report ingestion into analytics tables.

    Returns
    -------
    TestsToolOutput
        Tool output containing test catalog rows.
    """
    failure, warnings = _modules_precheck(t__modules, module_records)
    if failure is not None:
        return TestsToolOutput(result=failure)

    context = ToolRunContext(
        env=env,
        catalog=catalog,
        target_name=TESTS_INGEST_TARGET_NAME,
    )

    def _execute() -> TestsToolOutput:
        report_path = _resolve_report_file(env)
        if report_path is None:
            log.info("No pytest report found, writing empty test rows")
            result = ExecutionResult.ok(warnings=warnings)
            return TestsToolOutput(result=result, rows=())

        step = TestsIngestStep()
        ingest_result = step.execute(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            json_report_path=report_path,
        )
        return TestsToolOutput(
            result=_merge_result_warnings(ingest_result.result, warnings),
            rows=ingest_result.rows,
        )

    output = run_tool_step(context=context, run=_execute)
    return _coerce_tests_output(output, warnings)


@tag_compute(domain="ingestion", target=TESTS_INGEST_TARGET_NAME)
def t__tests_ingest__ingest(
    t__tests_ingest__run: TestsToolOutput,
) -> IngestStep[dict[str, tuple[tuple[object, ...], ...]]]:
    """Package tests ingestion rows for table materialization.

    Returns
    -------
    IngestStep[dict[str, tuple[tuple[object, ...], ...]]]
        Ingest result with table row payloads.
    """
    result = t__tests_ingest__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "Tests ingest skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "Tests ingest failed",
                warnings=result.warnings,
            )
        )

    payload = {TEST_CATALOG_TABLE_KEY: t__tests_ingest__run.rows}
    table_counts = {TEST_CATALOG_TABLE_KEY: len(t__tests_ingest__run.rows)}
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


@save_rows(
    context=TESTS_INGEST_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=TEST_CATALOG_TABLE_KEY),
)
@tag_compute(domain="ingestion", target=TESTS_INGEST_TARGET_NAME, target_="tests__rows")
def tests__rows(
    t__tests_ingest__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.test_catalog.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the test_catalog table, or None when ingestion is skipped or failed.

    Raises
    ------
    ValueError
        If the ingest payload is missing expected row data.
    """
    if t__tests_ingest__ingest.result.skipped or not t__tests_ingest__ingest.result.success:
        return None

    payload = t__tests_ingest__ingest.payload
    if payload is None:
        msg = "Missing tests ingest payload"
        raise ValueError(msg)
    rows = payload.get(TEST_CATALOG_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {TEST_CATALOG_TABLE_KEY}"
        raise ValueError(msg)
    return rows


@tag_helper(domain="ingestion", target=TESTS_INGEST_TARGET_NAME)
def tests_ingest__table_materializations(
    m__analytics__test_catalog: MaterializationResult,
) -> dict[str, MaterializationResult]:
    """Collect tests ingest table materializations.

    Returns
    -------
    dict[str, MaterializationResult]
        Mapping of table keys to materialization results.
    """
    return {TEST_CATALOG_TABLE_KEY: m__analytics__test_catalog}


@tag_helper(domain="ingestion", target=TESTS_INGEST_TARGET_NAME)
def tests_ingest__finalize_context(
    env: BuildEnv,
    catalog: DagCatalog,
) -> ToolFinalizeContext:
    """Build finalization context for tests ingest.

    Returns
    -------
    ToolFinalizeContext
        Finalization context for tests ingest.
    """
    return ToolFinalizeContext(
        env=env,
        catalog=catalog,
        target_name=TESTS_INGEST_TARGET_NAME,
    )


@codeintel_target(domain="ingestion", target=TESTS_INGEST_TARGET_NAME)
def t__tests_ingest(
    tests_ingest__finalize_context: ToolFinalizeContext,
    t__tests_ingest__run: TestsToolOutput,
    t__tests_ingest__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
    tests_ingest__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Ingest test catalog from pytest.

    Returns
    -------
    TargetRunRecord
        Record describing the execution outcome.
    """
    return finalize_target_from_materializations(
        context=tests_ingest__finalize_context,
        tool_step=t__tests_ingest__run,
        ingest_step=t__tests_ingest__ingest,
        artifact_materializations=None,
        table_materializations=tests_ingest__table_materializations,
    )


# ---------------------------------------------------------------------------
# typing target
# ---------------------------------------------------------------------------


def _coerce_typing_output(
    output: ToolStepOutput,
    warnings: tuple[str, ...],
) -> TypingToolOutput:
    if isinstance(output, TypingToolOutput):
        if warnings:
            return TypingToolOutput(
                result=_merge_result_warnings(output.result, warnings),
                typedness_rows=output.typedness_rows,
                diagnostic_rows=output.diagnostic_rows,
            )
        return output
    merged = _merge_result_warnings(output.result, warnings)
    return TypingToolOutput(result=merged, typedness_rows=(), diagnostic_rows=())


@tag_tool(domain="ingestion", target=TYPING_TARGET_NAME)
def t__typing__run(
    env: BuildEnv,
    catalog: DagCatalog,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> TypingToolOutput:
    """Execute typing analysis and persist typedness + diagnostics tables.

    Returns
    -------
    TypingToolOutput
        Tool output containing typedness and diagnostics rows.
    """
    failure, warnings = _modules_precheck(t__modules, module_records)
    if failure is not None:
        return TypingToolOutput(result=failure)

    context = ToolRunContext(
        env=env,
        catalog=catalog,
        target_name=TYPING_TARGET_NAME,
    )

    def _execute() -> TypingToolOutput:
        if not module_records:
            result = ExecutionResult.ok(warnings=warnings)
            return TypingToolOutput(result=result, typedness_rows=(), diagnostic_rows=())

        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
        tools = ToolRunnerAdapter(env.providers.tool_service)
        step = TypingIngestStep(discovery=discovery, tools=tools)
        ingest_result = asyncio.run(
            step.execute_async(
                module_records,
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
                repo_root=str(env.snapshot.repo_root),
                run_diagnostics=True,
            )
        )
        return TypingToolOutput(
            result=_merge_result_warnings(ingest_result.result, warnings),
            typedness_rows=ingest_result.typedness_rows,
            diagnostic_rows=ingest_result.diagnostic_rows,
        )

    output = run_tool_step(context=context, run=_execute)
    return _coerce_typing_output(output, warnings)


@tag_compute(domain="ingestion", target=TYPING_TARGET_NAME)
def t__typing__ingest(
    t__typing__run: TypingToolOutput,
) -> IngestStep[dict[str, tuple[tuple[object, ...], ...]]]:
    """Package typing rows for table materialization.

    Returns
    -------
    IngestStep[dict[str, tuple[tuple[object, ...], ...]]]
        Ingest result with table row payloads.
    """
    result = t__typing__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "Typing ingest skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "Typing ingest failed",
                warnings=result.warnings,
            )
        )

    payload = {
        TYPEDNESS_TABLE_KEY: t__typing__run.typedness_rows,
        STATIC_DIAGNOSTICS_TABLE_KEY: t__typing__run.diagnostic_rows,
    }
    table_counts = {
        TYPEDNESS_TABLE_KEY: len(t__typing__run.typedness_rows),
        STATIC_DIAGNOSTICS_TABLE_KEY: len(t__typing__run.diagnostic_rows),
    }
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


@save_rows(
    context=TYPING_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=TYPEDNESS_TABLE_KEY),
)
@tag_compute(domain="ingestion", target=TYPING_TARGET_NAME, target_="typing__typedness_rows")
def typing__typedness_rows(
    t__typing__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.typedness.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the typedness table, or None when ingestion is skipped or failed.

    Raises
    ------
    ValueError
        If the ingest payload is missing expected row data.
    """
    if t__typing__ingest.result.skipped or not t__typing__ingest.result.success:
        return None

    payload = t__typing__ingest.payload
    if payload is None:
        msg = "Missing typing ingest payload"
        raise ValueError(msg)
    rows = payload.get(TYPEDNESS_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {TYPEDNESS_TABLE_KEY}"
        raise ValueError(msg)
    return rows


@save_rows(
    context=TYPING_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=STATIC_DIAGNOSTICS_TABLE_KEY),
)
@tag_compute(domain="ingestion", target=TYPING_TARGET_NAME, target_="typing__diagnostic_rows")
def typing__diagnostic_rows(
    t__typing__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.static_diagnostics.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the static_diagnostics table, or None when ingestion is skipped or failed.

    Raises
    ------
    ValueError
        If the ingest payload is missing expected row data.
    """
    if t__typing__ingest.result.skipped or not t__typing__ingest.result.success:
        return None

    payload = t__typing__ingest.payload
    if payload is None:
        msg = "Missing typing ingest payload"
        raise ValueError(msg)
    rows = payload.get(STATIC_DIAGNOSTICS_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {STATIC_DIAGNOSTICS_TABLE_KEY}"
        raise ValueError(msg)
    return rows


@tag_helper(domain="ingestion", target=TYPING_TARGET_NAME)
def typing__table_materializations(
    m__analytics__typedness: MaterializationResult,
    m__analytics__static_diagnostics: MaterializationResult,
) -> dict[str, MaterializationResult]:
    """Collect typing table materializations.

    Returns
    -------
    dict[str, MaterializationResult]
        Mapping of table keys to materialization results.
    """
    return {
        TYPEDNESS_TABLE_KEY: m__analytics__typedness,
        STATIC_DIAGNOSTICS_TABLE_KEY: m__analytics__static_diagnostics,
    }


@tag_helper(domain="ingestion", target=TYPING_TARGET_NAME)
def typing__finalize_context(
    env: BuildEnv,
    catalog: DagCatalog,
) -> ToolFinalizeContext:
    """Build finalization context for typing ingest.

    Returns
    -------
    ToolFinalizeContext
        Finalization context for typing ingest.
    """
    return ToolFinalizeContext(
        env=env,
        catalog=catalog,
        target_name=TYPING_TARGET_NAME,
    )


@codeintel_target(
    domain="ingestion",
    target=TYPING_TARGET_NAME,
    spec=TargetSpecDescriptor(
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
)
def t__typing(
    typing__finalize_context: ToolFinalizeContext,
    t__typing__run: TypingToolOutput,
    t__typing__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
    typing__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Analyze type annotations and static diagnostics.

    Returns
    -------
    TargetRunRecord
        Record describing the execution outcome.
    """
    return finalize_target_from_materializations(
        context=typing__finalize_context,
        tool_step=t__typing__run,
        ingest_step=t__typing__ingest,
        artifact_materializations=None,
        table_materializations=typing__table_materializations,
    )


__all__: list[str] = [
    "ConfigScanResult",
    "ConfigToolOutput",
    "CoverageToolOutput",
    "ModuleToolOutput",
    "TestsToolOutput",
    "TypingToolOutput",
    "t__config_ingest",
    "t__config_ingest__ingest",
    "t__config_ingest__run",
    "t__config_ingest__scan",
    "t__coverage_ingest",
    "t__coverage_ingest__ingest",
    "t__coverage_ingest__run",
    "t__modules",
    "t__modules__ingest",
    "t__modules__run",
    "t__tests_ingest",
    "t__tests_ingest__ingest",
    "t__tests_ingest__run",
    "t__typing",
    "t__typing__ingest",
    "t__typing__run",
]
