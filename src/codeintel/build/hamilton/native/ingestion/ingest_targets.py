"""Native Hamilton ingestion targets.

This module consolidates ingestion-domain targets that share similar execution
patterns (tool invocations + table writes) and are frequently evolved together:

- ``modules``: Scan repository modules and write ``core.repo_map``.
- ``config_ingest``: Discover and ingest configuration files.
- ``tests_ingest``: Ingest pytest report data into analytics tables.
- ``typing``: Run typing diagnostics and persist diagnostics.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
from hamilton.function_modifiers import (
    apply_to,
    cache,
    parameterize,
    resolve_from_config,
    source,
    value,
)
from hamilton.function_modifiers.base import NodeTransformLifecycle
from hamilton.htypes import Collect, Parallelizable

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
from codeintel.build.hamilton.native.ingestion.pipelines import (
    mutate_ingest_rows,
    pipe_ingest_rows,
)
from codeintel.build.hamilton.native.options.ingestion import ModuleIngestOptions
from codeintel.build.hamilton.native.patterns import (
    IngestStep,
    RelationTableSaveSpec,
    TableTargetSpec,
    TableTargetTableSpec,
    ToolFinalizeContext,
    ToolRunContext,
    attach_table_target_template,
    finalize_target_from_materializations,
    run_tool_step,
)
from codeintel.build.hamilton.native.target_decorators import (
    TargetSpecDescriptor,
    codeintel_target,
)
from codeintel.build.hamilton.native.tool_results import ToolStepOutput
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_compute, tag_helper, tag_tool
from codeintel.build.hamilton.transforms.ingestion_normalize import normalize_ingest_frame
from codeintel.build.hamilton.transforms.registry_inject import inject_from_registry
from codeintel.build.hashing import compute_options_hash
from codeintel.build.resources import TOOL_EXECUTION, TargetResources
from codeintel.build.tabular.arrow_ops import dedupe_table_for_table
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import (
    ColumnarRows,
    columnar_buffer_for_table_key,
    columnar_row_count,
    empty_table_for_table,
)
from codeintel.core.paths import normalize_path
from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
)
from codeintel.ingestion.adapters.tool_runner import ToolRunnerAdapter
from codeintel.ingestion.compute.config_ingest import ConfigIngestStep
from codeintel.ingestion.compute.repo_scan import RepoScanStep
from codeintel.ingestion.compute.tests_ingest import TestsIngestStep
from codeintel.ingestion.compute.typing_ingest import TypingIngestStep
from codeintel.ingestion.infrastructure.scanning import default_config_profile
from codeintel.ingestion.ports.change_detection import ChangeSet, FileDigest
from codeintel.ingestion.ports.discovery import ModuleRecord

if TYPE_CHECKING:
    from codeintel.ingestion.ports.change_detection import ChangeRequest

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
TESTS_INGEST_TARGET_NAME = "tests_ingest"
TYPING_TARGET_NAME = "typing"

MODULES_TABLE_KEY = "core.modules"
FILE_STATE_TABLE_KEY = "core.file_state"
REPO_MAP_TABLE_KEY = "core.repo_map"
MODULES_TABLE_KEYS = (MODULES_TABLE_KEY, FILE_STATE_TABLE_KEY, REPO_MAP_TABLE_KEY)

CONFIG_VALUES_TABLE_KEY = "analytics.config_values"
TEST_CATALOG_TABLE_KEY = "analytics.test_catalog"
STATIC_DIAGNOSTICS_TABLE_KEY = "analytics.static_diagnostics"

_MODULE = sys.modules[__name__]


class _NoopChangeDetectionAdapter:
    def compute_changes(
        self,
        request: ChangeRequest,
        current_modules: Sequence[ModuleRecord],
    ) -> ChangeSet:
        state: dict[str, FileDigest] = {}
        for module in current_modules:
            digest = self.compute_file_digest(module.file_path)
            if digest is None:
                continue
            state[normalize_path(module.rel_path)] = digest
        return ChangeSet(
            added=list(current_modules),
            modified=[],
            deleted=[],
            state_hash=HashChangeDetectionAdapter.compute_state_hash(state),
            state_rows=self._build_state_rows(
                repo=request.repo,
                commit=request.commit,
                language=request.language,
                state=state,
            ),
        )

    def load_previous_state(self, repo: str, language: str) -> dict[str, FileDigest]:
        _ = (self, repo, language)
        return {}

    def save_current_state(
        self,
        repo: str,
        commit: str,
        language: str,
        state: Mapping[str, FileDigest],
    ) -> None:
        _ = (self, repo, commit, language, state)

    def compute_file_digest(self, path: Path) -> FileDigest | None:
        _ = self
        return HashChangeDetectionAdapter.compute_file_digest(path)

    def _build_state_rows(
        self,
        *,
        repo: str,
        commit: str,
        language: str,
        state: Mapping[str, FileDigest],
    ) -> ColumnarRows:
        _ = self
        if not state:
            return {}
        buffer = columnar_buffer_for_table_key(FILE_STATE_TABLE_KEY)
        for rel_path, digest in sorted(state.items()):
            buffer.append(
                {
                    "repo": repo,
                    "commit": commit,
                    "rel_path": rel_path,
                    "language": language,
                    "size_bytes": digest.size_bytes,
                    "mtime_ns": digest.mtime_ns,
                    "content_hash": digest.content_hash,
                }
            )
        return buffer.data


@dataclass(frozen=True)
class ModuleToolOutput(ToolStepOutput):
    """Tool step output for repository module scanning."""

    modules: tuple[ModuleRecord, ...] = field(default_factory=tuple)
    change_set: ChangeSet | None = None
    file_state_hash: str | None = None
    module_rows: pa.Table = field(default_factory=lambda: empty_table_for_table(MODULES_TABLE_KEY))
    file_state_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(FILE_STATE_TABLE_KEY)
    )
    repo_map_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(REPO_MAP_TABLE_KEY)
    )
    module_row_count: int = 0
    file_state_row_count: int = 0
    repo_map_row_count: int = 0


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

    rows: pa.Table = field(default_factory=lambda: empty_table_for_table(CONFIG_VALUES_TABLE_KEY))
    row_count: int = 0


@dataclass(frozen=True)
class TestsToolOutput(ToolStepOutput):
    """Tool step output for tests ingestion."""

    rows: pa.Table = field(default_factory=lambda: empty_table_for_table(TEST_CATALOG_TABLE_KEY))
    row_count: int = 0


@dataclass(frozen=True)
class TypingToolOutput(ToolStepOutput):
    """Tool step output for typing ingestion."""

    diagnostic_rows: pa.Table = field(
        default_factory=lambda: empty_table_for_table(STATIC_DIAGNOSTICS_TABLE_KEY)
    )
    diagnostic_row_count: int = 0


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
    paths = get_module_paths_from_env(env)
    if paths:
        if t__modules.status != "succeeded":
            log.warning(
                "Using stored module paths despite modules target %s: %s",
                t__modules.status,
                t__modules.error or "no error recorded",
            )
        return tuple(paths)
    try:
        options = load_target_options(
            env,
            target_name=MODULES_TARGET_NAME,
            options_type=ModuleIngestOptions,
        )
        profile = build_scan_profile(env.snapshot.repo_root, options)
        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
        discovered = discovery.discover_modules(env.snapshot.repo_root, profile)
        filtered = filter_modules(discovered, options)
        fallback = sorted({module.rel_path for module in filtered})
        if fallback:
            if t__modules.status != "succeeded":
                log.warning(
                    "Falling back to filesystem module scan after modules target %s: %s",
                    t__modules.status,
                    t__modules.error or "unknown error",
                )
            else:
                log.warning("Falling back to filesystem module scan after empty storage query.")
            return tuple(fallback)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        log.warning("Module path fallback scan failed: %s", exc)
    return tuple(paths)


@tag_helper(domain="ingestion")
def module_records_static(
    env: BuildEnv,
    module_paths: tuple[str, ...],
) -> tuple[ModuleRecord, ...]:
    """Convert module paths into ModuleRecord objects (static execution).

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


@tag_helper(domain="ingestion")
def module_record_inputs(
    module_paths: tuple[str, ...],
) -> Parallelizable[tuple[int, int, str]]:
    """Yield module record inputs for dynamic execution.

    Parameters
    ----------
    module_paths
        Module paths loaded from storage.

    Yields
    ------
    tuple[int, int, str]
        Tuple of (index, total, relative_path).
    """
    total = len(module_paths)
    for index, path in enumerate(module_paths, start=1):
        yield (index, total, path)


@tag_helper(domain="ingestion")
def module_record(
    env: BuildEnv,
    module_record_inputs: tuple[int, int, str],
) -> ModuleRecord:
    """Build a ModuleRecord from dynamic inputs.

    Parameters
    ----------
    env
        Build environment with snapshot metadata.
    module_record_inputs
        Tuple of (index, total, relative_path).

    Returns
    -------
    ModuleRecord
        Module record for downstream ingestion steps.
    """
    index, total, rel_path = module_record_inputs
    return ModuleRecord(
        rel_path=rel_path,
        module_name=rel_path.replace("/", ".").removesuffix(".py"),
        file_path=env.snapshot.repo_root / rel_path,
        index=index,
        total=total,
    )


@tag_helper(domain="ingestion")
def module_records_dynamic(
    module_record: Collect[ModuleRecord],
) -> tuple[ModuleRecord, ...]:
    """Collect dynamic ModuleRecord outputs into a stable tuple.

    Parameters
    ----------
    module_record
        Collected ModuleRecord values from dynamic execution.

    Returns
    -------
    tuple[ModuleRecord, ...]
        Module records for downstream ingestion steps.
    """
    records = list(module_record)
    if not records:
        return ()
    ordered = sorted(records, key=lambda record: record.index)
    return tuple(ordered)


def _pick_module_records(
    *,
    ci_dynamic_module_records: bool = False,
) -> NodeTransformLifecycle:
    if ci_dynamic_module_records:
        return inject_from_registry(
            param_name="records",
            node_name="module_records_dynamic",
        )
    return inject_from_registry(
        param_name="records",
        node_name="module_records_static",
    )


@resolve_from_config(decorate_with=_pick_module_records)
@tag_helper(domain="ingestion")
def module_records(records: tuple[ModuleRecord, ...]) -> tuple[ModuleRecord, ...]:
    """Return module records from the configured execution path.

    Parameters
    ----------
    records
        Module records from either static or dynamic execution.

    Returns
    -------
    tuple[ModuleRecord, ...]
        Module records for downstream ingestion steps.
    """
    return records


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
        if env.gateway is None:
            change_detection = _NoopChangeDetectionAdapter()
        else:
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

        module_rows = scan_result.module_rows_reader
        file_state_rows = scan_result.file_state_rows_reader
        repo_map_rows = scan_result.repo_map_rows_reader
        return ModuleToolOutput(
            result=ExecutionResult.ok(),
            modules=scan_result.modules,
            change_set=scan_result.change_set,
            file_state_hash=scan_result.change_set.state_hash,
            module_rows=module_rows,
            file_state_rows=file_state_rows,
            repo_map_rows=repo_map_rows,
            module_row_count=columnar_row_count(scan_result.module_rows),
            file_state_row_count=columnar_row_count(scan_result.file_state_rows),
            repo_map_row_count=columnar_row_count(scan_result.repo_map_rows),
        )

    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        log.exception("Module scan failed")
        return ModuleToolOutput(result=ExecutionResult.failed(str(exc)))


@tag_compute(domain="ingestion", target=MODULES_TARGET_NAME)
def t__modules__ingest(
    t__modules__run: ModuleToolOutput,
) -> IngestStep[dict[str, InferableTabularInput]]:
    """Package module scan rows for table materialization.

    Returns
    -------
    IngestStep[dict[str, InferableTabularInput]]
        Ingest result with table inputs.
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

    module_table = tabular_to_arrow_table(t__modules__run.module_rows)
    module_table = dedupe_table_for_table(MODULES_TABLE_KEY, module_table)
    module_rows = module_table
    file_state_table = tabular_to_arrow_table(t__modules__run.file_state_rows)
    file_state_table = dedupe_table_for_table(
        FILE_STATE_TABLE_KEY,
        file_state_table,
        prefer_columns=("mtime_ns", "content_hash"),
    )
    file_state_rows = file_state_table
    repo_map_rows = t__modules__run.repo_map_rows
    payload = {
        MODULES_TABLE_KEY: module_rows,
        FILE_STATE_TABLE_KEY: file_state_rows,
        REPO_MAP_TABLE_KEY: repo_map_rows,
    }
    table_counts = {
        MODULES_TABLE_KEY: t__modules__run.module_row_count,
        FILE_STATE_TABLE_KEY: t__modules__run.file_state_row_count,
        REPO_MAP_TABLE_KEY: t__modules__run.repo_map_row_count,
    }
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


def modules__module_rows__base(
    t__modules__ingest: IngestStep[dict[str, InferableTabularInput]],
) -> InferableTabularInput | None:
    """Extract rows for core.modules.

    Returns
    -------
    InferableTabularInput | None
        Tabular input for the modules table, or None when ingestion skipped or failed.

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
    frame = payload.get(MODULES_TABLE_KEY)
    if frame is None:
        msg = f"Missing frame for {MODULES_TABLE_KEY}"
        raise ValueError(msg)
    return frame


def modules__file_state_rows__base(
    t__modules__ingest: IngestStep[dict[str, InferableTabularInput]],
) -> InferableTabularInput | None:
    """Extract rows for core.file_state.

    Returns
    -------
    InferableTabularInput | None
        Tabular input for the file_state table, or None when ingestion skipped or failed.

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
    frame = payload.get(FILE_STATE_TABLE_KEY)
    if frame is None:
        msg = f"Missing frame for {FILE_STATE_TABLE_KEY}"
        raise ValueError(msg)
    return frame


def modules__repo_map_rows__base(
    t__modules__ingest: IngestStep[dict[str, InferableTabularInput]],
) -> InferableTabularInput | None:
    """Extract rows for core.repo_map.

    Returns
    -------
    InferableTabularInput | None
        Tabular input for the repo_map table, or None when ingestion skipped or failed.

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
    frame = payload.get(REPO_MAP_TABLE_KEY)
    if frame is None:
        msg = f"Missing frame for {REPO_MAP_TABLE_KEY}"
        raise ValueError(msg)
    return frame


_MODULES_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="ingestion",
    target_name=MODULES_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=MODULES_TABLE_KEY,
            base_node="modules__module_rows__base",
            save_spec=RelationTableSaveSpec(table_key=MODULES_TABLE_KEY),
            node_name="modules__module_rows",
            input_type=InferableTabularInput | None,
        ),
        TableTargetTableSpec(
            table_key=FILE_STATE_TABLE_KEY,
            base_node="modules__file_state_rows__base",
            save_spec=RelationTableSaveSpec(table_key=FILE_STATE_TABLE_KEY),
            node_name="modules__file_state_rows",
            input_type=InferableTabularInput | None,
        ),
        TableTargetTableSpec(
            table_key=REPO_MAP_TABLE_KEY,
            base_node="modules__repo_map_rows__base",
            save_spec=RelationTableSaveSpec(table_key=REPO_MAP_TABLE_KEY),
            node_name="modules__repo_map_rows",
            input_type=InferableTabularInput | None,
        ),
    ),
    table_materializations_node="modules__table_materializations",
    attach_anchor=False,
)
attach_table_target_template(_MODULE, spec=_MODULES_TABLE_TARGET_SPEC)
modules__module_rows = _MODULE.modules__module_rows
modules__file_state_rows = _MODULE.modules__file_state_rows
modules__repo_map_rows = _MODULE.modules__repo_map_rows
modules__table_materializations = _MODULE.modules__table_materializations


@cache(behavior="ignore")
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
    t__modules__ingest: IngestStep[dict[str, InferableTabularInput]],
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
            return ConfigToolOutput(
                result=ExecutionResult.ok(),
                rows=empty_table_for_table(CONFIG_VALUES_TABLE_KEY),
                row_count=0,
            )
        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
        step = ConfigIngestStep(discovery=discovery)
        ingest_result = step.execute(
            config_files,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
        return ConfigToolOutput(
            result=ingest_result.result,
            rows=ingest_result.rows_reader,
            row_count=ingest_result.row_count,
        )

    output = run_tool_step(context=context, run=_execute)
    if isinstance(output, ConfigToolOutput):
        return output
    return ConfigToolOutput(
        result=output.result,
        rows=empty_table_for_table(CONFIG_VALUES_TABLE_KEY),
        row_count=0,
    )


@tag_compute(domain="ingestion", target=CONFIG_INGEST_TARGET_NAME)
def t__config_ingest__ingest(
    t__config_ingest__run: ConfigToolOutput,
) -> IngestStep[dict[str, InferableTabularInput]]:
    """Package config ingestion rows for table materialization.

    Returns
    -------
    IngestStep[dict[str, InferableTabularInput]]
        Ingest result with table inputs.
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
    table_counts = {CONFIG_VALUES_TABLE_KEY: t__config_ingest__run.row_count}
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


@parameterize(
    config_ingest__raw_rows={
        "ingest_step": source("t__config_ingest__ingest"),
        "table_key": value(CONFIG_VALUES_TABLE_KEY),
        "label": value("config ingest"),
    },
    tests__raw_rows={
        "ingest_step": source("t__tests_ingest__ingest"),
        "table_key": value(TEST_CATALOG_TABLE_KEY),
        "label": value("tests ingest"),
    },
)
def extract_ingest_rows(
    ingest_step: IngestStep[dict[str, InferableTabularInput]],
    table_key: str,
    label: str,
) -> InferableTabularInput | None:
    """Extract raw rows for ingestion tables.

    Parameters
    ----------
    ingest_step
        Ingestion step containing table payloads.
    table_key
        Table key to extract from the payload.
    label
        Human-readable label for error messages.

    Returns
    -------
    InferableTabularInput | None
        Extracted tabular input or None when the ingest step skipped/failed.

    Raises
    ------
    ValueError
        If the ingest payload or table frame is missing.
    """
    if ingest_step.result.skipped or not ingest_step.result.success:
        return None

    payload = ingest_step.payload
    if payload is None:
        msg = f"Missing {label} payload"
        raise ValueError(msg)
    frame = payload.get(table_key)
    if frame is None:
        msg = f"Missing frame for {table_key}"
        raise ValueError(msg)
    return frame


@pipe_ingest_rows(
    required_cols=("repo", "commit", "config_path", "format", "key"),
    input_name="config_ingest__raw_rows",
)
def config_ingest__rows__base(
    config_ingest__raw_rows: InferableTabularInput | None,
) -> InferableTabularInput:
    """Return cleaned rows for analytics.config_values.

    Returns
    -------
    InferableTabularInput
        Cleaned tabular input for the config values table.
    """
    if config_ingest__raw_rows is None:
        return empty_table_for_table(CONFIG_VALUES_TABLE_KEY)
    return config_ingest__raw_rows


_CONFIG_INGEST_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="ingestion",
    target_name=CONFIG_INGEST_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=CONFIG_VALUES_TABLE_KEY,
            base_node="config_ingest__rows__base",
            save_spec=RelationTableSaveSpec(table_key=CONFIG_VALUES_TABLE_KEY),
            node_name="config_ingest__rows",
            input_type=InferableTabularInput,
        ),
    ),
    table_materializations_node="config_ingest__table_materializations",
    attach_anchor=False,
)
attach_table_target_template(_MODULE, spec=_CONFIG_INGEST_TABLE_TARGET_SPEC)
config_ingest__rows = _MODULE.config_ingest__rows
config_ingest__table_materializations = _MODULE.config_ingest__table_materializations


@cache(behavior="ignore")
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
    t__config_ingest__ingest: IngestStep[dict[str, InferableTabularInput]],
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
                row_count=output.row_count,
            )
        return output
    merged = _merge_result_warnings(output.result, warnings)
    return TestsToolOutput(
        result=merged,
        rows=empty_table_for_table(TEST_CATALOG_TABLE_KEY),
        row_count=0,
    )


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
            return TestsToolOutput(
                result=result,
                rows=empty_table_for_table(TEST_CATALOG_TABLE_KEY),
                row_count=0,
            )

        step = TestsIngestStep()
        ingest_result = step.execute(
            module_records,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            json_report_path=report_path,
        )
        return TestsToolOutput(
            result=_merge_result_warnings(ingest_result.result, warnings),
            rows=ingest_result.rows_reader,
            row_count=ingest_result.row_count,
        )

    output = run_tool_step(context=context, run=_execute)
    return _coerce_tests_output(output, warnings)


@tag_compute(domain="ingestion", target=TESTS_INGEST_TARGET_NAME)
def t__tests_ingest__ingest(
    t__tests_ingest__run: TestsToolOutput,
) -> IngestStep[dict[str, InferableTabularInput]]:
    """Package tests ingestion rows for table materialization.

    Returns
    -------
    IngestStep[dict[str, InferableTabularInput]]
        Ingest result with table inputs.
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
    table_counts = {TEST_CATALOG_TABLE_KEY: t__tests_ingest__run.row_count}
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


def tests__rows__base(
    tests__raw_rows: InferableTabularInput | None,
) -> InferableTabularInput | None:
    """Extract rows for analytics.test_catalog.

    Returns
    -------
    InferableTabularInput | None
        Tabular input for the test_catalog table, or None when ingestion is skipped or failed.
    """
    return tests__raw_rows


_TESTS_INGEST_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="ingestion",
    target_name=TESTS_INGEST_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=TEST_CATALOG_TABLE_KEY,
            base_node="tests__rows__base",
            save_spec=RelationTableSaveSpec(table_key=TEST_CATALOG_TABLE_KEY),
            node_name="tests__rows",
            input_type=InferableTabularInput | None,
        ),
    ),
    table_materializations_node="tests_ingest__table_materializations",
    attach_anchor=False,
)
attach_table_target_template(_MODULE, spec=_TESTS_INGEST_TABLE_TARGET_SPEC)
tests__rows = _MODULE.tests__rows
tests_ingest__table_materializations = _MODULE.tests_ingest__table_materializations


@cache(behavior="ignore")
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
    t__tests_ingest__ingest: IngestStep[dict[str, InferableTabularInput]],
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
                diagnostic_rows=output.diagnostic_rows,
                diagnostic_row_count=output.diagnostic_row_count,
            )
        return output
    merged = _merge_result_warnings(output.result, warnings)
    return TypingToolOutput(
        result=merged,
        diagnostic_rows=empty_table_for_table(STATIC_DIAGNOSTICS_TABLE_KEY),
        diagnostic_row_count=0,
    )


@tag_tool(domain="ingestion", target=TYPING_TARGET_NAME)
def t__typing__run(
    env: BuildEnv,
    catalog: DagCatalog,
    t__modules: TargetRunRecord,
    module_records: tuple[ModuleRecord, ...],
) -> TypingToolOutput:
    """Execute typing analysis and persist diagnostics tables.

    Returns
    -------
    TypingToolOutput
        Tool output containing diagnostics rows.
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
            return TypingToolOutput(
                result=result,
                diagnostic_rows=empty_table_for_table(STATIC_DIAGNOSTICS_TABLE_KEY),
                diagnostic_row_count=0,
            )

        tools = ToolRunnerAdapter(env.providers.tool_service)
        step = TypingIngestStep(tools=tools)
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
            diagnostic_rows=ingest_result.diagnostic_rows_reader,
            diagnostic_row_count=ingest_result.diagnostic_row_count,
        )

    output = run_tool_step(context=context, run=_execute)
    return _coerce_typing_output(output, warnings)


@tag_compute(domain="ingestion", target=TYPING_TARGET_NAME)
def t__typing__ingest(
    t__typing__run: TypingToolOutput,
) -> IngestStep[dict[str, InferableTabularInput]]:
    """Package typing rows for table materialization.

    Returns
    -------
    IngestStep[dict[str, InferableTabularInput]]
        Ingest result with table inputs.
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
        STATIC_DIAGNOSTICS_TABLE_KEY: t__typing__run.diagnostic_rows,
    }
    table_counts = {
        STATIC_DIAGNOSTICS_TABLE_KEY: t__typing__run.diagnostic_row_count,
    }
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


def typing__diagnostic_rows__base(
    t__typing__ingest: IngestStep[dict[str, InferableTabularInput]],
) -> InferableTabularInput | None:
    """Extract rows for analytics.static_diagnostics.

    Returns
    -------
    InferableTabularInput | None
        Tabular input for the static_diagnostics table, or None when ingestion is skipped or failed.

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
    frame = payload.get(STATIC_DIAGNOSTICS_TABLE_KEY)
    if frame is None:
        msg = f"Missing frame for {STATIC_DIAGNOSTICS_TABLE_KEY}"
        raise ValueError(msg)
    return frame


_TYPING_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="ingestion",
    target_name=TYPING_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=STATIC_DIAGNOSTICS_TABLE_KEY,
            base_node="typing__diagnostic_rows__base",
            save_spec=RelationTableSaveSpec(table_key=STATIC_DIAGNOSTICS_TABLE_KEY),
            node_name="typing__diagnostic_rows",
            input_type=InferableTabularInput | None,
        ),
    ),
    table_materializations_node="typing__table_materializations",
    attach_anchor=False,
)
attach_table_target_template(_MODULE, spec=_TYPING_TABLE_TARGET_SPEC)
typing__diagnostic_rows = _MODULE.typing__diagnostic_rows
typing__table_materializations = _MODULE.typing__table_materializations


@cache(behavior="ignore")
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
    t__typing__ingest: IngestStep[dict[str, InferableTabularInput]],
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


@tag_helper(domain="ingestion")
@mutate_ingest_rows(
    apply_to(modules__module_rows, table_key=value(MODULES_TABLE_KEY)),
    apply_to(modules__file_state_rows, table_key=value(FILE_STATE_TABLE_KEY)),
    apply_to(modules__repo_map_rows, table_key=value(REPO_MAP_TABLE_KEY)),
    apply_to(tests__rows, table_key=value(TEST_CATALOG_TABLE_KEY)),
    apply_to(typing__diagnostic_rows, table_key=value(STATIC_DIAGNOSTICS_TABLE_KEY)),
)
def _normalize_ingest_rows(
    rows: InferableTabularInput | None,
    table_key: str,
) -> pa.Table | None:
    """Normalize ingestion outputs with shared alignment/dedupe logic.

    Parameters
    ----------
    rows
        Ingestion rows to normalize.
    table_key
        Table key used for schema alignment.

    Returns
    -------
    pa.Table | None
        Normalized rows for the table.
    """
    return normalize_ingest_frame(rows, table_key=table_key)


@tag_helper(domain="ingestion")
@mutate_ingest_rows(
    apply_to(config_ingest__rows, table_key=value(CONFIG_VALUES_TABLE_KEY)),
)
def _normalize_required_ingest_rows(
    rows: InferableTabularInput,
    table_key: str,
) -> pa.Table:
    """Normalize required ingestion outputs with shared alignment/dedupe logic.

    Parameters
    ----------
    rows
        Ingestion rows to normalize.
    table_key
        Table key used for schema alignment.

    Returns
    -------
    pa.Table
        Normalized rows for the table.
    """
    normalized = normalize_ingest_frame(rows, table_key=table_key)
    if normalized is None:
        return empty_table_for_table(table_key)
    return normalized


__all__: list[str] = [
    "ConfigScanResult",
    "ConfigToolOutput",
    "ModuleToolOutput",
    "TestsToolOutput",
    "TypingToolOutput",
    "t__config_ingest",
    "t__config_ingest__ingest",
    "t__config_ingest__run",
    "t__config_ingest__scan",
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
