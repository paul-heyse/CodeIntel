"""Native Hamilton implementation for scip target.

This target produces the SCIP artifacts declared in the build contract:
- ``scip_index``: ``{scip_dir}/index.scip``

Tool execution is delegated to the ingestion tool runtime (ToolRunner),
while persistence is DAG-visible via ``FileArtifactSaver``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hamilton.node as h_node
from hamilton.function_modifiers import parameterize_values, source, value
from hamilton.function_modifiers.base import NodeTransformer

from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.materializers import DuckDBRowsSaver, FileArtifactSaver
from codeintel.build.hamilton.materializers.metadata import DuckDBMaterializationMetadata
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.ingestion.ingest_targets import ModuleScanResult
from codeintel.build.hamilton.native.materialization_records import (
    record_from_file_artifact_materializations,
)
from codeintel.build.hamilton.native.options.ingestion import ScipIngestOptions
from codeintel.build.hamilton.native.table_counts import normalize_table_counts
from codeintel.build.hamilton.native.target_decorators import (
    TargetSpecDescriptor,
    codeintel_target,
)
from codeintel.build.hamilton.native.tool_results import ToolStepOutput
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_helper, tag_tool
from codeintel.build.hashing import InputHashOptions, compute_options_hash
from codeintel.build.resources import TOOL_EXECUTION, TargetResources
from codeintel.build.schemas import deferred_columns_for_table_key
from codeintel.build.targets import TargetGraph
from codeintel.core.errors import CodeIntelStorageError, ColumnNotFoundError, TableNotFoundError
from codeintel.core.schemas.row_serialization import row_serializer_for_table_key
from codeintel.core.tools import ToolName
from codeintel.ingestion.engine.infrastructure import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolRunOptions,
)
from codeintel.ingestion.ports.change_detection import ChangeSet
from codeintel.ingestion.scip import (
    ScipParsedIndex,
    ScipRowContext,
    build_diagnostic_rows,
    build_external_symbol_rows,
    build_module_state_rows,
    build_occurrence_rows,
    build_symbol_information_rows,
    build_symbol_relationship_rows,
    build_symbol_rows,
    parse_index,
)
from codeintel.ingestion.scip.incremental import (
    ScipIncrementalConfig,
    update_index_incremental,
)
from codeintel.ingestion.scip.manifest import (
    ScipShardManifest,
    ScipShardRecord,
    load_manifest,
    manifest_from_state_rows,
    manifest_path,
    write_manifest,
)
from codeintel.storage.io import IbisIOConfig, load_table_as_dataframe

if TYPE_CHECKING:
    import pandas as pd

    from codeintel.config.models import ToolsConfig

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, Path, TargetRunRecord)

SCIP_TARGET_NAME = "scip"
SCIP_ARTIFACT_INDEX = "scip_index"

SCIP_SYMBOLS_TABLE_KEY = "core.scip_symbols"
SCIP_OCCURRENCES_TABLE_KEY = "core.scip_occurrences"
SCIP_SYMBOL_INFO_TABLE_KEY = "core.scip_symbol_information"
SCIP_RELATIONSHIPS_TABLE_KEY = "core.scip_symbol_relationships"
SCIP_DIAGNOSTICS_TABLE_KEY = "core.scip_diagnostics"
SCIP_EXTERNAL_SYMBOLS_TABLE_KEY = "core.scip_external_symbols"
SCIP_MODULE_STATE_TABLE_KEY = "core.scip_module_state"
SCIP_TABLE_KEYS = (
    SCIP_SYMBOLS_TABLE_KEY,
    SCIP_OCCURRENCES_TABLE_KEY,
    SCIP_SYMBOL_INFO_TABLE_KEY,
    SCIP_RELATIONSHIPS_TABLE_KEY,
    SCIP_DIAGNOSTICS_TABLE_KEY,
    SCIP_EXTERNAL_SYMBOLS_TABLE_KEY,
    SCIP_MODULE_STATE_TABLE_KEY,
)


ScipRunResult = ToolStepOutput


@dataclass(frozen=True)
class ScipIngestResult:
    """Result from SCIP ingestion row preparation.

    Attributes
    ----------
    result
        Execution status for ingest step.
    symbol_rows
        Row tuples for core.scip_symbols.
    occurrence_rows
        Row tuples for core.scip_occurrences.
    symbol_info_rows
        Row tuples for core.scip_symbol_information.
    relationship_rows
        Row tuples for core.scip_symbol_relationships.
    diagnostic_rows
        Row tuples for core.scip_diagnostics.
    external_symbol_rows
        Row tuples for core.scip_external_symbols.
    """

    result: ExecutionResult
    symbol_rows: tuple[tuple[object, ...], ...] = ()
    occurrence_rows: tuple[tuple[object, ...], ...] = ()
    symbol_info_rows: tuple[tuple[object, ...], ...] = ()
    relationship_rows: tuple[tuple[object, ...], ...] = ()
    diagnostic_rows: tuple[tuple[object, ...], ...] = ()
    external_symbol_rows: tuple[tuple[object, ...], ...] = ()


@dataclass(frozen=True)
class ScipRowPayload:
    """Row payloads for SCIP ingestion tables."""

    symbol_rows: tuple[tuple[object, ...], ...]
    occurrence_rows: tuple[tuple[object, ...], ...]
    symbol_info_rows: tuple[tuple[object, ...], ...]
    relationship_rows: tuple[tuple[object, ...], ...]
    diagnostic_rows: tuple[tuple[object, ...], ...]
    external_symbol_rows: tuple[tuple[object, ...], ...]


@dataclass(frozen=True)
class ScipModuleInputs:
    """Module target inputs required for SCIP execution."""

    modules: TargetRunRecord
    scan: ModuleScanResult


@dataclass(frozen=True)
class ScipRunConfig:
    """Configuration inputs for SCIP execution."""

    proto_module_path: Path | None
    options: ScipIngestOptions
    hash_options: InputHashOptions


@dataclass(frozen=True)
class ScipIngestInputs:
    """Inputs required to build SCIP ingestion rows."""

    modules: TargetRunRecord
    run: ToolStepOutput
    proto_module_path: Path | None
    options: ScipIngestOptions


@dataclass(frozen=True)
class ScipTargetInputs:
    """Inputs required to finalize the SCIP target."""

    modules: TargetRunRecord
    hash_options: InputHashOptions


@dataclass(frozen=True)
class ScipMaterializationInputs:
    """Aggregated inputs for scip materialization."""

    run: ToolStepOutput
    ingest: ScipIngestResult
    artifact_materializations: dict[str, MaterializationMetadata]
    table_materializations: dict[str, MaterializationMetadata]


@dataclass(frozen=True)
class ScipSymbolTableMaterializations:
    """Materialization metadata for symbol-related tables."""

    symbols: MaterializationMetadata
    occurrences: MaterializationMetadata
    symbol_information: MaterializationMetadata


@dataclass(frozen=True)
class ScipAuxTableMaterializations:
    """Materialization metadata for auxiliary SCIP tables."""

    relationships: MaterializationMetadata
    diagnostics: MaterializationMetadata
    external_symbols: MaterializationMetadata
    module_state: MaterializationMetadata


@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__options(env: BuildEnv) -> ScipIngestOptions:
    """Load SCIP ingestion options for the target.

    Returns
    -------
    ScipIngestOptions
        Resolved ingestion options for the target.
    """
    return load_target_options(env, target_name=SCIP_TARGET_NAME, options_type=ScipIngestOptions)


def _scip_tool_version(env: BuildEnv) -> str | None:
    try:
        result = asyncio.run(
            env.providers.tool_runner.run_async(
                ToolName.SCIP_PYTHON,
                ["--version"],
                options=ToolRunOptions(
                    timeout_s=float(env.providers.tool_runner.tools_config.default_timeout_s),
                ),
            )
        )
    except (ToolExecutionError, ToolNotFoundError, RuntimeError, OSError, ValueError):
        return None
    if not result.ok:
        return None
    stdout = result.stdout.strip()
    return stdout.splitlines()[0] if stdout else None


def _scip_options_hash(
    options: ScipIngestOptions,
    tools_config: ToolsConfig,
    tool_version: str | None,
) -> str | None:
    payload = asdict(options)
    payload["project_name"] = tools_config.scip_project_name
    payload["tool_version"] = tool_version
    return compute_options_hash(payload)


@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__hash_options(
    env: BuildEnv,
    t__modules__scan: ModuleScanResult,
    scip__options: ScipIngestOptions,
) -> InputHashOptions:
    """Build input hash options for SCIP execution.

    Returns
    -------
    InputHashOptions
        Hash inputs used to gate SCIP execution.
    """
    tool_version = _scip_tool_version(env)
    options_hash = _scip_options_hash(
        scip__options,
        env.providers.tool_runner.tools_config,
        tool_version,
    )
    return InputHashOptions(
        options_hash=options_hash,
        manifests=env.manifest_index,
        file_state_hash=t__modules__scan.file_state_hash,
    )


@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__module_inputs(
    t__modules: TargetRunRecord,
    t__modules__scan: ModuleScanResult,
) -> ScipModuleInputs:
    """Bundle module target outputs for SCIP execution.

    Returns
    -------
    ScipModuleInputs
        Inputs containing module target and scan results.
    """
    return ScipModuleInputs(modules=t__modules, scan=t__modules__scan)


@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__run_config(
    scip__proto_module_path: Path | None,
    scip__options: ScipIngestOptions,
    scip__hash_options: InputHashOptions,
    t__scip_proto: TargetRunRecord,
) -> ScipRunConfig:
    """Bundle configuration inputs for SCIP execution.

    Returns
    -------
    ScipRunConfig
        Configuration inputs for the scip tool run.
    """
    _ = t__scip_proto
    return ScipRunConfig(
        proto_module_path=scip__proto_module_path,
        options=scip__options,
        hash_options=scip__hash_options,
    )


def _scip_run_precheck(inputs: ScipModuleInputs) -> ExecutionResult | None:
    if inputs.modules.status != "succeeded":
        return ExecutionResult.failed(
            f"Upstream modules target failed: {inputs.modules.error}"
        )
    if not inputs.scan.success:
        return ExecutionResult.failed(inputs.scan.error or "Module scan failed")
    return None


def _scip_output_path(env: BuildEnv, options: ScipIngestOptions) -> Path:
    output_dir = options.scip_output_dir or env.paths.scip_dir
    return output_dir / "index.scip"


def _scip_index_output(run: ToolStepOutput) -> Path | None:
    return run.path_for(SCIP_ARTIFACT_INDEX)


def _skip_scip_run(
    executor: NativeTargetExecutor,
    output_scip: Path,
) -> ScipRunResult | None:
    if not executor.should_skip():
        return None
    if output_scip.is_file():
        return ScipRunResult(
            result=ExecutionResult.skip("SCIP target skipped"),
            outputs={SCIP_ARTIFACT_INDEX: output_scip},
        )
    log.warning("SCIP target marked up-to-date but index.scip is missing; rebuilding")
    return None


def _resolve_change_set(scan: ModuleScanResult) -> tuple[ChangeSet, bool]:
    change_set = scan.change_set
    if change_set is None:
        log.warning("Module change set missing; forcing full SCIP rebuild")
        return ChangeSet(), True
    return change_set, False


def _load_module_state_rows(env: BuildEnv) -> list[dict[str, object]]:
    io_config = IbisIOConfig(gateway=env.gateway)
    df, _meta = load_table_as_dataframe(SCIP_MODULE_STATE_TABLE_KEY, io_config)
    frame: pd.DataFrame = df
    if frame.empty:
        return []
    filtered = frame[
        (frame["repo"] == env.snapshot.repo) & (frame["commit"] == env.snapshot.commit)
    ]
    if filtered.empty:
        return []
    rows = filtered.to_dict(orient="records")
    return cast("list[dict[str, object]]", rows)


def _manifest_records_match(
    current: ScipShardManifest,
    expected: ScipShardManifest,
) -> bool:
    if current.version != expected.version:
        return False
    if len(current.records) != len(expected.records):
        return False
    for rel_path, expected_record in expected.records.items():
        current_record = current.records.get(rel_path)
        if current_record is None:
            return False
        if not _shard_record_matches(current_record, expected_record):
            return False
    return True


def _shard_record_matches(left: ScipShardRecord, right: ScipShardRecord) -> bool:
    return (
        left.rel_path == right.rel_path
        and left.content_hash == right.content_hash
        and left.options_hash == right.options_hash
        and left.tool_version == right.tool_version
        and left.shard_path == right.shard_path
        and left.updated_at == right.updated_at
    )


def _ensure_manifest_from_module_state(env: BuildEnv, scip_dir: Path) -> None:
    manifest_file = manifest_path(scip_dir)
    try:
        rows = _load_module_state_rows(env)
    except (
        CodeIntelStorageError,
        ColumnNotFoundError,
        TableNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        log.warning("Unable to restore SCIP shard manifest from module state: %s", exc)
        return
    if not rows:
        return
    manifest = manifest_from_state_rows(rows)
    if not manifest.records:
        return
    current_manifest = load_manifest(manifest_file)
    if _manifest_records_match(current_manifest, manifest):
        return
    write_manifest(manifest_file, manifest)


def _execute_scip_incremental(
    env: BuildEnv,
    run_config: ScipRunConfig,
    module_inputs: ScipModuleInputs,
    output_scip: Path,
) -> ScipRunResult:
    if run_config.proto_module_path is None:
        return ScipRunResult(
            result=ExecutionResult.failed("SCIP proto module path is missing"),
        )

    change_set, force_full_rebuild = _resolve_change_set(module_inputs.scan)
    try:
        config = ScipIncrementalConfig(
            repo_root=env.snapshot.repo_root,
            output_scip=output_scip,
            proto_module_path=run_config.proto_module_path,
            change_set=change_set,
            modules=module_inputs.scan.modules,
            options_hash=run_config.hash_options.options_hash,
            tools_config=env.providers.tool_runner.tools_config,
            tool_runner=env.providers.tool_runner,
            scope_paths=run_config.options.scope_paths,
            max_file_size_kb=run_config.options.max_file_size_kb,
            timeout_seconds=run_config.options.timeout_seconds,
            target_dir=None,
            force_full_rebuild=force_full_rebuild,
        )
        result = update_index_incremental(config=config)
    except (OSError, RuntimeError, ToolExecutionError, ToolNotFoundError, ValueError) as exc:
        log.exception("SCIP execution failed")
        return ScipRunResult(result=ExecutionResult.failed(str(exc)))

    if not result.success:
        return ScipRunResult(
            result=ExecutionResult.failed(result.error or "SCIP indexing failed"),
        )
    return ScipRunResult(
        result=ExecutionResult.ok(),
        outputs={SCIP_ARTIFACT_INDEX: result.index_path or output_scip},
    )


@tag_tool(domain="ingestion", target=SCIP_TARGET_NAME)
def t__scip__run(
    env: BuildEnv,
    graph: TargetGraph,
    scip__module_inputs: ScipModuleInputs,
    scip__run_config: ScipRunConfig,
) -> ScipRunResult:
    """Execute scip-python to produce SCIP artifacts.

    Returns
    -------
    ScipRunResult
        Run outcome including output paths or error details.
    """
    precheck = _scip_run_precheck(scip__module_inputs)
    if precheck is not None:
        return ScipRunResult(result=precheck)

    output_scip = _scip_output_path(env, scip__run_config.options)
    _ensure_manifest_from_module_state(env, output_scip.parent)
    executor = NativeTargetExecutor.for_target(
        env,
        graph,
        SCIP_TARGET_NAME,
        hash_options=scip__run_config.hash_options,
    )
    skipped = _skip_scip_run(executor, output_scip)
    if skipped is not None:
        return skipped

    return _execute_scip_incremental(
        env,
        scip__run_config,
        scip__module_inputs,
        output_scip,
    )


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node("artifact.scip_index"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SCIP_TARGET_NAME),
    artifact_name=value(SCIP_ARTIFACT_INDEX),
    path_template=value("{scip_dir}/index.scip"),
    hash_options=source("scip__hash_options"),
)
@tag_compute(
    domain="ingestion",
    target=SCIP_TARGET_NAME,
    target_="scip__index_artifact",
)
def scip__index_artifact(t__scip__run: ScipRunResult) -> Path | None:
    """Return the Path to index.scip for materialization.

    Returns
    -------
    Path | None
        Artifact path, or None when the run was skipped/failed.
    """
    if not t__scip__run.result.success or t__scip__run.result.skipped:
        return None
    return _scip_index_output(t__scip__run)


@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__ingest_inputs(
    t__modules: TargetRunRecord,
    t__scip__run: ScipRunResult,
    scip__proto_module_path: Path | None,
    scip__options: ScipIngestOptions,
) -> ScipIngestInputs:
    """Bundle inputs required for SCIP ingestion.

    Returns
    -------
    ScipIngestInputs
        Inputs used to parse SCIP and build row payloads.
    """
    return ScipIngestInputs(
        modules=t__modules,
        run=t__scip__run,
        proto_module_path=scip__proto_module_path,
        options=scip__options,
    )


def _scip_ingest_precheck(inputs: ScipIngestInputs) -> ExecutionResult | None:
    if inputs.modules.status != "succeeded":
        return ExecutionResult.failed(
            f"Upstream modules target failed: {inputs.modules.error}"
        )
    if inputs.run.result.skipped:
        return ExecutionResult.skip("SCIP target skipped")
    if not inputs.run.result.success:
        return ExecutionResult.failed(inputs.run.result.error or "SCIP execution failed")
    if inputs.proto_module_path is None:
        return ExecutionResult.failed("SCIP proto module path is missing")
    return None


def _build_scip_row_payload(
    env: BuildEnv,
    parsed: ScipParsedIndex,
    options: ScipIngestOptions,
) -> ScipRowPayload:
    created_at = datetime.now(tz=UTC)
    row_context = ScipRowContext(
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        created_at=created_at,
        include_references=options.should_include_references(),
        include_implementations=options.should_include_implementations(),
    )
    symbol_rows = tuple(
        build_symbol_rows(
            parsed.documents,
            row_context,
            serializer=row_serializer_for_table_key(SCIP_SYMBOLS_TABLE_KEY),
        )
    )
    occurrence_rows = tuple(
        build_occurrence_rows(
            parsed.documents,
            row_context,
            serializer=row_serializer_for_table_key(SCIP_OCCURRENCES_TABLE_KEY),
        )
    )
    symbol_info_rows = tuple(
        build_symbol_information_rows(
            parsed.symbol_infos,
            row_context,
            serializer=row_serializer_for_table_key(SCIP_SYMBOL_INFO_TABLE_KEY),
        )
    )
    relationship_rows = tuple(
        build_symbol_relationship_rows(
            parsed.relationships,
            row_context,
            serializer=row_serializer_for_table_key(SCIP_RELATIONSHIPS_TABLE_KEY),
        )
    )
    diagnostic_rows = tuple(
        build_diagnostic_rows(
            parsed.diagnostics,
            row_context,
            serializer=row_serializer_for_table_key(SCIP_DIAGNOSTICS_TABLE_KEY),
        )
    )
    external_symbol_rows = tuple(
        build_external_symbol_rows(
            parsed.external_symbols,
            row_context,
            serializer=row_serializer_for_table_key(SCIP_EXTERNAL_SYMBOLS_TABLE_KEY),
        )
    )
    return ScipRowPayload(
        symbol_rows=symbol_rows,
        occurrence_rows=occurrence_rows,
        symbol_info_rows=symbol_info_rows,
        relationship_rows=relationship_rows,
        diagnostic_rows=diagnostic_rows,
        external_symbol_rows=external_symbol_rows,
    )


def _scip_table_counts(payload: ScipRowPayload) -> dict[str, int]:
    return {
        SCIP_SYMBOLS_TABLE_KEY: len(payload.symbol_rows),
        SCIP_OCCURRENCES_TABLE_KEY: len(payload.occurrence_rows),
        SCIP_SYMBOL_INFO_TABLE_KEY: len(payload.symbol_info_rows),
        SCIP_RELATIONSHIPS_TABLE_KEY: len(payload.relationship_rows),
        SCIP_DIAGNOSTICS_TABLE_KEY: len(payload.diagnostic_rows),
        SCIP_EXTERNAL_SYMBOLS_TABLE_KEY: len(payload.external_symbol_rows),
    }


def _build_scip_ingest_result(env: BuildEnv, inputs: ScipIngestInputs) -> ScipIngestResult:
    precheck = _scip_ingest_precheck(inputs)
    if precheck is not None:
        return ScipIngestResult(result=precheck)

    output_scip = _scip_index_output(inputs.run) or (env.paths.scip_dir / "index.scip")
    proto_module_path = cast("Path", inputs.proto_module_path)
    try:
        parsed: ScipParsedIndex = parse_index(output_scip, proto_module_path)
        payload = _build_scip_row_payload(env, parsed, inputs.options)
    except (OSError, AttributeError, TypeError, ValueError):
        log.exception("SCIP ingestion failed")
        return ScipIngestResult(
            result=ExecutionResult.failed("SCIP ingestion failed with exception")
        )

    if not payload.symbol_rows or not payload.occurrence_rows:
        return ScipIngestResult(
            result=ExecutionResult.failed(
                "SCIP ingestion produced empty symbols or occurrences"
            )
        )

    table_counts = _scip_table_counts(payload)
    result = ExecutionResult.ok(
        table_counts=normalize_table_counts(SCIP_TABLE_KEYS, table_counts),
    )
    return ScipIngestResult(
        result=result,
        symbol_rows=payload.symbol_rows,
        occurrence_rows=payload.occurrence_rows,
        symbol_info_rows=payload.symbol_info_rows,
        relationship_rows=payload.relationship_rows,
        diagnostic_rows=payload.diagnostic_rows,
        external_symbol_rows=payload.external_symbol_rows,
    )


@tag_compute(domain="ingestion", target=SCIP_TARGET_NAME)
def t__scip__ingest(
    env: BuildEnv,
    scip__ingest_inputs: ScipIngestInputs,
) -> ScipIngestResult:
    """Build SCIP row payloads for core.scip_* tables.

    Returns
    -------
    ScipIngestResult
        Ingestion status and row tuples.
    """
    return _build_scip_ingest_result(env, scip__ingest_inputs)



@dataclass(frozen=True)
class _ScipIngestRowSpec:
    """Specification for parameterized SCIP row nodes."""

    table_key: str
    rows_attr: str
    node_name: str


_SCIP_INGEST_ROW_SPECS = (
    _ScipIngestRowSpec(
        table_key=SCIP_SYMBOLS_TABLE_KEY,
        rows_attr="symbol_rows",
        node_name="scip__symbol_rows",
    ),
    _ScipIngestRowSpec(
        table_key=SCIP_OCCURRENCES_TABLE_KEY,
        rows_attr="occurrence_rows",
        node_name="scip__occurrence_rows",
    ),
    _ScipIngestRowSpec(
        table_key=SCIP_SYMBOL_INFO_TABLE_KEY,
        rows_attr="symbol_info_rows",
        node_name="scip__symbol_info_rows",
    ),
    _ScipIngestRowSpec(
        table_key=SCIP_RELATIONSHIPS_TABLE_KEY,
        rows_attr="relationship_rows",
        node_name="scip__relationship_rows",
    ),
    _ScipIngestRowSpec(
        table_key=SCIP_DIAGNOSTICS_TABLE_KEY,
        rows_attr="diagnostic_rows",
        node_name="scip__diagnostic_rows",
    ),
    _ScipIngestRowSpec(
        table_key=SCIP_EXTERNAL_SYMBOLS_TABLE_KEY,
        rows_attr="external_symbol_rows",
        node_name="scip__external_symbol_rows",
    ),
)


def _scip_row_docstring(table_key: str) -> str:
    return (
        f"Return rows for {table_key}.\n\n"
        "Returns\n"
        "-------\n"
        "tuple[tuple[object, ...], ...] | None\n"
        f"    Row tuples for {table_key}, or None when ingestion skipped or failed.\n"
    )


_SCIP_INGEST_ROW_VALUES = {
    (spec.node_name, _scip_row_docstring(spec.table_key)): spec.rows_attr
    for spec in _SCIP_INGEST_ROW_SPECS
}
_SCIP_INGEST_ROW_TABLES = {spec.node_name: spec.table_key for spec in _SCIP_INGEST_ROW_SPECS}


def _resolve_scip_ingest_rows(
    t__scip__ingest: ScipIngestResult,
    rows_attr: str,
) -> tuple[tuple[object, ...], ...] | None:
    if t__scip__ingest.result.skipped or not t__scip__ingest.result.success:
        return None
    rows = getattr(t__scip__ingest, rows_attr)
    return cast("tuple[tuple[object, ...], ...]", rows)


class _SaveToScipRowTables(NodeTransformer):
    def __init__(self, table_keys: Mapping[str, str]) -> None:
        super().__init__(target=None)
        self.table_keys = dict(table_keys)

    def transform_node(
        self,
        node_: h_node.Node,
        config: dict[str, object],
        fn: Callable[..., object],
    ) -> list[h_node.Node]:
        name = node_.name
        if not isinstance(name, str):
            return [node_]
        table_key = self.table_keys.get(name)
        if table_key is None:
            return [node_]
        decorator = SaveToObjectMetadataDecorator(
            [DuckDBRowsSaver],
            output_name_=materialize_node(table_key),
            env=source("env"),
            graph=source("graph"),
            target_name=value(SCIP_TARGET_NAME),
            table_key=value(table_key),
            columns=value(deferred_columns_for_table_key(table_key)),
            hash_options=source("scip__hash_options"),
        )
        return list(decorator.transform_node(node_, config, fn))

    @staticmethod
    def validate(fn: Callable[..., object]) -> None:
        _ = fn


@_SaveToScipRowTables(_SCIP_INGEST_ROW_TABLES)
@parameterize_values(
    parameter="rows_attr",
    assigned_output=_SCIP_INGEST_ROW_VALUES,
)
@tag_compute(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__rows_for_table(
    t__scip__ingest: ScipIngestResult,
    rows_attr: str,
) -> tuple[tuple[object, ...], ...] | None:
    """Return rows for a SCIP table payload.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for the configured table, or None when ingestion skipped or failed.
    """
    return _resolve_scip_ingest_rows(t__scip__ingest, rows_attr)



@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(SCIP_MODULE_STATE_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SCIP_TARGET_NAME),
    table_key=value(SCIP_MODULE_STATE_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(SCIP_MODULE_STATE_TABLE_KEY)),
    hash_options=source("scip__hash_options"),
)
@tag_compute(
    domain="ingestion",
    target=SCIP_TARGET_NAME,
    target_="scip__module_state_rows",
)
def scip__module_state_rows(
    env: BuildEnv,
    t__scip__run: ScipRunResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Return rows for core.scip_module_state.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for scip_module_state, or None when execution skipped or failed.
    """
    if not t__scip__run.result.success or t__scip__run.result.skipped:
        return None
    index_path = _scip_index_output(t__scip__run)
    scip_dir = index_path.parent if index_path is not None else env.paths.scip_dir
    manifest = load_manifest(manifest_path(scip_dir))
    rows = build_module_state_rows(
        manifest,
        env.snapshot.repo,
        env.snapshot.commit,
        serializer=row_serializer_for_table_key(SCIP_MODULE_STATE_TABLE_KEY),
    )
    return tuple(rows)


@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__materializations(
    m__artifact__scip_index: MaterializationMetadata,
) -> dict[str, MaterializationMetadata]:
    """Collect scip artifact materialization payloads into a single mapping.

    Returns
    -------
    dict[str, MaterializationMetadata]
        Materialization metadata keyed by artifact name.
    """
    return {
        SCIP_ARTIFACT_INDEX: m__artifact__scip_index,
    }


@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__symbol_table_materializations(
    m__core__scip_symbols: MaterializationMetadata,
    m__core__scip_occurrences: MaterializationMetadata,
    m__core__scip_symbol_information: MaterializationMetadata,
) -> ScipSymbolTableMaterializations:
    """Collect symbol-related table materializations.

    Returns
    -------
    ScipSymbolTableMaterializations
        Materialization metadata for symbols, occurrences, and symbol information.
    """
    return ScipSymbolTableMaterializations(
        symbols=m__core__scip_symbols,
        occurrences=m__core__scip_occurrences,
        symbol_information=m__core__scip_symbol_information,
    )


@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__aux_table_materializations(
    m__core__scip_symbol_relationships: MaterializationMetadata,
    m__core__scip_diagnostics: MaterializationMetadata,
    m__core__scip_external_symbols: MaterializationMetadata,
    m__core__scip_module_state: MaterializationMetadata,
) -> ScipAuxTableMaterializations:
    """Collect auxiliary table materializations.

    Returns
    -------
    ScipAuxTableMaterializations
        Materialization metadata for relationships, diagnostics, external symbols, and
        module state.
    """
    return ScipAuxTableMaterializations(
        relationships=m__core__scip_symbol_relationships,
        diagnostics=m__core__scip_diagnostics,
        external_symbols=m__core__scip_external_symbols,
        module_state=m__core__scip_module_state,
    )


@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__table_materializations(
    scip__symbol_table_materializations: ScipSymbolTableMaterializations,
    scip__aux_table_materializations: ScipAuxTableMaterializations,
) -> dict[str, MaterializationMetadata]:
    """Collect scip table materialization payloads into a single mapping.

    Returns
    -------
    dict[str, MaterializationMetadata]
        Mapping from table key to saver metadata.
    """
    return {
        SCIP_SYMBOLS_TABLE_KEY: scip__symbol_table_materializations.symbols,
        SCIP_OCCURRENCES_TABLE_KEY: scip__symbol_table_materializations.occurrences,
        SCIP_SYMBOL_INFO_TABLE_KEY: scip__symbol_table_materializations.symbol_information,
        SCIP_RELATIONSHIPS_TABLE_KEY: scip__aux_table_materializations.relationships,
        SCIP_DIAGNOSTICS_TABLE_KEY: scip__aux_table_materializations.diagnostics,
        SCIP_EXTERNAL_SYMBOLS_TABLE_KEY: scip__aux_table_materializations.external_symbols,
        SCIP_MODULE_STATE_TABLE_KEY: scip__aux_table_materializations.module_state,
    }


@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__inputs(
    t__scip__run: ScipRunResult,
    t__scip__ingest: ScipIngestResult,
    scip__materializations: dict[str, MaterializationMetadata],
    scip__table_materializations: dict[str, MaterializationMetadata],
) -> ScipMaterializationInputs:
    """Bundle scip execution and materialization inputs.

    Returns
    -------
    ScipMaterializationInputs
        Aggregated inputs for scip materialization.
    """
    return ScipMaterializationInputs(
        run=t__scip__run,
        ingest=t__scip__ingest,
        artifact_materializations=scip__materializations,
        table_materializations=scip__table_materializations,
    )


@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__target_inputs(
    t__modules: TargetRunRecord,
    scip__hash_options: InputHashOptions,
) -> ScipTargetInputs:
    """Bundle inputs required to finalize the SCIP target.

    Returns
    -------
    ScipTargetInputs
        Inputs containing upstream module status and hash options.
    """
    return ScipTargetInputs(modules=t__modules, hash_options=scip__hash_options)


def _summarize_scip_table_materializations(
    materializations: dict[str, MaterializationMetadata],
) -> tuple[str, dict[str, int] | None, str | None]:
    parsed: dict[str, DuckDBMaterializationMetadata] = {}
    for table_key in SCIP_TABLE_KEYS:
        meta = materializations.get(table_key)
        if meta is None:
            parsed[table_key] = DuckDBMaterializationMetadata(
                status="failed",
                table_key=table_key,
                row_count=None,
                duration_ms=0.0,
                input_hash="",
                error=f"Missing materialization metadata for table: {table_key}",
            )
            continue

        result = DuckDBMaterializationMetadata.from_mapping(
            meta,
            default_table_key=table_key,
        )
        if result.status != "failed" and result.table_key != table_key:
            parsed[table_key] = DuckDBMaterializationMetadata(
                status="failed",
                table_key=table_key,
                row_count=None,
                duration_ms=result.duration_ms,
                input_hash=result.input_hash,
                error=(
                    "DuckDB materialization metadata table_key mismatch: "
                    f"expected={table_key} got={result.table_key}"
                ),
            )
            continue

        parsed[table_key] = result

    statuses = {result.status for result in parsed.values()}
    if "failed" in statuses:
        errors = [result.error for result in parsed.values() if result.error]
        message = errors[0] if errors else "One or more table writes failed"
        return "failed", None, message

    if statuses == {"skipped"}:
        return "skipped", None, None

    row_counts: dict[str, int] = {}
    for table_key, result in parsed.items():
        if result.status == "succeeded":
            row_counts[table_key] = result.row_count or 0
        else:
            row_counts[table_key] = 0

    return "succeeded", row_counts, None


@codeintel_target(
    domain="ingestion",
    target=SCIP_TARGET_NAME,
    spec=TargetSpecDescriptor(
        resources=TargetResources(
            tracker=True,
            modules=True,
            tools=(
                "scip-python",
            ),
        ),
        execution=TOOL_EXECUTION,
    ),
)
def t__scip(
    env: BuildEnv,
    graph: TargetGraph,
    scip__inputs: ScipMaterializationInputs,
    scip__target_inputs: ScipTargetInputs,
) -> TargetRunRecord:
    """SCIP index ingestion and GOID generation.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    executor = NativeTargetExecutor.for_target(
        env,
        graph,
        SCIP_TARGET_NAME,
        hash_options=scip__target_inputs.hash_options,
    )
    if scip__target_inputs.modules.status != "succeeded":
        return executor.fail(
            RuntimeError(
                scip__target_inputs.modules.error or "Upstream modules target failed"
            )
        )
    if not scip__inputs.run.result.success:
        return executor.fail(
            RuntimeError(scip__inputs.run.result.error or "SCIP execution failed")
        )
    if not scip__inputs.ingest.result.success and not scip__inputs.ingest.result.skipped:
        return executor.fail(
            RuntimeError(scip__inputs.ingest.result.error or "SCIP ingestion failed")
        )

    table_status, row_counts, table_error = _summarize_scip_table_materializations(
        scip__inputs.table_materializations
    )
    if table_status == "failed":
        return executor.fail(RuntimeError(table_error or "SCIP table writes failed"))

    return record_from_file_artifact_materializations(
        env=env,
        graph=graph,
        target_name="scip",
        materializations=scip__inputs.artifact_materializations,
        row_counts=row_counts,
    )


__all__ = [
    "ScipIngestResult",
    "ScipRunResult",
    "t__scip",
    "t__scip__ingest",
    "t__scip__run",
]
