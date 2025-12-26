"""Native Hamilton implementation for scip target.

This target produces the SCIP artifacts declared in the build contract:
- ``scip_index``: ``{scip_dir}/index.scip``

Tool execution is delegated to the ingestion tool runtime (ToolRunner),
while persistence is DAG-visible via artifact saver nodes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.ingestion.ingest_targets import ModuleToolOutput
from codeintel.build.hamilton.native.options.ingestion import ScipIngestOptions
from codeintel.build.hamilton.native.patterns import (
    ArtifactSaveSpec,
    IngestStep,
    SaverContext,
    TableSaveSpec,
    ToolFinalizeContext,
    ToolRunContext,
    finalize_target_from_materializations,
    run_tool_step,
    save_artifact,
    save_rows,
)
from codeintel.build.hamilton.native.table_counts import normalize_table_counts
from codeintel.build.hamilton.native.target_decorators import (
    TargetSpecDescriptor,
    codeintel_target,
)
from codeintel.build.hamilton.native.tool_results import ToolStepOutput
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_compute, tag_helper, tag_tool
from codeintel.build.hashing import InputHashOptions, compute_options_hash
from codeintel.build.resources import TOOL_EXECUTION, TargetResources
from codeintel.build.targets import TargetGraph
from codeintel.core.errors import CodeIntelStorageError, ColumnNotFoundError, TableNotFoundError
from codeintel.core.execution.ids import new_run_id
from codeintel.core.runtime.loader import load_runtime_settings
from codeintel.core.schemas.row_serialization import row_serializer_for_table_key
from codeintel.core.tools import ToolName
from codeintel.ingestion.engine.infrastructure import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolRunOptions,
)
from codeintel.ingestion.ports.change_detection import ChangeSet, FileDigest
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
from codeintel.ingestion.scip.telemetry import ScipRunTelemetry
from codeintel.observability.semconv_keys import (
    SCIP_COMMIT,
    SCIP_MODE,
    SCIP_REPO,
    SCIP_RUN_ID,
)
from codeintel.observability.teardown import (
    ScipTeardownStatus,
    ScipTeardownTelemetry,
    emit_scip_teardown_telemetry,
    emit_shutdown_error_event,
)
from codeintel.storage.io import IbisIOConfig, load_table_as_dataframe
from codeintel.storage.tracking.build_tracking import ScipRunRecord

if TYPE_CHECKING:
    import pandas as pd

    from codeintel.config.models import ToolsConfig

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, Path, TargetRunRecord)

ScipRunMode = Literal["incremental", "full", "skipped", "precheck_failed", "unknown"]

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
_FILE_STATE_ROW_MIN_COLUMNS = 7

SCIP_SAVE_CONTEXT = SaverContext(
    domain="ingestion",
    target=SCIP_TARGET_NAME,
    hash_options_node="scip__hash_options",
)


@dataclass(frozen=True)
class ScipRunResult(ToolStepOutput):
    """Outcome for a SCIP tool run with telemetry metadata."""

    run_id: str = ""
    mode: ScipRunMode = "unknown"


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
    scan: ModuleToolOutput


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
    run: ScipRunResult
    proto_module_path: Path | None
    options: ScipIngestOptions


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
    t__modules__run: ModuleToolOutput,
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
        file_state_hash=t__modules__run.file_state_hash,
    )


@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__module_inputs(
    t__modules: TargetRunRecord,
    t__modules__run: ModuleToolOutput,
) -> ScipModuleInputs:
    """Bundle module target outputs for SCIP execution.

    Returns
    -------
    ScipModuleInputs
        Inputs containing module target and scan results.
    """
    return ScipModuleInputs(modules=t__modules, scan=t__modules__run)


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
    if inputs.modules.status not in {"succeeded", "skipped"}:
        return ExecutionResult.failed(f"Upstream modules target failed: {inputs.modules.error}")
    scan_result = inputs.scan.result
    if scan_result.skipped:
        return ExecutionResult.skip(scan_result.skip_reason or "Module scan skipped")
    if not scan_result.success:
        return ExecutionResult.failed(scan_result.error or "Module scan failed")
    return None


def _resolve_scip_run_id(env: BuildEnv) -> str:
    if env.run_context is not None:
        return env.run_context.run_id
    return new_run_id("scip")


def _scip_output_path(env: BuildEnv, options: ScipIngestOptions) -> Path:
    output_dir = options.scip_output_dir or env.paths.scip_dir
    return output_dir / "index.scip"


def _scip_index_output(run: ScipRunResult) -> Path | None:
    return run.path_for(SCIP_ARTIFACT_INDEX)


def _resolve_change_set(scan: ModuleToolOutput) -> tuple[ChangeSet, bool]:
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


def _coerce_int(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


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


def _build_file_state_map(
    file_state_rows: tuple[tuple[object, ...], ...],
) -> dict[str, FileDigest]:
    digest_by_path: dict[str, FileDigest] = {}
    for row in file_state_rows:
        if len(row) < _FILE_STATE_ROW_MIN_COLUMNS:
            continue
        rel_path_raw = row[2]
        size_raw = row[4]
        mtime_raw = row[5]
        hash_raw = row[6]
        if rel_path_raw is None or hash_raw is None:
            continue
        rel_path = str(rel_path_raw)
        size_bytes = _coerce_int(size_raw) or 0
        mtime_ns = _coerce_int(mtime_raw) or 0
        content_hash = str(hash_raw)
        digest_by_path[rel_path] = FileDigest(
            size_bytes=size_bytes,
            mtime_ns=mtime_ns,
            content_hash=content_hash,
        )
    return digest_by_path


def _persist_scip_telemetry(env: BuildEnv, telemetry: ScipRunTelemetry) -> None:
    record = ScipRunRecord(
        run_id=telemetry.run_id,
        repo=telemetry.repo,
        commit=telemetry.commit,
        mode=telemetry.mode,
        options_hash=telemetry.options_hash,
        tool_version=telemetry.tool_version,
        total_modules=telemetry.total_modules,
        changed_modules=telemetry.changed_modules,
        deleted_modules=telemetry.deleted_modules,
        changed_ratio=telemetry.changed_ratio,
        batch_size=telemetry.batch_size,
        batch_count=telemetry.batch_count,
        decision=telemetry.decision,
        ratio_gate_applied=telemetry.ratio_gate_applied,
        ratio_gate_min_modules=telemetry.ratio_gate_min_modules,
        ratio_gate_min_changed=telemetry.ratio_gate_min_changed,
        hash_source=telemetry.hash_source,
        hash_source_breakdown=telemetry.hash_source_breakdown,
        hash_reused=telemetry.hash_reused,
        hash_computed=telemetry.hash_computed,
        plan_ms=telemetry.plan_ms,
        hash_ms=telemetry.hash_ms,
        tool_ms=telemetry.tool_ms,
        parse_ms=telemetry.parse_ms,
        merge_ms=telemetry.merge_ms,
        write_ms=telemetry.write_ms,
        total_ms=telemetry.total_ms,
        status=telemetry.status,
        error_summary=telemetry.error_summary,
        output_scip=telemetry.output_scip,
        recorded_at=telemetry.recorded_at,
    )
    env.gateway.build.record_scip_run(record)
    _write_scip_run_report(env.paths.scip_dir, telemetry)


def _write_scip_run_report(scip_dir: Path, telemetry: ScipRunTelemetry) -> None:
    run_dir = scip_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = telemetry.recorded_at.strftime("%Y%m%dT%H%M%SZ")
    output_path = run_dir / f"scip_run_{timestamp}.json"
    payload = telemetry.to_payload()
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _persist_scip_telemetry_safe(env: BuildEnv, telemetry: ScipRunTelemetry) -> None:
    try:
        _persist_scip_telemetry(env, telemetry)
    except (OSError, RuntimeError, ValueError) as exc:
        log.warning("Failed to persist SCIP telemetry: %s", exc)


def _execute_scip_incremental(
    env: BuildEnv,
    run_config: ScipRunConfig,
    module_inputs: ScipModuleInputs,
    output_scip: Path,
    *,
    run_id: str,
) -> ScipRunResult:
    if run_config.proto_module_path is None:
        return ScipRunResult(
            result=ExecutionResult.failed("SCIP proto module path is missing"),
            run_id=run_id,
            mode="unknown",
        )

    change_set, force_full_rebuild = _resolve_change_set(module_inputs.scan)
    telemetry = ScipRunTelemetry.create(
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        run_id=run_id,
        options_hash=run_config.hash_options.options_hash,
    )
    file_state_rows = module_inputs.scan.file_state_rows or tuple(change_set.state_rows)
    file_state_by_path = _build_file_state_map(file_state_rows)
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
            batch_size=run_config.options.batch_size,
            batch_max_bytes=run_config.options.batch_max_bytes,
            full_rebuild_threshold_count=run_config.options.full_rebuild_threshold_count,
            full_rebuild_threshold_ratio=run_config.options.full_rebuild_threshold_ratio,
            full_rebuild_ratio_min_modules=run_config.options.full_rebuild_ratio_min_modules,
            full_rebuild_ratio_min_changed=run_config.options.full_rebuild_ratio_min_changed,
            file_state_by_path=file_state_by_path,
            module_state_by_path=None,
            telemetry=telemetry,
        )
        result = update_index_incremental(config=config)
    except (OSError, RuntimeError, ToolExecutionError, ToolNotFoundError, ValueError) as exc:
        log.exception("SCIP execution failed")
        telemetry.status = "failed"
        telemetry.error_summary = str(exc)
        telemetry.total_ms = 0.0
        _persist_scip_telemetry_safe(env, telemetry)
        return ScipRunResult(
            result=ExecutionResult.failed(str(exc)),
            run_id=run_id,
            mode=_normalize_scip_run_mode(telemetry.mode or "incremental"),
        )

    if not result.success:
        telemetry.status = "failed"
        telemetry.error_summary = result.error or "SCIP indexing failed"
        _persist_scip_telemetry_safe(env, telemetry)
        return ScipRunResult(
            result=ExecutionResult.failed(result.error or "SCIP indexing failed"),
            run_id=run_id,
            mode=_normalize_scip_run_mode(telemetry.mode or "incremental"),
        )
    _persist_scip_telemetry_safe(env, telemetry)
    return ScipRunResult(
        result=ExecutionResult.ok(),
        outputs={SCIP_ARTIFACT_INDEX: result.index_path or output_scip},
        run_id=run_id,
        mode=_normalize_scip_run_mode(telemetry.mode or "incremental"),
    )


def _coerce_scip_run_output(
    output: ToolStepOutput,
    *,
    run_id: str,
    mode: ScipRunMode,
) -> ScipRunResult:
    if isinstance(output, ScipRunResult):
        return output
    return ScipRunResult(
        result=output.result,
        outputs=output.outputs,
        run_id=run_id,
        mode=mode,
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
    run_id = _resolve_scip_run_id(env)
    precheck = _scip_run_precheck(scip__module_inputs)
    if precheck is not None:
        mode = "skipped" if precheck.skipped else "precheck_failed"
        return ScipRunResult(result=precheck, run_id=run_id, mode=mode)

    output_scip = _scip_output_path(env, scip__run_config.options)
    _ensure_manifest_from_module_state(env, output_scip.parent)

    context = ToolRunContext(
        env=env,
        graph=graph,
        target_name=SCIP_TARGET_NAME,
        hash_options=scip__run_config.hash_options,
        skip_reason="SCIP target skipped",
    )

    def _execute() -> ScipRunResult:
        return _execute_scip_incremental(
            env,
            scip__run_config,
            scip__module_inputs,
            output_scip,
            run_id=run_id,
        )

    output = run_tool_step(context=context, run=_execute)
    scip_output = _coerce_scip_run_output(output, run_id=run_id, mode="unknown")
    if scip_output.result.skipped:
        if output_scip.is_file():
            return ScipRunResult(
                result=ExecutionResult.skip("SCIP target skipped"),
                outputs={SCIP_ARTIFACT_INDEX: output_scip},
                run_id=run_id,
                mode="skipped",
            )
        log.warning("SCIP target marked up-to-date but index.scip is missing; rebuilding")
        return _execute()

    return scip_output


@save_artifact(
    context=SCIP_SAVE_CONTEXT,
    spec=ArtifactSaveSpec(
        artifact_name=SCIP_ARTIFACT_INDEX,
        path_template="{scip_dir}/index.scip",
    ),
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
    if inputs.modules.status not in {"succeeded", "skipped"}:
        return ExecutionResult.failed(f"Upstream modules target failed: {inputs.modules.error}")
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


def _scip_table_counts(
    payload: ScipRowPayload,
    module_state_rows: tuple[tuple[object, ...], ...],
) -> dict[str, int]:
    return {
        SCIP_SYMBOLS_TABLE_KEY: len(payload.symbol_rows),
        SCIP_OCCURRENCES_TABLE_KEY: len(payload.occurrence_rows),
        SCIP_SYMBOL_INFO_TABLE_KEY: len(payload.symbol_info_rows),
        SCIP_RELATIONSHIPS_TABLE_KEY: len(payload.relationship_rows),
        SCIP_DIAGNOSTICS_TABLE_KEY: len(payload.diagnostic_rows),
        SCIP_EXTERNAL_SYMBOLS_TABLE_KEY: len(payload.external_symbol_rows),
        SCIP_MODULE_STATE_TABLE_KEY: len(module_state_rows),
    }


def _build_module_state_rows(
    env: BuildEnv,
    run: ScipRunResult,
) -> tuple[tuple[object, ...], ...]:
    index_path = _scip_index_output(run)
    scip_dir = index_path.parent if index_path is not None else env.paths.scip_dir
    manifest = load_manifest(manifest_path(scip_dir))
    rows = build_module_state_rows(
        manifest,
        env.snapshot.repo,
        env.snapshot.commit,
        serializer=row_serializer_for_table_key(SCIP_MODULE_STATE_TABLE_KEY),
    )
    return tuple(rows)


def _build_scip_ingest_result(
    env: BuildEnv,
    inputs: ScipIngestInputs,
) -> IngestStep[dict[str, tuple[tuple[object, ...], ...]]]:
    precheck = _scip_ingest_precheck(inputs)
    if precheck is not None:
        return IngestStep(result=precheck)

    output_scip = _scip_index_output(inputs.run) or (env.paths.scip_dir / "index.scip")
    proto_module_path = cast("Path", inputs.proto_module_path)
    try:
        parsed: ScipParsedIndex = parse_index(output_scip, proto_module_path)
        payload = _build_scip_row_payload(env, parsed, inputs.options)
    except (OSError, AttributeError, TypeError, ValueError):
        log.exception("SCIP ingestion failed")
        return IngestStep(result=ExecutionResult.failed("SCIP ingestion failed with exception"))

    if not payload.symbol_rows or not payload.occurrence_rows:
        return IngestStep(
            result=ExecutionResult.failed("SCIP ingestion produced empty symbols or occurrences")
        )

    module_state_rows = _build_module_state_rows(env, inputs.run)
    table_counts = _scip_table_counts(payload, module_state_rows)
    result = ExecutionResult.ok(
        table_counts=normalize_table_counts(SCIP_TABLE_KEYS, table_counts),
    )
    payload_by_table = {
        SCIP_SYMBOLS_TABLE_KEY: payload.symbol_rows,
        SCIP_OCCURRENCES_TABLE_KEY: payload.occurrence_rows,
        SCIP_SYMBOL_INFO_TABLE_KEY: payload.symbol_info_rows,
        SCIP_RELATIONSHIPS_TABLE_KEY: payload.relationship_rows,
        SCIP_DIAGNOSTICS_TABLE_KEY: payload.diagnostic_rows,
        SCIP_EXTERNAL_SYMBOLS_TABLE_KEY: payload.external_symbol_rows,
        SCIP_MODULE_STATE_TABLE_KEY: module_state_rows,
    }
    return IngestStep(result=result, payload=payload_by_table)


@tag_compute(domain="ingestion", target=SCIP_TARGET_NAME)
def t__scip__ingest(
    env: BuildEnv,
    scip__ingest_inputs: ScipIngestInputs,
) -> IngestStep[dict[str, tuple[tuple[object, ...], ...]]]:
    """Build SCIP row payloads for core.scip_* tables.

    Returns
    -------
    IngestStep[dict[str, tuple[tuple[object, ...], ...]]]
        Ingestion status and row payloads.
    """
    return _build_scip_ingest_result(env, scip__ingest_inputs)


def _scip_payload_rows(
    t__scip__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
    table_key: str,
) -> tuple[tuple[object, ...], ...] | None:
    if t__scip__ingest.result.skipped or not t__scip__ingest.result.success:
        return None
    payload = t__scip__ingest.payload
    if payload is None:
        msg = "Missing SCIP ingest payload"
        raise ValueError(msg)
    rows = payload.get(table_key)
    if rows is None:
        msg = f"Missing rows for {table_key}"
        raise ValueError(msg)
    return rows


@save_rows(
    context=SCIP_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=SCIP_SYMBOLS_TABLE_KEY),
)
@tag_compute(domain="ingestion", target=SCIP_TARGET_NAME, target_="scip__symbol_rows")
def scip__symbol_rows(
    t__scip__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Return rows for core.scip_symbols.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for core.scip_symbols, or None when ingestion skipped or failed.
    """
    return _scip_payload_rows(t__scip__ingest, SCIP_SYMBOLS_TABLE_KEY)


@save_rows(
    context=SCIP_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=SCIP_OCCURRENCES_TABLE_KEY),
)
@tag_compute(domain="ingestion", target=SCIP_TARGET_NAME, target_="scip__occurrence_rows")
def scip__occurrence_rows(
    t__scip__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Return rows for core.scip_occurrences.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for core.scip_occurrences, or None when ingestion skipped or failed.
    """
    return _scip_payload_rows(t__scip__ingest, SCIP_OCCURRENCES_TABLE_KEY)


@save_rows(
    context=SCIP_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=SCIP_SYMBOL_INFO_TABLE_KEY),
)
@tag_compute(domain="ingestion", target=SCIP_TARGET_NAME, target_="scip__symbol_info_rows")
def scip__symbol_info_rows(
    t__scip__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Return rows for core.scip_symbol_information.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for core.scip_symbol_information, or None when ingestion skipped or failed.
    """
    return _scip_payload_rows(t__scip__ingest, SCIP_SYMBOL_INFO_TABLE_KEY)


@save_rows(
    context=SCIP_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=SCIP_RELATIONSHIPS_TABLE_KEY),
)
@tag_compute(domain="ingestion", target=SCIP_TARGET_NAME, target_="scip__relationship_rows")
def scip__relationship_rows(
    t__scip__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Return rows for core.scip_symbol_relationships.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for core.scip_symbol_relationships, or None when ingestion skipped or failed.
    """
    return _scip_payload_rows(t__scip__ingest, SCIP_RELATIONSHIPS_TABLE_KEY)


@save_rows(
    context=SCIP_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=SCIP_DIAGNOSTICS_TABLE_KEY),
)
@tag_compute(domain="ingestion", target=SCIP_TARGET_NAME, target_="scip__diagnostic_rows")
def scip__diagnostic_rows(
    t__scip__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Return rows for core.scip_diagnostics.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for core.scip_diagnostics, or None when ingestion skipped or failed.
    """
    return _scip_payload_rows(t__scip__ingest, SCIP_DIAGNOSTICS_TABLE_KEY)


@save_rows(
    context=SCIP_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=SCIP_EXTERNAL_SYMBOLS_TABLE_KEY),
)
@tag_compute(domain="ingestion", target=SCIP_TARGET_NAME, target_="scip__external_symbol_rows")
def scip__external_symbol_rows(
    t__scip__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Return rows for core.scip_external_symbols.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for core.scip_external_symbols, or None when ingestion skipped or failed.
    """
    return _scip_payload_rows(t__scip__ingest, SCIP_EXTERNAL_SYMBOLS_TABLE_KEY)


@save_rows(
    context=SCIP_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=SCIP_MODULE_STATE_TABLE_KEY),
)
@tag_compute(domain="ingestion", target=SCIP_TARGET_NAME, target_="scip__module_state_rows")
def scip__module_state_rows(
    t__scip__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Return rows for core.scip_module_state.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for core.scip_module_state, or None when ingestion skipped or failed.
    """
    return _scip_payload_rows(t__scip__ingest, SCIP_MODULE_STATE_TABLE_KEY)


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
def scip__finalize_context(
    env: BuildEnv,
    graph: TargetGraph,
    scip__hash_options: InputHashOptions,
) -> ToolFinalizeContext:
    """Build finalization context for the SCIP target.

    Returns
    -------
    ToolFinalizeContext
        Finalization context for SCIP ingestion.
    """
    return ToolFinalizeContext(
        env=env,
        graph=graph,
        target_name=SCIP_TARGET_NAME,
        hash_options=scip__hash_options,
    )


def _should_emit_scip_teardown(run: ScipRunResult) -> bool:
    if run.result.skipped:
        return False
    return run.mode not in {"skipped", "precheck_failed"}


def _normalize_scip_run_mode(mode: str | None) -> ScipRunMode:
    if mode == "full":
        return "full"
    if mode == "incremental":
        return "incremental"
    if mode == "precheck_failed":
        return "precheck_failed"
    if mode == "skipped":
        return "skipped"
    if mode == "unknown":
        return "unknown"
    return "unknown"


def _normalize_scip_teardown_status(status: str) -> ScipTeardownStatus:
    if status in {"failed", "skipped", "succeeded"}:
        return cast("ScipTeardownStatus", status)
    return "unknown"


def _emit_scip_teardown(
    env: BuildEnv,
    run: ScipRunResult,
    record: TargetRunRecord,
) -> None:
    settings = load_runtime_settings().observability
    if not settings.teardown_enabled:
        return
    if not _should_emit_scip_teardown(run):
        return
    error_summary = record.error or run.result.error
    telemetry = ScipTeardownTelemetry(
        run_id=run.run_id or None,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        scip_mode=run.mode or None,
        status=_normalize_scip_teardown_status(record.status),
        error_summary=error_summary,
        duration_ms=record.duration_ms,
    )
    emit_scip_teardown_telemetry(telemetry, logger=log)


def _emit_scip_teardown_safe(
    env: BuildEnv,
    run: ScipRunResult,
    record: TargetRunRecord,
) -> None:
    try:
        _emit_scip_teardown(env, run, record)
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        emit_shutdown_error_event(
            span_name="scip.teardown",
            error=exc,
            logger=log,
            attributes={
                key: value
                for key, value in {
                    SCIP_RUN_ID: run.run_id or None,
                    SCIP_REPO: env.snapshot.repo,
                    SCIP_COMMIT: env.snapshot.commit,
                    SCIP_MODE: run.mode or None,
                }.items()
                if value is not None
            },
        )
        log.warning("Failed to emit SCIP teardown telemetry: %s", exc)


@codeintel_target(
    domain="ingestion",
    target=SCIP_TARGET_NAME,
    spec=TargetSpecDescriptor(
        resources=TargetResources(
            tracker=True,
            modules=True,
            tools=("scip-python",),
        ),
        execution=TOOL_EXECUTION,
    ),
)
def t__scip(
    scip__finalize_context: ToolFinalizeContext,
    t__scip__run: ScipRunResult,
    t__scip__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
    scip__materializations: dict[str, MaterializationMetadata],
    scip__table_materializations: dict[str, MaterializationMetadata],
) -> TargetRunRecord:
    """SCIP index ingestion and GOID generation.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    for warning in t__scip__run.result.warnings:
        log.warning("SCIP run warning: %s", warning)
    for warning in t__scip__ingest.result.warnings:
        log.warning("SCIP ingest warning: %s", warning)

    record = finalize_target_from_materializations(
        context=scip__finalize_context,
        tool_step=t__scip__run,
        ingest_step=t__scip__ingest,
        artifact_materializations=scip__materializations,
        table_materializations=scip__table_materializations,
    )
    _emit_scip_teardown_safe(scip__finalize_context.env, t__scip__run, record)
    return record


__all__ = [
    "ScipRunResult",
    "t__scip",
    "t__scip__ingest",
    "t__scip__run",
]
