"""Native Hamilton implementation for scip target.

This target produces the SCIP artifacts declared in the build contract:
- ``scip_index``: ``{scip_dir}/index.scip``

Tool execution is delegated to the ingestion tool runtime (ToolRunner),
while persistence is DAG-visible via artifact saver nodes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import pyarrow as pa
import pyarrow.dataset as ds
from hamilton.function_modifiers import cache

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.ingestion.ingest_targets import ModuleToolOutput
from codeintel.build.hamilton.native.ingestion.manifesting import (
    IngestManifestDetails,
    finalize_ingest_reader_with_manifest,
)
from codeintel.build.hamilton.native.options.ingestion import ScipIngestOptions
from codeintel.build.hamilton.native.patterns import (
    ArtifactSaveSpec,
    IngestStep,
    MultiTableTargetContext,
    RelationTableSaveSpec,
    SaverContext,
    TableTargetTableContext,
    ToolFinalizeContext,
    ToolRunContext,
    attach_table_target_template,
    build_multi_table_target_spec_from_contexts,
    finalize_target_from_materializations,
    run_tool_step,
    save_artifact,
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
from codeintel.build.hashing import compute_options_hash
from codeintel.build.resources import TOOL_EXECUTION, TargetResources
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.arrow_ops import (
    iter_rows,
    normalize_table_for_join,
)
from codeintel.build.tabular.compute_columns import append_constant_columns
from codeintel.build.tabular.conversion import table_to_reader, tabular_to_arrow_table
from codeintel.build.tabular.finalize_ops import (
    FinalizeDedupe,
    FinalizeResult,
    finalize_join_keys,
    finalize_spec_for_table,
    finalize_table,
    record_join_precheck_errors,
)
from codeintel.build.tabular.plan_ops import HashJoinSpec
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.arrowdsl import ExecutionPlan, PipelineRunOptions, run_pipeline
from codeintel.core.columnar.execution_context import (
    ExecutionContext,
    resolve_columnar_context,
    resolve_execution_context,
)
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.finalize_ops import (
    finalize_spec_for_table as columnar_finalize_spec_for_table,
)
from codeintel.core.columnar.plan_builder import build_grouped_rollup_plan, build_table_plan
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.rows import (
    columnar_row_count,
    empty_table_for_table,
    table_for_columnar_rows,
    table_for_rows,
)
from codeintel.core.columnar.schema_ops import concat_tables_unified
from codeintel.core.columnar.streaming import ScanTelemetry
from codeintel.core.config.settings import ObservabilitySettings
from codeintel.core.datasets.arrow_store import scan_dataset
from codeintel.core.execution.ids import new_run_id
from codeintel.core.query_results import iter_records_from_arrow_reader
from codeintel.core.schemas.primitives import resolve_join_safe_columns
from codeintel.core.spans import normalize_byte_span
from codeintel.core.tools import ToolName
from codeintel.ingestion.compute.plan_surface import (
    IngestQuery,
    ingest_reader_for_dataset,
    ingest_scan_telemetry_for_dataset,
)
from codeintel.ingestion.engine.infrastructure import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolRunOptions,
)
from codeintel.ingestion.ports.change_detection import ChangeSet, FileDigest
from codeintel.ingestion.ports.tools import ScipDocument, ScipOccurrence
from codeintel.ingestion.scip import (
    SCIP_DIAGNOSTICS_TABLE_KEY,
    SCIP_EXTERNAL_SYMBOLS_TABLE_KEY,
    SCIP_INDEX_METADATA_TABLE_KEY,
    SCIP_MODULE_STATE_TABLE_KEY,
    SCIP_OCCURRENCES_TABLE_KEY,
    SCIP_RELATIONSHIPS_TABLE_KEY,
    SCIP_SYMBOL_INFO_TABLE_KEY,
    SCIP_SYMBOLS_TABLE_KEY,
    ScipParsedIndex,
    ScipRowContext,
    parse_index,
    rebase_parsed_index,
)
from codeintel.ingestion.scip.environment import resolve_environment_json
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
from codeintel.ingestion.scip.rows import (
    iter_diagnostic_rows,
    iter_external_symbol_rows,
    iter_index_metadata_rows,
    iter_module_state_rows,
    iter_occurrence_rows,
    iter_symbol_information_rows,
    iter_symbol_relationship_rows,
    iter_symbol_rows,
)
from codeintel.ingestion.scip.telemetry import ScipRunIdentity, ScipRunTelemetry
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

if TYPE_CHECKING:
    from codeintel.config.models import ToolsConfig

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, Path, TargetRunRecord)

ScipRunMode = Literal["incremental", "full", "skipped", "precheck_failed", "unknown"]

SCIP_TARGET_NAME = "scip"
SCIP_ARTIFACT_INDEX = "scip_index"

SCIP_TABLE_KEYS = (
    SCIP_SYMBOLS_TABLE_KEY,
    SCIP_OCCURRENCES_TABLE_KEY,
    SCIP_SYMBOL_INFO_TABLE_KEY,
    SCIP_RELATIONSHIPS_TABLE_KEY,
    SCIP_DIAGNOSTICS_TABLE_KEY,
    SCIP_EXTERNAL_SYMBOLS_TABLE_KEY,
    SCIP_INDEX_METADATA_TABLE_KEY,
    SCIP_MODULE_STATE_TABLE_KEY,
)
FILE_STATE_TABLE_KEY = "core.file_state"
FILE_LINE_INDEX_TABLE_KEY = "core.file_line_index"
_INTERNAL_PLAN_TABLE_KEY = "internal.plan_materialize"

_MODULE = sys.modules[__name__]
SCIP_SAVE_CONTEXT = SaverContext(domain="ingestion", target=SCIP_TARGET_NAME)


_POSITION_ENCODING_UTF8 = 1
_POSITION_ENCODING_UTF16 = 2
_POSITION_ENCODING_UTF32 = 3

_BOM_UTF8 = b"\xef\xbb\xbf"
_BOM_UTF16_LE = b"\xff\xfe"
_BOM_UTF16_BE = b"\xfe\xff"


@dataclass(frozen=True)
class ScipRunResult(ToolStepOutput):
    """Outcome for a SCIP tool run with telemetry metadata."""

    run_id: str = ""
    mode: ScipRunMode = "unknown"


@dataclass(frozen=True)
class ScipRunRecord:
    """Structured record for build.scip_runs telemetry rows."""

    run_id: str
    repo: str
    commit: str
    mode: str
    options_hash: str | None
    project_version: str | None
    project_namespace: str | None
    tool_version: str | None
    total_modules: int
    changed_modules: int
    deleted_modules: int
    changed_ratio: float | None
    batch_size: int | None
    batch_count: int
    decision: str | None
    ratio_gate_applied: bool | None
    ratio_gate_min_modules: int | None
    ratio_gate_min_changed: int | None
    hash_source: str | None
    hash_source_breakdown: str | None
    hash_reused: int
    hash_computed: int
    plan_ms: float | None
    hash_ms: float | None
    tool_ms: float | None
    parse_ms: float | None
    merge_ms: float | None
    write_ms: float | None
    total_ms: float | None
    status: str
    error_summary: str | None
    output_scip: str | None
    recorded_at: datetime


@dataclass(frozen=True)
class ScipRowPayload:
    """Row payloads for SCIP ingestion tables."""

    symbol_rows: InferableTabularInput
    occurrence_rows: InferableTabularInput
    symbol_info_rows: InferableTabularInput
    relationship_rows: InferableTabularInput
    diagnostic_rows: InferableTabularInput
    external_symbol_rows: InferableTabularInput
    index_metadata_rows: InferableTabularInput
    symbol_row_count: int
    occurrence_row_count: int
    symbol_info_row_count: int
    relationship_row_count: int
    diagnostic_row_count: int
    external_symbol_row_count: int
    index_metadata_row_count: int


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


@dataclass(frozen=True)
class ScipIngestInputs:
    """Inputs required to build SCIP ingestion rows."""

    modules: TargetRunRecord
    run: ScipRunResult
    proto_module_path: Path | None
    options: ScipIngestOptions


@cache(behavior="ignore")
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


@dataclass(frozen=True)
class _ScipOptionsHashInputs:
    options: ScipIngestOptions
    tools_config: ToolsConfig
    tool_version: str | None
    project_version: str | None
    project_namespace: str | None
    environment_json: Path | None
    environment_source: str | None
    environment_json_hash: str | None


def _scip_options_hash(
    inputs: _ScipOptionsHashInputs,
) -> str | None:
    payload = asdict(inputs.options)
    payload["environment_json"] = inputs.environment_json
    payload["environment_json_hash"] = inputs.environment_json_hash
    payload["environment_source"] = inputs.environment_source
    payload["project_name"] = inputs.tools_config.scip_project_name
    payload["tool_version"] = inputs.tool_version
    payload["project_version"] = inputs.project_version
    payload["project_namespace"] = inputs.project_namespace
    return compute_options_hash(payload)


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _hash_file(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        payload = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(payload).hexdigest()


def _resolve_project_version(
    options: ScipIngestOptions,
    *,
    commit: str,
    default_value: str | None,
) -> str | None:
    mode = options.project_version_mode.strip().lower()
    if mode == "commit":
        return commit
    if mode == "constant":
        return _normalize_optional_text(options.project_version_value or default_value)
    if mode == "unset":
        return _normalize_optional_text(default_value)
    return None


def _normalize_project_namespace(
    project_namespace: str | None,
    *,
    default_value: str | None,
) -> str | None:
    return _normalize_optional_text(project_namespace or default_value)


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


@dataclass(frozen=True)
class _FileLineIndex:
    encoding: str | None
    lines: dict[int, tuple[int, int]]


@dataclass(frozen=True)
class _EncodingContext:
    decode_encoding: str
    encode_encoding: str
    bom: bytes


def _chunked(values: Iterable[str], size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _dataset_for_snapshot(env: BuildEnv, *, table_key: str) -> ds.Dataset | None:
    dataset_root = env.paths.dataset_root_dir
    if dataset_root is None:
        return None
    try:
        return scan_dataset(
            dataset_root=dataset_root,
            table_key=table_key,
            snapshot_id=env.snapshot.commit,
        )
    except FileNotFoundError:
        return None
    except (OSError, ValueError, pa.ArrowInvalid) as exc:
        log.warning("Dataset scan failed for %s: %s", table_key, exc)
        return None


def _resolve_ingest_execution_ctx(env: BuildEnv | None) -> ExecutionContext:
    if env is not None:
        resolved = resolve_columnar_context(env.execution_context)
        if resolved is not None:
            return resolved
    fallback: ExecutionContext | None = None
    return resolve_execution_context(fallback)


def _join_safe_allowlist(table_key: str | None) -> tuple[str, ...]:
    if table_key is None:
        return ()
    schema = get_schema_service().get_table_schema(table_key)
    return resolve_join_safe_columns(schema)


def _file_line_index_columns(available: set[str]) -> tuple[str, ...] | None:
    required = ("rel_path", "line", "start_byte", "end_byte")
    if not set(required).issubset(available):
        return None
    columns: list[str] = list(required)
    if "encoding" in available:
        columns.append("encoding")
    return tuple(columns)


def _file_line_index_reader(
    dataset: ds.Dataset,
    *,
    columns: Sequence[str] | None,
    rel_paths: Sequence[str],
    execution_ctx: ExecutionContext,
    env: BuildEnv,
) -> pa.RecordBatchReader:
    query = IngestQuery(
        table_key=FILE_LINE_INDEX_TABLE_KEY,
        columns=columns,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        rel_paths=rel_paths,
    )
    return ingest_reader_for_dataset(dataset, query=query, ctx=execution_ctx)


def _scan_telemetry_payload(telemetry: ScanTelemetry) -> dict[str, int | None]:
    return {
        "fragment_count": telemetry.fragment_count,
        "estimated_rows": telemetry.estimated_rows,
    }


@lru_cache(maxsize=16)
def _scan_telemetry_for_table(
    dataset_root: str,
    table_key: str,
    repo: str,
    commit: str,
) -> dict[str, int | None] | None:
    try:
        dataset = scan_dataset(
            dataset_root=Path(dataset_root),
            table_key=table_key,
            snapshot_id=commit,
        )
    except FileNotFoundError:
        return None
    except (OSError, ValueError, pa.ArrowInvalid) as exc:
        log.warning("Scan telemetry unavailable for %s: %s", table_key, exc)
        return None
    query = IngestQuery(
        table_key=table_key,
        repo=repo,
        commit=commit,
    )
    try:
        telemetry = ingest_scan_telemetry_for_dataset(dataset, query=query)
    except ValueError as exc:
        log.warning("Scan telemetry spec failed for %s: %s", table_key, exc)
        return None
    return _scan_telemetry_payload(telemetry)


def _scip_scan_telemetry(env: BuildEnv) -> dict[str, dict[str, int | None]]:
    dataset_root = env.paths.dataset_root_dir
    if dataset_root is None:
        return {}
    repo = env.snapshot.repo
    commit = env.snapshot.commit
    payload: dict[str, dict[str, int | None]] = {}
    for table_key in (FILE_LINE_INDEX_TABLE_KEY, SCIP_MODULE_STATE_TABLE_KEY):
        telemetry = _scan_telemetry_for_table(str(dataset_root), table_key, repo, commit)
        if telemetry is not None:
            payload[table_key] = telemetry
    return payload


def _merge_file_line_index_rows(
    results: dict[str, _FileLineIndex],
    reader: pa.RecordBatchReader,
) -> None:
    for row in iter_records_from_arrow_reader(reader):
        rel_path = row.get("rel_path")
        if not isinstance(rel_path, str):
            continue
        line = _coerce_int(row.get("line"))
        start_byte = _coerce_int(row.get("start_byte"))
        end_byte = _coerce_int(row.get("end_byte"))
        if line is None or start_byte is None or end_byte is None:
            continue
        encoding = row.get("encoding")
        file_index = results.get(rel_path)
        if file_index is None:
            file_index = _FileLineIndex(
                encoding=encoding if isinstance(encoding, str) else None,
                lines={},
            )
            results[rel_path] = file_index
        file_index.lines[line] = (start_byte, end_byte)


def _load_file_line_index(
    env: BuildEnv,
    rel_paths: Iterable[str],
) -> dict[str, _FileLineIndex]:
    rel_path_list = sorted({path for path in rel_paths if path})
    if not rel_path_list:
        return {}
    dataset = _dataset_for_snapshot(env, table_key=FILE_LINE_INDEX_TABLE_KEY)
    if dataset is None:
        return {}
    available = set(dataset.schema.names)
    columns = _file_line_index_columns(available)
    if columns is None:
        return {}
    execution_ctx = _resolve_ingest_execution_ctx(env)
    results: dict[str, _FileLineIndex] = {}
    try:
        for chunk in _chunked(rel_path_list, 500):
            reader = _file_line_index_reader(
                dataset,
                columns=None,
                rel_paths=chunk,
                execution_ctx=execution_ctx,
                env=env,
            )
            _merge_file_line_index_rows(results, reader)
    except (OSError, RuntimeError, ValueError, pa.ArrowInvalid, pa.ArrowTypeError):
        log.warning("File line index unavailable; skipping SCIP byte span normalization")
        return {}
    return results


def _resolve_encoding_context(
    file_bytes: bytes,
    *,
    encoding_hint: str | None,
    text_document_encoding: str | None,
) -> _EncodingContext:
    encoding = (encoding_hint or text_document_encoding or "utf-8").lower()
    if encoding in {"utf-8", "utf8", "utf-8-sig", "utf8-sig"}:
        bom = _BOM_UTF8 if file_bytes.startswith(_BOM_UTF8) else b""
        return _EncodingContext(decode_encoding="utf-8", encode_encoding="utf-8", bom=bom)
    if encoding in {"utf-16", "utf16", "utf-16-le", "utf16le", "utf-16-be", "utf16be"}:
        if file_bytes.startswith(_BOM_UTF16_LE):
            return _EncodingContext(
                decode_encoding="utf-16-le",
                encode_encoding="utf-16-le",
                bom=_BOM_UTF16_LE,
            )
        if file_bytes.startswith(_BOM_UTF16_BE):
            return _EncodingContext(
                decode_encoding="utf-16-be",
                encode_encoding="utf-16-be",
                bom=_BOM_UTF16_BE,
            )
        if "be" in encoding:
            return _EncodingContext(
                decode_encoding="utf-16-be",
                encode_encoding="utf-16-be",
                bom=b"",
            )
        return _EncodingContext(
            decode_encoding="utf-16-le",
            encode_encoding="utf-16-le",
            bom=b"",
        )
    return _EncodingContext(decode_encoding=encoding, encode_encoding=encoding, bom=b"")


def _prefix_for_position(
    text: str,
    col: int,
    position_encoding: int,
) -> str | None:
    if col < 0:
        return None
    prefix: str | None = None
    if position_encoding == _POSITION_ENCODING_UTF8:
        encoded = text.encode("utf-8")
        if col <= len(encoded):
            prefix = encoded[:col].decode("utf-8", errors="ignore")
    elif position_encoding == _POSITION_ENCODING_UTF16:
        encoded = text.encode("utf-16-le")
        byte_len = col * 2
        if byte_len <= len(encoded):
            prefix = encoded[:byte_len].decode("utf-16-le", errors="ignore")
    elif position_encoding == _POSITION_ENCODING_UTF32 and col <= len(text):
        prefix = text[:col]
    return prefix


def _byte_offset_for_position(
    *,
    line_bytes: bytes,
    line_start: int,
    col: int,
    position_encoding: int,
    context: _EncodingContext,
) -> int | None:
    trimmed = line_bytes
    offset_adjust = 0
    if line_start == 0 and context.bom and trimmed.startswith(context.bom):
        trimmed = trimmed[len(context.bom) :]
        offset_adjust = len(context.bom)
    if position_encoding == _POSITION_ENCODING_UTF8:
        if col < 0 or col > len(trimmed):
            return None
        return line_start + offset_adjust + col
    try:
        text = trimmed.decode(context.decode_encoding, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return None
    prefix = _prefix_for_position(text, col, position_encoding)
    if prefix is None:
        return None
    try:
        prefix_bytes = prefix.encode(context.encode_encoding)
    except (LookupError, UnicodeEncodeError):
        return None
    return line_start + offset_adjust + len(prefix_bytes)


def _byte_span_for_occurrence(
    *,
    occ: ScipOccurrence,
    position_encoding: int | None,
    file_index: _FileLineIndex,
    file_bytes: bytes,
    context: _EncodingContext,
) -> tuple[int, int] | None:
    if position_encoding is None:
        return None
    start_line = occ.range_start_line
    end_line = occ.range_end_line
    start_info = file_index.lines.get(start_line)
    end_info = file_index.lines.get(end_line)
    if start_info is None or end_info is None:
        return None
    start_line_start, start_line_end = start_info
    end_line_start, end_line_end = end_info
    start_line_bytes = file_bytes[start_line_start:start_line_end]
    end_line_bytes = file_bytes[end_line_start:end_line_end]
    start_byte = _byte_offset_for_position(
        line_bytes=start_line_bytes,
        line_start=start_line_start,
        col=occ.range_start_col,
        position_encoding=position_encoding,
        context=context,
    )
    if start_byte is None:
        return None
    end_byte = _byte_offset_for_position(
        line_bytes=end_line_bytes,
        line_start=end_line_start,
        col=occ.range_end_col,
        position_encoding=position_encoding,
        context=context,
    )
    if end_byte is None:
        return None
    normalized = normalize_byte_span(start_byte, end_byte)
    if normalized is None:
        return None
    return normalized


def _document_has_byte_spans(doc: ScipDocument) -> bool:
    return all(occ.start_byte is not None and occ.end_byte is not None for occ in doc.occurrences)


def _load_document_bytes(repo_root: Path, rel_path: str) -> bytes | None:
    file_path = repo_root / Path(rel_path)
    try:
        return file_path.read_bytes()
    except OSError as exc:
        log.warning("Failed to read file for SCIP byte spans: %s", exc)
        return None


def _update_occurrence_span(
    *,
    occ: ScipOccurrence,
    doc: ScipDocument,
    file_index: _FileLineIndex,
    file_bytes: bytes,
    context: _EncodingContext,
) -> tuple[ScipOccurrence, bool]:
    if occ.start_byte is not None and occ.end_byte is not None:
        return occ, False
    span = _byte_span_for_occurrence(
        occ=occ,
        position_encoding=occ.position_encoding or doc.position_encoding,
        file_index=file_index,
        file_bytes=file_bytes,
        context=context,
    )
    if span is None:
        return occ, False
    return (
        ScipOccurrence(
            symbol=occ.symbol,
            range_start_line=occ.range_start_line,
            range_start_col=occ.range_start_col,
            range_end_line=occ.range_end_line,
            range_end_col=occ.range_end_col,
            symbol_roles=occ.symbol_roles,
            syntax_kind=occ.syntax_kind,
            enclosing_start_line=occ.enclosing_start_line,
            enclosing_start_col=occ.enclosing_start_col,
            enclosing_end_line=occ.enclosing_end_line,
            enclosing_end_col=occ.enclosing_end_col,
            override_documentation=occ.override_documentation,
            position_encoding=occ.position_encoding,
            text_document_encoding=occ.text_document_encoding,
            start_byte=span[0],
            end_byte=span[1],
        ),
        True,
    )


def _apply_line_index_to_doc(
    *,
    doc: ScipDocument,
    file_index: _FileLineIndex,
    file_bytes: bytes,
    context: _EncodingContext,
) -> tuple[ScipDocument, bool]:
    updated_occurrences: list[ScipOccurrence] = []
    changed = False
    for occ in doc.occurrences:
        updated, occ_changed = _update_occurrence_span(
            occ=occ,
            doc=doc,
            file_index=file_index,
            file_bytes=file_bytes,
            context=context,
        )
        updated_occurrences.append(updated)
        if occ_changed:
            changed = True
    if not changed:
        return doc, False
    return (
        ScipDocument(
            relative_path=doc.relative_path,
            symbols=doc.symbols,
            occurrences=tuple(updated_occurrences),
            position_encoding=doc.position_encoding,
            text_document_encoding=doc.text_document_encoding,
        ),
        True,
    )


def _apply_file_line_index(
    env: BuildEnv,
    parsed: ScipParsedIndex,
) -> ScipParsedIndex:
    rel_paths = {doc.relative_path for doc in parsed.documents}
    if not rel_paths:
        return parsed
    line_indexes = _load_file_line_index(env, rel_paths)
    if not line_indexes:
        return parsed

    repo_root = Path(env.snapshot.repo_root)
    updated_docs: list[ScipDocument] = []
    changed = False
    for doc in parsed.documents:
        file_index = line_indexes.get(doc.relative_path)
        if file_index is None or not doc.occurrences:
            updated_docs.append(doc)
            continue
        if _document_has_byte_spans(doc):
            updated_docs.append(doc)
            continue
        file_bytes = _load_document_bytes(repo_root, doc.relative_path)
        if file_bytes is None:
            updated_docs.append(doc)
            continue
        context = _resolve_encoding_context(
            file_bytes,
            encoding_hint=file_index.encoding,
            text_document_encoding=doc.text_document_encoding,
        )
        updated_doc, doc_changed = _apply_line_index_to_doc(
            doc=doc,
            file_index=file_index,
            file_bytes=file_bytes,
            context=context,
        )
        if doc_changed:
            changed = True
        updated_docs.append(updated_doc)
    if not changed:
        return parsed
    return ScipParsedIndex(
        documents=tuple(updated_docs),
        symbol_infos=parsed.symbol_infos,
        relationships=parsed.relationships,
        diagnostics=parsed.diagnostics,
        external_symbols=parsed.external_symbols,
        metadata=parsed.metadata,
        project_root=parsed.project_root,
    )


def _resolve_change_set(scan: ModuleToolOutput) -> tuple[ChangeSet, bool]:
    change_set = scan.change_set
    if change_set is None:
        log.warning("Module change set missing; forcing full SCIP rebuild")
        return ChangeSet(), True
    return change_set, False


def _load_module_state_rows(env: BuildEnv) -> list[dict[str, object]]:
    dataset = _dataset_for_snapshot(env, table_key=SCIP_MODULE_STATE_TABLE_KEY)
    if dataset is None:
        return []
    available = set(dataset.schema.names)
    required = {"rel_path", "content_hash", "shard_path", "updated_at"}
    if not required.issubset(available):
        return []
    execution_ctx = _resolve_ingest_execution_ctx(env)
    query = IngestQuery(
        table_key=SCIP_MODULE_STATE_TABLE_KEY,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
    )
    reader = ingest_reader_for_dataset(dataset, query=query, ctx=execution_ctx)
    return list(iter_records_from_arrow_reader(reader))


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
        and left.environment_source == right.environment_source
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
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        pa.ArrowInvalid,
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
    file_state_rows: pa.RecordBatchReader | pa.Table,
) -> dict[str, FileDigest]:
    digest_by_path: dict[str, FileDigest] = {}
    if isinstance(file_state_rows, pa.Table):
        row_iter = iter_rows(file_state_rows)
    else:
        row_iter = iter_records_from_arrow_reader(file_state_rows)
    for row in row_iter:
        rel_path_raw = row.get("rel_path")
        size_raw = row.get("size_bytes")
        mtime_raw = row.get("mtime_ns")
        hash_raw = row.get("content_hash")
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


@dataclass(frozen=True)
class _ScipIncrementalContext:
    change_set: ChangeSet
    force_full_rebuild: bool
    tool_version: str | None
    tools_config: ToolsConfig
    project_version: str | None
    project_namespace: str | None
    environment_json: Path | None
    environment_source: str | None
    environment_json_hash: str | None
    options_hash: str | None


def _build_incremental_context(
    env: BuildEnv,
    run_config: ScipRunConfig,
    module_inputs: ScipModuleInputs,
    output_scip: Path,
    *,
    run_id: str,
) -> _ScipIncrementalContext | ScipRunResult:
    change_set, force_full_rebuild = _resolve_change_set(module_inputs.scan)
    tool_version = _scip_tool_version(env)
    tools_config = env.providers.tool_runner.tools_config
    project_version = _resolve_project_version(
        run_config.options,
        commit=env.snapshot.commit,
        default_value=tools_config.scip_project_version,
    )
    project_namespace = _normalize_project_namespace(
        run_config.options.project_namespace,
        default_value=tools_config.scip_project_namespace,
    )
    scip_dir = output_scip.parent
    try:
        env_resolution = resolve_environment_json(
            environment_json=run_config.options.environment_json,
            scip_dir=scip_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        telemetry = ScipRunTelemetry.create(
            identity=ScipRunIdentity(
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
                run_id=run_id,
                options_hash=None,
                project_version=project_version,
                project_namespace=project_namespace,
                environment_source=None,
            )
        )
        telemetry.status = "failed"
        telemetry.error_summary = str(exc)
        telemetry.total_ms = 0.0
        _persist_scip_telemetry_safe(env, telemetry)
        return ScipRunResult(
            result=ExecutionResult.failed(str(exc)),
            run_id=run_id,
            mode=_normalize_scip_run_mode("incremental"),
        )

    environment_json = env_resolution.environment_json
    environment_source = env_resolution.source
    environment_json_hash = _hash_file(environment_json)
    options_hash = _scip_options_hash(
        _ScipOptionsHashInputs(
            options=run_config.options,
            tools_config=tools_config,
            tool_version=tool_version,
            project_version=project_version,
            project_namespace=project_namespace,
            environment_json=environment_json,
            environment_source=environment_source,
            environment_json_hash=environment_json_hash,
        )
    )
    return _ScipIncrementalContext(
        change_set=change_set,
        force_full_rebuild=force_full_rebuild,
        tool_version=tool_version,
        tools_config=tools_config,
        project_version=project_version,
        project_namespace=project_namespace,
        environment_json=environment_json,
        environment_source=environment_source,
        environment_json_hash=environment_json_hash,
        options_hash=options_hash,
    )


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

    context = _build_incremental_context(
        env,
        run_config,
        module_inputs,
        output_scip,
        run_id=run_id,
    )
    if isinstance(context, ScipRunResult):
        return context
    telemetry = ScipRunTelemetry.create(
        identity=ScipRunIdentity(
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            run_id=run_id,
            options_hash=context.options_hash,
            project_version=context.project_version,
            project_namespace=context.project_namespace,
            environment_source=context.environment_source,
        )
    )
    file_state_rows = module_inputs.scan.file_state_rows
    if (
        module_inputs.scan.file_state_row_count == 0
        and columnar_row_count(context.change_set.state_rows) > 0
    ):
        file_state_rows, _ = table_for_columnar_rows(
            FILE_STATE_TABLE_KEY,
            context.change_set.state_rows,
        )
    file_state_by_path = _build_file_state_map(file_state_rows)
    try:
        config = ScipIncrementalConfig(
            repo_root=env.snapshot.repo_root,
            output_scip=output_scip,
            proto_module_path=run_config.proto_module_path,
            change_set=context.change_set,
            modules=module_inputs.scan.modules,
            options_hash=context.options_hash,
            tools_config=context.tools_config,
            tool_runner=env.providers.tool_runner,
            scope_paths=run_config.options.scope_paths,
            environment_json=context.environment_json,
            pyright_config_path=run_config.options.pyright_config_path,
            project_version=context.project_version,
            project_namespace=context.project_namespace,
            max_file_size_kb=run_config.options.max_file_size_kb,
            timeout_seconds=run_config.options.timeout_seconds,
            scip_node_max_old_space_mb=run_config.options.scip_node_max_old_space_mb,
            target_dir=None,
            force_full_rebuild=context.force_full_rebuild,
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
    catalog: DagCatalog,
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
        catalog=catalog,
        target_name=SCIP_TARGET_NAME,
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
    if scip_output.result.skipped and output_scip.is_file():
        return ScipRunResult(
            result=ExecutionResult.skip("SCIP target skipped"),
            outputs={SCIP_ARTIFACT_INDEX: output_scip},
            run_id=run_id,
            mode="skipped",
        )
    if scip_output.result.skipped:
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
    table_specs = {
        "symbol": (
            SCIP_SYMBOLS_TABLE_KEY,
            iter_symbol_rows(parsed.documents, row_context),
        ),
        "occurrence": (
            SCIP_OCCURRENCES_TABLE_KEY,
            iter_occurrence_rows(parsed.documents, row_context),
        ),
        "symbol_info": (
            SCIP_SYMBOL_INFO_TABLE_KEY,
            iter_symbol_information_rows(parsed.symbol_infos, row_context),
        ),
        "relationship": (
            SCIP_RELATIONSHIPS_TABLE_KEY,
            iter_symbol_relationship_rows(parsed.relationships, row_context),
        ),
        "diagnostic": (
            SCIP_DIAGNOSTICS_TABLE_KEY,
            iter_diagnostic_rows(parsed.diagnostics, row_context),
        ),
        "external_symbol": (
            SCIP_EXTERNAL_SYMBOLS_TABLE_KEY,
            iter_external_symbol_rows(parsed.external_symbols, row_context),
        ),
        "index_metadata": (
            SCIP_INDEX_METADATA_TABLE_KEY,
            iter_index_metadata_rows(parsed.metadata, row_context),
        ),
    }
    tables: dict[str, tuple[pa.Table, int]] = {}
    for name, (table_key, iterator) in table_specs.items():
        tables[name] = table_for_rows(table_key, iterator, extras_policy="retain")
    return ScipRowPayload(
        symbol_rows=tables["symbol"][0],
        occurrence_rows=tables["occurrence"][0],
        symbol_info_rows=tables["symbol_info"][0],
        relationship_rows=tables["relationship"][0],
        diagnostic_rows=tables["diagnostic"][0],
        external_symbol_rows=tables["external_symbol"][0],
        index_metadata_rows=tables["index_metadata"][0],
        symbol_row_count=tables["symbol"][1],
        occurrence_row_count=tables["occurrence"][1],
        symbol_info_row_count=tables["symbol_info"][1],
        relationship_row_count=tables["relationship"][1],
        diagnostic_row_count=tables["diagnostic"][1],
        external_symbol_row_count=tables["external_symbol"][1],
        index_metadata_row_count=tables["index_metadata"][1],
    )


def _scip_table_counts(
    payload: ScipRowPayload,
    module_state_count: int,
) -> dict[str, int]:
    return {
        SCIP_SYMBOLS_TABLE_KEY: payload.symbol_row_count,
        SCIP_OCCURRENCES_TABLE_KEY: payload.occurrence_row_count,
        SCIP_SYMBOL_INFO_TABLE_KEY: payload.symbol_info_row_count,
        SCIP_RELATIONSHIPS_TABLE_KEY: payload.relationship_row_count,
        SCIP_DIAGNOSTICS_TABLE_KEY: payload.diagnostic_row_count,
        SCIP_EXTERNAL_SYMBOLS_TABLE_KEY: payload.external_symbol_row_count,
        SCIP_INDEX_METADATA_TABLE_KEY: payload.index_metadata_row_count,
        SCIP_MODULE_STATE_TABLE_KEY: module_state_count,
    }


def _build_module_state_frame(
    env: BuildEnv,
    run: ScipRunResult,
) -> tuple[InferableTabularInput, int]:
    index_path = _scip_index_output(run)
    scip_dir = index_path.parent if index_path is not None else env.paths.scip_dir
    manifest = load_manifest(manifest_path(scip_dir))
    rows_iter = iter_module_state_rows(
        manifest,
        env.snapshot.repo,
        env.snapshot.commit,
    )
    reader, row_count = table_for_rows(
        SCIP_MODULE_STATE_TABLE_KEY,
        rows_iter,
        extras_policy="retain",
    )
    return reader, row_count


def _build_scip_ingest_result(
    env: BuildEnv,
    inputs: ScipIngestInputs,
) -> IngestStep[dict[str, InferableTabularInput]]:
    precheck = _scip_ingest_precheck(inputs)
    if precheck is not None:
        return IngestStep(result=precheck)

    output_scip = _scip_index_output(inputs.run) or (env.paths.scip_dir / "index.scip")
    proto_module_path = cast("Path", inputs.proto_module_path)
    try:
        parsed: ScipParsedIndex = parse_index(output_scip, proto_module_path)
        parsed = rebase_parsed_index(parsed, Path(env.snapshot.repo_root))
        parsed = _apply_file_line_index(env, parsed)
        payload = _build_scip_row_payload(env, parsed, inputs.options)
    except (OSError, AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        log.exception("SCIP ingestion failed")
        return IngestStep(result=ExecutionResult.failed("SCIP ingestion failed with exception"))

    warnings: tuple[str, ...] | None = None
    if payload.symbol_row_count == 0 and payload.occurrence_row_count == 0:
        warnings = ("SCIP scope empty: no symbols or occurrences",)
    elif payload.symbol_row_count == 0 or payload.occurrence_row_count == 0:
        return IngestStep(
            result=ExecutionResult.failed("SCIP ingestion produced empty symbols or occurrences")
        )

    module_state_frame, module_state_count = _build_module_state_frame(env, inputs.run)
    table_counts = _scip_table_counts(payload, module_state_count)
    result = ExecutionResult.ok(
        table_counts=normalize_table_counts(SCIP_TABLE_KEYS, table_counts),
        warnings=warnings,
    )
    payload_by_table = {
        SCIP_SYMBOLS_TABLE_KEY: payload.symbol_rows,
        SCIP_OCCURRENCES_TABLE_KEY: payload.occurrence_rows,
        SCIP_SYMBOL_INFO_TABLE_KEY: payload.symbol_info_rows,
        SCIP_RELATIONSHIPS_TABLE_KEY: payload.relationship_rows,
        SCIP_DIAGNOSTICS_TABLE_KEY: payload.diagnostic_rows,
        SCIP_EXTERNAL_SYMBOLS_TABLE_KEY: payload.external_symbol_rows,
        SCIP_INDEX_METADATA_TABLE_KEY: payload.index_metadata_rows,
        SCIP_MODULE_STATE_TABLE_KEY: module_state_frame,
    }
    return IngestStep(result=result, payload=payload_by_table)


@tag_compute(domain="ingestion", target=SCIP_TARGET_NAME)
def t__scip__ingest(
    env: BuildEnv,
    scip__ingest_inputs: ScipIngestInputs,
) -> IngestStep[dict[str, InferableTabularInput]]:
    """Build SCIP row payloads for core.scip_* tables.

    Returns
    -------
    IngestStep[dict[str, InferableTabularInput]]
        Ingestion status and tabular payloads.
    """
    return _build_scip_ingest_result(env, scip__ingest_inputs)


def _scip_payload_frame(
    t__scip__ingest: IngestStep[dict[str, InferableTabularInput]],
    table_key: str,
) -> InferableTabularInput:
    if t__scip__ingest.result.skipped or not t__scip__ingest.result.success:
        return empty_table_for_table(table_key)
    payload = t__scip__ingest.payload
    if payload is None:
        msg = "Missing SCIP ingest payload"
        raise ValueError(msg)
    frame = payload.get(table_key)
    if frame is None:
        msg = f"Missing frame for {table_key}"
        raise ValueError(msg)
    return frame


def _scip_payload_table(
    t__scip__ingest: IngestStep[dict[str, InferableTabularInput]],
    table_key: str,
) -> pa.Table:
    return tabular_to_arrow_table(_scip_payload_frame(t__scip__ingest, table_key))


def _scip_manifest_extras(result: ExecutionResult, *, env: BuildEnv) -> dict[str, object]:
    status = "failed"
    if result.skipped:
        status = "skipped"
    elif result.success:
        status = "success"
    extras: dict[str, object] = {
        "tool_status": status,
        "input_table_counts": dict(result.table_counts),
    }
    if result.error is not None:
        extras["error"] = result.error
    if result.skip_reason is not None:
        extras["skip_reason"] = result.skip_reason
    if result.warnings:
        extras["warnings"] = list(result.warnings)
    scan_telemetry = _scip_scan_telemetry(env)
    if scan_telemetry:
        extras["scan_telemetry"] = scan_telemetry
    return extras


def _scip_manifest_details(
    t__scip__ingest: IngestStep[dict[str, InferableTabularInput]],
    *,
    env: BuildEnv,
) -> IngestManifestDetails:
    extras = _scip_manifest_extras(t__scip__ingest.result, env=env)
    return IngestManifestDetails(manifest_extras=extras)


def _finalize_scip_table(
    env: BuildEnv,
    table_key: str,
    table: pa.Table,
    *,
    details: IngestManifestDetails,
) -> pa.Table:
    reader = table_to_reader(table, batch_size=None)
    return finalize_ingest_reader_with_manifest(
        env=env,
        table_key=table_key,
        reader=reader,
        target_name=SCIP_TARGET_NAME,
        details=details,
    )


def scip__symbol_rows__base(
    env: BuildEnv,
    t__scip__ingest: IngestStep[dict[str, InferableTabularInput]],
) -> InferableTabularInput:
    """Return rows for core.scip_symbols.

    Parameters
    ----------
    env
        Build environment with snapshot metadata.
    t__scip__ingest
        Ingest payload containing SCIP tables.

    Returns
    -------
    InferableTabularInput
        Tabular input for core.scip_symbols.
    """
    table = _scip_payload_table(t__scip__ingest, SCIP_SYMBOLS_TABLE_KEY)
    details = _scip_manifest_details(t__scip__ingest, env=env)
    return _finalize_scip_table(env, SCIP_SYMBOLS_TABLE_KEY, table, details=details)


def scip__occurrence_rows__base(
    env: BuildEnv,
    t__scip__ingest: IngestStep[dict[str, InferableTabularInput]],
) -> InferableTabularInput:
    """Return rows for core.scip_occurrences.

    Parameters
    ----------
    env
        Build environment with snapshot metadata.
    t__scip__ingest
        Ingest payload containing SCIP tables.

    Returns
    -------
    InferableTabularInput
        Tabular input for core.scip_occurrences.
    """
    table = _scip_payload_table(t__scip__ingest, SCIP_OCCURRENCES_TABLE_KEY)
    details = _scip_manifest_details(t__scip__ingest, env=env)
    return _finalize_scip_table(env, SCIP_OCCURRENCES_TABLE_KEY, table, details=details)


def scip__symbol_info_rows__base(
    env: BuildEnv,
    t__scip__ingest: IngestStep[dict[str, InferableTabularInput]],
) -> InferableTabularInput:
    """Return rows for core.scip_symbol_information.

    Parameters
    ----------
    env
        Build environment with snapshot metadata.
    t__scip__ingest
        Ingest payload containing SCIP tables.

    Returns
    -------
    InferableTabularInput
        Tabular input for core.scip_symbol_information.
    """
    table = _scip_payload_table(t__scip__ingest, SCIP_SYMBOL_INFO_TABLE_KEY)
    details = _scip_manifest_details(t__scip__ingest, env=env)
    return _finalize_scip_table(env, SCIP_SYMBOL_INFO_TABLE_KEY, table, details=details)


def scip__relationship_rows__base(
    env: BuildEnv,
    t__scip__ingest: IngestStep[dict[str, InferableTabularInput]],
) -> InferableTabularInput:
    """Return rows for core.scip_symbol_relationships.

    Parameters
    ----------
    env
        Build environment with snapshot metadata.
    t__scip__ingest
        Ingest payload containing SCIP tables.

    Returns
    -------
    InferableTabularInput
        Tabular input for core.scip_symbol_relationships.
    """
    table = _scip_payload_table(t__scip__ingest, SCIP_RELATIONSHIPS_TABLE_KEY)
    details = _scip_manifest_details(t__scip__ingest, env=env)
    return _finalize_scip_table(env, SCIP_RELATIONSHIPS_TABLE_KEY, table, details=details)


def scip__diagnostic_rows__base(
    env: BuildEnv,
    t__scip__ingest: IngestStep[dict[str, InferableTabularInput]],
) -> InferableTabularInput:
    """Return rows for core.scip_diagnostics.

    Parameters
    ----------
    env
        Build environment with snapshot metadata.
    t__scip__ingest
        Ingest payload containing SCIP tables.

    Returns
    -------
    InferableTabularInput
        Tabular input for core.scip_diagnostics.
    """
    table = _scip_payload_table(t__scip__ingest, SCIP_DIAGNOSTICS_TABLE_KEY)
    details = _scip_manifest_details(t__scip__ingest, env=env)
    return _finalize_scip_table(env, SCIP_DIAGNOSTICS_TABLE_KEY, table, details=details)


def _derive_external_symbol_rows(
    occurrences: pa.Table,
    relationships: pa.Table,
    symbol_info: pa.Table,
    *,
    execution_ctx: ExecutionContext,
) -> pa.Table:
    table_key = SCIP_EXTERNAL_SYMBOLS_TABLE_KEY
    candidates: list[pa.Table] = []
    occ_required = {"repo", "commit", "symbol"}
    if occurrences.num_rows > 0 and occ_required.issubset(occurrences.column_names):
        candidates.append(occurrences.select(["repo", "commit", "symbol"]))
    rel_required = {"repo", "commit", "symbol", "related_symbol"}
    if relationships.num_rows > 0 and rel_required.issubset(relationships.column_names):
        candidates.append(relationships.select(["repo", "commit", "symbol"]))
        related = relationships.select(["repo", "commit", "related_symbol"]).rename_columns(
            ["repo", "commit", "symbol"]
        )
        candidates.append(related)
    if not candidates:
        return empty_table_for_table(table_key)
    combined = normalize_table_for_join(
        concat_tables_unified(candidates),
        allowed_columns=_join_safe_allowlist(SCIP_EXTERNAL_SYMBOLS_TABLE_KEY),
    )
    distinct = _distinct_external_symbol_rows(combined, execution_ctx=execution_ctx)
    info_required = {"repo", "commit", "symbol"}
    if symbol_info.num_rows == 0 or not info_required.issubset(symbol_info.column_names):
        missing = distinct
    else:
        info = normalize_table_for_join(
            symbol_info.select(["repo", "commit", "symbol"]),
            allowed_columns=_join_safe_allowlist(SCIP_SYMBOL_INFO_TABLE_KEY),
        )
        missing = _left_anti_external_symbols(distinct, info, execution_ctx=execution_ctx)
    if missing.num_rows == 0:
        return empty_table_for_table(table_key)
    return append_constant_columns(
        missing,
        {
            "package_manager": None,
            "package_name": None,
            "package_version": None,
            "created_at": datetime.now(UTC),
        },
    )


def _precheck_join_table(
    table: pa.Table,
    *,
    table_key: str | None,
    join_keys: Sequence[str],
) -> pa.Table:
    if table.num_rows == 0 or not join_keys:
        return table
    if table_key is None:
        result = finalize_join_keys(
            table,
            required_non_null=join_keys,
            key_fields=join_keys,
        )
    else:
        result = finalize_table(
            table,
            spec=finalize_spec_for_table(
                table_key,
                mode="tolerant",
                required_non_null=join_keys,
                key_fields=join_keys,
                dedupe=FinalizeDedupe(enabled=False),
                target_name=SCIP_TARGET_NAME,
            ),
        )
    record_join_precheck_errors(
        result,
        table_key=table_key,
        target_name=SCIP_TARGET_NAME,
        join_keys=join_keys,
    )
    _log_join_precheck_errors(result, table_key=table_key, join_keys=join_keys)
    return result.good


def _log_join_precheck_errors(
    result: FinalizeResult,
    *,
    table_key: str | None,
    join_keys: Sequence[str],
) -> None:
    if result.errors.num_rows == 0:
        return
    table_label = table_key or "derived"
    log.warning(
        "Join key precheck dropped %d rows table=%s keys=%s",
        result.errors.num_rows,
        table_label,
        ",".join(join_keys),
    )


def _plan_to_table(plan: Plan, *, execution_ctx: ExecutionContext) -> pa.Table:
    resolved_ctx = resolve_execution_context(execution_ctx)
    result = run_pipeline(
        plan=ExecutionPlan.from_plan(plan),
        finalize=columnar_finalize_spec_for_table(_INTERNAL_PLAN_TABLE_KEY, mode="tolerant"),
        options=PipelineRunOptions(ctx=resolved_ctx),
    )
    return result.good


def _distinct_external_symbol_rows(
    symbols: pa.Table,
    *,
    execution_ctx: ExecutionContext,
) -> pa.Table:
    if symbols.num_rows == 0:
        return symbols
    join_keys = ("repo", "commit", "symbol")
    checked = _precheck_join_table(
        symbols,
        table_key=SCIP_EXTERNAL_SYMBOLS_TABLE_KEY,
        join_keys=join_keys,
    )
    project = {name: E.field(name) for name in join_keys}
    plan = build_table_plan(table=checked).project(project)
    plan = build_grouped_rollup_plan(
        plan,
        keys=join_keys,
        aggregates=(),
        order_by=tuple((key, "ascending") for key in join_keys),
    )
    return _plan_to_table(plan, execution_ctx=execution_ctx)


def _left_anti_external_symbols(
    left: pa.Table,
    right: pa.Table,
    *,
    execution_ctx: ExecutionContext,
) -> pa.Table:
    if left.num_rows == 0:
        return left
    join_keys = ("repo", "commit", "symbol")
    project = {name: E.field(name) for name in join_keys}
    left_checked = _precheck_join_table(
        left,
        table_key=SCIP_EXTERNAL_SYMBOLS_TABLE_KEY,
        join_keys=join_keys,
    )
    right_checked = _precheck_join_table(
        right,
        table_key=SCIP_SYMBOL_INFO_TABLE_KEY,
        join_keys=join_keys,
    )
    left_checked = normalize_table_for_join(
        left_checked,
        allowed_columns=_join_safe_allowlist(SCIP_EXTERNAL_SYMBOLS_TABLE_KEY),
    )
    right_checked = normalize_table_for_join(
        right_checked,
        allowed_columns=_join_safe_allowlist(SCIP_SYMBOL_INFO_TABLE_KEY),
    )
    left_plan = build_table_plan(table=left_checked).project(project)
    right_plan = build_table_plan(table=right_checked).project(project)
    joined = left_plan.hash_join(
        right=right_plan,
        spec=HashJoinSpec(
            left_keys=list(join_keys),
            right_keys=list(join_keys),
            how="left anti",
            left_output=list(join_keys),
            right_output=[],
        ),
    )
    ordered = joined.order_by(sort_keys=[(key, "ascending") for key in join_keys])
    return _plan_to_table(ordered, execution_ctx=execution_ctx)


def scip__external_symbol_rows__base(
    env: BuildEnv,
    t__scip__ingest: IngestStep[dict[str, InferableTabularInput]],
    scip__occurrence_rows__base: InferableTabularInput,
    scip__relationship_rows__base: InferableTabularInput,
    scip__symbol_info_rows__base: InferableTabularInput,
) -> InferableTabularInput:
    """Return rows for core.scip_external_symbols.

    Parameters
    ----------
    env
        Build environment with snapshot metadata.
    t__scip__ingest
        Ingest payload containing SCIP tables.
    scip__occurrence_rows__base
        Occurrence rows used for external symbol derivation.
    scip__relationship_rows__base
        Relationship rows used for external symbol derivation.
    scip__symbol_info_rows__base
        Symbol info rows used for external symbol derivation.

    Returns
    -------
    InferableTabularInput
        Tabular input for core.scip_external_symbols.
    """
    execution_ctx = _resolve_ingest_execution_ctx(env)
    base_table = _scip_payload_table(t__scip__ingest, SCIP_EXTERNAL_SYMBOLS_TABLE_KEY)
    details = _scip_manifest_details(t__scip__ingest, env=env)
    derived = _derive_external_symbol_rows(
        tabular_to_arrow_table(scip__occurrence_rows__base),
        tabular_to_arrow_table(scip__relationship_rows__base),
        tabular_to_arrow_table(scip__symbol_info_rows__base),
        execution_ctx=execution_ctx,
    )
    if derived.num_rows == 0:
        return _finalize_scip_table(
            env,
            SCIP_EXTERNAL_SYMBOLS_TABLE_KEY,
            base_table,
            details=details,
        )
    combined = concat_tables_unified([base_table, derived])
    return _finalize_scip_table(
        env,
        SCIP_EXTERNAL_SYMBOLS_TABLE_KEY,
        combined,
        details=details,
    )


def scip__index_metadata_rows__base(
    env: BuildEnv,
    t__scip__ingest: IngestStep[dict[str, InferableTabularInput]],
) -> InferableTabularInput:
    """Return rows for core.scip_index_metadata.

    Parameters
    ----------
    env
        Build environment with snapshot metadata.
    t__scip__ingest
        Ingest payload containing SCIP tables.

    Returns
    -------
    InferableTabularInput
        Tabular input for core.scip_index_metadata.
    """
    table = _scip_payload_table(t__scip__ingest, SCIP_INDEX_METADATA_TABLE_KEY)
    details = _scip_manifest_details(t__scip__ingest, env=env)
    return _finalize_scip_table(env, SCIP_INDEX_METADATA_TABLE_KEY, table, details=details)


def scip__module_state_rows__base(
    env: BuildEnv,
    t__scip__ingest: IngestStep[dict[str, InferableTabularInput]],
) -> InferableTabularInput:
    """Return rows for core.scip_module_state.

    Parameters
    ----------
    env
        Build environment with snapshot metadata.
    t__scip__ingest
        Ingest payload containing SCIP tables.

    Returns
    -------
    InferableTabularInput
        Tabular input for core.scip_module_state.
    """
    table = _scip_payload_table(t__scip__ingest, SCIP_MODULE_STATE_TABLE_KEY)
    details = _scip_manifest_details(t__scip__ingest, env=env)
    return _finalize_scip_table(env, SCIP_MODULE_STATE_TABLE_KEY, table, details=details)


@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__materializations(
    m__artifact__scip_index: MaterializationResult,
) -> dict[str, MaterializationResult]:
    """Collect scip artifact materialization payloads into a single mapping.

    Returns
    -------
    dict[str, MaterializationResult]
        Materialization results keyed by artifact name.
    """
    return {
        SCIP_ARTIFACT_INDEX: m__artifact__scip_index,
    }


_SCIP_TABLE_TARGET_SPEC = build_multi_table_target_spec_from_contexts(
    context=MultiTableTargetContext(
        domain="ingestion",
        target_name=SCIP_TARGET_NAME,
        tables=(),
        table_materializations_node="scip__table_materializations",
        attach_anchor=False,
        save_spec_factory=RelationTableSaveSpec,
        default_input_type=InferableTabularInput,
    ),
    table_contexts=(
        TableTargetTableContext(
            table_key=SCIP_SYMBOLS_TABLE_KEY,
            base_node="scip__symbol_rows__base",
            node_name="scip__symbol_rows",
        ),
        TableTargetTableContext(
            table_key=SCIP_OCCURRENCES_TABLE_KEY,
            base_node="scip__occurrence_rows__base",
            node_name="scip__occurrence_rows",
        ),
        TableTargetTableContext(
            table_key=SCIP_SYMBOL_INFO_TABLE_KEY,
            base_node="scip__symbol_info_rows__base",
            node_name="scip__symbol_info_rows",
        ),
        TableTargetTableContext(
            table_key=SCIP_RELATIONSHIPS_TABLE_KEY,
            base_node="scip__relationship_rows__base",
            node_name="scip__relationship_rows",
        ),
        TableTargetTableContext(
            table_key=SCIP_DIAGNOSTICS_TABLE_KEY,
            base_node="scip__diagnostic_rows__base",
            node_name="scip__diagnostic_rows",
        ),
        TableTargetTableContext(
            table_key=SCIP_EXTERNAL_SYMBOLS_TABLE_KEY,
            base_node="scip__external_symbol_rows__base",
            node_name="scip__external_symbol_rows",
        ),
        TableTargetTableContext(
            table_key=SCIP_INDEX_METADATA_TABLE_KEY,
            base_node="scip__index_metadata_rows__base",
            node_name="scip__index_metadata_rows",
        ),
        TableTargetTableContext(
            table_key=SCIP_MODULE_STATE_TABLE_KEY,
            base_node="scip__module_state_rows__base",
            node_name="scip__module_state_rows",
        ),
    ),
)
attach_table_target_template(_MODULE, spec=_SCIP_TABLE_TARGET_SPEC)
scip__symbol_rows = _MODULE.scip__symbol_rows
scip__occurrence_rows = _MODULE.scip__occurrence_rows
scip__symbol_info_rows = _MODULE.scip__symbol_info_rows
scip__relationship_rows = _MODULE.scip__relationship_rows
scip__diagnostic_rows = _MODULE.scip__diagnostic_rows
scip__external_symbol_rows = _MODULE.scip__external_symbol_rows
scip__index_metadata_rows = _MODULE.scip__index_metadata_rows
scip__module_state_rows = _MODULE.scip__module_state_rows
scip__table_materializations = _MODULE.scip__table_materializations


@cache(behavior="ignore")
@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__finalize_context(
    env: BuildEnv,
    catalog: DagCatalog,
) -> ToolFinalizeContext:
    """Build finalization context for the SCIP target.

    Returns
    -------
    ToolFinalizeContext
        Finalization context for SCIP ingestion.
    """
    return ToolFinalizeContext(
        env=env,
        catalog=catalog,
        target_name=SCIP_TARGET_NAME,
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


def _resolve_observability_settings(env: BuildEnv) -> ObservabilitySettings:
    if env.execution_context is not None:
        return env.execution_context.observability_settings
    return ObservabilitySettings()


def _emit_scip_teardown(
    env: BuildEnv,
    run: ScipRunResult,
    record: TargetRunRecord,
) -> None:
    settings = _resolve_observability_settings(env)
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
    t__scip__ingest: IngestStep[dict[str, InferableTabularInput]],
    scip__materializations: dict[str, MaterializationResult],
    scip__table_materializations: dict[str, MaterializationResult],
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
