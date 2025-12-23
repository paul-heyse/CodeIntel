"""Native Hamilton implementation for scip target.

This target produces the SCIP artifacts declared in the build contract:
- ``scip_index``: ``{scip_dir}/index.scip``
- ``scip_json``: ``{scip_dir}/index.json``

Tool execution is delegated to the ingestion tool runtime (ToolService),
while persistence is DAG-visible via ``FileArtifactSaver``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from hamilton.function_modifiers import source, value

from codeintel.build.contracts import ArtifactSpec
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.materializers import DuckDBRowsSaver, FileArtifactSaver
from codeintel.build.hamilton.materializers.metadata import DuckDBMaterializationMetadata
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    record_from_file_artifact_materializations,
)
from codeintel.build.hamilton.native.table_counts import normalize_table_counts
from codeintel.build.hamilton.native.target_override_tables import SCIP_OVERRIDE_TABLES
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
    register_output_targets,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_helper, tag_materialize, tag_tool
from codeintel.build.resources import TOOL_EXECUTION, TargetResources
from codeintel.build.schemas import deferred_columns_for_table_key
from codeintel.build.targets import TargetGraph
from codeintel.core.schemas.row_serialization import row_serializer_for_table_key
from codeintel.ingestion.scip import (
    build_occurrence_rows,
    build_symbol_rows,
    parse_scip_json_file,
)

if TYPE_CHECKING:
    from codeintel.ingestion.ports.tools import ScipDocument

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, Path, TargetRunRecord)

SCIP_TARGET_NAME = "scip"
SCIP_ARTIFACT_INDEX = "scip_index"
SCIP_ARTIFACT_JSON = "scip_json"

SCIP_SYMBOLS_TABLE_KEY = "core.scip_symbols"
SCIP_OCCURRENCES_TABLE_KEY = "core.scip_occurrences"
SCIP_TABLE_KEYS = (SCIP_SYMBOLS_TABLE_KEY, SCIP_OCCURRENCES_TABLE_KEY)

SCIP_ARTIFACT_SPECS = (
    ArtifactSpec(SCIP_ARTIFACT_INDEX, "{scip_dir}/index.scip", "SCIP index file"),
    ArtifactSpec(SCIP_ARTIFACT_JSON, "{scip_dir}/index.json", "SCIP JSON export"),
)

register_output_targets(
    make_output_target(
        name=SCIP_TARGET_NAME,
        module="ingestion",
        description="SCIP index ingestion and GOID generation.",
        options=TargetSpecOptions(
            table_keys=SCIP_TABLE_KEYS,
            override_tables=SCIP_OVERRIDE_TABLES,
            artifacts=SCIP_ARTIFACT_SPECS,
            resources=TargetResources(
                tracker=True,
                modules=True,
                tools=(
                    "scip-python",
                    "scip",
                ),
            ),
            execution=TOOL_EXECUTION,
        ),
    ),
)


@dataclass(frozen=True)
class ScipRunResult:
    """Result from running SCIP tooling.

    Attributes
    ----------
    success
        Whether execution completed successfully.
    skipped
        Whether execution was skipped due to manifest match.
    documents
        Parsed SCIP documents, if available.
    index_path
        Path to the generated index.scip.
    json_path
        Path to the generated index.json.
    error
        Error message on failure.
    """

    success: bool
    skipped: bool = False
    documents: tuple[ScipDocument, ...] = ()
    index_path: Path | None = None
    json_path: Path | None = None
    error: str | None = None


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
    """

    result: ExecutionResult
    symbol_rows: tuple[tuple[object, ...], ...] = ()
    occurrence_rows: tuple[tuple[object, ...], ...] = ()


@dataclass(frozen=True)
class ScipMaterializationInputs:
    """Aggregated inputs for scip materialization."""

    run: ScipRunResult
    ingest: ScipIngestResult
    artifact_materializations: dict[str, MaterializationMetadata]
    table_materializations: dict[str, MaterializationMetadata]


@tag_tool(domain="ingestion", target=SCIP_TARGET_NAME)
def t__scip__run(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules: TargetRunRecord,
) -> ScipRunResult:
    """Execute scip-python + scip print to produce SCIP artifacts.

    Returns
    -------
    ScipRunResult
        Run outcome including output paths or error details.
    """
    if t__modules.status != "succeeded":
        return ScipRunResult(
            success=False,
            error=f"Upstream modules target failed: {t__modules.error}",
        )

    executor = NativeTargetExecutor.for_target(env, graph, SCIP_TARGET_NAME)
    if executor.should_skip():
        return ScipRunResult(success=True, skipped=True)

    output_scip = env.paths.scip_dir / "index.scip"
    output_json = env.paths.scip_dir / "index.json"

    if output_scip.exists() and output_json.exists():
        return ScipRunResult(
            success=True,
            index_path=output_scip,
            json_path=output_json,
        )

    try:
        result = asyncio.run(
            env.providers.tool_service.run_scip_full(
                env.snapshot.repo_root,
                output_scip=output_scip,
            )
        )
        index_path = result.index_scip_path or output_scip
        json_path = result.index_json_path or output_json
        documents = tuple(doc.to_port_document() for doc in result.documents)
        return ScipRunResult(
            success=True,
            documents=documents,
            index_path=index_path,
            json_path=json_path,
        )
    except Exception as exc:
        log.exception("SCIP execution failed")
        return ScipRunResult(success=False, error=str(exc))


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node("artifact.scip_index"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SCIP_TARGET_NAME),
    artifact_name=value(SCIP_ARTIFACT_INDEX),
    path_template=value("{scip_dir}/index.scip"),
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
    if not t__scip__run.success or t__scip__run.skipped:
        return None
    return t__scip__run.index_path


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node("artifact.scip_json"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SCIP_TARGET_NAME),
    artifact_name=value(SCIP_ARTIFACT_JSON),
    path_template=value("{scip_dir}/index.json"),
)
@tag_compute(
    domain="ingestion",
    target=SCIP_TARGET_NAME,
    target_="scip__json_artifact",
)
def scip__json_artifact(t__scip__run: ScipRunResult) -> Path | None:
    """Return the Path to index.json for materialization.

    Returns
    -------
    Path | None
        Artifact path, or None when the run was skipped/failed.
    """
    if not t__scip__run.success or t__scip__run.skipped:
        return None
    return t__scip__run.json_path


@tag_compute(domain="ingestion", target=SCIP_TARGET_NAME)
def t__scip__ingest(
    env: BuildEnv,
    t__modules: TargetRunRecord,
    t__scip__run: ScipRunResult,
) -> ScipIngestResult:
    """Build SCIP row payloads for core.scip_* tables.

    Returns
    -------
    ScipIngestResult
        Ingestion status and row tuples.
    """
    if t__modules.status != "succeeded":
        return ScipIngestResult(
            result=ExecutionResult.failed(f"Upstream modules target failed: {t__modules.error}")
        )
    if not t__scip__run.success:
        return ScipIngestResult(
            result=ExecutionResult.failed(t__scip__run.error or "SCIP execution failed")
        )
    if t__scip__run.skipped:
        return ScipIngestResult(result=ExecutionResult.skip("SCIP target skipped"))

    try:
        output_scip = t__scip__run.index_path or (env.paths.scip_dir / "index.scip")
        documents = list(t__scip__run.documents)
        if not documents:
            documents = parse_scip_json_file(t__scip__run.json_path, output_scip)

        created_at = datetime.now(tz=UTC)
        symbol_serializer = row_serializer_for_table_key(SCIP_SYMBOLS_TABLE_KEY)
        occurrence_serializer = row_serializer_for_table_key(SCIP_OCCURRENCES_TABLE_KEY)

        symbol_rows = tuple(
            build_symbol_rows(
                documents,
                env.snapshot.repo,
                env.snapshot.commit,
                created_at,
                serializer=symbol_serializer,
            )
        )
        occurrence_rows = tuple(
            build_occurrence_rows(
                documents,
                env.snapshot.repo,
                env.snapshot.commit,
                created_at,
                serializer=occurrence_serializer,
            )
        )
        if not symbol_rows or not occurrence_rows:
            return ScipIngestResult(
                result=ExecutionResult.failed(
                    "SCIP ingestion produced empty symbols or occurrences"
                )
            )

        table_counts = {
            SCIP_SYMBOLS_TABLE_KEY: len(symbol_rows),
            SCIP_OCCURRENCES_TABLE_KEY: len(occurrence_rows),
        }
        return ScipIngestResult(
            result=ExecutionResult.ok(
                table_counts=normalize_table_counts(SCIP_TABLE_KEYS, table_counts),
            ),
            symbol_rows=symbol_rows,
            occurrence_rows=occurrence_rows,
        )
    except Exception:
        log.exception("SCIP ingestion failed")
        return ScipIngestResult(
            result=ExecutionResult.failed("SCIP ingestion failed with exception")
        )


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(SCIP_SYMBOLS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SCIP_TARGET_NAME),
    table_key=value(SCIP_SYMBOLS_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(SCIP_SYMBOLS_TABLE_KEY)),
)
@tag_compute(domain="ingestion", target=SCIP_TARGET_NAME, target_="scip__symbol_rows")
def scip__symbol_rows(
    t__scip__ingest: ScipIngestResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Return rows for core.scip_symbols.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for scip_symbols, or None when ingestion skipped or failed.
    """
    if t__scip__ingest.result.skipped or not t__scip__ingest.result.success:
        return None
    return t__scip__ingest.symbol_rows


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(SCIP_OCCURRENCES_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SCIP_TARGET_NAME),
    table_key=value(SCIP_OCCURRENCES_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(SCIP_OCCURRENCES_TABLE_KEY)),
)
@tag_compute(domain="ingestion", target=SCIP_TARGET_NAME, target_="scip__occurrence_rows")
def scip__occurrence_rows(
    t__scip__ingest: ScipIngestResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Return rows for core.scip_occurrences.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for scip_occurrences, or None when ingestion skipped or failed.
    """
    if t__scip__ingest.result.skipped or not t__scip__ingest.result.success:
        return None
    return t__scip__ingest.occurrence_rows


@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__materializations(
    m__artifact__scip_index: MaterializationMetadata,
    m__artifact__scip_json: MaterializationMetadata,
) -> dict[str, MaterializationMetadata]:
    """Collect scip artifact materialization payloads into a single mapping.

    Returns
    -------
    dict[str, MaterializationMetadata]
        Materialization metadata keyed by artifact name.
    """
    return {
        SCIP_ARTIFACT_INDEX: m__artifact__scip_index,
        SCIP_ARTIFACT_JSON: m__artifact__scip_json,
    }


@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__table_materializations(
    m__core__scip_symbols: MaterializationMetadata,
    m__core__scip_occurrences: MaterializationMetadata,
) -> dict[str, MaterializationMetadata]:
    """Collect scip table materialization payloads into a single mapping.

    Returns
    -------
    dict[str, MaterializationMetadata]
        Mapping from table key to saver metadata.
    """
    return {
        SCIP_SYMBOLS_TABLE_KEY: m__core__scip_symbols,
        SCIP_OCCURRENCES_TABLE_KEY: m__core__scip_occurrences,
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


@tag_materialize(domain="ingestion", target=SCIP_TARGET_NAME)
def t__scip(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules: TargetRunRecord,
    scip__inputs: ScipMaterializationInputs,
) -> TargetRunRecord:
    """Finalize scip target from artifact materialization metadata.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    executor = NativeTargetExecutor.for_target(env, graph, SCIP_TARGET_NAME)
    if t__modules.status != "succeeded":
        return executor.fail(RuntimeError(t__modules.error or "Upstream modules target failed"))
    if not scip__inputs.run.success:
        return executor.fail(RuntimeError(scip__inputs.run.error or "SCIP execution failed"))
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
