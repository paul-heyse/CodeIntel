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
from codeintel.build.hamilton.materializers import FileArtifactSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    record_from_file_artifact_materializations,
)
from codeintel.build.hamilton.native.table_counts import normalize_table_counts
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_helper, tag_materialize, tag_tool
from codeintel.build.resources import TOOL_EXECUTION, TargetResources
from codeintel.build.targets import TargetGraph
from codeintel.core.schemas.row_serialization import row_serializer_for_table_key
from codeintel.ingestion.adapters import DuckDBStorageAdapter
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

TARGET_SPECS = (
    make_output_target(
        name=SCIP_TARGET_NAME,
        module="ingestion",
        description="SCIP index ingestion and GOID generation.",
        options=TargetSpecOptions(
            table_keys=SCIP_TABLE_KEYS,
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
class ScipMaterializationInputs:
    """Aggregated inputs for scip materialization."""

    run: ScipRunResult
    ingest: ExecutionResult
    materializations: dict[str, MaterializationMetadata]


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
) -> ExecutionResult:
    """Ingest SCIP JSON payloads into core.scip_* tables.

    Returns
    -------
    ExecutionResult
        Ingestion status and per-table row counts.
    """
    if t__modules.status != "succeeded":
        return ExecutionResult.failed(f"Upstream modules target failed: {t__modules.error}")
    if not t__scip__run.success:
        return ExecutionResult.failed(t__scip__run.error or "SCIP execution failed")
    if t__scip__run.skipped:
        return ExecutionResult.skip("SCIP target skipped")

    try:
        output_scip = t__scip__run.index_path or (env.paths.scip_dir / "index.scip")
        documents = list(t__scip__run.documents)
        if not documents:
            documents = parse_scip_json_file(t__scip__run.json_path, output_scip)

        created_at = datetime.now(tz=UTC)
        symbol_serializer = row_serializer_for_table_key(SCIP_SYMBOLS_TABLE_KEY)
        occurrence_serializer = row_serializer_for_table_key(SCIP_OCCURRENCES_TABLE_KEY)

        symbol_rows = build_symbol_rows(
            documents,
            env.snapshot.repo,
            env.snapshot.commit,
            created_at,
            serializer=symbol_serializer,
        )
        occurrence_rows = build_occurrence_rows(
            documents,
            env.snapshot.repo,
            env.snapshot.commit,
            created_at,
            serializer=occurrence_serializer,
        )

        storage = DuckDBStorageAdapter(env.gateway)
        storage.delete_by_params(SCIP_SYMBOLS_TABLE_KEY, [env.snapshot.repo, env.snapshot.commit])
        storage.delete_by_params(
            SCIP_OCCURRENCES_TABLE_KEY, [env.snapshot.repo, env.snapshot.commit]
        )

        table_counts: dict[str, int] = {}
        scope = f"{env.snapshot.repo}@{env.snapshot.commit}"

        if symbol_rows:
            result = storage.write_batch(SCIP_SYMBOLS_TABLE_KEY, symbol_rows, scope=scope)
            table_counts[SCIP_SYMBOLS_TABLE_KEY] = result.rows_affected

        if occurrence_rows:
            result = storage.write_batch(SCIP_OCCURRENCES_TABLE_KEY, occurrence_rows, scope=scope)
            table_counts[SCIP_OCCURRENCES_TABLE_KEY] = result.rows_affected

        return ExecutionResult.ok(
            table_counts=normalize_table_counts(SCIP_TABLE_KEYS, table_counts),
        )
    except Exception:
        log.exception("SCIP ingestion failed")
        return ExecutionResult.failed("SCIP ingestion failed with exception")


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
def scip__inputs(
    t__scip__run: ScipRunResult,
    t__scip__ingest: ExecutionResult,
    scip__materializations: dict[str, MaterializationMetadata],
) -> ScipMaterializationInputs:
    """Bundle scip execution and materialization inputs.

    Returns
    -------
    ScipMaterializationInputs
        Collected inputs for the scip materialization node.
    """
    return ScipMaterializationInputs(
        run=t__scip__run,
        ingest=t__scip__ingest,
        materializations=scip__materializations,
    )


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
    if executor.should_skip():
        return executor.skip()
    if t__modules.status != "succeeded":
        return executor.fail(RuntimeError(t__modules.error or "Upstream modules target failed"))
    if not scip__inputs.run.success:
        return executor.fail(RuntimeError(scip__inputs.run.error or "SCIP execution failed"))
    if scip__inputs.ingest.skipped:
        return executor.skip()
    if not scip__inputs.ingest.success:
        return executor.fail(RuntimeError(scip__inputs.ingest.error or "SCIP ingestion failed"))

    return record_from_file_artifact_materializations(
        env=env,
        graph=graph,
        target_name="scip",
        materializations=scip__inputs.materializations,
        row_counts=normalize_table_counts(SCIP_TABLE_KEYS, scip__inputs.ingest.table_counts),
    )


__all__ = [
    "ScipRunResult",
    "t__scip",
    "t__scip__ingest",
    "t__scip__run",
]
