"""Convert Hamilton materialization metadata into TargetRunRecord results.

This module bridges Hamilton's DataSaver/materializer nodes (which return a
metadata dict) to the build system's ``TargetRunRecord`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
from codeintel.build.hamilton.native.runner import (
    NativeRunInfo,
    RunRecordInputs,
    create_run_record,
    save_manifest,
)
from codeintel.hamilton.records import TargetRunRecord

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.targets import OutputTarget, TargetGraph

MaterializationStatus = Literal["succeeded", "skipped", "failed"]
_MATERIALIZATION_STATUS: dict[str, MaterializationStatus] = {
    "failed": "failed",
    "skipped": "skipped",
    "succeeded": "succeeded",
}


@dataclass(frozen=True)
class DuckDBMaterializationResult:
    """Parsed materialization metadata for a single DuckDB table write."""

    status: MaterializationStatus
    table_key: str
    row_count: int | None
    duration_ms: float
    input_hash: str
    error: str | None


@dataclass(frozen=True)
class FileArtifactMaterializationResult:
    """Parsed materialization metadata for a single file artifact write."""

    status: MaterializationStatus
    artifact_name: str
    path: str | None
    size_bytes: int | None
    duration_ms: float
    input_hash: str
    error: str | None


def record_from_duckdb_materialization(
    *,
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    expected_table_key: str,
    materialization: Mapping[str, object],
) -> TargetRunRecord:
    """Build a TargetRunRecord from a DuckDB materializer metadata dict.

    Parameters
    ----------
    env
        Build environment for manifest persistence and expected output refs.
    graph
        Target graph used to resolve the OutputTarget contract.
    target_name
        Target name for which the record is being produced.
    expected_table_key
        Table key expected to be materialized for this target.
    materialization
        Materialization metadata dict returned by the Hamilton saver node.

    Returns
    -------
    TargetRunRecord
        Record describing succeeded/skipped/failed completion.
    """
    parsed = _parse_materialization(materialization, default_table_key=expected_table_key)
    target = graph.get(target_name)
    if target is None:
        msg = f"Target not found: {target_name}"
        return TargetRunRecord(
            target=target_name,
            plugin_name=f"native:{target_name}",
            status="failed",
            input_hash=parsed.input_hash,
            options_hash=None,
            duration_ms=parsed.duration_ms,
            row_counts={},
            error=msg,
            datasets=(),
            artifacts=(),
        )

    run = NativeRunInfo(
        input_hash=parsed.input_hash,
        options_hash=None,
        duration_ms=parsed.duration_ms,
        row_counts={parsed.table_key: parsed.row_count or 0}
        if parsed.status == "succeeded"
        else None,
    )

    if parsed.status == "failed":
        return create_run_record(
            target,
            "failed",
            parsed.input_hash,
            inputs=RunRecordInputs(
                env=env, run=run, error=RuntimeError(parsed.error or "Write failed")
            ),
        )

    if parsed.status == "skipped":
        return create_run_record(
            target,
            "skipped",
            parsed.input_hash,
            inputs=RunRecordInputs(env=env, run=run),
        )

    record = create_run_record(
        target,
        "succeeded",
        parsed.input_hash,
        inputs=RunRecordInputs(env=env, run=run),
    )
    save_manifest(env, record)
    return record


def record_from_file_artifact_materialization(
    *,
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    expected_artifact_name: str,
    materialization: Mapping[str, object],
) -> TargetRunRecord:
    """Build a TargetRunRecord from a file artifact saver metadata dict.

    Parameters
    ----------
    env
        Build environment for manifest persistence and expected output refs.
    graph
        Target graph used to resolve the OutputTarget contract.
    target_name
        Target name for which the record is being produced.
    expected_artifact_name
        Artifact name expected to be written for this target.
    materialization
        Materialization metadata dict returned by the Hamilton saver node.

    Returns
    -------
    TargetRunRecord
        Record describing succeeded/skipped/failed completion.
    """
    parsed = _parse_file_artifact_materialization(
        materialization,
        default_artifact_name=expected_artifact_name,
    )
    target = graph.get(target_name)
    if target is None:
        msg = f"Target not found: {target_name}"
        return TargetRunRecord(
            target=target_name,
            plugin_name=f"native:{target_name}",
            status="failed",
            input_hash=parsed.input_hash,
            options_hash=None,
            duration_ms=parsed.duration_ms,
            row_counts={},
            error=msg,
            datasets=(),
            artifacts=(),
        )

    run = NativeRunInfo(
        input_hash=parsed.input_hash,
        options_hash=None,
        duration_ms=parsed.duration_ms,
        row_counts=None,
    )

    if parsed.status == "failed":
        return create_run_record(
            target,
            "failed",
            parsed.input_hash,
            inputs=RunRecordInputs(
                env=env, run=run, error=RuntimeError(parsed.error or "Artifact write failed")
            ),
        )

    if parsed.status == "skipped":
        return create_run_record(
            target,
            "skipped",
            parsed.input_hash,
            inputs=RunRecordInputs(env=env, run=run),
        )

    record = create_run_record(
        target,
        "succeeded",
        parsed.input_hash,
        inputs=RunRecordInputs(env=env, run=run),
    )

    updated_artifacts: list[ArtifactRef] = []
    for art in record.artifacts:
        if isinstance(art, ArtifactRef) and art.name == parsed.artifact_name:
            updated = art
            if parsed.path is not None:
                updated = updated.with_path(parsed.path)
            if parsed.size_bytes is not None:
                updated = updated.with_metadata("size_bytes", parsed.size_bytes)
            updated_artifacts.append(updated)
        elif isinstance(art, ArtifactRef):
            updated_artifacts.append(art)

    if updated_artifacts:
        record = TargetRunRecord(
            target=record.target,
            plugin_name=record.plugin_name,
            status=record.status,
            input_hash=record.input_hash,
            options_hash=record.options_hash,
            duration_ms=record.duration_ms,
            row_counts=record.row_counts,
            error=record.error,
            datasets=record.datasets,
            artifacts=tuple(updated_artifacts),
        )

    save_manifest(env, record)
    return record


def record_from_duckdb_materializations(
    *,
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    materializations: Mapping[str, Mapping[str, object]],
) -> TargetRunRecord:
    """Build a TargetRunRecord from multiple DuckDB saver metadata dicts.

    Parameters
    ----------
    env
        Build environment for manifest persistence and expected output refs.
    graph
        Target graph used to resolve the OutputTarget contract.
    target_name
        Target name for which the record is being produced.
    materializations
        Mapping of table_key to materialization metadata dict returned by saver nodes.

    Returns
    -------
    TargetRunRecord
        Record describing succeeded/skipped/failed completion for the target.
    """
    target = graph.get(target_name)
    if target is None:
        msg = f"Target not found: {target_name}"
        return TargetRunRecord(
            target=target_name,
            plugin_name=f"native:{target_name}",
            status="failed",
            input_hash="",
            options_hash=None,
            duration_ms=0.0,
            row_counts={},
            error=msg,
            datasets=(),
            artifacts=(),
        )

    parsed: dict[str, DuckDBMaterializationResult] = {}
    for expected_table_key in target.contract.table_keys:
        meta = materializations.get(expected_table_key)
        if meta is None:
            parsed[expected_table_key] = DuckDBMaterializationResult(
                status="failed",
                table_key=expected_table_key,
                row_count=None,
                duration_ms=0.0,
                input_hash="",
                error=f"Missing materialization metadata for table: {expected_table_key}",
            )
            continue
        parsed[expected_table_key] = _parse_materialization(
            meta, default_table_key=expected_table_key
        )

    statuses = {r.status for r in parsed.values()}
    input_hash = next((r.input_hash for r in parsed.values() if r.input_hash), "")
    duration_ms = sum(r.duration_ms for r in parsed.values())

    if "failed" in statuses:
        errors = [r.error for r in parsed.values() if r.status == "failed" and r.error]
        message = errors[0] if errors else "One or more table writes failed"
        run = NativeRunInfo(
            input_hash=input_hash,
            options_hash=None,
            duration_ms=duration_ms,
            row_counts=None,
        )
        return create_run_record(
            target,
            "failed",
            input_hash,
            inputs=RunRecordInputs(env=env, run=run, error=RuntimeError(message)),
        )

    if statuses == {"skipped"}:
        run = NativeRunInfo(
            input_hash=input_hash,
            options_hash=None,
            duration_ms=duration_ms,
            row_counts=None,
        )
        return create_run_record(
            target,
            "skipped",
            input_hash,
            inputs=RunRecordInputs(env=env, run=run),
        )

    row_counts: dict[str, int] = {}
    for table_key, result in parsed.items():
        if result.status == "succeeded":
            row_counts[table_key] = result.row_count or 0
        else:
            row_counts[table_key] = 0

    run = NativeRunInfo(
        input_hash=input_hash,
        options_hash=None,
        duration_ms=duration_ms,
        row_counts=row_counts,
    )
    record = create_run_record(
        target,
        "succeeded",
        input_hash,
        inputs=RunRecordInputs(env=env, run=run),
    )
    save_manifest(env, record)
    return record


def record_from_file_artifact_materializations(
    *,
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    materializations: Mapping[str, Mapping[str, object]],
) -> TargetRunRecord:
    """Build a TargetRunRecord from multiple file artifact saver metadata dicts.

    Parameters
    ----------
    env
        Build environment for manifest persistence and expected output refs.
    graph
        Target graph used to resolve the OutputTarget contract.
    target_name
        Target name for which the record is being produced.
    materializations
        Mapping of artifact_name to materialization metadata dict returned by saver nodes.

    Returns
    -------
    TargetRunRecord
        Record describing succeeded/skipped/failed completion for the target.
    """
    target = graph.get(target_name)
    if target is None:
        msg = f"Target not found: {target_name}"
        return TargetRunRecord(
            target=target_name,
            plugin_name=f"native:{target_name}",
            status="failed",
            input_hash="",
            options_hash=None,
            duration_ms=0.0,
            row_counts={},
            error=msg,
            datasets=(),
            artifacts=(),
        )

    parsed = _parse_expected_artifact_materializations(target, materializations)
    statuses, input_hash, duration_ms = _summarize_file_artifact_results(parsed)

    if "failed" in statuses:
        errors = [
            result.error for result in parsed.values() if result.status == "failed" and result.error
        ]
        message = errors[0] if errors else "One or more artifact writes failed"
        run = NativeRunInfo(
            input_hash=input_hash,
            options_hash=None,
            duration_ms=duration_ms,
            row_counts=None,
        )
        return create_run_record(
            target,
            "failed",
            input_hash,
            inputs=RunRecordInputs(env=env, run=run, error=RuntimeError(message)),
        )

    if statuses == {"skipped"}:
        run = NativeRunInfo(
            input_hash=input_hash,
            options_hash=None,
            duration_ms=duration_ms,
            row_counts=None,
        )
        return create_run_record(
            target,
            "skipped",
            input_hash,
            inputs=RunRecordInputs(env=env, run=run),
        )

    run = NativeRunInfo(
        input_hash=input_hash,
        options_hash=None,
        duration_ms=duration_ms,
        row_counts=None,
    )
    record = create_run_record(
        target,
        "succeeded",
        input_hash,
        inputs=RunRecordInputs(env=env, run=run),
    )
    record = _apply_file_artifact_results(record, parsed)

    save_manifest(env, record)
    return record


def _parse_expected_artifact_materializations(
    target: OutputTarget,
    materializations: Mapping[str, Mapping[str, object]],
) -> dict[str, FileArtifactMaterializationResult]:
    parsed: dict[str, FileArtifactMaterializationResult] = {}
    for expected_artifact_name in target.contract.artifact_names:
        meta = materializations.get(expected_artifact_name)
        if meta is None:
            parsed[expected_artifact_name] = FileArtifactMaterializationResult(
                status="failed",
                artifact_name=expected_artifact_name,
                path=None,
                size_bytes=None,
                duration_ms=0.0,
                input_hash="",
                error=f"Missing materialization metadata for artifact: {expected_artifact_name}",
            )
            continue

        parsed[expected_artifact_name] = _parse_file_artifact_materialization(
            meta,
            default_artifact_name=expected_artifact_name,
        )
    return parsed


def _summarize_file_artifact_results(
    parsed: Mapping[str, FileArtifactMaterializationResult],
) -> tuple[set[MaterializationStatus], str, float]:
    statuses: set[MaterializationStatus] = {result.status for result in parsed.values()}
    input_hash = next((result.input_hash for result in parsed.values() if result.input_hash), "")
    duration_ms = sum((result.duration_ms for result in parsed.values()), 0.0)
    return statuses, input_hash, duration_ms


def _apply_file_artifact_results(
    record: TargetRunRecord,
    parsed: Mapping[str, FileArtifactMaterializationResult],
) -> TargetRunRecord:
    parsed_by_name = {result.artifact_name: result for result in parsed.values()}
    updated_artifacts: list[ArtifactRef] = []

    for artifact in record.artifacts:
        if not isinstance(artifact, ArtifactRef):
            continue

        result = parsed_by_name.get(artifact.name)
        if result is None:
            updated_artifacts.append(artifact)
            continue

        updated = artifact
        if result.path is not None:
            updated = updated.with_path(result.path)
        if result.size_bytes is not None:
            updated = updated.with_metadata("size_bytes", result.size_bytes)
        updated_artifacts.append(updated)

    if not updated_artifacts:
        return record

    return TargetRunRecord(
        target=record.target,
        plugin_name=record.plugin_name,
        status=record.status,
        input_hash=record.input_hash,
        options_hash=record.options_hash,
        duration_ms=record.duration_ms,
        row_counts=record.row_counts,
        error=record.error,
        datasets=record.datasets,
        artifacts=tuple(updated_artifacts),
    )


def _parse_materialization(
    materialization: Mapping[str, object],
    *,
    default_table_key: str,
) -> DuckDBMaterializationResult:
    status_raw = materialization.get("status")
    if isinstance(status_raw, str) and status_raw in _MATERIALIZATION_STATUS:
        status = _MATERIALIZATION_STATUS[status_raw]
    else:
        status = "failed"

    table_key = materialization.get("table_key")
    if not isinstance(table_key, str) or not table_key:
        table_key = default_table_key

    row_count_raw = materialization.get("row_count")
    row_count = row_count_raw if isinstance(row_count_raw, int) else None

    duration_raw = materialization.get("duration_ms")
    duration_ms = float(duration_raw) if isinstance(duration_raw, (int, float)) else 0.0

    input_hash_raw = materialization.get("input_hash")
    input_hash = input_hash_raw if isinstance(input_hash_raw, str) else ""

    error_raw = materialization.get("error")
    error = error_raw if isinstance(error_raw, str) else None

    return DuckDBMaterializationResult(
        status=status,
        table_key=table_key,
        row_count=row_count,
        duration_ms=duration_ms,
        input_hash=input_hash,
        error=error,
    )


def _parse_file_artifact_materialization(
    materialization: Mapping[str, object],
    *,
    default_artifact_name: str,
) -> FileArtifactMaterializationResult:
    status_raw = materialization.get("status")
    if isinstance(status_raw, str) and status_raw in _MATERIALIZATION_STATUS:
        status = _MATERIALIZATION_STATUS[status_raw]
    else:
        status = "failed"

    artifact_name = materialization.get("artifact_name")
    if not isinstance(artifact_name, str) or not artifact_name:
        artifact_name = default_artifact_name

    path_raw = materialization.get("path")
    path = path_raw if isinstance(path_raw, str) else None

    size_bytes_raw = materialization.get("size_bytes")
    size_bytes = size_bytes_raw if isinstance(size_bytes_raw, int) else None

    duration_raw = materialization.get("duration_ms")
    duration_ms = float(duration_raw) if isinstance(duration_raw, (int, float)) else 0.0

    input_hash_raw = materialization.get("input_hash")
    input_hash = input_hash_raw if isinstance(input_hash_raw, str) else ""

    error_raw = materialization.get("error")
    error = error_raw if isinstance(error_raw, str) else None

    return FileArtifactMaterializationResult(
        status=status,
        artifact_name=artifact_name,
        path=path,
        size_bytes=size_bytes,
        duration_ms=duration_ms,
        input_hash=input_hash,
        error=error,
    )


__all__ = [
    "DuckDBMaterializationResult",
    "FileArtifactMaterializationResult",
    "MaterializationStatus",
    "record_from_duckdb_materialization",
    "record_from_duckdb_materializations",
    "record_from_file_artifact_materialization",
    "record_from_file_artifact_materializations",
]
