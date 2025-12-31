"""Convert Hamilton materialization results into TargetRunRecord outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.build.hamilton.native.outputs import (
    expected_artifact_names_for_target,
    expected_table_keys_for_target,
)
from codeintel.build.hamilton.run_records import (
    NativeRunInfo,
    RunRecordInputs,
    TargetRunRecord,
    create_run_record,
    options_hash_for_target,
    save_manifest,
)
from codeintel.core.execution.materialization import (
    MaterializationStatus,
    failed_artifact_result,
    failed_table_result,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.hamilton.dag_catalog import DagCatalog, TargetDescriptor
    from codeintel.core.hamilton.records import DatasetRefProtocol


@dataclass(frozen=True)
class FileArtifactRecordContext:
    """Context bundle for file artifact run record creation."""

    env: BuildEnv
    catalog: DagCatalog
    target_name: str


@dataclass(frozen=True, slots=True)
class MaterializationRecordContext:
    """Context bundle for mixed materialization run record creation."""

    env: BuildEnv
    catalog: DagCatalog
    target_name: str
    change_delta: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class _MaterializationInputs:
    table_materializations: dict[str, MaterializationResult]
    artifact_materializations: dict[str, MaterializationResult]


@dataclass(frozen=True, slots=True)
class _MaterializationSummary:
    statuses: set[MaterializationStatus]
    input_hash: str
    duration_ms: float
    row_counts: dict[str, int] | None
    errors: list[str]
    hash_mismatch: str | None


@dataclass(frozen=True, slots=True)
class _RunRecordContext:
    context: MaterializationRecordContext
    target: TargetDescriptor
    options_hash: str | None


def record_from_duckdb_materialization(
    *,
    env: BuildEnv,
    catalog: DagCatalog,
    target_name: str,
    expected_table_key: str,
    materialization: MaterializationResult,
) -> TargetRunRecord:
    """Build a TargetRunRecord from a DuckDB materializer result.

    Parameters
    ----------
    env
        Build environment for manifest persistence and expected output refs.
    catalog
        DAG catalog used to resolve the target outputs.
    target_name
        Target name for which the record is being produced.
    expected_table_key
        Table key expected to be materialized for this target.
    materialization
        Materialization result returned by the Hamilton saver node.

    Returns
    -------
    TargetRunRecord
        Record describing succeeded/skipped/failed completion.
    """
    result = materialization
    options_hash = options_hash_for_target(env, target_name)
    target = catalog.get_target(target_name)
    if target is None:
        msg = f"Target not found: {target_name}"
        return TargetRunRecord(
            target=target_name,
            impl_kind="native",
            status="failed",
            input_hash=result.input_hash,
            options_hash=options_hash,
            duration_ms=result.duration_ms,
            row_counts={},
            error=msg,
            datasets=(),
            artifacts=(),
        )

    expected_table_keys = _expected_table_keys(target.name, catalog=catalog)
    error_message: str | None = None
    if expected_table_keys != (expected_table_key,):
        if expected_table_key not in expected_table_keys:
            error_message = (
                "DuckDB materialization table_key is not declared by saver tags: "
                f"target={target_name} table_key={expected_table_key}"
            )
        else:
            error_message = (
                "record_from_duckdb_materialization requires a single table output: "
                f"target={target_name} expected_table_keys={expected_table_keys} "
                f"expected_table_key={expected_table_key}"
            )
    elif result.table_key is None:
        error_message = (
            "DuckDB materialization missing table_key: "
            f"target={target_name} expected_table_key={expected_table_key}"
        )
    elif result.table_key != expected_table_key:
        error_message = (
            "DuckDB materialization table_key does not match expected table_key: "
            f"target={target_name} table_key={result.table_key} "
            f"expected_table_key={expected_table_key}"
        )
    if error_message is not None:
        run = NativeRunInfo(
            input_hash=result.input_hash,
            options_hash=options_hash,
            duration_ms=result.duration_ms,
            row_counts=None,
        )
        return create_run_record(
            target,
            "failed",
            result.input_hash,
            inputs=RunRecordInputs(
                env=env,
                run=run,
                error=ValueError(error_message),
                catalog=catalog,
            ),
        )

    table_key = cast("str", result.table_key)
    run = NativeRunInfo(
        input_hash=result.input_hash,
        options_hash=options_hash,
        duration_ms=result.duration_ms,
        row_counts={table_key: result.row_count or 0} if result.status == "succeeded" else None,
    )

    if result.status == "failed":
        return create_run_record(
            target,
            "failed",
            result.input_hash,
            inputs=RunRecordInputs(
                env=env,
                run=run,
                error=RuntimeError(result.error or "Write failed"),
                catalog=catalog,
            ),
        )

    if result.status == "skipped":
        return create_run_record(
            target,
            "skipped",
            result.input_hash,
            inputs=RunRecordInputs(env=env, run=run, catalog=catalog),
        )

    record = create_run_record(
        target,
        "succeeded",
        result.input_hash,
        inputs=RunRecordInputs(env=env, run=run, catalog=catalog),
    )
    record = _apply_dataset_manifest_paths(
        record,
        _dataset_manifest_paths({table_key: result}),
    )
    save_manifest(env, record)
    return record


def record_from_file_artifact_materialization(
    *,
    context: FileArtifactRecordContext,
    expected_artifact_name: str,
    materialization: MaterializationResult,
    row_counts: dict[str, int] | None = None,
) -> TargetRunRecord:
    """Build a TargetRunRecord from a file artifact saver result.

    Parameters
    ----------
    context
        Context bundle containing env, catalog, and target name.
    expected_artifact_name
        Artifact name expected to be written for this target.
    materialization
        Materialization result returned by the Hamilton saver node.
    row_counts
        Optional table row counts for mixed artifact/table targets.

    Returns
    -------
    TargetRunRecord
        Record describing succeeded/skipped/failed completion.
    """
    env = context.env
    catalog = context.catalog
    target_name = context.target_name

    result = materialization
    options_hash = options_hash_for_target(env, target_name)
    target = catalog.get_target(target_name)
    if target is None:
        msg = f"Target not found: {target_name}"
        return TargetRunRecord(
            target=target_name,
            impl_kind="native",
            status="failed",
            input_hash=result.input_hash,
            options_hash=options_hash,
            duration_ms=result.duration_ms,
            row_counts={},
            error=msg,
            datasets=(),
            artifacts=(),
        )

    expected_names = _expected_artifact_names(target.name, catalog=catalog)
    error_message: str | None = None
    if expected_names != (expected_artifact_name,):
        if expected_artifact_name not in expected_names:
            error_message = (
                "Artifact materialization name is not declared by saver tags: "
                f"target={target_name} artifact_name={expected_artifact_name}"
            )
        else:
            error_message = (
                "record_from_file_artifact_materialization requires a single artifact output: "
                f"target={target_name} expected_artifact_names={expected_names} "
                f"expected_artifact_name={expected_artifact_name}"
            )
    elif result.artifact_name is None:
        error_message = (
            "File artifact materialization missing artifact_name: "
            f"target={target_name} expected_artifact_name={expected_artifact_name}"
        )
    elif result.artifact_name != expected_artifact_name:
        error_message = (
            "File artifact materialization artifact_name does not match expected: "
            f"target={target_name} artifact_name={result.artifact_name} "
            f"expected_artifact_name={expected_artifact_name}"
        )
    if error_message is not None:
        run = NativeRunInfo(
            input_hash=result.input_hash,
            options_hash=options_hash,
            duration_ms=result.duration_ms,
            row_counts=None,
        )
        return create_run_record(
            target,
            "failed",
            result.input_hash,
            inputs=RunRecordInputs(
                env=env,
                run=run,
                error=ValueError(error_message),
                catalog=catalog,
            ),
        )

    run = NativeRunInfo(
        input_hash=result.input_hash,
        options_hash=options_hash,
        duration_ms=result.duration_ms,
        row_counts=row_counts,
    )

    if result.status == "failed":
        return create_run_record(
            target,
            "failed",
            result.input_hash,
            inputs=RunRecordInputs(
                env=env,
                run=run,
                error=RuntimeError(result.error or "Artifact write failed"),
                catalog=catalog,
            ),
        )

    if result.status == "skipped":
        return create_run_record(
            target,
            "skipped",
            result.input_hash,
            inputs=RunRecordInputs(env=env, run=run, catalog=catalog),
        )

    record = create_run_record(
        target,
        "succeeded",
        result.input_hash,
        inputs=RunRecordInputs(env=env, run=run, catalog=catalog),
    )

    record = _apply_file_artifact_results(record, {expected_artifact_name: result})

    save_manifest(env, record)
    return record


def record_from_duckdb_materializations(
    *,
    env: BuildEnv,
    catalog: DagCatalog,
    target_name: str,
    materializations: dict[str, MaterializationResult],
    change_delta: Mapping[str, object] | None = None,
) -> TargetRunRecord:
    """Build a TargetRunRecord from multiple DuckDB saver results.

    Parameters
    ----------
    env
        Build environment for manifest persistence and expected output refs.
    catalog
        DAG catalog used to resolve the target outputs.
    target_name
        Target name for which the record is being produced.
    materializations
        Mapping of table_key to materialization results returned by saver nodes.
    change_delta
        Optional change-detection payload to store with the manifest.

    Returns
    -------
    TargetRunRecord
        Record describing succeeded/skipped/failed completion for the target.
    """
    options_hash = options_hash_for_target(env, target_name)
    target = catalog.get_target(target_name)
    if target is None:
        msg = f"Target not found: {target_name}"
        return TargetRunRecord(
            target=target_name,
            impl_kind="native",
            status="failed",
            input_hash="",
            options_hash=options_hash,
            duration_ms=0.0,
            row_counts={},
            error=msg,
            datasets=(),
            artifacts=(),
        )

    expected_table_keys = _expected_table_keys(target.name, catalog=catalog)
    extra_table_keys = set(materializations) - set(expected_table_keys)
    if extra_table_keys:
        msg = f"Unexpected materialization results for tables: {sorted(extra_table_keys)}"
        run = NativeRunInfo(
            input_hash="",
            options_hash=options_hash,
            duration_ms=0.0,
            row_counts=None,
        )
        return create_run_record(
            target,
            "failed",
            "",
            inputs=RunRecordInputs(
                env=env,
                run=run,
                error=RuntimeError(msg),
                catalog=catalog,
            ),
        )

    parsed = _parse_expected_table_materializations(expected_table_keys, materializations)
    statuses, input_hash, duration_ms, row_counts = _summarize_table_results(parsed)

    if "failed" in statuses:
        errors = _materialization_errors(parsed)
        message = errors[0] if errors else "One or more table writes failed"
        run = NativeRunInfo(
            input_hash=input_hash,
            options_hash=options_hash,
            duration_ms=duration_ms,
            row_counts=None,
        )
        return create_run_record(
            target,
            "failed",
            input_hash,
            inputs=RunRecordInputs(
                env=env,
                run=run,
                error=RuntimeError(message),
                catalog=catalog,
            ),
        )

    if statuses == {"skipped"}:
        run = NativeRunInfo(
            input_hash=input_hash,
            options_hash=options_hash,
            duration_ms=duration_ms,
            row_counts=None,
        )
        return create_run_record(
            target,
            "skipped",
            input_hash,
            inputs=RunRecordInputs(env=env, run=run, catalog=catalog),
        )

    row_counts = cast("dict[str, int]", row_counts)

    run = NativeRunInfo(
        input_hash=input_hash,
        options_hash=options_hash,
        duration_ms=duration_ms,
        row_counts=row_counts,
    )
    record = create_run_record(
        target,
        "succeeded",
        input_hash,
        inputs=RunRecordInputs(env=env, run=run, catalog=catalog),
    )
    record = _apply_dataset_manifest_paths(record, _dataset_manifest_paths(parsed))
    save_manifest(env, record, change_delta=change_delta)
    return record


def record_from_file_artifact_materializations(
    *,
    env: BuildEnv,
    catalog: DagCatalog,
    target_name: str,
    materializations: dict[str, MaterializationResult],
    row_counts: dict[str, int] | None = None,
) -> TargetRunRecord:
    """Build a TargetRunRecord from multiple file artifact saver results.

    Parameters
    ----------
    env
        Build environment for manifest persistence and expected output refs.
    catalog
        DAG catalog used to resolve the target outputs.
    target_name
        Target name for which the record is being produced.
    materializations
        Mapping of artifact_name to materialization results returned by saver nodes.
    row_counts
        Optional table row counts for mixed artifact/table targets.

    Returns
    -------
    TargetRunRecord
        Record describing succeeded/skipped/failed completion for the target.
    """
    options_hash = options_hash_for_target(env, target_name)
    target = catalog.get_target(target_name)
    if target is None:
        msg = f"Target not found: {target_name}"
        return TargetRunRecord(
            target=target_name,
            impl_kind="native",
            status="failed",
            input_hash="",
            options_hash=options_hash,
            duration_ms=0.0,
            row_counts={},
            error=msg,
            datasets=(),
            artifacts=(),
        )

    expected_names = _expected_artifact_names(target.name, catalog=catalog)
    extra_artifacts = set(materializations) - set(expected_names)
    if extra_artifacts:
        msg = f"Unexpected materialization results for artifacts: {sorted(extra_artifacts)}"
        run = NativeRunInfo(
            input_hash="",
            options_hash=options_hash,
            duration_ms=0.0,
            row_counts=None,
        )
        return create_run_record(
            target,
            "failed",
            "",
            inputs=RunRecordInputs(
                env=env,
                run=run,
                error=RuntimeError(msg),
                catalog=catalog,
            ),
        )

    parsed = _parse_expected_artifact_materializations(expected_names, materializations)
    statuses, input_hash, duration_ms = _summarize_file_artifact_results(parsed)

    if "failed" in statuses:
        errors = [
            result.error for result in parsed.values() if result.status == "failed" and result.error
        ]
        message = errors[0] if errors else "One or more artifact writes failed"
        run = NativeRunInfo(
            input_hash=input_hash,
            options_hash=options_hash,
            duration_ms=duration_ms,
            row_counts=None,
        )
        return create_run_record(
            target,
            "failed",
            input_hash,
            inputs=RunRecordInputs(
                env=env,
                run=run,
                error=RuntimeError(message),
                catalog=catalog,
            ),
        )

    if statuses == {"skipped"}:
        run = NativeRunInfo(
            input_hash=input_hash,
            options_hash=options_hash,
            duration_ms=duration_ms,
            row_counts=None,
        )
        return create_run_record(
            target,
            "skipped",
            input_hash,
            inputs=RunRecordInputs(env=env, run=run, catalog=catalog),
        )

    run = NativeRunInfo(
        input_hash=input_hash,
        options_hash=options_hash,
        duration_ms=duration_ms,
        row_counts=row_counts,
    )
    record = create_run_record(
        target,
        "succeeded",
        input_hash,
        inputs=RunRecordInputs(env=env, run=run, catalog=catalog),
    )
    record = _apply_file_artifact_results(record, parsed)

    save_manifest(env, record)
    return record


def record_from_materializations(
    *,
    context: MaterializationRecordContext,
    artifact_materializations: Mapping[str, MaterializationResult] | None,
    table_materializations: Mapping[str, MaterializationResult] | None,
) -> TargetRunRecord:
    """Build a TargetRunRecord from mixed artifact/table materializations.

    Parameters
    ----------
    context
        Context containing the build environment and target metadata.
    artifact_materializations
        Materialization results for artifact outputs.
    table_materializations
        Materialization results for table outputs.

    Returns
    -------
    TargetRunRecord
        Run record synthesized from the provided materialization results.
    """
    options_hash = options_hash_for_target(context.env, context.target_name)
    target = context.catalog.get_target(context.target_name)
    if target is None:
        return _missing_target_record(target_name=context.target_name, options_hash=options_hash)

    record_context = _RunRecordContext(
        context=context,
        target=target,
        options_hash=options_hash,
    )
    expected_table_keys = _expected_table_keys(target.name, catalog=context.catalog)
    expected_artifact_names = _expected_artifact_names(target.name, catalog=context.catalog)
    inputs = _normalize_materializations(
        record=record_context,
        expected_table_keys=expected_table_keys,
        expected_artifact_names=expected_artifact_names,
        artifact_materializations=artifact_materializations,
        table_materializations=table_materializations,
    )
    if isinstance(inputs, TargetRunRecord):
        return inputs

    parsed_tables = _parse_expected_table_materializations(
        expected_table_keys,
        inputs.table_materializations,
    )
    parsed_artifacts = _parse_expected_artifact_materializations(
        expected_artifact_names,
        inputs.artifact_materializations,
    )
    summary = _summarize_materializations(parsed_tables, parsed_artifacts)
    failure_record = _build_failure_record(record=record_context, summary=summary)
    if failure_record is not None:
        return failure_record

    row_counts = _filter_row_counts(
        summary.row_counts,
        expected_table_keys=expected_table_keys,
    )
    run = NativeRunInfo(
        input_hash=summary.input_hash,
        options_hash=options_hash,
        duration_ms=summary.duration_ms,
        row_counts=row_counts,
    )
    record = create_run_record(
        target,
        "succeeded",
        summary.input_hash,
        inputs=RunRecordInputs(env=context.env, run=run, catalog=context.catalog),
    )
    record = _apply_file_artifact_results(record, parsed_artifacts)
    record = _apply_dataset_manifest_paths(record, _dataset_manifest_paths(parsed_tables))
    save_manifest(context.env, record, change_delta=context.change_delta)
    return record


def _missing_target_record(*, target_name: str, options_hash: str | None) -> TargetRunRecord:
    msg = f"Target not found: {target_name}"
    return TargetRunRecord(
        target=target_name,
        impl_kind="native",
        status="failed",
        input_hash="",
        options_hash=options_hash,
        duration_ms=0.0,
        row_counts={},
        error=msg,
        datasets=(),
        artifacts=(),
    )


def _filter_row_counts(
    row_counts: dict[str, int] | None,
    *,
    expected_table_keys: tuple[str, ...],
) -> dict[str, int] | None:
    if row_counts is None:
        return None
    if not expected_table_keys:
        return {}
    return {key: row_counts[key] for key in expected_table_keys if key in row_counts}


def _error_run_record(
    *,
    record: _RunRecordContext,
    message: str,
    input_hash: str = "",
    duration_ms: float = 0.0,
    row_counts: dict[str, int] | None = None,
) -> TargetRunRecord:
    run = NativeRunInfo(
        input_hash=input_hash,
        options_hash=record.options_hash,
        duration_ms=duration_ms,
        row_counts=row_counts,
    )
    return create_run_record(
        record.target,
        "failed",
        input_hash,
        inputs=RunRecordInputs(
            env=record.context.env,
            run=run,
            error=RuntimeError(message),
            catalog=record.context.catalog,
        ),
    )


def _normalize_materializations(
    *,
    record: _RunRecordContext,
    expected_table_keys: tuple[str, ...],
    expected_artifact_names: tuple[str, ...],
    artifact_materializations: Mapping[str, MaterializationResult] | None,
    table_materializations: Mapping[str, MaterializationResult] | None,
) -> _MaterializationInputs | TargetRunRecord:
    if table_materializations is None:
        if expected_table_keys:
            msg = f"Missing materialization results for tables: {list(expected_table_keys)}"
            return _error_run_record(
                record=record,
                message=msg,
            )
        table_materializations = {}

    if artifact_materializations is None:
        if expected_artifact_names:
            msg = f"Missing materialization results for artifacts: {list(expected_artifact_names)}"
            return _error_run_record(
                record=record,
                message=msg,
            )
        artifact_materializations = {}

    extra_table_keys = set(table_materializations) - set(expected_table_keys)
    if extra_table_keys:
        msg = f"Unexpected materialization results for tables: {sorted(extra_table_keys)}"
        return _error_run_record(
            record=record,
            message=msg,
        )

    extra_artifacts = set(artifact_materializations) - set(expected_artifact_names)
    if extra_artifacts:
        msg = f"Unexpected materialization results for artifacts: {sorted(extra_artifacts)}"
        return _error_run_record(
            record=record,
            message=msg,
        )

    return _MaterializationInputs(
        table_materializations=dict(table_materializations),
        artifact_materializations=dict(artifact_materializations),
    )


def _resolve_input_hash(
    table_input_hash: str,
    artifact_input_hash: str,
) -> tuple[str, str | None]:
    input_hashes = {table_input_hash, artifact_input_hash}
    input_hashes.discard("")
    if len(input_hashes) > 1:
        return "", f"Materialization input_hash mismatch: {sorted(input_hashes)}"
    return next(iter(input_hashes), ""), None


def _summarize_materializations(
    parsed_tables: dict[str, MaterializationResult],
    parsed_artifacts: dict[str, MaterializationResult],
) -> _MaterializationSummary:
    table_statuses, table_input_hash, table_duration, row_counts = _summarize_table_results(
        parsed_tables
    )
    artifact_statuses, artifact_input_hash, artifact_duration = _summarize_file_artifact_results(
        parsed_artifacts
    )
    input_hash, mismatch = _resolve_input_hash(table_input_hash, artifact_input_hash)
    duration_ms = table_duration + artifact_duration
    statuses = cast("set[MaterializationStatus]", table_statuses | artifact_statuses)
    errors = _materialization_errors(parsed_tables) + _materialization_errors(parsed_artifacts)
    return _MaterializationSummary(
        statuses=statuses,
        input_hash=input_hash,
        duration_ms=duration_ms,
        row_counts=row_counts,
        errors=errors,
        hash_mismatch=mismatch,
    )


def _build_failure_record(
    *,
    record: _RunRecordContext,
    summary: _MaterializationSummary,
) -> TargetRunRecord | None:
    if summary.hash_mismatch is not None:
        return _error_run_record(
            record=record,
            message=summary.hash_mismatch,
            duration_ms=summary.duration_ms,
        )
    if "failed" in summary.statuses:
        message = summary.errors[0] if summary.errors else "One or more writes failed"
        return _error_run_record(
            record=record,
            message=message,
            input_hash=summary.input_hash,
            duration_ms=summary.duration_ms,
        )
    if summary.statuses == {"skipped"} or not summary.statuses:
        run = NativeRunInfo(
            input_hash=summary.input_hash,
            options_hash=record.options_hash,
            duration_ms=summary.duration_ms,
            row_counts=None,
        )
        return create_run_record(
            record.target,
            "skipped",
            summary.input_hash,
            inputs=RunRecordInputs(
                env=record.context.env,
                run=run,
                catalog=record.context.catalog,
            ),
        )
    return None


def _parse_expected_artifact_materializations(
    expected_artifact_names: tuple[str, ...],
    materializations: dict[str, MaterializationResult],
) -> dict[str, MaterializationResult]:
    parsed: dict[str, MaterializationResult] = {}
    for expected_artifact_name in expected_artifact_names:
        result = materializations.get(expected_artifact_name)
        if result is None:
            parsed[expected_artifact_name] = failed_artifact_result(
                artifact_name=expected_artifact_name,
                duration_ms=0.0,
                input_hash="",
                error=f"Missing materialization result for artifact: {expected_artifact_name}",
            )
            continue

        if result.artifact_name is None:
            parsed[expected_artifact_name] = failed_artifact_result(
                artifact_name=expected_artifact_name,
                duration_ms=result.duration_ms,
                input_hash=result.input_hash,
                error=(
                    "File artifact materialization missing artifact_name: "
                    f"expected={expected_artifact_name}"
                ),
            )
            continue

        if result.status != "failed" and result.artifact_name != expected_artifact_name:
            parsed[expected_artifact_name] = MaterializationResult(
                status="failed",
                artifact_name=expected_artifact_name,
                path=result.path,
                size_bytes=result.size_bytes,
                duration_ms=result.duration_ms,
                input_hash=result.input_hash,
                error=(
                    "File artifact materialization artifact_name mismatch: "
                    f"expected={expected_artifact_name} got={result.artifact_name}"
                ),
            )
            continue

        parsed[expected_artifact_name] = result
    return parsed


def _expected_artifact_names(
    target_name: str,
    *,
    catalog: DagCatalog,
) -> tuple[str, ...]:
    return expected_artifact_names_for_target(target_name, outputs=catalog)


def _expected_table_keys(
    target_name: str,
    *,
    catalog: DagCatalog,
) -> tuple[str, ...]:
    return expected_table_keys_for_target(target_name, outputs=catalog)


def _parse_expected_table_materializations(
    expected_table_keys: tuple[str, ...],
    materializations: dict[str, MaterializationResult],
) -> dict[str, MaterializationResult]:
    parsed: dict[str, MaterializationResult] = {}
    for expected_table_key in expected_table_keys:
        result = materializations.get(expected_table_key)
        if result is None:
            parsed[expected_table_key] = failed_table_result(
                table_key=expected_table_key,
                duration_ms=0.0,
                input_hash="",
                error=f"Missing materialization result for table: {expected_table_key}",
            )
            continue

        if result.table_key is None:
            parsed[expected_table_key] = failed_table_result(
                table_key=expected_table_key,
                duration_ms=result.duration_ms,
                input_hash=result.input_hash,
                error=(f"DuckDB materialization missing table_key: expected={expected_table_key}"),
            )
            continue

        if result.status != "failed" and result.table_key != expected_table_key:
            parsed[expected_table_key] = MaterializationResult(
                status="failed",
                table_key=expected_table_key,
                row_count=None,
                duration_ms=result.duration_ms,
                input_hash=result.input_hash,
                error=(
                    "DuckDB materialization table_key mismatch: "
                    f"expected={expected_table_key} got={result.table_key}"
                ),
            )
            continue
        parsed[expected_table_key] = result
    return parsed


def _summarize_file_artifact_results(
    parsed: dict[str, MaterializationResult],
) -> tuple[set[MaterializationStatus], str, float]:
    statuses: set[MaterializationStatus] = {result.status for result in parsed.values()}
    input_hash = next((result.input_hash for result in parsed.values() if result.input_hash), "")
    duration_ms = sum((result.duration_ms for result in parsed.values()), 0.0)
    return statuses, input_hash, duration_ms


def _summarize_table_results(
    parsed: dict[str, MaterializationResult],
) -> tuple[set[MaterializationStatus], str, float, dict[str, int] | None]:
    statuses: set[MaterializationStatus] = {result.status for result in parsed.values()}
    input_hash = next((result.input_hash for result in parsed.values() if result.input_hash), "")
    duration_ms = sum((result.duration_ms for result in parsed.values()), 0.0)
    if "failed" in statuses or statuses == {"skipped"}:
        return statuses, input_hash, duration_ms, None
    row_counts: dict[str, int] = {}
    for table_key, result in parsed.items():
        if result.status == "succeeded":
            row_counts[table_key] = result.row_count or 0
        else:
            row_counts[table_key] = 0
    return statuses, input_hash, duration_ms, row_counts


def _materialization_errors(
    parsed: Mapping[str, MaterializationResult],
) -> list[str]:
    return [
        result.error
        for result in parsed.values()
        if result.status == "failed" and result.error is not None
    ]


def _apply_file_artifact_results(
    record: TargetRunRecord,
    parsed: dict[str, MaterializationResult],
) -> TargetRunRecord:
    parsed_by_name = {
        result.artifact_name: result
        for result in parsed.values()
        if result.artifact_name is not None
    }
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
        impl_kind=record.impl_kind,
        status=record.status,
        input_hash=record.input_hash,
        options_hash=record.options_hash,
        duration_ms=record.duration_ms,
        row_counts=record.row_counts,
        error=record.error,
        datasets=record.datasets,
        artifacts=tuple(updated_artifacts),
        drift_summaries=record.drift_summaries,
    )


def _dataset_manifest_paths(
    parsed: Mapping[str, MaterializationResult],
) -> dict[str, str]:
    results: dict[str, str] = {}
    for table_key, result in parsed.items():
        path = result.dataset_manifest_path
        if path:
            results[table_key] = path
    return results


def _apply_dataset_manifest_paths(
    record: TargetRunRecord,
    manifest_paths: Mapping[str, str],
) -> TargetRunRecord:
    if not manifest_paths or not record.datasets:
        return record

    updated_datasets: list[DatasetRefProtocol] = []
    for dataset in record.datasets:
        table_key = dataset.table_key
        path = manifest_paths.get(table_key)
        if path is None:
            updated_datasets.append(dataset)
            continue
        updated_datasets.append(_apply_dataset_manifest_path(dataset, path))

    return TargetRunRecord(
        target=record.target,
        impl_kind=record.impl_kind,
        status=record.status,
        input_hash=record.input_hash,
        options_hash=record.options_hash,
        duration_ms=record.duration_ms,
        row_counts=record.row_counts,
        error=record.error,
        datasets=tuple(updated_datasets),
        artifacts=record.artifacts,
        drift_summaries=record.drift_summaries,
    )


def _apply_dataset_manifest_path(
    dataset: DatasetRefProtocol,
    path: str,
) -> DatasetRefProtocol:
    if isinstance(dataset, DatasetRef):
        return dataset.with_metadata("dataset_manifest_path", path)
    with_metadata = getattr(dataset, "with_metadata", None)
    if callable(with_metadata):
        updated = with_metadata("dataset_manifest_path", path)
        return cast("DatasetRefProtocol", updated)
    return dataset


__all__ = [
    "FileArtifactRecordContext",
    "MaterializationRecordContext",
    "MaterializationStatus",
    "record_from_duckdb_materialization",
    "record_from_duckdb_materializations",
    "record_from_file_artifact_materialization",
    "record_from_file_artifact_materializations",
    "record_from_materializations",
]
