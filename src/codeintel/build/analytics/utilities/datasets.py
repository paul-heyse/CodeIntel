"""Analytics dataset contract and persistence helpers.

This module is a thin layer around the canonical build-time contract and schema
providers, plus convenience helpers for validating and inserting rows.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
from sqlglot import exp

from codeintel.build.analytics.utilities.finalize import (
    finalize_analytics_result,
    finalize_artifact_counts,
    finalize_artifact_table_key,
)
from codeintel.build.analytics.utilities.pipeline import (
    AnalyticsPipelineRunRequest,
    QuerySource,
    run_analytics_pipeline,
)
from codeintel.build.analytics.utilities.snapshot import (
    SnapshotContext,
)
from codeintel.build.analytics.utilities.snapshot import (
    snapshot_plan as _snapshot_plan,
)
from codeintel.build.analytics.utilities.snapshot import (
    snapshot_reader as _snapshot_reader,
)
from codeintel.build.analytics.utilities.snapshot import (
    snapshot_table as _snapshot_table,
)
from codeintel.build.schemas import (
    ContractResolutionMode,
    ContractResolutionSettings,
    get_contract_for_table_key,
)
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.plan_ops import Plan
from codeintel.build.validation.columnar import ColumnarValidationContext, validate_table
from codeintel.config.datasets.columns import load_columns_by_table
from codeintel.core.columnar.conversion import record_batch_reader_from_iterable, table_to_reader
from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.queryspec import QuerySpec
from codeintel.core.columnar.readers import empty_reader_from_schema
from codeintel.core.columnar.rows import table_for_rows
from codeintel.core.columnar.run_manifest import RunManifestOptions
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.datasets.arrow_store import ArrowDatasetWriteOptions, write_dataset
from codeintel.core.datasets.parquet_metadata import DatasetMetadataContext
from codeintel.core.datasets.paths import SnapshotIdError, dataset_snapshot_dir
from codeintel.core.schemas.arrow_polars import table_schema_from_arrow_schema
from codeintel.core.schemas.hashing import schema_digest, schema_hash
from codeintel.core.schemas.primitives import resolve_canonical_sort_keys
from codeintel.core.schemas.resolution import resolve_table_schema
from codeintel.core.schemas.row_models import normalize_row_value_for_type
from codeintel.core.validation.profiles import ValidationProfile

if TYPE_CHECKING:
    from codeintel.build.analytics.utilities.persistence import DeleteScope
    from codeintel.build.tabular.finalize_ops import FinalizeResult
    from codeintel.core.gateway import BuildGateway
    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.core.schemas.primitives import ColumnType, TableSchema
    from codeintel.core.schemas.schema_catalog_models import SchemaObservationRecord

LOG = logging.getLogger(__name__)

_FULL_CONTRACT_SETTINGS = ContractResolutionSettings(mode=ContractResolutionMode.FULL)


class DatasetSchemaMissingError(ValueError):
    """Raised when a dataset schema is missing."""

    def __init__(self, table_key: str) -> None:
        message = f"Dataset schema missing for {table_key}"
        super().__init__(message)
        self.table_key = table_key


class UnsupportedDeleteScopeError(ValueError):
    """Raised when delete_scope is unsupported for a table."""

    def __init__(self, table_key: str) -> None:
        message = f"Unsupported delete target: {table_key}"
        super().__init__(message)
        self.table_key = table_key


def snapshot_plan(
    table: pa.Table,
    *,
    columns: Sequence[str] | None = None,
    context: SnapshotContext | None = None,
) -> Plan:
    """Return a snapshot-scoped Plan for analytics tables.

    Returns
    -------
    Plan
        Snapshot-scoped plan with optional projection applied.
    """
    return _snapshot_plan(table, columns=columns, context=context)


def snapshot_table(
    table: pa.Table,
    *,
    columns: Sequence[str] | None = None,
    context: SnapshotContext | None = None,
) -> pa.Table:
    """Materialize a snapshot-scoped Plan.

    Returns
    -------
    pyarrow.Table
        Snapshot-scoped table.
    """
    return _snapshot_table(
        table,
        columns=columns,
        context=context,
    )


def snapshot_reader(
    table: pa.Table,
    *,
    columns: Sequence[str] | None = None,
    context: SnapshotContext | None = None,
) -> pa.RecordBatchReader:
    """Materialize a snapshot-scoped Plan as a reader.

    Returns
    -------
    pyarrow.RecordBatchReader
        Snapshot-scoped reader.
    """
    return _snapshot_reader(
        table,
        columns=columns,
        context=context,
    )


def _table_supports_snapshot_delete(table_key: str) -> bool:
    """Check if a table supports repo/commit scoped deletion.

    Parameters
    ----------
    table_key
        Fully qualified table key (e.g., 'analytics.function_types').

    Returns
    -------
    bool
        True if the table has repo and commit columns.
    """
    columns = load_columns_by_table().get(table_key)
    if columns is None:
        return False
    return "repo" in columns and "commit" in columns


def _delete_sql_for_table(table_key: str) -> str:
    schema, table = table_key.split(".", 1)
    table_expr = exp.Table(this=exp.to_identifier(table), db=exp.to_identifier(schema))
    condition = exp.and_(
        exp.EQ(this=exp.column("repo"), expression=exp.Parameter()),
        exp.EQ(this=exp.column("commit"), expression=exp.Parameter()),
    )
    statement = exp.Delete(this=table_expr, where=condition)
    return statement.sql(dialect="duckdb")


@lru_cache(maxsize=1)
def get_delete_sql_by_table() -> dict[str, str]:
    """Return per-table DELETE statements scoped by repo+commit.

    This is computed lazily to avoid importing the full schema provider during
    module import (which can create circular imports when building the unified
    registry).

    Returns
    -------
    dict[str, str]
        Mapping from table_key to a parametrized DELETE statement.
    """
    return {
        table_key: _delete_sql_for_table(table_key)
        for table_key in load_columns_by_table()
        if _table_supports_snapshot_delete(table_key)
    }


def get_analytics_dataset_contract(
    gateway: BuildGateway,
    table_key: str,
) -> DatasetContract:
    """
    Return the canonical DatasetContract for a table key.

    Returns
    -------
    DatasetContract
        Contract for the requested table key.
    """
    _ = gateway
    return get_contract_for_table_key(table_key, settings=_FULL_CONTRACT_SETTINGS)


def get_function_ast_features_contract(
    gateway: BuildGateway,
) -> DatasetContract:
    """
    Return the dataset contract for function AST features.

    Returns
    -------
    DatasetContract
        Contract describing analytics.function_ast_features.
    """
    _ = gateway
    return get_contract_for_table_key(
        "analytics.function_ast_features",
        settings=_FULL_CONTRACT_SETTINGS,
    )


def _partition_columns_for_schema(table_schema: TableSchema) -> tuple[str, ...]:
    column_names = table_schema.column_names()
    if "repo" in column_names and "commit" in column_names:
        return ("repo", "commit")
    return ()

def _resolve_manifest_sort_keys(table_schema: TableSchema) -> tuple[str, ...] | None:
    return resolve_canonical_sort_keys(table_schema)


def _manifest_extras(
    table_schema: TableSchema,
    *,
    finalize_counts: Mapping[str, int] | None = None,
    artifact_for: str | None = None,
    artifact_type: str | None = None,
) -> dict[str, object]:
    extras: dict[str, object] = {
        "table_schema": table_schema.to_json_obj(),
        "write_source": "analytics_insert",
        "written_at": datetime.now(tz=UTC).isoformat(),
    }
    if finalize_counts is not None:
        extras["finalize"] = dict(finalize_counts)
    if artifact_for is not None:
        extras["artifact_for"] = artifact_for
    if artifact_type is not None:
        extras["artifact_type"] = artifact_type
    return extras


@dataclass(frozen=True, slots=True)
class _ParquetMetadataContext:
    table_schema: TableSchema
    schema_hash_value: str
    schema_digest_value: str
    partition_columns: tuple[str, ...]
    repo: str
    commit: str
    snapshot_id: str


@dataclass(frozen=True, slots=True)
class _WriteContext:
    dataset_root: Path
    snapshot_id: str
    repo: str
    commit: str


def _parquet_metadata_payload(
    *,
    context: _ParquetMetadataContext,
) -> dict[str, object]:
    table_schema = context.table_schema
    columns_json = {col.name: col.type for col in table_schema.columns}
    nullability_json = {col.name: col.nullable for col in table_schema.columns}
    return {
        "codeintel.table_key": table_schema.table_key,
        "codeintel.domain": table_schema.schema,
        "codeintel.schema_hash": context.schema_hash_value,
        "codeintel.schema_digest": context.schema_digest_value,
        "codeintel.columns_json": columns_json,
        "codeintel.nullability_json": nullability_json,
        "codeintel.primary_keys_json": list(table_schema.primary_key),
        "codeintel.partition_columns_json": list(context.partition_columns),
        "codeintel.build_id": context.snapshot_id,
        "codeintel.repo": context.repo,
        "codeintel.commit": context.commit,
        "codeintel.snapshot_id": context.snapshot_id,
        "codeintel.generated_at": datetime.now(tz=UTC).isoformat(),
        "codeintel.write_source": "analytics_insert",
    }


def _resolve_parquet_context(
    gateway: BuildGateway,
) -> tuple[Path, str, str, str]:
    config = gateway.config
    dataset_root_dir = config.dataset_root_dir
    commit_value = getattr(config, "commit", None)
    snapshot_id = commit_value if isinstance(commit_value, str) and commit_value else None
    if dataset_root_dir is None or snapshot_id is None:
        msg = "Parquet dataset writes require dataset_root_dir and commit metadata"
        raise RuntimeError(msg)
    repo_value = getattr(config, "repo", None)
    repo = repo_value if isinstance(repo_value, str) else ""
    commit = snapshot_id
    return dataset_root_dir, snapshot_id, repo, commit


def _metadata_context_for_snapshot(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
) -> DatasetMetadataContext | None:
    try:
        snapshot_dir = dataset_snapshot_dir(
            dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
    except SnapshotIdError as exc:
        LOG.warning("Invalid snapshot_id for %s: %s", table_key, exc)
        return None
    return DatasetMetadataContext(dataset_root=snapshot_dir, table_key=table_key)


def _log_missing_metadata(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
) -> None:
    metadata_ctx = _metadata_context_for_snapshot(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    if metadata_ctx is None:
        return
    if metadata_ctx.read_schema() is None:
        LOG.debug("Parquet metadata missing for %s@%s", table_key, snapshot_id)


def _write_parquet_dataset(
    *,
    gateway: BuildGateway,
    contract: DatasetContract,
    rows: Sequence[Mapping[str, object]],
    delete_scope: DeleteScope | None,
) -> int:
    table_schema = contract.schema
    if table_schema is None:
        msg = f"Dataset schema missing for {contract.table_key}"
        raise ValueError(msg)
    if delete_scope is not None and not _table_supports_snapshot_delete(contract.table_key):
        message = f"Unsupported delete target: {contract.table_key}"
        raise ValueError(message)
    normalized = validate_contract_rows(
        contract.table_key,
        rows,
        gateway=gateway,
        validation_profile=contract.validation_profile,
    )
    if not normalized:
        return 0
    dataset_root_dir, snapshot_id, repo, commit = _resolve_parquet_context(gateway)
    write_context = _WriteContext(
        dataset_root=dataset_root_dir,
        snapshot_id=snapshot_id,
        repo=repo,
        commit=commit,
    )
    result = _finalize_rows_for_parquet(
        contract.table_key,
        rows=normalized,
    )
    stable_sort_keys = _resolve_manifest_sort_keys(table_schema)
    manifest_extras = _manifest_extras(
        table_schema,
        finalize_counts=finalize_artifact_counts(result),
    )
    options = _parquet_write_options(
        table_schema=table_schema,
        stable_sort_keys=stable_sort_keys,
        context=write_context,
        manifest_extras=manifest_extras,
    )
    reader = table_to_reader(result.good, batch_size=DEFAULT_ARROW_BATCH_SIZE)
    write_dataset(
        dataset_root=write_context.dataset_root,
        table_key=contract.table_key,
        snapshot_id=write_context.snapshot_id,
        data=reader,
        options=options,
    )
    _write_finalize_artifacts(
        context=write_context,
        base_table_key=contract.table_key,
        result=result,
    )
    _log_missing_metadata(
        dataset_root=write_context.dataset_root,
        table_key=contract.table_key,
        snapshot_id=write_context.snapshot_id,
    )
    return len(normalized)


def _finalize_rows_for_parquet(
    table_key: str,
    *,
    rows: Sequence[Mapping[str, object]],
) -> FinalizeResult:
    table, _ = table_for_rows(table_key, rows)
    result = finalize_analytics_result(table_key, table)
    if result.errors.num_rows:
        LOG.warning(
            "Finalize produced %d error rows for %s; persisting good rows only",
            result.errors.num_rows,
            table_key,
        )
    return result


def _write_finalize_artifacts(
    *,
    context: _WriteContext,
    base_table_key: str,
    result: FinalizeResult,
) -> None:
    _write_finalize_artifact(
        context=context,
        base_table_key=base_table_key,
        artifact="errors",
        table=result.errors,
    )
    _write_finalize_artifact(
        context=context,
        base_table_key=base_table_key,
        artifact="alignment",
        table=result.alignment,
    )
    _write_finalize_artifact(
        context=context,
        base_table_key=base_table_key,
        artifact="stats",
        table=result.stats,
    )


def _write_finalize_artifact(
    *,
    context: _WriteContext,
    base_table_key: str,
    artifact: str,
    table: pa.Table,
) -> None:
    artifact_table_key = finalize_artifact_table_key(base_table_key, artifact)
    try:
        table_schema = table_schema_from_arrow_schema(
            arrow_schema=table.schema,
            table_key=artifact_table_key,
        )
        stable_sort_keys = _resolve_manifest_sort_keys(table_schema)
        manifest_extras = _manifest_extras(
            table_schema,
            artifact_for=base_table_key,
            artifact_type=artifact,
        )
        options = _parquet_write_options(
            table_schema=table_schema,
            stable_sort_keys=stable_sort_keys,
            context=context,
            manifest_extras=manifest_extras,
        )
        write_dataset(
            dataset_root=context.dataset_root,
            table_key=artifact_table_key,
            snapshot_id=context.snapshot_id,
            data=table,
            options=options,
        )
    except (OSError, ValueError, pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        LOG.warning(
            "Finalize artifact write failed; table_key=%s artifact=%s error=%s",
            base_table_key,
            artifact,
            exc,
        )


def _parquet_write_options(
    *,
    table_schema: TableSchema,
    stable_sort_keys: tuple[str, ...] | None,
    context: _WriteContext,
    manifest_extras: Mapping[str, object],
) -> ArrowDatasetWriteOptions:
    schema_hash_value = schema_hash(table_schema)
    schema_digest_value = schema_digest(table_schema)
    partition_columns = _partition_columns_for_schema(table_schema)
    metadata = _parquet_metadata_payload(
        context=_ParquetMetadataContext(
            table_schema=table_schema,
            schema_hash_value=schema_hash_value,
            schema_digest_value=schema_digest_value,
            partition_columns=partition_columns,
            repo=context.repo,
            commit=context.commit,
            snapshot_id=context.snapshot_id,
        )
    )
    return ArrowDatasetWriteOptions(
        partition_columns=partition_columns,
        schema_hash=schema_hash_value,
        manifest_extras=manifest_extras,
        schema_metadata=metadata,
        stable_sort_keys=stable_sort_keys,
    )


def insert_analytics_rows(
    gateway: BuildGateway,
    contract: DatasetContract,
    rows: Sequence[Mapping[str, object]],
    *,
    delete_scope: DeleteScope | None = None,
    scope: str | None = None,
) -> int:
    """Persist rows for a dataset contract to parquet datasets.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    contract
        Dataset contract describing the target table.
    rows
        Rows to insert.
    delete_scope
        Optional deletion scope for clearing existing data.
    scope
        Optional scope label for logging.

    Returns
    -------
    int
        Number of rows inserted.

    """
    _ = scope
    return _write_parquet_dataset(
        gateway=gateway,
        contract=contract,
        rows=rows,
        delete_scope=delete_scope,
    )


@dataclass(frozen=True, slots=True)
class AnalyticsPipelineRequest:
    """Inputs required to execute and persist an analytics QuerySpec."""

    source: QuerySource
    spec: QuerySpec
    table_key: str
    ctx: ExecutionContext
    delete_scope: DeleteScope | None = None
    manifest_dir: Path | None = None
    manifest_options: RunManifestOptions | None = None


def run_analytics_pipeline_to_parquet(
    gateway: BuildGateway,
    *,
    request: AnalyticsPipelineRequest,
) -> int:
    """Execute a QuerySpec and persist the finalized output to parquet datasets.

    Parameters
    ----------
    gateway
        Storage gateway for dataset access.
    request
        Pipeline execution inputs and optional deletion scope.

    Returns
    -------
    int
        Number of rows written from the finalized output.

    Raises
    ------
    DatasetSchemaMissingError
        If the dataset schema is missing.
    UnsupportedDeleteScopeError
        If delete_scope is unsupported for the target table.
    """
    contract = get_analytics_dataset_contract(gateway, request.table_key)
    table_schema = contract.schema
    if table_schema is None:
        raise DatasetSchemaMissingError(contract.table_key)
    if request.delete_scope is not None and not _table_supports_snapshot_delete(
        contract.table_key
    ):
        raise UnsupportedDeleteScopeError(contract.table_key)
    dataset_root_dir, snapshot_id, repo, commit = _resolve_parquet_context(gateway)
    write_context = _WriteContext(
        dataset_root=dataset_root_dir,
        snapshot_id=snapshot_id,
        repo=repo,
        commit=commit,
    )
    manifest_dir = request.manifest_dir or _manifest_dir_for_snapshot(
        dataset_root=write_context.dataset_root,
        table_key=request.table_key,
        snapshot_id=write_context.snapshot_id,
    )
    manifest_options = request.manifest_options or _manifest_options_for_snapshot(
        table_key=request.table_key,
        snapshot_id=write_context.snapshot_id,
        repo=write_context.repo,
        commit=write_context.commit,
    )
    result = run_analytics_pipeline(
        AnalyticsPipelineRunRequest(
            source=request.source,
            spec=request.spec,
            table_key=request.table_key,
            ctx=request.ctx,
            manifest_dir=manifest_dir,
            manifest_options=manifest_options,
        )
    )
    stable_sort_keys = _resolve_manifest_sort_keys(table_schema)
    manifest_extras = _manifest_extras(
        table_schema,
        finalize_counts=finalize_artifact_counts(result),
    )
    options = _parquet_write_options(
        table_schema=table_schema,
        stable_sort_keys=stable_sort_keys,
        context=write_context,
        manifest_extras=manifest_extras,
    )
    reader = table_to_reader(result.good, batch_size=DEFAULT_ARROW_BATCH_SIZE)
    write_dataset(
        dataset_root=write_context.dataset_root,
        table_key=contract.table_key,
        snapshot_id=write_context.snapshot_id,
        data=reader,
        options=options,
    )
    _write_finalize_artifacts(
        context=write_context,
        base_table_key=contract.table_key,
        result=result,
    )
    _log_missing_metadata(
        dataset_root=write_context.dataset_root,
        table_key=contract.table_key,
        snapshot_id=write_context.snapshot_id,
    )
    return result.good.num_rows


def _manifest_dir_for_snapshot(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
) -> Path | None:
    try:
        return dataset_snapshot_dir(
            dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
    except SnapshotIdError:
        return None


def _manifest_options_for_snapshot(
    *,
    table_key: str,
    snapshot_id: str,
    repo: str,
    commit: str,
) -> RunManifestOptions:
    filename = f"run_manifest_{table_key.replace('.', '_')}.json"
    extras = {
        "table_key": table_key,
        "snapshot_id": snapshot_id,
        "repo": repo,
        "commit": commit,
        "write_source": "analytics_insert",
    }
    return RunManifestOptions(extras=extras, filename=filename)


def validate_contract_rows(
    table_key: str,
    rows: Sequence[Mapping[str, object]],
    *,
    gateway: BuildGateway | None = None,
    validation_profile: ValidationProfile | None = None,
) -> list[dict[str, object]]:
    """
    Validate rows for a dataset using Arrow/Polars checks and return normalized dicts.

    Missing values are normalized to ``None`` for safe DuckDB insertion.

    Returns
    -------
    list[dict[str, object]]
        Validated rows coerced to serializable dictionaries.

    Raises
    ------
    ValueError
        If rows include columns not present in the dataset schema.
    """
    if not rows:
        return []
    observation_provider = gateway.schemas if gateway is not None else None
    resolution = resolve_table_schema(table_key, observation_provider=observation_provider)
    resolved_profile = _resolved_validation_profile(table_key, gateway, validation_profile)
    try:
        records = _validated_records(
            table_key,
            rows,
            table_schema=resolution.table_schema,
            observation=resolution.observation,
            validation_profile=resolved_profile,
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    column_types = _column_types_for_schema(resolution.table_schema)
    return _normalize_records(records, column_types)


def _resolved_validation_profile(
    table_key: str,
    gateway: BuildGateway | None,
    validation_profile: ValidationProfile | None,
) -> ValidationProfile | None:
    if validation_profile is not None:
        return validation_profile
    if gateway is None:
        return None
    dataset = gateway.datasets.by_table_key.get(table_key)
    if dataset is None:
        return None
    return dataset.validation_profile


def _validated_records(
    table_key: str,
    rows: Sequence[Mapping[str, object]],
    *,
    table_schema: TableSchema | None,
    observation: SchemaObservationRecord | None,
    validation_profile: ValidationProfile | None,
) -> list[dict[str, object]]:
    if table_schema is None:
        table = pa.table(rows)
        return list(iter_rows(table))
    _validate_extra_columns(table_key, rows, table_schema=table_schema)
    table, _ = table_for_rows(table_key, rows)
    table = table.select([col.name for col in table_schema.columns])
    context = ColumnarValidationContext(
        table_schema=table_schema,
        schema_observation=observation,
        validation_profile=validation_profile,
    )
    validate_table(
        table_key,
        table,
        context=context,
        mode="strict",
    )
    return list(iter_rows(table))


def _validate_extra_columns(
    table_key: str,
    rows: Sequence[Mapping[str, object]],
    *,
    table_schema: TableSchema,
) -> None:
    expected = {col.name for col in table_schema.columns}
    extra = {str(name) for row in rows for name in row if name not in expected}
    if not extra:
        return
    extras = ", ".join(sorted(extra))
    message = f"Unexpected columns for {table_key}: {extras}"
    raise ValueError(message)


def _column_types_for_schema(table_schema: TableSchema | None) -> dict[str, ColumnType]:
    if table_schema is None:
        return {}
    return {col.name: col.type for col in table_schema.columns}


def _normalize_records(
    records: list[dict[str, object]],
    column_types: Mapping[str, ColumnType],
) -> list[dict[str, object]]:
    return [
        {
            str(key): normalize_row_value_for_type(value, column_types.get(str(key)))
            for key, value in record.items()
        }
        for record in records
    ]


def _record_batch_reader_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    schema: pa.Schema,
    batch_size: int,
) -> pa.RecordBatchReader:
    if not rows:
        return empty_reader_from_schema(schema)

    def _iter_batches() -> Iterable[pa.RecordBatch]:
        for chunk in _chunked_rows(rows, batch_size=batch_size):
            yield pa.RecordBatch.from_pylist(chunk, schema=schema)

    reader = record_batch_reader_from_iterable(_iter_batches(), empty_policy="none")
    if reader is None:
        return empty_reader_from_schema(schema)
    return reader


def _chunked_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    batch_size: int,
) -> Iterable[Sequence[Mapping[str, object]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


__all__ = [
    "AnalyticsPipelineRequest",
    "get_analytics_dataset_contract",
    "get_delete_sql_by_table",
    "get_function_ast_features_contract",
    "insert_analytics_rows",
    "run_analytics_pipeline_to_parquet",
    "snapshot_plan",
    "snapshot_reader",
    "snapshot_table",
    "validate_contract_rows",
]
