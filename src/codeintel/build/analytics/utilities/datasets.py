"""Analytics dataset contract and persistence helpers.

This module is a thin layer around the canonical build-time contract and schema
providers, plus convenience helpers for validating and inserting rows.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
from sqlglot import exp

from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.compute_columns import append_constant_columns

if TYPE_CHECKING:
    from codeintel.build.analytics.utilities.persistence import DeleteScope
    from codeintel.core.gateway import BuildGateway
    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.core.schemas.primitives import ColumnType, TableSchema

from codeintel.build.schemas import (
    ContractResolutionMode,
    ContractResolutionSettings,
    get_contract_for_table_key,
)
from codeintel.build.validation.columnar import ColumnarValidationContext, validate_table
from codeintel.config.datasets.columns import load_columns_by_table
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.datasets.arrow_store import ArrowDatasetWriteOptions, write_dataset
from codeintel.core.schemas.arrow_gen import arrow_contract_for_table_schema
from codeintel.core.schemas.hashing import schema_digest, schema_hash
from codeintel.core.schemas.resolution import resolve_table_schema
from codeintel.core.schemas.row_models import normalize_row_value_for_type
from codeintel.core.validation.profiles import ValidationProfile

_FULL_CONTRACT_SETTINGS = ContractResolutionSettings(mode=ContractResolutionMode.FULL)


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


def _manifest_extras(table_schema: TableSchema) -> dict[str, object]:
    return {
        "table_schema": table_schema.to_json_obj(),
        "write_source": "analytics_insert",
        "written_at": datetime.now(tz=UTC).isoformat(),
    }


@dataclass(frozen=True, slots=True)
class _ParquetMetadataContext:
    table_schema: TableSchema
    schema_hash_value: str
    schema_digest_value: str
    partition_columns: tuple[str, ...]
    repo: str
    commit: str
    snapshot_id: str


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
    arrow_schema = arrow_contract_for_table_schema(table_schema=table_schema)
    reader = _record_batch_reader_from_rows(
        normalized,
        schema=arrow_schema,
        batch_size=DEFAULT_ARROW_BATCH_SIZE,
    )
    schema_hash_value = schema_hash(table_schema)
    schema_digest_value = schema_digest(table_schema)
    partition_columns = _partition_columns_for_schema(table_schema)
    metadata = _parquet_metadata_payload(
        context=_ParquetMetadataContext(
            table_schema=table_schema,
            schema_hash_value=schema_hash_value,
            schema_digest_value=schema_digest_value,
            partition_columns=partition_columns,
            repo=repo,
            commit=commit,
            snapshot_id=snapshot_id,
        )
    )
    options = ArrowDatasetWriteOptions(
        partition_columns=partition_columns,
        schema_hash=schema_hash_value,
        manifest_extras=_manifest_extras(table_schema),
        schema_metadata=metadata,
    )
    write_dataset(
        dataset_root=dataset_root_dir,
        table_key=contract.table_key,
        snapshot_id=snapshot_id,
        data=reader,
        options=options,
    )
    return len(normalized)


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
        If rows include columns that are not present in the dataset schema.
    """
    if not rows:
        return []
    observation_provider = gateway.schemas if gateway is not None else None
    resolution = resolve_table_schema(
        table_key,
        observation_provider=observation_provider,
    )
    table_schema = resolution.table_schema
    observation = resolution.observation
    resolved_profile = validation_profile
    if resolved_profile is None and gateway is not None:
        dataset = gateway.datasets.by_table_key.get(table_key)
        if dataset is not None:
            resolved_profile = dataset.validation_profile
    records: list[dict[str, object]]
    if table_schema is None:
        table = pa.Table.from_pylist(rows)
        records = list(iter_rows(table))
    else:
        expected_columns = [col.name for col in table_schema.columns]
        table = pa.Table.from_pylist(rows)
        extra = [name for name in table.column_names if name not in expected_columns]
        if extra:
            extras = ", ".join(sorted(extra))
            message = f"Unexpected columns for {table_key}: {extras}"
            raise ValueError(message)
        missing = [name for name in expected_columns if name not in table.column_names]
        if missing:
            table = append_constant_columns(table, dict.fromkeys(missing))
        table = table.select(expected_columns)
        context = ColumnarValidationContext(
            table_schema=table_schema,
            schema_observation=observation,
            validation_profile=resolved_profile,
        )
        validate_table(
            table_key,
            table,
            context=context,
            mode="strict",
        )
        records = list(iter_rows(table))
    column_types: dict[str, ColumnType] = (
        {col.name: col.type for col in table_schema.columns} if table_schema is not None else {}
    )
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
        return pa.RecordBatchReader.from_batches(schema, [])

    def _iter_batches() -> Iterable[pa.RecordBatch]:
        for chunk in _chunked_rows(rows, batch_size=batch_size):
            yield pa.RecordBatch.from_pylist(chunk, schema=schema)

    return pa.RecordBatchReader.from_batches(schema, _iter_batches())


def _chunked_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    batch_size: int,
) -> Iterable[Sequence[Mapping[str, object]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


__all__ = [
    "get_analytics_dataset_contract",
    "get_delete_sql_by_table",
    "get_function_ast_features_contract",
    "insert_analytics_rows",
    "validate_contract_rows",
]
