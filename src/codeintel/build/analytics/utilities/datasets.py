"""Analytics dataset contract and persistence helpers.

This module is a thin layer around the canonical build-time contract and schema
providers, plus convenience helpers for validating and inserting rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

import polars as pl
from sqlglot import exp

if TYPE_CHECKING:
    from codeintel.build.analytics.utilities.persistence import DeleteScope
    from codeintel.core.gateway import BuildGateway
    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.core.schemas.primitives import ColumnType

from codeintel.build.schemas import (
    ContractResolutionMode,
    ContractResolutionSettings,
    get_contract_for_table_key,
)
from codeintel.config.datasets.columns import load_columns_by_table
from codeintel.core.schemas.resolution import resolve_table_schema
from codeintel.core.schemas.row_models import normalize_row_value_for_type
from codeintel.storage.validation.columnar import validate_table

_FULL_CONTRACT_SETTINGS = ContractResolutionSettings(mode=ContractResolutionMode.FULL)


def _table_supports_snapshot_delete(table_key: str) -> bool:
    """Check if a table supports repo/commit scoped deletion.

    Parameters
    ----------
    table_key
        Fully qualified table key (e.g., 'analytics.function_metrics').

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


def insert_analytics_rows(
    gateway: BuildGateway,
    contract: DatasetContract,
    rows: Sequence[Mapping[str, object]],
    *,
    delete_scope: DeleteScope | None = None,
    scope: str | None = None,
) -> int:
    """Insert rows for a dataset contract using DuckDBPolicyBackend.

    Deletions are routed through DuckDBPolicyBackend for centralized SQL
    generation.

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

    Raises
    ------
    ValueError
        If delete columns cannot be determined for the requested dataset.
    """
    _ = scope
    backend = gateway.policy
    backend.ensure_table(contract.table_key)

    if delete_scope is not None:
        if not _table_supports_snapshot_delete(contract.table_key):
            message = f"Unsupported delete target: {contract.table_key}"
            raise ValueError(message)

        backend.delete_for_snapshot(
            contract.table_key,
            repo=delete_scope.repo,
            commit=delete_scope.commit,
        )

    return backend.bulk_insert_mappings(contract.table_key, rows) if rows else 0


def validate_contract_rows(
    table_key: str,
    rows: Sequence[Mapping[str, object]],
    *,
    gateway: BuildGateway | None = None,
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
    records: list[dict[str, object]]
    if table_schema is None:
        frame = pl.from_dicts(rows)
        records = frame.to_dicts()
    else:
        expected_columns = [col.name for col in table_schema.columns]
        frame = pl.from_dicts(rows)
        extra = [name for name in frame.columns if name not in expected_columns]
        if extra:
            extras = ", ".join(sorted(extra))
            message = f"Unexpected columns for {table_key}: {extras}"
            raise ValueError(message)
        missing = [name for name in expected_columns if name not in frame.columns]
        for name in missing:
            frame = frame.with_columns(pl.lit(None).alias(name))
        frame = frame.select(expected_columns)
        validate_table(
            table_key,
            frame.to_arrow(),
            table_schema=table_schema,
            schema_observation=observation,
            mode="strict",
        )
        records = frame.to_dicts()
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


__all__ = [
    "get_analytics_dataset_contract",
    "get_delete_sql_by_table",
    "get_function_ast_features_contract",
    "insert_analytics_rows",
    "validate_contract_rows",
]
