"""Analytics dataset contract and persistence helpers.

This module is a thin layer around the canonical build-time contract and schema
providers, plus convenience helpers for validating and inserting rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, cast

import pandas as pd
from sqlglot import exp

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.analytics.utilities.persistence import DeleteScope
    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.storage.gateway import StorageGateway

from codeintel.build.hamilton.contracts.schemas.validation import validate_df
from codeintel.build.schemas import get_contract_for_table_key
from codeintel.config.datasets.columns import load_columns_by_table


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
    gateway: StorageGateway,
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
    return get_contract_for_table_key(table_key)


def get_function_ast_features_contract(
    gateway: StorageGateway,
) -> DatasetContract:
    """
    Return the dataset contract for function AST features.

    Returns
    -------
    DatasetContract
        Contract describing analytics.function_ast_features.
    """
    _ = gateway
    return get_contract_for_table_key("analytics.function_ast_features")


def insert_analytics_rows(
    gateway: StorageGateway,
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
    table_key: str, rows: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    """
    Validate rows for a dataset using Pandera and return normalized dicts.

    Missing values are normalized to ``None`` for safe DuckDB insertion.

    Returns
    -------
    list[dict[str, object]]
        Pandera-validated rows coerced to serializable dictionaries.
    """
    if not rows:
        return []
    df = validate_df(table_key, pd.DataFrame(rows))
    normalized = df.where(pd.notna(df), None)
    return normalized.to_dict(orient="records")


__all__ = [
    "get_analytics_dataset_contract",
    "get_delete_sql_by_table",
    "get_function_ast_features_contract",
    "insert_analytics_rows",
    "validate_contract_rows",
    "validate_tuple_rows",
]


def validate_tuple_rows(
    table_key: str,
    rows: Sequence[Mapping[str, object] | Sequence[object]],
    *,
    columns: Sequence[str] | None = None,
    schema: TableSchema | None = None,
) -> list[tuple[object, ...]]:
    """
    Validate tuple rows for a dataset and return normalized tuples.

    Parameters
    ----------
    table_key
        Fully qualified dataset key.
    rows
        Iterable of rows as mappings or positional sequences.
    columns
        Column order corresponding to the tuples.
    schema
        Optional TableSchema used to derive column order.

    Returns
    -------
    list[tuple[object, ...]]
        Pandera-validated rows with ``None`` for missing values.

    Raises
    ------
    ValueError
        If both ``columns`` and ``schema`` are provided or if column names cannot be
        determined.
    """
    if not rows:
        return []
    if columns is not None and schema is not None:
        message = "Specify either schema or columns, not both"
        raise ValueError(message)
    column_names = tuple(columns or (schema.column_names() if schema else ()))
    if not column_names:
        message = f"Column names required to validate rows for {table_key}"
        raise ValueError(message)

    first = rows[0]
    columns_index = pd.Index(column_names)
    if isinstance(first, Mapping):
        mapping_rows = cast("Sequence[Mapping[str, object]]", rows)
        df = pd.DataFrame(mapping_rows, columns=columns_index)
    else:
        tuple_rows = cast("Sequence[Sequence[object]]", rows)
        df = pd.DataFrame(tuple_rows, columns=columns_index)

    validated = validate_df(table_key, df)
    normalized = validated.where(pd.notna(validated), None)
    ordered = normalized.loc[:, columns_index]
    return [tuple(row) for row in ordered.itertuples(index=False, name=None)]
