"""Pandera schemas and validation helpers for CodeIntel datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from pandera import Check, Column, DataFrameSchema

from codeintel.config.datasets import DATASET_CONTRACTS_BY_TABLE_KEY, DatasetContract
from codeintel.config.datasets.primitives import ColumnType, TableSchema

__all__ = ["DATASET_SCHEMAS", "get_dataset_schema", "validate_dataset_df"]

_STRING_DTYPE = pd.StringDtype()
_INT_DTYPE = pd.Int64Dtype()
_FLOAT_DTYPE = pd.Float64Dtype()
_BOOL_DTYPE = pd.BooleanDtype()

# Mapping from DuckDB column types to Pandera-friendly dtypes.
_COLUMN_TYPE_TO_DTYPE: Mapping[str, object] = {
    "BOOLEAN": _BOOL_DTYPE,
    "INTEGER": _INT_DTYPE,
    "BIGINT": _INT_DTYPE,
    "DOUBLE": _FLOAT_DTYPE,
    "DECIMAL": object,
    "DECIMAL(38,0)": object,
    "VARCHAR": _STRING_DTYPE,
    "JSON": _STRING_DTYPE,
    "TIMESTAMP": "datetime64[ns]",
    "TIMESTAMPTZ": "datetime64[ns]",
}

# Table-level checks keyed by fully-qualified table name.
_DATAFRAME_CHECKS: dict[str, list[Check]] = {
    "core.goids": [
        Check(
            lambda df: ~df.duplicated(subset=["repo", "commit", "goid_h128"]).any(),
            error="Duplicate (repo, commit, goid_h128) in core.goids",
        ),
        Check(
            lambda df: ~df.duplicated(subset=["repo", "commit", "urn"]).any(),
            error="Duplicate (repo, commit, urn) in core.goids",
        ),
        Check(
            lambda df: df["end_line"].isna() | (df["end_line"] >= df["start_line"]),
            error="end_line must be >= start_line when present",
        ),
    ],
    "core.goid_crosswalk": [
        Check(
            lambda df: df["end_line"].isna() | (df["end_line"] >= df["start_line"]),
            error="end_line must be >= start_line when present",
        ),
    ],
}

# Column-level checks keyed by table -> column -> list of checks.
_COLUMN_CHECKS: dict[str, dict[str, list[Check]]] = {
    "analytics.function_metrics": {
        "function_goid_h128": [Check(lambda s: s.isna() | (s >= 0))],
        "loc": [Check(lambda s: s.isna() | (s >= 0))],
        "logical_loc": [Check(lambda s: s.isna() | (s >= 0))],
        "cyclomatic_complexity": [Check(lambda s: s.isna() | (s >= 0))],
        "start_line": [Check(lambda s: s.isna() | (s >= 1))],
        "end_line": [Check(lambda s: s.isna() | (s >= 1))],
    },
    "analytics.function_types": {
        "function_goid_h128": [Check(lambda s: s.isna() | (s >= 0))],
        "start_line": [Check(lambda s: s.isna() | (s >= 1))],
        "end_line": [Check(lambda s: s.isna() | (s >= 1))],
    },
    "graph.call_graph_edges": {
        "caller_goid_h128": [Check(lambda s: s.isna() | (s >= 0))],
        "callee_goid_h128": [Check(lambda s: s.isna() | (s >= 0))],
    },
    "graph.call_graph_nodes": {
        "goid_h128": [Check(lambda s: s.isna() | (s >= 0))],
    },
    "core.goids": {
        "goid_h128": [Check(lambda s: s.isna() | (s >= 0))],
        "start_line": [Check(lambda s: s.isna() | (s >= 1))],
        "end_line": [Check(lambda s: s.isna() | (s >= 1))],
    },
    "core.goid_crosswalk": {
        "start_line": [Check(lambda s: s.isna() | (s >= 1))],
        "end_line": [Check(lambda s: s.isna() | (s >= 1))],
    },
}

_SCHEMA_TABLE_KEYS: tuple[str, ...] = (
    "analytics.function_metrics",
    "analytics.function_types",
    "graph.call_graph_nodes",
    "graph.call_graph_edges",
    "core.goids",
    "core.goid_crosswalk",
)


def _dtype_for_column_type(col_type: ColumnType) -> object:
    normalized = col_type.upper()
    if normalized.startswith("DECIMAL"):
        return object
    return _COLUMN_TYPE_TO_DTYPE.get(normalized, object)


def _build_columns(
    table_key: str,
    schema: TableSchema,
    *,
    column_checks: Mapping[str, list[Check]],
) -> dict[str, Column[Any]]:
    columns: dict[str, Column[Any]] = {}
    for col in schema.columns:
        checks = list(column_checks.get(col.name, ()))
        columns[col.name] = Column(
            _dtype_for_column_type(col.type),
            nullable=col.nullable,
            checks=checks,
        )
    return columns


def _build_schema(contract: DatasetContract) -> DataFrameSchema:
    if contract.schema is None:
        message = f"Dataset {contract.table_key} is missing a TableSchema"
        raise ValueError(message)
    table_key = contract.table_key
    column_checks = _COLUMN_CHECKS.get(table_key, {})
    columns = _build_columns(table_key, contract.schema, column_checks=column_checks)
    dataframe_checks = _DATAFRAME_CHECKS.get(table_key, [])
    return DataFrameSchema(
        columns,
        strict=True,
        coerce=True,
        checks=dataframe_checks,
        name=table_key,
    )


def _materialize_schemas() -> dict[str, DataFrameSchema]:
    schemas: dict[str, DataFrameSchema] = {}
    for table_key in _SCHEMA_TABLE_KEYS:
        contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
        if contract is None:
            continue
        schemas[table_key] = _build_schema(contract)
    return schemas


DATASET_SCHEMAS: dict[str, DataFrameSchema] = _materialize_schemas()


def get_dataset_schema(table_key: str) -> DataFrameSchema | None:
    """Return the Pandera schema for a dataset when registered."""
    return DATASET_SCHEMAS.get(table_key)


def validate_dataset_df(table_key: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a dataset DataFrame against its Pandera schema when available.

    Parameters
    ----------
    table_key
        Fully qualified table name (e.g., ``analytics.function_metrics``).
    df
        DataFrame to validate.

    Returns
    -------
    pandas.DataFrame
        Validated (and possibly coerced) DataFrame.
    """
    schema = get_dataset_schema(table_key)
    if schema is None:
        return df
    return schema.validate(df, lazy=True)
