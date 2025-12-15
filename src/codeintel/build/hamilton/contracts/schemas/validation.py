"""Centralized Pandera validation helpers backed by SCHEMA_REGISTRY."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pandas as pd
from pandera.errors import SchemaErrors

from codeintel.build.hamilton.contracts.schemas.registry import SCHEMA_REGISTRY
from codeintel.core.schemas.json_schema_gen import pandera_to_json_schema

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandera import DataFrameSchema

_log = logging.getLogger(__name__)

ValidationMode = str


def get_pandera_schema(table_key: str) -> DataFrameSchema | None:
    """Return the Pandera schema for a dataset if registered.

    Returns
    -------
    DataFrameSchema | None
        Registered schema for the dataset, or ``None`` when absent.
    """
    schema = SCHEMA_REGISTRY.get(table_key)
    if schema is None:
        return None
    return schema.pandera_schema


def validate_df(
    table_key: str,
    df: pd.DataFrame,
    *,
    mode: ValidationMode = "strict",
) -> pd.DataFrame:
    """
    Validate a DataFrame against the registered Pandera schema when present.

    Parameters
    ----------
    table_key
        Fully qualified dataset name.
    df
        DataFrame to validate.
    mode
        Validation handling: ``"strict"`` raises on errors, ``"warn"`` logs and
        returns the original frame, ``"skip"`` bypasses validation.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame, or the original frame when validation is skipped.

    Raises
    ------
    SchemaErrors
        When validation fails in ``"strict"`` mode.
    """
    schema = get_pandera_schema(table_key)
    if schema is None or mode == "skip":
        return df

    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        if mode == "warn":
            _log.warning(
                "Pandera validation warning for %s: %s",
                table_key,
                str(exc)[:200],
            )
            return df
        raise


def validate_rows(
    table_key: str,
    rows: Sequence[Mapping[str, object]] | Sequence[Sequence[object]],
) -> list[dict[str, Any]]:
    """Validate row-oriented data and return normalized dictionaries.

    This helper coerces ``NaN``/``NaT`` to ``None`` for safe serialization.

    Parameters
    ----------
    table_key
        Fully qualified dataset name.
    rows
        Row-oriented data (mapping rows or positional rows).

    Returns
    -------
    list[dict[str, Any]]
        Normalized rows validated against the registered schema.

    Raises
    ------
    ValueError
        Raised when positional rows are provided without a registered schema.
    TypeError
        Raised when a mix of mapping and positional rows is provided.
    """
    if not rows:
        return []

    schema = get_pandera_schema(table_key)
    if schema is None:
        first = rows[0]
        if not isinstance(first, Mapping):
            message = (
                f"Cannot normalize sequence-style rows for {table_key}; "
                "register a schema or provide mapping rows"
            )
            raise ValueError(message)
        normalized: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                message = f"Mixed row types for {table_key}; expected mapping rows"
                raise TypeError(message)
            normalized.append(dict(row))
        return normalized

    column_names = pd.Index(list(schema.columns.keys()))
    first = rows[0]
    if isinstance(first, Mapping):
        df = pd.DataFrame(rows, columns=column_names)
    else:
        df = pd.DataFrame(rows, columns=column_names)

    validated = validate_df(table_key, df, mode="strict")
    normalized_df = validated.where(pd.notna(validated), None)
    return normalized_df.to_dict(orient="records")


def dataset_json_schema(table_key: str) -> dict[str, Any] | None:
    """Return JSON Schema for a dataset when available.

    Parameters
    ----------
    table_key
        Fully qualified dataset table key (schema.table).

    Returns
    -------
    dict[str, Any] | None
        JSON Schema when a Pandera schema is registered, otherwise ``None``.
    """
    schema = SCHEMA_REGISTRY.get(table_key)
    if schema is None:
        return None
    return pandera_to_json_schema(schema.pandera_schema)


def dataset_json_schemas() -> dict[str, dict[str, Any]]:
    """Return JSON Schemas for all registered datasets.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from table_key to JSON Schema dictionary.
    """
    return {
        table_key: pandera_to_json_schema(schema.pandera_schema)
        for table_key, schema in SCHEMA_REGISTRY.all().items()
    }


__all__ = [
    "ValidationMode",
    "dataset_json_schema",
    "dataset_json_schemas",
    "get_pandera_schema",
    "pandera_to_json_schema",
    "validate_df",
    "validate_rows",
]
