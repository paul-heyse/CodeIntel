"""Pandera contract integration for Hamilton nodes.

This module provides utilities to attach Pandera validation to Hamilton
node outputs using the existing SCHEMA_REGISTRY as the schema source.

Design Principles
-----------------
1. Pandera schemas come from SCHEMA_REGISTRY (single source of truth).
2. Integration uses Hamilton's @check_output decorator.
3. Validation errors propagate with full Pandera error context.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

import pandas as pd

from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY

if TYPE_CHECKING:
    import pandera as pa

    from codeintel.build.hamilton.io.dataset_ref import DatasetRef
    from codeintel.storage.gateway import StorageGateway

__all__ = [
    "get_pandera_schema",
    "validate_dataframe",
    "validate_dataset_ref",
    "with_contract",
]

log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def _ensure_dataframe(result: object, table_key: str) -> pd.DataFrame:
    """Ensure validation inputs are DataFrames.

    Parameters
    ----------
    result
        Object to validate.
    table_key
        Table key used for error context.

    Returns
    -------
    pd.DataFrame
        DataFrame to validate.

    Raises
    ------
    TypeError
        If the provided result is not a pandas DataFrame.
    """
    if isinstance(result, pd.DataFrame):
        return result
    msg = f"Expected pandas.DataFrame for {table_key}, got {type(result).__name__}"
    raise TypeError(msg)


def get_pandera_schema(table_key: str) -> pa.DataFrameSchema | None:
    """Retrieve Pandera schema from registry.

    Parameters
    ----------
    table_key
        Fully-qualified table name.

    Returns
    -------
    pa.DataFrameSchema | None
        Pandera schema if registered, None otherwise.

    Examples
    --------
    >>> schema = get_pandera_schema("analytics.function_metrics")
    >>> schema is not None
    True
    """
    dataset_schema = SCHEMA_REGISTRY.get(table_key)
    if dataset_schema is None:
        return None
    return dataset_schema.pandera_schema


def validate_dataframe(df: pd.DataFrame, table_key: str) -> pd.DataFrame:
    """Validate a DataFrame against its registered schema.

    Parameters
    ----------
    df
        DataFrame to validate.
    table_key
        Table key for schema lookup.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame (may be coerced).

    Raises
    ------
    ValueError
        If no schema is registered for the table key, or if validation fails.
    TypeError
        If the provided object is not a pandas DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({"repo": ["r1"], "loc": [100]})
    >>> validated = validate_dataframe(df, "test.table")
    """
    schema = get_pandera_schema(table_key)
    if schema is None:
        msg = f"No Pandera schema registered for {table_key}"
        raise ValueError(msg)
    if not isinstance(df, pd.DataFrame):
        msg = f"Expected pandas.DataFrame for {table_key}, got {type(df).__name__}"
        raise TypeError(msg)
    return schema.validate(df)


def with_contract(table_key: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function to validate its output against a Pandera schema.

    This is a lightweight decorator that validates the function's return
    value (expected to be a DataFrame) against the schema from SCHEMA_REGISTRY.

    Parameters
    ----------
    table_key
        Table key for schema lookup.

    Returns
    -------
    Callable[[F], F]
        Decorated function with Pandera validation.

    Notes
    -----
    If no schema is registered for the table key, the decorator passes
    through without validation and logs a warning.

    Examples
    --------
    >>> @with_contract("analytics.function_metrics")
    ... def compute_metrics(data: pd.DataFrame) -> pd.DataFrame:
    ...     return process(data)
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            result = func(*args, **kwargs)
            if not isinstance(result, pd.DataFrame):
                return result

            schema = get_pandera_schema(table_key)
            if schema is None:
                log.warning(
                    "No Pandera schema for %s; skipping validation for %s",
                    table_key,
                    func.__name__,
                )
                return result

            return cast("R", schema.validate(result))

        return cast("Callable[P, R]", wrapper)

    return decorator


def validate_dataset_ref(
    ref: DatasetRef,
    gateway: StorageGateway,
) -> tuple[bool, str | None]:
    """Validate a DatasetRef's underlying table against its schema.

    Loads the table data and validates it against the Pandera schema
    from SCHEMA_REGISTRY.

    Parameters
    ----------
    ref
        Dataset reference to validate.
    gateway
        Storage gateway for table access.

    Returns
    -------
    tuple[bool, str | None]
        (is_valid, error_message) tuple. If valid, error_message is None.

    Examples
    --------
    >>> ref = DatasetRef(table_key="analytics.function_metrics")
    >>> is_valid, error = validate_dataset_ref(ref, gateway)
    >>> if not is_valid:
    ...     print(f"Validation failed: {error}")
    """
    schema = get_pandera_schema(ref.table_key)
    if schema is None:
        return True, None

    try:
        table = gateway.ibis.table(ref.table_key)
        df = table.execute()
        frame = _ensure_dataframe(df, ref.table_key)
        schema.validate(frame)
    except (ValueError, TypeError, RuntimeError) as e:
        return False, str(e)
    else:
        return True, None


def contract_status_for_table(table_key: str) -> dict[str, Any]:
    """Get contract status information for a table.

    Parameters
    ----------
    table_key
        Fully-qualified table name.

    Returns
    -------
    dict[str, Any]
        Status information including whether schema exists and column info.

    Examples
    --------
    >>> status = contract_status_for_table("analytics.function_metrics")
    >>> status["has_schema"]
    True
    """
    schema = get_pandera_schema(table_key)
    if schema is None:
        return {
            "table_key": table_key,
            "has_schema": False,
            "columns": [],
        }

    return {
        "table_key": table_key,
        "has_schema": True,
        "columns": list(schema.columns.keys()),
        "coerce": schema.coerce,
        "strict": schema.strict,
    }
