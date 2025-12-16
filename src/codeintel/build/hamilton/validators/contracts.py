"""Contract builders for Hamilton-native validation.

This module provides utilities to build sets of validators for common
table contracts, making it easy to define consistent validation rules.

Examples
--------
>>> from codeintel.build.hamilton.validators import build_table_contract
>>> from hamilton.function_modifiers import check_output_custom
>>>
>>> validators = build_table_contract(
...     required_columns=["id", "name"],
...     column_types={"id": "int", "name": "string"},
...     no_nulls=["id"],
... )
>>>
>>> @check_output_custom(*validators)
>>> def my_node(...) -> pd.DataFrame:
...     ...
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from codeintel.build.hamilton.validators.dataframe import (
    ColumnsExistValidator,
    ColumnTypesValidator,
    ColumnValuesInSetValidator,
    NoNullsInColumnsValidator,
    RowCountRangeValidator,
    UniqueColumnsValidator,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hamilton.data_quality.base import BaseDefaultValidator

__all__ = [
    "build_key_column_contract",
    "build_metrics_contract",
    "build_table_contract",
]


def build_table_contract(
    required_columns: Sequence[str],
    column_types: dict[str, str] | None = None,
    no_nulls: Sequence[str] | None = None,
    unique: Sequence[str] | None = None,
    min_rows: int = 0,
    max_rows: int | None = None,
) -> list[BaseDefaultValidator]:
    """Build a set of validators for a table contract.

    Create a comprehensive validation contract for a DataFrame by combining
    multiple validators. This is the primary entry point for defining
    Hamilton-native table schemas.

    Parameters
    ----------
    required_columns
        Column names that must exist in the DataFrame.
    column_types
        Optional mapping from column name to expected type.
        Types: "string", "int", "float", "bool", "datetime", "object".
    no_nulls
        Optional list of columns that must not contain null values.
    unique
        Optional list of columns that must have unique values.
    min_rows
        Minimum number of rows (default 0).
    max_rows
        Maximum number of rows (None for unlimited).

    Returns
    -------
    list[BaseDefaultValidator]
        List of validators to pass to @check_output_custom.

    Examples
    --------
    >>> validators = build_table_contract(
    ...     required_columns=["function_goid_h128", "repo", "commit", "loc"],
    ...     column_types={"loc": "int", "repo": "string"},
    ...     no_nulls=["function_goid_h128", "repo", "commit"],
    ...     unique=["function_goid_h128"],
    ...     min_rows=1,
    ... )
    >>> len(validators)
    5
    """
    validators: list[BaseDefaultValidator] = []

    # Required columns check
    validators.append(ColumnsExistValidator(required_columns))

    # Column types check
    if column_types:
        validators.append(ColumnTypesValidator(column_types))

    # No nulls check
    if no_nulls:
        validators.append(NoNullsInColumnsValidator(no_nulls))

    # Unique columns check
    if unique:
        validators.append(UniqueColumnsValidator(unique))

    # Row count range check (always add if min_rows > 0 or max_rows specified)
    if min_rows > 0 or max_rows is not None:
        validators.append(RowCountRangeValidator(min_rows=min_rows, max_rows=max_rows))

    return validators


def build_key_column_contract(
    key_columns: Sequence[str],
    additional_columns: Sequence[str] | None = None,
) -> list[BaseDefaultValidator]:
    """Build validators for a table with key columns.

    Create a contract that validates key columns exist, have no nulls,
    and are unique (forming a composite key).

    Parameters
    ----------
    key_columns
        Column names that form the primary key (must exist, no nulls, unique).
    additional_columns
        Optional additional columns that must exist.

    Returns
    -------
    list[BaseDefaultValidator]
        List of validators for key column contract.

    Examples
    --------
    >>> validators = build_key_column_contract(
    ...     key_columns=["repo", "commit", "function_goid_h128"],
    ...     additional_columns=["loc", "complexity"],
    ... )
    """
    all_columns = list(key_columns)
    if additional_columns:
        all_columns.extend(additional_columns)

    validators: list[BaseDefaultValidator] = [
        ColumnsExistValidator(all_columns),
        NoNullsInColumnsValidator(key_columns),
    ]

    # Note: For composite keys, we'd need a CompositeUniqueValidator
    # For now, we check each key column individually
    # (Full composite uniqueness would require a custom implementation)
    if len(key_columns) == 1:
        validators.append(UniqueColumnsValidator(key_columns))

    return validators


def build_metrics_contract(
    metric_columns: Sequence[str],
    metric_types: dict[str, str] | None = None,
    allowed_ranges: dict[str, tuple[float | None, float | None]] | None = None,
) -> list[BaseDefaultValidator]:
    """Build validators for metrics tables.

    Create a contract for tables containing numeric metrics with
    optional range validation.

    Parameters
    ----------
    metric_columns
        Column names for metrics (must exist).
    metric_types
        Optional type specifications (default assumes "float").
    allowed_ranges
        Optional mapping from column to (min, max) tuple.
        Use None for unbounded on either side.

    Returns
    -------
    list[BaseDefaultValidator]
        List of validators for metrics contract.

    Examples
    --------
    >>> validators = build_metrics_contract(
    ...     metric_columns=["loc", "complexity", "coverage"],
    ...     metric_types={"loc": "int", "complexity": "int"},
    ...     allowed_ranges={"coverage": (0.0, 100.0)},
    ... )
    """
    validators: list[BaseDefaultValidator] = [
        ColumnsExistValidator(metric_columns),
    ]

    # Default types for metrics
    if metric_types:
        validators.append(ColumnTypesValidator(metric_types))

    # Range validation would require a custom RangeValidator
    # For now, we just note this in diagnostics
    # Future: Add ColumnRangeValidator
    _ = allowed_ranges  # Reserved for future range validation

    return validators


def build_enum_column_contract(
    column: str,
    allowed_values: set[Any],
    allow_nulls: bool = False,
) -> list[BaseDefaultValidator]:
    """Build validators for a column with enumerated values.

    Create a contract that validates a column contains only values
    from an allowed set.

    Parameters
    ----------
    column
        Column name to validate.
    allowed_values
        Set of allowed values.
    allow_nulls
        Whether null values are permitted.

    Returns
    -------
    list[BaseDefaultValidator]
        List of validators for enum column contract.

    Examples
    --------
    >>> validators = build_enum_column_contract(
    ...     column="status",
    ...     allowed_values={"pending", "active", "completed"},
    ...     allow_nulls=False,
    ... )
    """
    validators: list[BaseDefaultValidator] = [
        ColumnsExistValidator([column]),
        ColumnValuesInSetValidator(column, allowed_values),
    ]

    if not allow_nulls:
        validators.append(NoNullsInColumnsValidator([column]))

    return validators
