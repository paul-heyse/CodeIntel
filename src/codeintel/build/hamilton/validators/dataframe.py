"""Hamilton-native DataFrame and Ibis table validators.

These validators integrate with Hamilton's @check_output_custom decorator
to provide framework-driven data validation at DAG execution time.

Supports both pandas DataFrames and Ibis table expressions:
- For DataFrames: Full validation (schema + data)
- For Ibis tables: Schema validation only (columns, types)

Examples
--------
>>> from hamilton.function_modifiers import check_output_custom
>>> from codeintel.build.hamilton.validators import ColumnsExistValidator
>>>
>>> @check_output_custom(ColumnsExistValidator(["id", "name"]))
>>> def my_node(...) -> pd.DataFrame:
...     ...
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd
from hamilton.data_quality.base import BaseDefaultValidator, ValidationResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    import ibis.expr.types as ir

__all__ = [
    "ColumnTypesValidator",
    "ColumnValuesInSetValidator",
    "ColumnsExistValidator",
    "NoNullsInColumnsValidator",
    "RowCountRangeValidator",
    "RowCountValidator",
    "UniqueColumnsValidator",
]

_log = logging.getLogger(__name__)

# Mapping from string dtype names to pandas dtype checks
_DTYPE_MAP: dict[str, tuple[str, ...]] = {
    "string": ("object", "string", "str"),
    "int": ("int64", "int32", "int16", "int8", "Int64", "Int32", "Int16", "Int8"),
    "int64": ("int64", "Int64"),
    "float": ("float64", "float32", "float16", "Float64", "Float32"),
    "float64": ("float64", "Float64"),
    "bool": ("bool", "boolean"),
    "datetime": ("datetime64[ns]", "datetime64[ns, UTC]"),
    "object": ("object",),
}

# Mapping from Ibis dtype strings to canonical type names
_IBIS_DTYPE_MAP: dict[str, tuple[str, ...]] = {
    "string": ("string", "String", "!string"),
    "int": ("int64", "int32", "int16", "int8", "Int64", "Int32", "Int16", "Int8"),
    "int64": ("int64", "Int64", "!int64"),
    "float": ("float64", "float32", "Float64", "Float32"),
    "float64": ("float64", "Float64", "!float64"),
    "bool": ("bool", "boolean", "Boolean"),
    "datetime": ("timestamp", "Timestamp"),
}


def _is_ibis_table(data: Any) -> bool:
    """Check if data is an Ibis table expression.

    Parameters
    ----------
    data
        The data to check.

    Returns
    -------
    bool
        True if data is an Ibis table expression.
    """
    try:
        import ibis.expr.types as ir

        return isinstance(data, ir.Table)
    except ImportError:
        return False


def _get_ibis_columns(data: ir.Table) -> list[str]:
    """Get column names from an Ibis table.

    Parameters
    ----------
    data
        Ibis table expression.

    Returns
    -------
    list[str]
        List of column names.
    """
    return list(data.columns)


def _get_ibis_schema(data: ir.Table) -> dict[str, str]:
    """Get column types from an Ibis table.

    Parameters
    ----------
    data
        Ibis table expression.

    Returns
    -------
    dict[str, str]
        Mapping from column name to type string.
    """
    schema = data.schema()
    return {name: str(dtype) for name, dtype in zip(schema.names, schema.types)}


class ColumnsExistValidator(BaseDefaultValidator):
    """Validate that required columns exist in a DataFrame or Ibis table.

    Supports both pandas DataFrames and Ibis table expressions.

    Parameters
    ----------
    columns
        List of column names that must be present.

    Examples
    --------
    >>> validator = ColumnsExistValidator(["id", "name", "value"])
    >>> result = validator.validate(pd.DataFrame({"id": [1], "name": ["a"]}))
    >>> result.passes
    False
    >>> "value" in result.message
    True
    """

    def __init__(self, columns: Sequence[str]) -> None:
        """Initialize with required column names."""
        self.required_columns = list(columns)

    @classmethod
    def applies_to(cls, datatype: type) -> bool:
        """Apply to DataFrame and Ibis table types.

        Returns
        -------
        bool
            True when the validator supports the provided datatype.
        """
        # Always return True - we check type at runtime in validate()
        # This allows support for both pd.DataFrame and ir.Table
        return True

    @classmethod
    def arg(cls) -> str:
        """Return the argument name for this validator.

        Returns
        -------
        str
            Argument name used in decorator configuration.
        """
        return "columns_exist"

    def description(self) -> str:
        """Return validator description.

        Returns
        -------
        str
            Human-readable description of the validation performed.
        """
        return f"Validates that columns {self.required_columns} exist"

    def validate(self, data: pd.DataFrame | Any) -> ValidationResult:
        """Validate that all required columns exist.

        Parameters
        ----------
        data
            DataFrame or Ibis table to validate.

        Returns
        -------
        ValidationResult
            Result with passes=True if all columns exist.
        """
        # Handle Ibis tables
        if _is_ibis_table(data):
            existing = set(_get_ibis_columns(data))
            data_type = "Ibis table"
        elif isinstance(data, pd.DataFrame):
            existing = set(data.columns)
            data_type = "DataFrame"
        else:
            # Unknown type - pass through with warning
            return ValidationResult(
                passes=True,
                message=f"Skipped validation for unsupported type: {type(data).__name__}",
                diagnostics={"skipped": True, "data_type": type(data).__name__},
            )

        missing = set(self.required_columns) - existing
        if missing:
            return ValidationResult(
                passes=False,
                message=f"Missing required columns in {data_type}: {sorted(missing)}",
                diagnostics={
                    "missing_columns": sorted(missing),
                    "existing_columns": sorted(existing),
                    "required_columns": self.required_columns,
                    "data_type": data_type,
                },
            )
        return ValidationResult(
            passes=True,
            message=f"All {len(self.required_columns)} required columns present in {data_type}",
            diagnostics={"required_columns": self.required_columns, "data_type": data_type},
        )


class ColumnTypesValidator(BaseDefaultValidator):
    """Validate that columns have expected dtypes.

    Supports both pandas DataFrames and Ibis table expressions.

    Parameters
    ----------
    column_types
        Mapping from column name to expected dtype string.
        Supported types: "string", "int", "float", "bool", "datetime", "object".

    Examples
    --------
    >>> validator = ColumnTypesValidator({"id": "int", "name": "string"})
    >>> df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
    >>> result = validator.validate(df)
    """

    def __init__(self, column_types: dict[str, str]) -> None:
        """Initialize with column type mapping."""
        self.column_types = column_types

    @classmethod
    def applies_to(cls, datatype: type) -> bool:
        """Apply to DataFrame and Ibis table types.

        Returns
        -------
        bool
            True when the validator supports the provided datatype.
        """
        return True  # Check type at runtime

    @classmethod
    def arg(cls) -> str:
        """Return the argument name for this validator.

        Returns
        -------
        str
            Argument name used in decorator configuration.
        """
        return "column_types"

    def description(self) -> str:
        """Return validator description.

        Returns
        -------
        str
            Human-readable description of the validation performed.
        """
        return f"Validates column types: {self.column_types}"

    def validate(self, data: pd.DataFrame | Any) -> ValidationResult:
        """Validate column dtypes.

        Parameters
        ----------
        data
            DataFrame or Ibis table to validate.

        Returns
        -------
        ValidationResult
            Result with details of any type mismatches.
        """
        mismatches: dict[str, dict[str, Any]] = {}

        # Handle Ibis tables
        if _is_ibis_table(data):
            schema = _get_ibis_schema(data)
            dtype_map = _IBIS_DTYPE_MAP
            data_type = "Ibis table"
            columns = set(schema.keys())
        elif isinstance(data, pd.DataFrame):
            schema = {col: str(data[col].dtype) for col in data.columns}
            dtype_map = _DTYPE_MAP
            data_type = "DataFrame"
            columns = set(data.columns)
        else:
            return ValidationResult(
                passes=True,
                message=f"Skipped type validation for unsupported type: {type(data).__name__}",
                diagnostics={"skipped": True, "data_type": type(data).__name__},
            )

        for col, expected_type in self.column_types.items():
            if col not in columns:
                # Column doesn't exist - skip (ColumnsExistValidator handles this)
                continue

            actual_dtype = schema[col]
            expected_dtypes = dtype_map.get(expected_type, (expected_type,))

            # Check if actual dtype matches any expected dtype (case-insensitive for Ibis)
            dtype_matches = any(exp.lower() in actual_dtype.lower() for exp in expected_dtypes)

            if not dtype_matches:
                mismatches[col] = {
                    "expected": expected_type,
                    "actual": actual_dtype,
                }

        if mismatches:
            return ValidationResult(
                passes=False,
                message=f"Column type mismatches in {data_type}: {list(mismatches.keys())}",
                diagnostics={"mismatches": mismatches, "data_type": data_type},
            )
        return ValidationResult(
            passes=True,
            message=f"All {len(self.column_types)} column types valid in {data_type}",
            diagnostics={"column_types": self.column_types, "data_type": data_type},
        )


class NoNullsInColumnsValidator(BaseDefaultValidator):
    """Validate that specified columns contain no null values.

    For Ibis tables, validation is skipped (requires query execution).

    Parameters
    ----------
    columns
        List of column names that must not contain nulls.

    Examples
    --------
    >>> validator = NoNullsInColumnsValidator(["id", "name"])
    >>> df = pd.DataFrame({"id": [1, None], "name": ["a", "b"]})
    >>> result = validator.validate(df)
    >>> result.passes
    False
    """

    def __init__(self, columns: Sequence[str]) -> None:
        """Initialize with column names to check for nulls."""
        self.columns = list(columns)

    @classmethod
    def applies_to(cls, datatype: type) -> bool:
        """Apply to DataFrame and Ibis table types.

        Returns
        -------
        bool
            True when the validator supports the provided datatype.
        """
        return True  # Check type at runtime

    @classmethod
    def arg(cls) -> str:
        """Return the argument name for this validator.

        Returns
        -------
        str
            Argument name used in decorator configuration.
        """
        return "no_nulls"

    def description(self) -> str:
        """Return validator description.

        Returns
        -------
        str
            Human-readable description of the validation performed.
        """
        return f"Validates no nulls in columns: {self.columns}"

    def validate(self, data: pd.DataFrame | Any) -> ValidationResult:
        """Validate that specified columns have no nulls.

        Parameters
        ----------
        data
            DataFrame or Ibis table to validate.

        Returns
        -------
        ValidationResult
            Result with null counts for any columns with nulls.
        """
        # For Ibis tables, skip data validation (requires execution)
        if _is_ibis_table(data):
            _log.debug("NoNullsInColumnsValidator: Skipping for Ibis table (lazy expression)")
            return ValidationResult(
                passes=True,
                message=f"Null check skipped for Ibis table (columns: {self.columns})",
                diagnostics={
                    "checked_columns": self.columns,
                    "skipped": True,
                    "reason": "Ibis tables are lazy; null checks require execution",
                },
            )

        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                passes=True,
                message=f"Skipped null validation for unsupported type: {type(data).__name__}",
                diagnostics={"skipped": True, "data_type": type(data).__name__},
            )

        null_counts: dict[str, int] = {}

        for col in self.columns:
            if col not in data.columns:
                # Column doesn't exist - skip
                continue

            null_count = data[col].isna().sum()
            if null_count > 0:
                null_counts[col] = int(null_count)

        if null_counts:
            return ValidationResult(
                passes=False,
                message=f"Columns with null values: {list(null_counts.keys())}",
                diagnostics={
                    "null_counts": null_counts,
                    "total_rows": len(data),
                },
            )
        return ValidationResult(
            passes=True,
            message=f"No nulls in {len(self.columns)} checked columns",
            diagnostics={"checked_columns": self.columns},
        )


class UniqueColumnsValidator(BaseDefaultValidator):
    """Validate that specified columns contain only unique values.

    For Ibis tables, validation is skipped (requires query execution).

    Parameters
    ----------
    columns
        List of column names that must have unique values.
        If multiple columns are specified, each is checked independently.

    Examples
    --------
    >>> validator = UniqueColumnsValidator(["id"])
    >>> df = pd.DataFrame({"id": [1, 1, 2]})
    >>> result = validator.validate(df)
    >>> result.passes
    False
    """

    def __init__(self, columns: Sequence[str]) -> None:
        """Initialize with column names that must be unique."""
        self.columns = list(columns)

    @classmethod
    def applies_to(cls, datatype: type) -> bool:
        """Apply to DataFrame and Ibis table types.

        Returns
        -------
        bool
            True when the validator supports the provided datatype.
        """
        return True  # Check type at runtime

    @classmethod
    def arg(cls) -> str:
        """Return the argument name for this validator.

        Returns
        -------
        str
            Argument name used in decorator configuration.
        """
        return "unique_columns"

    def description(self) -> str:
        """Return validator description.

        Returns
        -------
        str
            Human-readable description of the validation performed.
        """
        return f"Validates unique values in columns: {self.columns}"

    def validate(self, data: pd.DataFrame | Any) -> ValidationResult:
        """Validate that specified columns have unique values.

        Parameters
        ----------
        data
            DataFrame or Ibis table to validate.

        Returns
        -------
        ValidationResult
            Result with duplicate counts for any non-unique columns.
        """
        # For Ibis tables, skip data validation (requires execution)
        if _is_ibis_table(data):
            _log.debug("UniqueColumnsValidator: Skipping for Ibis table (lazy expression)")
            return ValidationResult(
                passes=True,
                message=f"Uniqueness check skipped for Ibis table (columns: {self.columns})",
                diagnostics={
                    "checked_columns": self.columns,
                    "skipped": True,
                    "reason": "Ibis tables are lazy; uniqueness checks require execution",
                },
            )

        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                passes=True,
                message=f"Skipped uniqueness validation for type: {type(data).__name__}",
                diagnostics={"skipped": True, "data_type": type(data).__name__},
            )

        duplicates: dict[str, int] = {}

        for col in self.columns:
            if col not in data.columns:
                # Column doesn't exist - skip
                continue

            duplicate_count = data[col].duplicated().sum()
            if duplicate_count > 0:
                duplicates[col] = int(duplicate_count)

        if duplicates:
            return ValidationResult(
                passes=False,
                message=f"Columns with duplicates: {list(duplicates.keys())}",
                diagnostics={
                    "duplicate_counts": duplicates,
                    "total_rows": len(data),
                },
            )
        return ValidationResult(
            passes=True,
            message=f"All {len(self.columns)} columns have unique values",
            diagnostics={"checked_columns": self.columns},
        )


class RowCountValidator(BaseDefaultValidator):
    """Validate that DataFrame has at least a minimum number of rows.

    For Ibis tables, validation is skipped (requires query execution).

    Parameters
    ----------
    min_rows
        Minimum number of rows required.

    Examples
    --------
    >>> validator = RowCountValidator(min_rows=10)
    >>> df = pd.DataFrame({"id": [1, 2, 3]})
    >>> result = validator.validate(df)
    >>> result.passes
    False
    """

    def __init__(self, min_rows: int) -> None:
        """Initialize with minimum row count."""
        self.min_rows = min_rows

    @classmethod
    def applies_to(cls, datatype: type) -> bool:
        """Apply to DataFrame and Ibis table types.

        Returns
        -------
        bool
            True when the validator supports the provided datatype.
        """
        return True  # Check type at runtime

    @classmethod
    def arg(cls) -> str:
        """Return the argument name for this validator.

        Returns
        -------
        str
            Argument name used in decorator configuration.
        """
        return "min_rows"

    def description(self) -> str:
        """Return validator description.

        Returns
        -------
        str
            Human-readable description of the validation performed.
        """
        return f"Validates minimum row count: {self.min_rows}"

    def validate(self, data: pd.DataFrame | Any) -> ValidationResult:
        """Validate that DataFrame has minimum rows.

        Parameters
        ----------
        data
            DataFrame or Ibis table to validate.

        Returns
        -------
        ValidationResult
            Result indicating if row count meets minimum.
        """
        # For Ibis tables, skip row count validation (requires execution)
        if _is_ibis_table(data):
            _log.debug("RowCountValidator: Skipping for Ibis table (lazy expression)")
            return ValidationResult(
                passes=True,
                message=f"Row count check skipped for Ibis table (min: {self.min_rows})",
                diagnostics={
                    "min_rows": self.min_rows,
                    "skipped": True,
                    "reason": "Ibis tables are lazy; row count requires execution",
                },
            )

        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                passes=True,
                message=f"Skipped row count validation for type: {type(data).__name__}",
                diagnostics={"skipped": True, "data_type": type(data).__name__},
            )

        actual_rows = len(data)
        if actual_rows < self.min_rows:
            return ValidationResult(
                passes=False,
                message=f"Row count {actual_rows} below minimum {self.min_rows}",
                diagnostics={
                    "actual_rows": actual_rows,
                    "min_rows": self.min_rows,
                },
            )
        return ValidationResult(
            passes=True,
            message=f"Row count {actual_rows} meets minimum {self.min_rows}",
            diagnostics={
                "actual_rows": actual_rows,
                "min_rows": self.min_rows,
            },
        )


class RowCountRangeValidator(BaseDefaultValidator):
    """Validate that DataFrame row count is within a range.

    For Ibis tables, validation is skipped (requires query execution).

    Parameters
    ----------
    min_rows
        Minimum number of rows (inclusive). Use 0 for no minimum.
    max_rows
        Maximum number of rows (inclusive). Use None for no maximum.

    Examples
    --------
    >>> validator = RowCountRangeValidator(min_rows=1, max_rows=1000)
    >>> df = pd.DataFrame({"id": range(500)})
    >>> result = validator.validate(df)
    >>> result.passes
    True
    """

    def __init__(self, min_rows: int = 0, max_rows: int | None = None) -> None:
        """Initialize with row count range."""
        self.min_rows = min_rows
        self.max_rows = max_rows

    @classmethod
    def applies_to(cls, datatype: type) -> bool:
        """Apply to DataFrame and Ibis table types.

        Returns
        -------
        bool
            True when the validator supports the provided datatype.
        """
        return True  # Check type at runtime

    @classmethod
    def arg(cls) -> str:
        """Return the argument name for this validator.

        Returns
        -------
        str
            Argument name used in decorator configuration.
        """
        return "row_count_range"

    def description(self) -> str:
        """Return validator description.

        Returns
        -------
        str
            Human-readable description of the validation performed.
        """
        return f"Validates row count in range [{self.min_rows}, {self.max_rows}]"

    def validate(self, data: pd.DataFrame | Any) -> ValidationResult:
        """Validate that DataFrame row count is in range.

        Parameters
        ----------
        data
            DataFrame or Ibis table to validate.

        Returns
        -------
        ValidationResult
            Result indicating if row count is within range.
        """
        # For Ibis tables, skip row count validation (requires execution)
        if _is_ibis_table(data):
            _log.debug("RowCountRangeValidator: Skipping for Ibis table (lazy expression)")
            return ValidationResult(
                passes=True,
                message="Row count range check skipped for Ibis table",
                diagnostics={
                    "min_rows": self.min_rows,
                    "max_rows": self.max_rows,
                    "skipped": True,
                    "reason": "Ibis tables are lazy; row count requires execution",
                },
            )

        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                passes=True,
                message=f"Skipped row count validation for type: {type(data).__name__}",
                diagnostics={"skipped": True, "data_type": type(data).__name__},
            )

        actual_rows = len(data)
        diagnostics: dict[str, Any] = {
            "actual_rows": actual_rows,
            "min_rows": self.min_rows,
            "max_rows": self.max_rows,
        }

        if actual_rows < self.min_rows:
            return ValidationResult(
                passes=False,
                message=f"Row count {actual_rows} below minimum {self.min_rows}",
                diagnostics=diagnostics,
            )

        if self.max_rows is not None and actual_rows > self.max_rows:
            return ValidationResult(
                passes=False,
                message=f"Row count {actual_rows} exceeds maximum {self.max_rows}",
                diagnostics=diagnostics,
            )

        return ValidationResult(
            passes=True,
            message=f"Row count {actual_rows} within range [{self.min_rows}, {self.max_rows}]",
            diagnostics=diagnostics,
        )


class ColumnValuesInSetValidator(BaseDefaultValidator):
    """Validate that column values are from an allowed set.

    For Ibis tables, validation is skipped (requires query execution).

    Parameters
    ----------
    column
        Column name to validate.
    allowed_values
        Set of allowed values.

    Examples
    --------
    >>> validator = ColumnValuesInSetValidator("status", {"active", "inactive"})
    >>> df = pd.DataFrame({"status": ["active", "pending"]})
    >>> result = validator.validate(df)
    >>> result.passes
    False
    """

    def __init__(self, column: str, allowed_values: set[Any]) -> None:
        """Initialize with column and allowed values."""
        self.column = column
        self.allowed_values = allowed_values

    @classmethod
    def applies_to(cls, datatype: type) -> bool:
        """Apply to DataFrame and Ibis table types.

        Returns
        -------
        bool
            True when the validator supports the provided datatype.
        """
        return True  # Check type at runtime

    @classmethod
    def arg(cls) -> str:
        """Return the argument name for this validator.

        Returns
        -------
        str
            Argument name used in decorator configuration.
        """
        return "values_in_set"

    def description(self) -> str:
        """Return validator description.

        Returns
        -------
        str
            Human-readable description of the validation performed.
        """
        return f"Validates {self.column} values in allowed set"

    def validate(self, data: pd.DataFrame | Any) -> ValidationResult:
        """Validate that column values are in the allowed set.

        Parameters
        ----------
        data
            DataFrame or Ibis table to validate.

        Returns
        -------
        ValidationResult
            Result indicating if all values are allowed.
        """
        # For Ibis tables, skip value validation (requires execution)
        if _is_ibis_table(data):
            # Check column exists in schema
            columns = _get_ibis_columns(data)
            if self.column not in columns:
                return ValidationResult(
                    passes=True,
                    message=f"Column {self.column} not present in Ibis table (skipped)",
                    diagnostics={"column": self.column, "skipped": True},
                )
            _log.debug("ColumnValuesInSetValidator: Skipping for Ibis table (lazy expression)")
            return ValidationResult(
                passes=True,
                message=f"Value set check skipped for Ibis table (column: {self.column})",
                diagnostics={
                    "column": self.column,
                    "allowed_values": list(self.allowed_values),
                    "skipped": True,
                    "reason": "Ibis tables are lazy; value checks require execution",
                },
            )

        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                passes=True,
                message=f"Skipped value validation for type: {type(data).__name__}",
                diagnostics={"skipped": True, "data_type": type(data).__name__},
            )

        if self.column not in data.columns:
            return ValidationResult(
                passes=True,
                message=f"Column {self.column} not present (skipped)",
                diagnostics={"column": self.column, "skipped": True},
            )

        actual_values = set(data[self.column].dropna().unique())
        invalid_values = actual_values - self.allowed_values

        if invalid_values:
            return ValidationResult(
                passes=False,
                message=f"Invalid values in {self.column}: {invalid_values}",
                diagnostics={
                    "column": self.column,
                    "invalid_values": list(invalid_values),
                    "allowed_values": list(self.allowed_values),
                },
            )
        return ValidationResult(
            passes=True,
            message=f"All values in {self.column} are valid",
            diagnostics={
                "column": self.column,
                "allowed_values": list(self.allowed_values),
            },
        )
