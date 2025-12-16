"""Schema documentation utilities for Hamilton-native validation.

This module provides utilities for generating Hamilton @schema.output
decorators from existing table metadata and for extracting schema
information from Hamilton nodes.

The goal is to make Hamilton the authoritative source for data schemas,
with utilities to bridge existing SCHEMA_REGISTRY during migration.

Examples
--------
Generate @schema.output arguments from table metadata:

>>> from codeintel.build.hamilton.schema_docs import schema_for_table
>>> columns = schema_for_table("analytics.function_metrics")
>>> # Use in decorator: @schema.output(*columns)

Extract schema from a Hamilton node:

>>> from codeintel.build.hamilton.schema_docs import extract_node_schema
>>> schema = extract_node_schema(my_module, "t__function_metrics__compute")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY

if TYPE_CHECKING:
    from types import ModuleType


__all__ = [
    "ColumnSchema",
    "TableSchema",
    "extract_node_schema",
    "schema_for_table",
    "schema_from_columns",
    "schema_output_tuple",
]

log = logging.getLogger(__name__)

_SCHEMA_TUPLE_MIN_LEN = 2


class ColumnSchema:
    """Schema definition for a single column.

    This class represents column metadata in a format compatible with
    Hamilton's @schema.output decorator.

    Parameters
    ----------
    name
        Column name.
    dtype
        Data type as string (e.g., "string", "int", "float").
    description
        Optional human-readable description.
    nullable
        Whether the column can contain null values.
    unique
        Whether the column values must be unique.

    Examples
    --------
    >>> col = ColumnSchema("id", "int", description="Primary key", unique=True)
    >>> col.to_tuple()
    ('id', 'int')
    >>> col.to_dict()
    {'name': 'id', 'dtype': 'int', 'description': 'Primary key', 'unique': True}
    """

    def __init__(
        self,
        name: str,
        dtype: str,
        description: str | None = None,
        *,
        nullable: bool = True,
        unique: bool = False,
    ) -> None:
        """Initialize column schema."""
        self.name = name
        self.dtype = dtype
        self.description = description
        self.nullable = nullable
        self.unique = unique

    def to_tuple(self) -> tuple[str, str]:
        """Convert to @schema.output tuple format.

        Returns
        -------
        tuple[str, str]
            Tuple of (column_name, dtype) for @schema.output.
        """
        return (self.name, self.dtype)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary representation.

        Returns
        -------
        dict[str, object]
            Dictionary with all schema attributes.
        """
        result: dict[str, object] = {
            "name": self.name,
            "dtype": self.dtype,
        }
        if self.description:
            result["description"] = self.description
        if not self.nullable:
            result["nullable"] = False
        if self.unique:
            result["unique"] = True
        return result


class TableSchema:
    """Schema definition for a table/DataFrame.

    This class represents table-level schema metadata, including
    columns and constraints.

    Parameters
    ----------
    table_key
        Fully-qualified table name (e.g., "analytics.function_metrics").
    columns
        List of column schemas.
    description
        Optional table description.
    primary_key
        Optional list of column names forming the primary key.

    Examples
    --------
    >>> columns = [
    ...     ColumnSchema("id", "int", unique=True),
    ...     ColumnSchema("name", "string"),
    ... ]
    >>> table = TableSchema("my.table", columns, primary_key=["id"])
    >>> table.schema_output_args()
    (('id', 'int'), ('name', 'string'))
    """

    def __init__(
        self,
        table_key: str,
        columns: list[ColumnSchema],
        description: str | None = None,
        primary_key: list[str] | None = None,
    ) -> None:
        """Initialize table schema."""
        self.table_key = table_key
        self.columns = columns
        self.description = description
        self.primary_key = primary_key or []

    def schema_output_args(self) -> tuple[tuple[str, str], ...]:
        """Generate arguments for @schema.output decorator.

        Returns
        -------
        tuple[tuple[str, str], ...]
            Tuple of (column_name, dtype) tuples for @schema.output.
        """
        return tuple(col.to_tuple() for col in self.columns)

    def column_names(self) -> list[str]:
        """Get list of column names.

        Returns
        -------
        list[str]
            All column names in order.
        """
        return [col.name for col in self.columns]

    def non_nullable_columns(self) -> list[str]:
        """Get list of columns that cannot be null.

        Returns
        -------
        list[str]
            Column names that have nullable=False.
        """
        return [col.name for col in self.columns if not col.nullable]

    def unique_columns(self) -> list[str]:
        """Get list of columns that must be unique.

        Returns
        -------
        list[str]
            Column names that have unique=True.
        """
        return [col.name for col in self.columns if col.unique]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary representation.

        Returns
        -------
        dict[str, object]
            Dictionary with all schema attributes.
        """
        result: dict[str, object] = {
            "table_key": self.table_key,
            "columns": [col.to_dict() for col in self.columns],
        }
        if self.description:
            result["description"] = self.description
        if self.primary_key:
            result["primary_key"] = self.primary_key
        return result


def schema_for_table(table_key: str) -> tuple[tuple[str, str], ...]:
    """Generate @schema.output arguments from existing table metadata.

    This function bridges existing SCHEMA_REGISTRY schemas to Hamilton's
    @schema.output format for migration purposes.

    Parameters
    ----------
    table_key
        Fully-qualified table name (e.g., "analytics.function_metrics").

    Returns
    -------
    tuple[tuple[str, str], ...]
        Tuple of (column_name, dtype) tuples for @schema.output.
        Returns empty tuple if table not found in registry.

    Examples
    --------
    >>> columns = schema_for_table("analytics.function_metrics")
    >>> # Use in decorator:
    >>> # @schema.output(*columns)
    >>> # def t__function_metrics__compute(...) -> pd.DataFrame:
    """
    dataset_schema = SCHEMA_REGISTRY.get(table_key)
    if dataset_schema is None:
        log.warning("Table %s not found in SCHEMA_REGISTRY", table_key)
        return ()

    pandera_schema = dataset_schema.pandera_schema
    columns = pandera_schema.columns
    if not hasattr(columns, "items"):
        log.warning("Pandera schema for %s has unexpected columns type", table_key)
        return ()

    return tuple((col_name, _pandera_dtype_to_string(col_spec.dtype)) for col_name, col_spec in columns.items())


def _pandera_dtype_to_string(dtype: object) -> str:
    """Convert Pandera dtype to string representation.

    Parameters
    ----------
    dtype
        Pandera dtype object.

    Returns
    -------
    str
        String representation of the dtype.
    """
    dtype_str = str(dtype)
    # Normalize common patterns
    if "int" in dtype_str.lower():
        return "int"
    if "float" in dtype_str.lower():
        return "float"
    if "bool" in dtype_str.lower():
        return "bool"
    if "datetime" in dtype_str.lower():
        return "datetime"
    if "string" in dtype_str.lower() or "object" in dtype_str.lower():
        return "string"
    return dtype_str


def schema_from_columns(
    columns: list[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    """Create schema output tuple from column definitions.

    Convenience function for defining inline schemas.

    Parameters
    ----------
    columns
        List of (column_name, dtype) tuples.

    Returns
    -------
    tuple[tuple[str, str], ...]
        Tuple suitable for @schema.output.

    Examples
    --------
    >>> schema = schema_from_columns(
    ...     [
    ...         ("id", "int"),
    ...         ("name", "string"),
    ...         ("value", "float"),
    ...     ]
    ... )
    >>> # @schema.output(*schema)
    """
    return tuple(columns)


def schema_output_tuple(
    *columns: tuple[str, str],
) -> tuple[tuple[str, str], ...]:
    """Create schema output tuple from variadic column arguments.

    Alternative syntax for defining inline schemas.

    Parameters
    ----------
    *columns
        Variable number of (column_name, dtype) tuples.

    Returns
    -------
    tuple[tuple[str, str], ...]
        Tuple suitable for @schema.output.

    Examples
    --------
    >>> schema = schema_output_tuple(
    ...     ("id", "int"),
    ...     ("name", "string"),
    ... )
    """
    return columns


def extract_node_schema(
    module: ModuleType,
    node_name: str,
) -> TableSchema | None:
    """Extract schema information from a Hamilton node.

    Reads @schema.output and @tag decorators from a Hamilton function
    to reconstruct the TableSchema.

    Parameters
    ----------
    module
        Python module containing the Hamilton node.
    node_name
        Name of the Hamilton function.

    Returns
    -------
    TableSchema | None
        Extracted schema, or None if not found or no schema defined.

    Examples
    --------
    >>> import my_hamilton_module
    >>> schema = extract_node_schema(my_hamilton_module, "t__function_metrics__compute")
    >>> if schema:
    ...     print(schema.column_names())
    """
    if not hasattr(module, node_name):
        log.warning("Node %s not found in module %s", node_name, module.__name__)
        return None

    func = getattr(module, node_name)

    # Look for schema decorator metadata
    # Hamilton stores decorator info in function attributes
    schema_info = getattr(func, "_schema", None)
    if schema_info is None:
        # Try looking for output_spec
        schema_info = getattr(func, "output_spec", None)

    if schema_info is None:
        log.debug("No schema found for node %s", node_name)
        return None

    # Extract table key from tags if available
    tags = getattr(func, "_tags", {})
    domain = tags.get("domain", "unknown")
    target = tags.get("target", node_name)
    table_key = f"{domain}.{target}"

    # Convert schema info to ColumnSchema objects
    columns: list[ColumnSchema] = []
    if isinstance(schema_info, (list, tuple)):
        columns.extend(
            ColumnSchema(name=str(item[0]), dtype=str(item[1]))
            for item in schema_info
            if isinstance(item, tuple) and len(item) >= _SCHEMA_TUPLE_MIN_LEN
        )

    if not columns:
        return None

    return TableSchema(
        table_key=table_key,
        columns=columns,
    )


# Common column type constants for easy reference
class ColumnTypes:
    """Common column type constants.

    Use these constants for consistent dtype specifications
    in schema definitions.

    Examples
    --------
    >>> from codeintel.build.hamilton.schema_docs import ColumnTypes
    >>> columns = [
    ...     ("id", ColumnTypes.INT),
    ...     ("name", ColumnTypes.STRING),
    ...     ("value", ColumnTypes.FLOAT),
    ... ]
    """

    STRING: str = "string"
    INT: str = "int"
    FLOAT: str = "float"
    BOOL: str = "bool"
    DATETIME: str = "datetime"
    OBJECT: str = "object"
