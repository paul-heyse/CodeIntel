"""Safe SQL query builder for DuckDB operations.

This module provides type-safe SQL construction utilities that ensure safety by:

1. Table names are validated against known patterns via SafeTable
2. Column names are validated identifiers via SafeColumn
3. All user values are parameterized (never interpolated)

Ruff's S608 rule flags f-string SQL construction in this module. These are
intentional false positives - identifiers are validated before interpolation,
and values are always parameterized. Targeted noqa comments explain each case

Example
-------
>>> from codeintel.storage.sql_builder import QueryBuilder, SafeTable
>>>
>>> # Build a safe COUNT query
>>> query, params = QueryBuilder.count(
...     "analytics.function_metrics", where={"repo": "org/repo", "commit": "abc123"}
... )
>>> result = con.execute(query, params)
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

# Type aliases for clarity
type SqlParams = Sequence[object]

# Valid identifier pattern: alphanumeric with underscores and dots (for schema.table)
_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?$")

# Maximum identifier length (DuckDB limit)
_MAX_IDENTIFIER_LENGTH = 128


class SqlBuilderError(Exception):
    """Base exception for SQL builder errors."""


class InvalidIdentifierError(SqlBuilderError):
    """Raised when an identifier (table/column name) is invalid."""

    def __init__(self, identifier: str, reason: str) -> None:
        """Initialize the error.

        Parameters
        ----------
        identifier
            The invalid identifier.
        reason
            Why the identifier is invalid.
        """
        super().__init__(f"Invalid SQL identifier '{identifier}': {reason}")
        self.identifier = identifier
        self.reason = reason


def validate_identifier(name: str) -> str:
    """Validate and return a SQL identifier (table or column name).

    Parameters
    ----------
    name
        The identifier to validate.

    Returns
    -------
    str
        The validated identifier.

    Raises
    ------
    InvalidIdentifierError
        If the identifier is invalid.
    """
    if not name:
        raise InvalidIdentifierError(name, "identifier cannot be empty")

    if len(name) > _MAX_IDENTIFIER_LENGTH:
        raise InvalidIdentifierError(
            name, f"identifier exceeds maximum length of {_MAX_IDENTIFIER_LENGTH}"
        )

    if not _IDENTIFIER_PATTERN.match(name):
        raise InvalidIdentifierError(
            name,
            "identifier must start with letter/underscore and contain only "
            "alphanumeric characters, underscores, and at most one dot for schema prefix",
        )

    return name


def _render_sql(parts: Sequence[str]) -> str:
    """Render SQL from validated parts without inline interpolation.

    Returns
    -------
    str
        The joined SQL string with whitespace normalization.
    """
    return " ".join(part for part in parts if part)


def render_sql(parts: Sequence[str]) -> str:
    """Public helper for rendering SQL from validated parts.

    Returns
    -------
    str
        The rendered SQL string.
    """
    return _render_sql(parts)


@dataclass(frozen=True)
class SafeTable:
    """A validated table name that's safe for SQL construction.

    Use this to wrap table names before using them in queries to ensure
    they've been validated.

    Parameters
    ----------
    name
        The table name (optionally with schema prefix like 'analytics.metrics').

    Raises
    ------
    InvalidIdentifierError
        If the table name is invalid.

    Example
    -------
    >>> table = SafeTable("analytics.function_metrics")
    >>> str(table)
    'analytics.function_metrics'
    """

    name: str

    def __post_init__(self) -> None:
        """Validate the table name on construction."""
        validate_identifier(self.name)

    def __str__(self) -> str:
        """Return the table name as a string.

        Returns
        -------
        str
            The validated table name.
        """
        return self.name


@dataclass(frozen=True)
class SafeColumn:
    """A validated column name that's safe for SQL construction.

    Parameters
    ----------
    name
        The column name.

    Raises
    ------
    InvalidIdentifierError
        If the column name is invalid.
    """

    name: str

    def __post_init__(self) -> None:
        """Validate the column name on construction.

        Raises
        ------
        InvalidIdentifierError
            If the column name is invalid or contains dots.
        """
        # Column names shouldn't have dots (no schema prefix)
        if "." in self.name:
            raise InvalidIdentifierError(self.name, "column names cannot contain dots")
        validate_identifier(self.name)

    def __str__(self) -> str:
        """Return the column name as a string.

        Returns
        -------
        str
            The validated column name.
        """
        return self.name


class QueryBuilder:
    """Static methods for building safe parameterized SQL queries.

    All methods return a tuple of (query_string, parameters) where
    the query uses `?` placeholders and parameters are the values
    to bind.

    Example
    -------
    >>> query, params = QueryBuilder.count(
    ...     "analytics.metrics", where={"repo": "org/repo", "commit": "abc123"}
    ... )
    >>> # query = "SELECT COUNT(*) FROM analytics.metrics WHERE repo = ? AND commit = ?"
    >>> # params = ["org/repo", "abc123"]
    """

    @staticmethod
    def count(
        table: str | SafeTable,
        *,
        where: dict[str, object] | None = None,
    ) -> tuple[str, list[object]]:
        """Build a COUNT(*) query with optional WHERE clause.

        Parameters
        ----------
        table
            Table name (will be validated if not already SafeTable).
        where
            Optional column=value conditions for WHERE clause.

        Returns
        -------
        tuple[str, list[object]]
            Query string and parameter values.
        """
        table_name = str(table) if isinstance(table, SafeTable) else str(SafeTable(table))
        params: list[object] = []

        query_parts = ["SELECT COUNT(*) FROM", table_name]
        query = _render_sql(query_parts)

        if where:
            conditions: list[str] = []
            for col, val in where.items():
                safe_col = SafeColumn(col)
                conditions.append(f"{safe_col} = ?")
                params.append(val)
            query += f" WHERE {' AND '.join(conditions)}"

        return query, params

    @staticmethod
    def count_where_null(
        table: str | SafeTable,
        column: str | SafeColumn,
        *,
        where: dict[str, object] | None = None,
    ) -> tuple[str, list[object]]:
        """Build a COUNT query for NULL values in a specific column.

        Parameters
        ----------
        table
            Table name.
        column
            Column to check for NULL values.
        where
            Additional WHERE conditions.

        Returns
        -------
        tuple[str, list[object]]
            Query string and parameter values.
        """
        table_name = str(table) if isinstance(table, SafeTable) else str(SafeTable(table))
        col_name = str(column) if isinstance(column, SafeColumn) else str(SafeColumn(column))
        params: list[object] = []

        conditions = [f"{col_name} IS NULL"]
        if where:
            for col, val in where.items():
                safe_col = SafeColumn(col)
                conditions.append(f"{safe_col} = ?")
                params.append(val)

        where_clause = " AND ".join(conditions)
        query = _render_sql(["SELECT COUNT(*) FROM", table_name, "WHERE", where_clause])
        return query, params

    @staticmethod
    def delete(
        table: str | SafeTable,
        *,
        where: dict[str, object],
    ) -> tuple[str, list[object]]:
        """Build a DELETE query with WHERE clause.

        Parameters
        ----------
        table
            Table name.
        where
            Column=value conditions for WHERE clause (required).

        Returns
        -------
        tuple[str, list[object]]
            Query string and parameter values.
        """
        table_name = str(table) if isinstance(table, SafeTable) else str(SafeTable(table))
        params: list[object] = []

        conditions: list[str] = []
        for col, val in where.items():
            safe_col = SafeColumn(col)
            conditions.append(f"{safe_col} = ?")
            params.append(val)

        where_clause = " AND ".join(conditions)
        query = _render_sql(["DELETE FROM", table_name, "WHERE", where_clause])
        return query, params

    @staticmethod
    def delete_in(
        table: str | SafeTable,
        column: str | SafeColumn,
        values: Sequence[object],
    ) -> tuple[str, list[object]]:
        """Build a DELETE query with IN clause.

        Parameters
        ----------
        table
            Table name.
        column
            Column for IN clause.
        values
            Values to match in the IN clause.

        Returns
        -------
        tuple[str, list[object]]
            Query string and parameter values.
        """
        table_name = str(table) if isinstance(table, SafeTable) else str(SafeTable(table))
        col_name = str(column) if isinstance(column, SafeColumn) else str(SafeColumn(column))

        placeholders = ", ".join("?" * len(values))
        in_clause = f"{col_name} IN ({placeholders})"
        query = _render_sql(["DELETE FROM", table_name, "WHERE", in_clause])
        return query, list(values)

    @staticmethod
    def select_all(table: str | SafeTable) -> str:
        """Build a SELECT * query.

        Parameters
        ----------
        table
            Table name.

        Returns
        -------
        str
            Query string (no parameters needed).
        """
        table_name = str(table) if isinstance(table, SafeTable) else str(SafeTable(table))
        return _render_sql(["SELECT * FROM", table_name])

    @staticmethod
    def insert(
        table: str | SafeTable,
        columns: Sequence[str | SafeColumn],
    ) -> str:
        """Build an INSERT query with placeholders.

        Parameters
        ----------
        table
            Table name.
        columns
            Column names for the INSERT.

        Returns
        -------
        str
            Query string with ? placeholders for values.
        """
        table_name = str(table) if isinstance(table, SafeTable) else str(SafeTable(table))
        safe_cols = [str(c) if isinstance(c, SafeColumn) else str(SafeColumn(c)) for c in columns]
        cols_str = ", ".join(safe_cols)
        placeholders = ", ".join("?" * len(columns))
        values_clause = f"({cols_str}) VALUES ({placeholders})"
        return _render_sql(["INSERT INTO", table_name, values_clause])

    @staticmethod
    def delete_repo_commit(table: str | SafeTable) -> str:
        """Build a standard repo/commit scoped delete query.

        Returns
        -------
        str
            Delete statement scoped by repository and commit.
        """
        table_name = str(table) if isinstance(table, SafeTable) else str(SafeTable(table))
        where_clause = " AND ".join((f"{SafeColumn('repo')} = ?", f"{SafeColumn('commit')} = ?"))
        return _render_sql(["DELETE FROM", table_name, "WHERE", where_clause])


def build_delete_query(table: str, *, has_scope: bool = True) -> str | None:
    """Build a standard DELETE query for a table.

    Helper function for generating delete queries for tables with
    optional repo/commit scoping.

    Parameters
    ----------
    table
        Table name.
    has_scope
        Whether the table has repo and commit columns.

    Returns
    -------
    str | None
        Delete query string, or None if not applicable.
    """
    if not has_scope:
        return None

    return QueryBuilder.delete_repo_commit(SafeTable(table))


__all__ = [
    "InvalidIdentifierError",
    "QueryBuilder",
    "SafeColumn",
    "SafeTable",
    "SqlBuilderError",
    "SqlParams",
    "build_delete_query",
    "render_sql",
    "validate_identifier",
]
