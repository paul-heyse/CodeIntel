"""Core SQL query building primitives with no external dependencies.

This module provides type-safe SQL construction utilities that ensure safety by:

1. Table names are validated against known patterns via SafeTable
2. Column names are validated identifiers via SafeColumn
3. All user values are parameterized (never interpolated)

This module has no dependencies on config.datasets or other schema-aware
modules, making it safe to import from anywhere without circular dependency
concerns.

.. deprecated::
    The following classes are deprecated and will be removed in a future version:

    - ``SafeTable``, ``SafeColumn``: Use ``DuckDBPolicyBackend`` methods instead
    - ``QueryBuilder``: Use ``DuckDBPolicyBackend.bulk_insert()``,
      ``DuckDBPolicyBackend.delete_for_snapshot()``, etc.
    - ``build_insert_sql``, ``build_delete_query``: Use policy backend methods

    The following utilities remain supported:
    - ``quote_identifier``, ``quote_table_key``: Used by macros and policy backend
    - ``safe_macro_call``, ``macro_select_sql``: Used for macro invocations
    - ``InvalidIdentifierError``, ``SqlBuilderError``: Exception types
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Collection, Sequence
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

    .. deprecated::
        Use ``DuckDBPolicyBackend`` methods instead of building SQL strings
        with SafeTable. The policy backend provides centralized, type-safe
        SQL generation via SQLGlot.

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
    """

    name: str

    def __post_init__(self) -> None:
        """Validate the table name on construction."""
        warnings.warn(
            "SafeTable is deprecated. Use DuckDBPolicyBackend methods instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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

    .. deprecated::
        Use ``DuckDBPolicyBackend`` methods instead of building SQL strings
        with SafeColumn. The policy backend provides centralized, type-safe
        SQL generation via SQLGlot.

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
        warnings.warn(
            "SafeColumn is deprecated. Use DuckDBPolicyBackend methods instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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


def _warn_query_builder_deprecated() -> None:
    """Emit deprecation warning for QueryBuilder usage."""
    warnings.warn(
        "QueryBuilder is deprecated. Use DuckDBPolicyBackend methods instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class QueryBuilder:
    """Static methods for building safe parameterized SQL queries.

    .. deprecated::
        Use ``DuckDBPolicyBackend`` from ``codeintel.storage.duckdb_policy_backend``
        instead. The policy backend provides centralized, type-safe SQL generation
        via SQLGlot.

        Migration:
        - ``QueryBuilder.count()`` -> Use Ibis expressions via ``gateway.ibis``
        - ``QueryBuilder.delete()`` -> ``DuckDBPolicyBackend.delete_for_snapshot()``
        - ``QueryBuilder.insert()`` -> ``DuckDBPolicyBackend.bulk_insert()``

    All methods return a tuple of (query_string, parameters) where
    the query uses `?` placeholders and parameters are the values
    to bind.
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
        _warn_query_builder_deprecated()
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


@dataclass(frozen=True)
class PreparedStatements:
    """Prepared insert/delete/select SQL for a table (registry-driven)."""

    insert_sql: str
    delete_sql: str | None = None
    select_sql: str | None = None
    select_params: list[object] | None = None


# --------------------------------------------------------------------------
# Quoting helpers
# --------------------------------------------------------------------------

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
TABLE_KEY_PARTS = 2


def quote_identifier(identifier: str) -> str:
    """Validate and quote a SQL identifier.

    Parameters
    ----------
    identifier
        Column or table component name.

    Returns
    -------
    str
        Quoted identifier.

    Raises
    ------
    ValueError
        When the identifier is unsafe.
    """
    if not _IDENTIFIER_RE.fullmatch(identifier):
        message = f"Unsafe identifier: {identifier}"
        raise ValueError(message)
    return f'"{identifier}"'


def quote_table_key(table_key: str) -> str:
    """Validate and quote a fully qualified table key (schema.table).

    Parameters
    ----------
    table_key
        Fully qualified table identifier (e.g., "core.modules").

    Returns
    -------
    str
        Quoted table identifier.

    Raises
    ------
    ValueError
        When the key is invalid.
    """
    parts = table_key.split(".")
    if len(parts) != TABLE_KEY_PARTS or not parts[0] or not parts[1]:
        message = f"Table key must include schema: {table_key}"
        raise ValueError(message)
    schema_name, table_name = parts
    return f"{quote_identifier(schema_name)}.{quote_identifier(table_name)}"


def quote_macro_name(macro_name: str) -> str:
    """Validate a fully qualified macro name without quoting.

    Parameters
    ----------
    macro_name
        Macro identifier, optionally schema-qualified (e.g., "analytics.ingest_modules").

    Returns
    -------
    str
        Validated macro name (unchanged).

    Raises
    ------
    ValueError
        If any macro component is unsafe or empty.
    """
    parts = macro_name.split(".")
    if not parts or any(not part or not _IDENTIFIER_RE.fullmatch(part) for part in parts):
        message = f"Unsafe macro name: {macro_name}"
        raise ValueError(message)
    return ".".join(parts)


# --------------------------------------------------------------------------
# Macro helpers
# --------------------------------------------------------------------------


def macro_select_sql(macro_name: str, placeholders: str) -> str:
    """Build a validated SELECT statement invoking a macro.

    Parameters
    ----------
    macro_name
        Fully qualified macro name (schema.macro).
    placeholders
        Placeholder string (e.g., "?, ?, ?").

    Returns
    -------
    str
        Safe SELECT statement invoking the macro.

    Raises
    ------
    ValueError
        If the macro name is unsafe.
    """
    if "." not in macro_name:
        message = f"Macro name must include schema: {macro_name}"
        raise ValueError(message)
    schema_name, macro = macro_name.split(".", maxsplit=1)
    macro_sql = ".".join((quote_identifier(schema_name), quote_identifier(macro)))
    return "".join(
        (
            "SELECT * FROM /*",
            macro_name,
            "*/ ",
            macro_sql,
            "(",
            placeholders,
            ")",
        )
    )


def safe_macro_call(
    macro_name: str,
    args: Sequence[object],
    *,
    allowed: Collection[str] | None = None,
) -> tuple[str, Sequence[object]]:
    """Return a safe SELECT statement and args for a macro invocation.

    Parameters
    ----------
    macro_name
        Fully qualified macro name (schema.macro).
    args
        Parameters to pass to the macro.
    allowed
        Optional allowlist of macro names; when provided, macro_name must be present.

    Returns
    -------
    tuple[str, Sequence[object]]
        Parameterized SQL and the original args.

    Raises
    ------
    ValueError
        If the macro name is unsafe or not allowlisted.
    """
    if allowed is not None and macro_name not in allowed:
        message = f"Macro {macro_name} is not allowlisted"
        raise ValueError(message)
    placeholders = ", ".join("?" for _ in args)
    sql = macro_select_sql(macro_name, placeholders)
    return sql, args


# --------------------------------------------------------------------------
# Insert SQL builder
# --------------------------------------------------------------------------


def build_insert_sql(
    table_identifier: str,
    columns: Sequence[str],
    *,
    identifier_is_quoted: bool = False,
) -> str:
    """Build a parameterized INSERT statement with validated identifiers.

    Parameters
    ----------
    table_identifier
        Fully qualified table name (schema.table) or a pre-quoted identifier
        when ``identifier_is_quoted`` is True.
    columns
        Ordered list of column names to insert into.
    identifier_is_quoted
        When True, ``table_identifier`` is treated as already quoted (e.g., a
        temporary view name). When False, it is validated and quoted.

    Returns
    -------
    str
        INSERT statement with placeholders for values.
    """
    table_sql = table_identifier if identifier_is_quoted else quote_table_key(table_identifier)
    cols_sql = ", ".join(quote_identifier(col) for col in columns)
    placeholders = ", ".join("?" for _ in columns)
    return "".join(
        (
            "INSERT INTO ",
            table_sql,
            " (",
            cols_sql,
            ") VALUES (",
            placeholders,
            ")",
        )
    )


__all__ = [
    "TABLE_KEY_PARTS",
    "InvalidIdentifierError",
    "PreparedStatements",
    "QueryBuilder",
    "SafeColumn",
    "SafeTable",
    "SqlBuilderError",
    "SqlParams",
    "build_delete_query",
    "build_insert_sql",
    "macro_select_sql",
    "quote_identifier",
    "quote_macro_name",
    "quote_table_key",
    "render_sql",
    "safe_macro_call",
    "validate_identifier",
]
