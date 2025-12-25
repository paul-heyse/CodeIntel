"""Table key utilities for schema-qualified name handling.

This module provides shared utilities for parsing and manipulating
schema-qualified table keys in the format "schema.table".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "ParsedTableKey",
    "TableKey",
    "TableKeyValidationError",
    "fully_qualified_table_ref",
    "is_valid_table_key",
    "parse_table_key",
    "split_table_key",
    "split_table_key_or_default",
    "table_name_from_key",
    "try_parse_table_key",
    "validate_table_key",
]


type TableKey = str


@dataclass(frozen=True, slots=True)
class ParsedTableKey:
    """Parsed schema-qualified table key."""

    schema: str
    name: str


_TABLE_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")


class TableKeyValidationError(ValueError):
    """Raised when a table key is invalid or not schema-qualified."""


def validate_table_key(table_key: TableKey) -> None:
    """Validate that a table key is schema-qualified and well-formed.

    Parameters
    ----------
    table_key
        Fully qualified table name (schema.table).

    Raises
    ------
    TableKeyValidationError
        If the table key is not schema-qualified or contains invalid characters.
    """
    if "." not in table_key:
        message = f"Table key must be schema-qualified: {table_key}"
        raise TableKeyValidationError(message)
    if not _TABLE_KEY_PATTERN.match(table_key):
        message = f"Invalid table key format: {table_key}"
        raise TableKeyValidationError(message)


def fully_qualified_table_ref(table_key: TableKey, *, catalog: str | None = None) -> str:
    """Return a fully qualified SQL table reference.

    Parameters
    ----------
    table_key
        Fully qualified table name (schema.table).
    catalog
        Optional catalog name. When None, omit the catalog prefix.

    Returns
    -------
    str
        SQL table reference in ``catalog.schema.table`` or ``schema.table`` form.

    Raises
    ------
    ValueError
        If the catalog name is invalid.
    """
    if catalog is not None and (not isinstance(catalog, str) or not catalog.strip()):
        msg = f"Invalid catalog name: {catalog!r}"
        raise ValueError(msg)
    parsed = parse_table_key(table_key)
    if catalog is None:
        return f"{parsed.schema}.{parsed.name}"
    return f"{catalog}.{parsed.schema}.{parsed.name}"


def is_valid_table_key(table_key: TableKey) -> bool:
    """Return True when a table key is schema-qualified and valid.

    Returns
    -------
    bool
        True when the table key is valid.
    """
    try:
        validate_table_key(table_key)
    except TableKeyValidationError:
        return False
    return True


def parse_table_key(table_key: TableKey) -> ParsedTableKey:
    """Parse a schema-qualified table key into structured components.

    Parameters
    ----------
    table_key
        Fully qualified table name (schema.table).

    Returns
    -------
    ParsedTableKey
        Parsed schema and table name.
    """
    validate_table_key(table_key)
    schema_name, table_name = table_key.split(".", maxsplit=1)
    return ParsedTableKey(schema=schema_name, name=table_name)


def try_parse_table_key(table_key: TableKey) -> ParsedTableKey | None:
    """Parse a table key, returning None when invalid.

    Returns
    -------
    ParsedTableKey | None
        Parsed table key, or None when invalid.
    """
    try:
        return parse_table_key(table_key)
    except TableKeyValidationError:
        return None


def split_table_key(table_key: TableKey) -> tuple[str, str]:
    """Split a schema-qualified table key into (schema, table).

    Parameters
    ----------
    table_key
        Fully qualified table name (schema.table).

    Returns
    -------
    tuple[str, str]
        Schema and table name.

    Examples
    --------
    >>> split_table_key("analytics.function_metrics")
    ('analytics', 'function_metrics')
    >>> split_table_key("core.goids")
    ('core', 'goids')
    """
    parsed = parse_table_key(table_key)
    return parsed.schema, parsed.name


def split_table_key_or_default(table_key: TableKey, *, default_schema: str) -> tuple[str, str]:
    """Split a table key, falling back to a default schema when unqualified.

    Parameters
    ----------
    table_key
        Table key, optionally schema-qualified.
    default_schema
        Schema to use when table_key is unqualified.

    Returns
    -------
    tuple[str, str]
        Schema and table name.
    """
    if "." not in table_key:
        return default_schema, table_key
    return split_table_key(table_key)


def table_name_from_key(table_key: TableKey) -> str:
    """Return the table name component from a table key.

    Parameters
    ----------
    table_key
        Table key, optionally schema-qualified.

    Returns
    -------
    str
        Table name component.
    """
    if "." not in table_key:
        return table_key
    _, name = split_table_key(table_key)
    return name
