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
    "parse_table_key",
    "split_table_key",
    "validate_table_key",
]


type TableKey = str


@dataclass(frozen=True, slots=True)
class ParsedTableKey:
    """Parsed schema-qualified table key."""

    schema: str
    name: str


_TABLE_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")


def validate_table_key(table_key: TableKey) -> None:
    """Validate that a table key is schema-qualified and well-formed.

    Parameters
    ----------
    table_key
        Fully qualified table name (schema.table).

    Raises
    ------
    ValueError
        If the table key is not schema-qualified or contains invalid characters.
    """
    if "." not in table_key:
        message = f"Table key must be schema-qualified: {table_key}"
        raise ValueError(message)
    if not _TABLE_KEY_PATTERN.match(table_key):
        message = f"Invalid table key format: {table_key}"
        raise ValueError(message)


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
