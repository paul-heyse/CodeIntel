"""Canonical table key parsing and validation helpers."""

from __future__ import annotations

from codeintel.storage.helpers.table_key import (
    ParsedTableKey,
)
from codeintel.storage.helpers.table_key import (
    parse_table_key as _parse_table_key,
)
from codeintel.storage.helpers.table_key import (
    split_table_key as _split_table_key,
)
from codeintel.storage.helpers.table_key import (
    validate_table_key as _validate_table_key,
)


def validate_table_key(table_key: str) -> None:
    """Validate that a table key is non-empty and well-formed.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Raises
    ------
    ValueError
        If the table key is empty or invalid.
    """
    if not table_key:
        msg = "table_key must be non-empty"
        raise ValueError(msg)
    _validate_table_key(table_key)


def split_table_key(table_key: str) -> tuple[str, str]:
    """Split a table key into schema and table name.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    tuple[str, str]
        Tuple of (schema_name, table_name).
    """
    return _split_table_key(table_key)


def parse_table_key(table_key: str) -> ParsedTableKey:
    """Parse a table key into its structured components.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    ParsedTableKey
        Parsed table key components.
    """
    return _parse_table_key(table_key)


__all__ = [
    "ParsedTableKey",
    "parse_table_key",
    "split_table_key",
    "validate_table_key",
]
