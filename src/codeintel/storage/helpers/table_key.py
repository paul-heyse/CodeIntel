"""Table key utilities for schema-qualified name handling.

This module provides shared utilities for parsing and manipulating
schema-qualified table keys in the format "schema.table".
"""

from __future__ import annotations

__all__ = ["split_table_key"]


def split_table_key(table_key: str) -> tuple[str, str]:
    """Split a schema-qualified table key into (schema, table).

    Parameters
    ----------
    table_key
        Fully qualified table name (schema.table).

    Returns
    -------
    tuple[str, str]
        Schema and table name.

    Raises
    ------
    ValueError
        If table_key is not schema-qualified.

    Examples
    --------
    >>> split_table_key("analytics.function_metrics")
    ('analytics', 'function_metrics')
    >>> split_table_key("core.goids")
    ('core', 'goids')
    """
    if "." not in table_key:
        message = f"Table key must be schema-qualified: {table_key}"
        raise ValueError(message)
    schema_name, table_name = table_key.split(".", maxsplit=1)
    return schema_name, table_name
