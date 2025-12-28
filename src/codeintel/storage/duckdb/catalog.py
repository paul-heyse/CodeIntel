"""DuckDB catalog helpers shared across storage modules."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from sqlglot import tokens

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

__all__ = [
    "duckdb_default_catalog",
    "duckdb_schema_exists",
    "is_valid_catalog_identifier",
    "normalize_catalog_identifier",
]

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_KEYWORDS = frozenset(tokens.Tokenizer.KEYWORDS)


def _is_valid_identifier(value: str) -> bool:
    return bool(_IDENTIFIER_RE.fullmatch(value))


def _is_reserved_keyword(value: str) -> bool:
    return value.upper() in _RESERVED_KEYWORDS


def is_valid_catalog_identifier(value: str) -> bool:
    """Return True when a catalog identifier is safe for unquoted SQL usage.

    Returns
    -------
    bool
        True when the identifier is valid and not a reserved keyword.
    """
    return _is_valid_identifier(value) and not _is_reserved_keyword(value)


def _hash_identifier(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"codeintel_{digest}"


def normalize_catalog_identifier(value: str | None) -> str | None:
    """Return a safe catalog identifier for the supplied value.

    Parameters
    ----------
    value
        Raw catalog identifier to normalize.

    Returns
    -------
    str | None
        Normalized identifier or None when the input is empty.
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if is_valid_catalog_identifier(stripped):
        return stripped

    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", stripped)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        return _hash_identifier(stripped)
    if sanitized[0].isdigit():
        sanitized = f"catalog_{sanitized}"
    if _is_reserved_keyword(sanitized):
        sanitized = f"{sanitized}_catalog"
    if _is_valid_identifier(sanitized) and not _is_reserved_keyword(sanitized):
        return sanitized
    return _hash_identifier(stripped)


def duckdb_default_catalog(con: DuckDBPyConnection) -> str | None:
    """Return the primary catalog name for a DuckDB connection.

    Parameters
    ----------
    con
        DuckDB connection to query.

    Returns
    -------
    str | None
        Primary catalog name, or None when unavailable.
    """
    row = con.execute("PRAGMA database_list").fetchone()
    if row is None:
        return None
    catalog = row[1]
    if (
        isinstance(catalog, str)
        and catalog.strip()
        and is_valid_catalog_identifier(catalog)
    ):
        return catalog
    return None


def duckdb_schema_exists(con: DuckDBPyConnection, *, schema: str) -> bool:
    """Return True when a DuckDB schema exists.

    Parameters
    ----------
    con
        DuckDB connection to query.
    schema
        Schema name to check.

    Returns
    -------
    bool
        True when the schema exists.
    """
    row = con.execute(
        "SELECT 1 FROM information_schema.schemata WHERE schema_name = ? LIMIT 1",
        [schema],
    ).fetchone()
    return row is not None
