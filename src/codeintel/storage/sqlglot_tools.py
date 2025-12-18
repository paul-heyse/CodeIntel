"""SQLGlot toolkit for DuckDB dialect operations.

This module centralizes SQLGlot-based utilities used across the storage layer:

- parsing and canonicalization
- scope-aware physical table reference extraction (CTE-safe)
- stable SQL fingerprinting

Keeping these primitives in one place prevents semantic drift between modules
that need to reason about compiled SQL (view dependencies, diffs, perimeters).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError
from sqlglot.optimizer.scope import traverse_scope

from codeintel.storage.constants import DUCKDB_DIALECT

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "ParseError",
    "canonical_sql_duckdb",
    "extract_table_keys_duckdb",
    "extract_table_refs",
    "fingerprint_canonical_sql",
    "fingerprint_sql_duckdb",
    "parse_one_duckdb",
]


def parse_one_duckdb(sql: str) -> exp.Expression:
    """Parse a DuckDB SQL string into a SQLGlot AST.

    Parameters
    ----------
    sql
        SQL string to parse.

    Returns
    -------
    sqlglot.expressions.Expression
        Parsed AST root.
    """
    return parse_one(sql, dialect=DUCKDB_DIALECT)


def canonical_sql_duckdb(sql: str) -> str:
    """Return a canonicalized DuckDB SQL string.

    Canonicalization is performed by parsing the SQL to an AST and re-rendering
    it using the DuckDB dialect.

    Returns
    -------
    str
        Canonicalized SQL string.
    """
    root = parse_one_duckdb(sql)
    return root.sql(dialect=DUCKDB_DIALECT)


def fingerprint_sql_duckdb(sql: str) -> str:
    """Return a stable SHA-256 fingerprint for a SQL string.

    Returns
    -------
    str
        Fingerprint of the canonicalized SQL.
    """
    canon = canonical_sql_duckdb(sql)
    return fingerprint_canonical_sql(canon)


def fingerprint_canonical_sql(canon: str) -> str:
    """Return a stable SHA-256 fingerprint for canonical SQL text.

    Parameters
    ----------
    canon
        Canonical SQL string.

    Returns
    -------
    str
        Stable fingerprint of the text.
    """
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def extract_table_refs(root: exp.Expression) -> tuple[exp.Table, ...]:
    """Extract physical table references from a parsed AST.

    Notes
    -----
    Uses scope traversal to avoid treating CTE names as physical tables.

    Returns
    -------
    tuple[sqlglot.expressions.Table, ...]
        Physical table nodes referenced by the query.
    """
    tables: list[exp.Table] = []
    for scope in traverse_scope(root):
        tables.extend(
            source for source in scope.sources.values() if isinstance(source, exp.Table)
        )
    return tuple(tables)


def extract_table_keys_duckdb(sql: str) -> frozenset[str]:
    """Extract referenced physical table keys from a DuckDB SQL string.

    Returns lowercased keys of the form ``schema.table`` when schema-qualified,
    otherwise ``table``.

    Returns
    -------
    frozenset[str]
        Referenced table keys.
    """
    root = parse_one_duckdb(sql)
    out: set[str] = set()
    for table in extract_table_refs(root):
        schema = table.db
        name = table.name
        out.add(f"{schema}.{name}".lower() if schema else name.lower())
    return frozenset(out)


def extract_table_keys_from_roots(roots: Iterable[exp.Expression]) -> frozenset[str]:
    """Extract referenced physical table keys from multiple SQLGlot roots.

    Returns
    -------
    frozenset[str]
        Referenced table keys.
    """
    out: set[str] = set()
    for root in roots:
        for table in extract_table_refs(root):
            schema = table.db
            name = table.name
            out.add(f"{schema}.{name}".lower() if schema else name.lower())
    return frozenset(out)
