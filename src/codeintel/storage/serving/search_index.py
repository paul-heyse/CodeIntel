"""DuckDB full-text search (FTS) index helpers for serving snapshots.

These helpers live in the storage layer so that DuckDB-specific details stay
isolated to ``codeintel.storage``.
"""

from __future__ import annotations

import duckdb

from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.storage.gateway.minimal import MinimalStorageGateway
from codeintel.storage.queries.execution import (
    duckdb_schema_exists,
    duckdb_table_exists,
    execute_sql,
)

DuckDBConnection = duckdb.DuckDBPyConnection

_SEARCH_DOCUMENTS_SCHEMA = TableSchema(
    schema="docs",
    name="search_documents",
    columns=[
        Column("doc_id", "VARCHAR", nullable=False),
        Column("kind", "VARCHAR", nullable=False),
        Column("name", "VARCHAR", nullable=False),
        Column("module", "VARCHAR"),
        Column("rel_path", "VARCHAR"),
        Column("text", "VARCHAR"),
        Column("ref_goid_h128", "VARCHAR"),
        Column("repo", "VARCHAR"),
        Column("commit", "VARCHAR"),
    ],
)

__all__ = [
    "build_search_documents_table",
    "ensure_fts_index",
]


def build_search_documents_table(con: DuckDBConnection) -> None:
    """Create and populate ``docs.search_documents`` in the current database."""
    backend = MinimalStorageGateway(con).policy
    backend.create_schema_if_not_exists("docs")
    backend.create_table_from_schema(_SEARCH_DOCUMENTS_SCHEMA, drop_existing=True)

    if duckdb_table_exists(con, schema="core", table="modules"):
        execute_sql(
            con,
            """
            INSERT INTO docs.search_documents
            SELECT
                'module:' || COALESCE(repo, '') || ':' || COALESCE(commit, '') || ':' ||
                    COALESCE(module, '') || ':' || COALESCE(path, '') AS doc_id,
                'module' AS kind,
                COALESCE(module, '') AS name,
                COALESCE(module, '') AS module,
                path AS rel_path,
                COALESCE(module, '') || ' ' || COALESCE(path, '') AS text,
                NULL AS ref_goid_h128,
                repo,
                commit
            FROM core.modules
            """,
        )

    if duckdb_table_exists(con, schema="core", table="docstrings"):
        execute_sql(
            con,
            """
            INSERT INTO docs.search_documents
            SELECT
                'docstring:' || repo || ':' || commit || ':' || rel_path || ':' ||
                    module || ':' || qualname || ':' || COALESCE(kind, '') || ':' ||
                    COALESCE(CAST(lineno AS VARCHAR), '') AS doc_id,
                'docstring' AS kind,
                COALESCE(qualname, '') AS name,
                COALESCE(module, '') AS module,
                rel_path,
                COALESCE(short_desc, '') || '\n' ||
                    COALESCE(long_desc, '') || '\n' ||
                    COALESCE(raw_docstring, '') AS text,
                NULL AS ref_goid_h128,
                repo,
                commit
            FROM core.docstrings
            """,
        )

    if duckdb_table_exists(con, schema="analytics", table="function_metrics"):
        execute_sql(
            con,
            """
            INSERT INTO docs.search_documents
            SELECT
                'function:' || COALESCE(CAST(function_goid_h128 AS VARCHAR), '') || ':' ||
                    COALESCE(repo, '') || ':' || COALESCE(commit, '') AS doc_id,
                'function' AS kind,
                COALESCE(qualname, '') AS name,
                NULL AS module,
                rel_path,
                COALESCE(qualname, '') || ' ' || COALESCE(urn, '') || ' ' || COALESCE(rel_path, '')
                    AS text,
                CAST(function_goid_h128 AS VARCHAR) AS ref_goid_h128,
                repo,
                commit
            FROM analytics.function_metrics
            """,
        )

    if duckdb_table_exists(con, schema="core", table="scip_symbols"):
        execute_sql(
            con,
            """
            INSERT INTO docs.search_documents
            SELECT
                'symbol:' || repo || ':' || commit || ':' || rel_path || ':' || symbol AS doc_id,
                'symbol' AS kind,
                symbol AS name,
                NULL AS module,
                rel_path,
                COALESCE(documentation, '') AS text,
                NULL AS ref_goid_h128,
                repo,
                commit
            FROM core.scip_symbols
            """,
        )


def _fts_schema_for_table_key(table_key: str) -> str:
    schema, name = table_key.split(".", 1)
    return f"fts_{schema}_{name}"


def ensure_fts_index(con: DuckDBConnection, *, table_key: str = "docs.search_documents") -> str:
    """Ensure a DuckDB FTS index exists for ``table_key``.

    Returns
    -------
    str
        The DuckDB schema name that holds the FTS index tables/macros.

    Raises
    ------
    ValueError
        If ``table_key`` is not schema-qualified or the target table does not exist.
    """
    if "." not in table_key:
        msg = f"Expected schema-qualified table_key, got: {table_key}"
        raise ValueError(msg)

    schema, name = table_key.split(".", 1)
    if not duckdb_table_exists(con, schema=schema, table=name):
        msg = f"Search documents table not found: {table_key}"
        raise ValueError(msg)

    fts_schema = _fts_schema_for_table_key(table_key)
    if duckdb_schema_exists(con, schema=fts_schema):
        return fts_schema

    try:
        execute_sql(con, "LOAD fts")
    except duckdb.Error:
        execute_sql(con, "INSTALL fts")
        execute_sql(con, "LOAD fts")

    execute_sql(
        con,
        f"""
        PRAGMA create_fts_index(
            '{table_key}',
            'doc_id',
            'text',
            'name',
            'module'
        )
        """,
    )
    return fts_schema
