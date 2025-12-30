"""DuckDB full-text search (FTS) index helpers for serving snapshots.

These helpers live in the storage layer so that DuckDB-specific details stay
isolated to ``codeintel.storage``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import duckdb

from codeintel.storage.duckdb_policy_backend import (
    duckdb_default_catalog,
    duckdb_schema_exists,
)
from codeintel.storage.gateway.extensions import require_extension
from codeintel.storage.gateway.minimal import MinimalStorageGateway
from codeintel.storage.helpers.table_key import (
    fully_qualified_table_ref,
    split_table_key,
)

if TYPE_CHECKING:
    from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

DuckDBConnection = duckdb.DuckDBPyConnection

__all__ = [
    "build_search_documents_table",
    "ensure_fts_index",
    "fts_index_available",
    "fts_schema_for_table_key",
]


def _modules_select(modules_ref: str) -> str:
    return f"""
        SELECT
            CAST(
                'module:' || COALESCE(repo, '') || ':' || COALESCE(commit, '') || ':' ||
                    COALESCE(module, '') || ':' || COALESCE(path, '')
                AS VARCHAR
            ) AS doc_id,
            CAST('module' AS VARCHAR) AS kind,
            CAST(COALESCE(module, '') AS VARCHAR) AS name,
            CAST(COALESCE(module, '') AS VARCHAR) AS module,
            CAST(path AS VARCHAR) AS rel_path,
            CAST(COALESCE(module, '') || ' ' || COALESCE(path, '') AS VARCHAR) AS text,
            CAST(NULL AS VARCHAR) AS ref_goid_h128,
            CAST(repo AS VARCHAR) AS repo,
            CAST(commit AS VARCHAR) AS commit
        FROM {modules_ref}
    """


def _docstrings_select(docstrings_ref: str) -> str:
    return f"""
        SELECT
            CAST(
                'docstring:' || repo || ':' || commit || ':' || rel_path || ':' ||
                    module || ':' || qualname || ':' || COALESCE(kind, '') || ':' ||
                    COALESCE(CAST(lineno AS VARCHAR), '')
                AS VARCHAR
            ) AS doc_id,
            CAST('docstring' AS VARCHAR) AS kind,
            CAST(COALESCE(qualname, '') AS VARCHAR) AS name,
            CAST(COALESCE(module, '') AS VARCHAR) AS module,
            CAST(rel_path AS VARCHAR) AS rel_path,
            CAST(
                COALESCE(short_desc, '') || '\n' ||
                    COALESCE(long_desc, '') || '\n' ||
                    COALESCE(raw_docstring, '')
                AS VARCHAR
            ) AS text,
            CAST(NULL AS VARCHAR) AS ref_goid_h128,
            CAST(repo AS VARCHAR) AS repo,
            CAST(commit AS VARCHAR) AS commit
        FROM {docstrings_ref}
    """


def _function_metrics_select(function_metrics_ref: str) -> str:
    return f"""
        SELECT
            CAST(
                'function:' || COALESCE(CAST(function_goid_h128 AS VARCHAR), '') || ':' ||
                    COALESCE(repo, '') || ':' || COALESCE(commit, '')
                AS VARCHAR
            ) AS doc_id,
            CAST('function' AS VARCHAR) AS kind,
            CAST(COALESCE(qualname, '') AS VARCHAR) AS name,
            CAST(NULL AS VARCHAR) AS module,
            CAST(rel_path AS VARCHAR) AS rel_path,
            CAST(
                COALESCE(qualname, '') || ' ' || COALESCE(urn, '') || ' ' ||
                    COALESCE(rel_path, '')
                AS VARCHAR
            ) AS text,
            CAST(function_goid_h128 AS VARCHAR) AS ref_goid_h128,
            CAST(repo AS VARCHAR) AS repo,
            CAST(commit AS VARCHAR) AS commit
        FROM {function_metrics_ref}
    """


def _scip_symbols_select(scip_symbols_ref: str) -> str:
    return f"""
        SELECT
            CAST(
                'symbol:' || repo || ':' || commit || ':' || rel_path || ':' || symbol
                AS VARCHAR
            ) AS doc_id,
            CAST('symbol' AS VARCHAR) AS kind,
            CAST(symbol AS VARCHAR) AS name,
            CAST(NULL AS VARCHAR) AS module,
            CAST(rel_path AS VARCHAR) AS rel_path,
            CAST(COALESCE(documentation, '') AS VARCHAR) AS text,
            CAST(NULL AS VARCHAR) AS ref_goid_h128,
            CAST(repo AS VARCHAR) AS repo,
            CAST(commit AS VARCHAR) AS commit
        FROM {scip_symbols_ref}
    """


def _empty_search_documents_select() -> str:
    return """
        SELECT
            CAST(NULL AS VARCHAR) AS doc_id,
            CAST(NULL AS VARCHAR) AS kind,
            CAST(NULL AS VARCHAR) AS name,
            CAST(NULL AS VARCHAR) AS module,
            CAST(NULL AS VARCHAR) AS rel_path,
            CAST(NULL AS VARCHAR) AS text,
            CAST(NULL AS VARCHAR) AS ref_goid_h128,
            CAST(NULL AS VARCHAR) AS repo,
            CAST(NULL AS VARCHAR) AS commit
    """


@dataclass(frozen=True, slots=True)
class _SearchDocumentsRefs:
    search_documents_ref: str
    modules_ref: str
    docstrings_ref: str
    function_metrics_ref: str
    scip_symbols_ref: str


def _create_search_documents_table(
    *,
    backend: DuckDBPolicyBackend,
    refs: _SearchDocumentsRefs,
) -> None:
    backend.execute_sql(f"DROP TABLE IF EXISTS {refs.search_documents_ref}")
    seed_select = None
    if backend.table_exists(schema="core", table="modules"):
        seed_select = _modules_select(refs.modules_ref)
    elif backend.table_exists(schema="core", table="docstrings"):
        seed_select = _docstrings_select(refs.docstrings_ref)
    elif backend.table_exists(schema="analytics", table="function_metrics"):
        seed_select = _function_metrics_select(refs.function_metrics_ref)
    elif backend.table_exists(schema="core", table="scip_symbols"):
        seed_select = _scip_symbols_select(refs.scip_symbols_ref)
    else:
        seed_select = _empty_search_documents_select()
    backend.execute_sql(
        f"""
        CREATE TABLE {refs.search_documents_ref} AS
        SELECT * FROM (
            {seed_select}
        )
        WHERE 1 = 0
        """
    )


def build_search_documents_table(con: DuckDBConnection) -> None:
    """Create and populate ``docs.search_documents`` in the current database."""
    catalog = duckdb_default_catalog(con)
    search_documents_ref = fully_qualified_table_ref(
        "docs.search_documents",
        catalog=catalog,
    )
    modules_ref = fully_qualified_table_ref("core.modules", catalog=catalog)
    docstrings_ref = fully_qualified_table_ref("core.docstrings", catalog=catalog)
    function_metrics_ref = fully_qualified_table_ref(
        "analytics.function_metrics",
        catalog=catalog,
    )
    scip_symbols_ref = fully_qualified_table_ref("core.scip_symbols", catalog=catalog)

    backend = MinimalStorageGateway(con).policy
    backend.create_schema_if_not_exists("docs")
    refs = _SearchDocumentsRefs(
        search_documents_ref=search_documents_ref,
        modules_ref=modules_ref,
        docstrings_ref=docstrings_ref,
        function_metrics_ref=function_metrics_ref,
        scip_symbols_ref=scip_symbols_ref,
    )
    _create_search_documents_table(backend=backend, refs=refs)

    if backend.table_exists(schema="core", table="modules"):
        backend.execute_sql(
            f"INSERT INTO {search_documents_ref} {_modules_select(modules_ref)}",
        )

    if backend.table_exists(schema="core", table="docstrings"):
        backend.execute_sql(
            f"INSERT INTO {search_documents_ref} {_docstrings_select(docstrings_ref)}",
        )

    if backend.table_exists(schema="analytics", table="function_metrics"):
        backend.execute_sql(
            f"INSERT INTO {search_documents_ref} {_function_metrics_select(function_metrics_ref)}",
        )

    if backend.table_exists(schema="core", table="scip_symbols"):
        backend.execute_sql(
            f"INSERT INTO {search_documents_ref} {_scip_symbols_select(scip_symbols_ref)}",
        )


def fts_schema_for_table_key(table_key: str) -> str:
    """Return the DuckDB schema name that holds the FTS index for ``table_key``.

    Parameters
    ----------
    table_key
        Schema-qualified table key (e.g., ``"docs.search_documents"``).

    Returns
    -------
    str
        Schema name that DuckDB will use for the FTS index objects.
    """
    schema, name = split_table_key(table_key)
    return f"fts_{schema}_{name}"


def fts_index_available(con: DuckDBConnection, *, table_key: str = "docs.search_documents") -> bool:
    """Return True when a DuckDB FTS index exists for ``table_key``.

    Parameters
    ----------
    con
        Active DuckDB connection.
    table_key
        Schema-qualified table key to check.

    Returns
    -------
    bool
        True when the FTS schema exists.
    """
    return duckdb_schema_exists(con, schema=fts_schema_for_table_key(table_key))


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
    schema, name = split_table_key(table_key)
    backend = MinimalStorageGateway(con).policy
    if not backend.table_exists(schema=schema, table=name):
        msg = f"Search documents table not found: {table_key}"
        raise ValueError(msg)

    fts_schema = fts_schema_for_table_key(table_key)
    if duckdb_schema_exists(con, schema=fts_schema):
        return fts_schema

    require_extension(con, "fts", allow_install=True)
    create_sql = f"""
    PRAGMA create_fts_index(
        '{table_key}',
        'doc_id',
        'text',
        'name',
        'module'
    )
    """
    backend.execute_sql(create_sql)
    return fts_schema
