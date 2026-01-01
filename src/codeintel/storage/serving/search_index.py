"""DuckDB full-text search (FTS) index helpers for serving snapshots.

These helpers live in the storage layer so that DuckDB-specific details stay
isolated to ``codeintel.storage``.
"""

from __future__ import annotations

import duckdb
from sqlglot import exp

from codeintel.core.schemas.primitives import Column, TableSchema
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
from codeintel.storage.sqlglot_tools import table_expr_from_ref

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
    "fts_index_available",
    "fts_schema_for_table_key",
]


def build_search_documents_table(con: DuckDBConnection) -> None:
    """Create and populate ``docs.search_documents`` in the current database."""
    catalog = duckdb_default_catalog(con)
    modules_ref = fully_qualified_table_ref("core.modules", catalog=catalog)
    docstrings_ref = fully_qualified_table_ref("core.docstrings", catalog=catalog)
    function_metrics_ref = fully_qualified_table_ref(
        "analytics.function_metrics",
        catalog=catalog,
    )
    scip_symbols_ref = fully_qualified_table_ref("core.scip_symbols", catalog=catalog)

    backend = MinimalStorageGateway(con).policy
    backend.create_schema_if_not_exists("docs")
    backend.create_table_from_schema(_SEARCH_DOCUMENTS_SCHEMA, drop_existing=True)

    if backend.table_exists(schema="core", table="modules"):
        backend.insert_select(
            "docs.search_documents",
            columns=_search_document_columns(),
            select_sql=_modules_select(modules_ref),
        )

    if backend.table_exists(schema="core", table="docstrings"):
        backend.insert_select(
            "docs.search_documents",
            columns=_search_document_columns(),
            select_sql=_docstrings_select(docstrings_ref),
        )

    if backend.table_exists(schema="analytics", table="function_metrics"):
        backend.insert_select(
            "docs.search_documents",
            columns=_search_document_columns(),
            select_sql=_function_metrics_select(function_metrics_ref),
        )

    if backend.table_exists(schema="core", table="scip_symbols"):
        backend.insert_select(
            "docs.search_documents",
            columns=_search_document_columns(),
            select_sql=_scip_symbols_select(scip_symbols_ref),
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


def _search_document_columns() -> list[str]:
    return [
        "doc_id",
        "kind",
        "name",
        "module",
        "rel_path",
        "text",
        "ref_goid_h128",
        "repo",
        "commit",
    ]


def _coalesce_text(expr: exp.Expression) -> exp.Expression:
    return exp.Coalesce(expressions=[expr, exp.Literal.string("")])


def _concat(expressions: list[exp.Expression]) -> exp.Expression:
    return exp.Concat(expressions=expressions)


def _varchar_cast(expr: exp.Expression) -> exp.Expression:
    return exp.Cast(this=expr, to=exp.DataType.build("VARCHAR", dialect="duckdb"))


def _modules_select(modules_ref: str) -> exp.Select:
    table_expr = table_expr_from_ref(modules_ref)
    doc_id = _concat(
        [
            exp.Literal.string("module:"),
            _coalesce_text(exp.column("repo")),
            exp.Literal.string(":"),
            _coalesce_text(exp.column("commit")),
            exp.Literal.string(":"),
            _coalesce_text(exp.column("module")),
            exp.Literal.string(":"),
            _coalesce_text(exp.column("path")),
        ]
    )
    text = _concat(
        [
            _coalesce_text(exp.column("module")),
            exp.Literal.string(" "),
            _coalesce_text(exp.column("path")),
        ]
    )
    return exp.select(
        exp.alias_(doc_id, "doc_id"),
        exp.alias_(exp.Literal.string("module"), "kind"),
        exp.alias_(_coalesce_text(exp.column("module")), "name"),
        exp.alias_(_coalesce_text(exp.column("module")), "module"),
        exp.alias_(exp.column("path"), "rel_path"),
        exp.alias_(text, "text"),
        exp.alias_(exp.null(), "ref_goid_h128"),
        exp.column("repo"),
        exp.column("commit"),
    ).from_(table_expr)


def _docstrings_select(docstrings_ref: str) -> exp.Select:
    table_expr = table_expr_from_ref(docstrings_ref)
    lineno = _coalesce_text(_varchar_cast(exp.column("lineno")))
    doc_id = _concat(
        [
            exp.Literal.string("docstring:"),
            exp.column("repo"),
            exp.Literal.string(":"),
            exp.column("commit"),
            exp.Literal.string(":"),
            exp.column("rel_path"),
            exp.Literal.string(":"),
            exp.column("module"),
            exp.Literal.string(":"),
            exp.column("qualname"),
            exp.Literal.string(":"),
            _coalesce_text(exp.column("kind")),
            exp.Literal.string(":"),
            lineno,
        ]
    )
    text = _concat(
        [
            _coalesce_text(exp.column("short_desc")),
            exp.Literal.string("\n"),
            _coalesce_text(exp.column("long_desc")),
            exp.Literal.string("\n"),
            _coalesce_text(exp.column("raw_docstring")),
        ]
    )
    return exp.select(
        exp.alias_(doc_id, "doc_id"),
        exp.alias_(exp.Literal.string("docstring"), "kind"),
        exp.alias_(_coalesce_text(exp.column("qualname")), "name"),
        exp.alias_(_coalesce_text(exp.column("module")), "module"),
        exp.column("rel_path"),
        exp.alias_(text, "text"),
        exp.alias_(exp.null(), "ref_goid_h128"),
        exp.column("repo"),
        exp.column("commit"),
    ).from_(table_expr)


def _function_metrics_select(function_metrics_ref: str) -> exp.Select:
    table_expr = table_expr_from_ref(function_metrics_ref)
    goid = _coalesce_text(_varchar_cast(exp.column("function_goid_h128")))
    doc_id = _concat(
        [
            exp.Literal.string("function:"),
            goid,
            exp.Literal.string(":"),
            _coalesce_text(exp.column("repo")),
            exp.Literal.string(":"),
            _coalesce_text(exp.column("commit")),
        ]
    )
    text = _concat(
        [
            _coalesce_text(exp.column("qualname")),
            exp.Literal.string(" "),
            _coalesce_text(exp.column("urn")),
            exp.Literal.string(" "),
            _coalesce_text(exp.column("rel_path")),
        ]
    )
    return exp.select(
        exp.alias_(doc_id, "doc_id"),
        exp.alias_(exp.Literal.string("function"), "kind"),
        exp.alias_(_coalesce_text(exp.column("qualname")), "name"),
        exp.alias_(exp.null(), "module"),
        exp.column("rel_path"),
        exp.alias_(text, "text"),
        exp.alias_(_varchar_cast(exp.column("function_goid_h128")), "ref_goid_h128"),
        exp.column("repo"),
        exp.column("commit"),
    ).from_(table_expr)


def _scip_symbols_select(scip_symbols_ref: str) -> exp.Select:
    table_expr = table_expr_from_ref(scip_symbols_ref)
    doc_id = _concat(
        [
            exp.Literal.string("symbol:"),
            exp.column("repo"),
            exp.Literal.string(":"),
            exp.column("commit"),
            exp.Literal.string(":"),
            exp.column("rel_path"),
            exp.Literal.string(":"),
            exp.column("symbol"),
        ]
    )
    return exp.select(
        exp.alias_(doc_id, "doc_id"),
        exp.alias_(exp.Literal.string("symbol"), "kind"),
        exp.alias_(exp.column("symbol"), "name"),
        exp.alias_(exp.null(), "module"),
        exp.column("rel_path"),
        exp.alias_(_coalesce_text(exp.column("documentation")), "text"),
        exp.alias_(exp.null(), "ref_goid_h128"),
        exp.column("repo"),
        exp.column("commit"),
    ).from_(table_expr)
