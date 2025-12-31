"""Search query planning for serving."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from codeintel.storage.duckdb_types import (
    ColumnExpression,
    ConstantExpression,
    DuckDBRelation,
    Expression,
    FunctionExpression,
)
from codeintel.storage.serving.search_index import DuckDBConnection, fts_index_available

if TYPE_CHECKING:
    from codeintel.serving.search.models import SearchQueryRequest

SEARCH_TABLE_SCHEMA: Final[str] = "docs"
SEARCH_TABLE_NAME: Final[str] = "search_documents"
SEARCH_TABLE_KEY: Final[str] = "docs.search_documents"

_SEARCH_SELECT_COLUMNS = (
    "kind",
    "name",
    "module",
    "rel_path",
    "ref_goid_h128",
)


def is_fts_available(con: DuckDBConnection) -> bool:
    """Return True when the search table has an FTS index available.

    Parameters
    ----------
    con
        Active DB-API connection.

    Returns
    -------
    bool
        True when the search index is available.
    """
    return fts_index_available(con, table_key=SEARCH_TABLE_KEY)


def build_search_relation(
    con: DuckDBConnection,
    request: SearchQueryRequest,
    *,
    fts_available: bool,
) -> DuckDBRelation:
    """Return a DuckDB relation for a search request.

    Parameters
    ----------
    con
        Active DuckDB connection.
    request
        Search request model.
    fts_available
        Whether the DuckDB FTS extension and search index are available.

    Returns
    -------
    DuckDBRelation
        DuckDB relation representing the search query.
    """
    relation = con.table(SEARCH_TABLE_KEY)
    if request.kinds:
        kind_literals = [ConstantExpression(value) for value in request.kinds]
        relation = relation.filter(ColumnExpression("kind").isin(*kind_literals))
    if fts_available:
        score_expr = _fts_score_expr(request.query)
        relation = relation.filter(score_expr.isnotnull())
        relation = relation.select(*_SEARCH_SELECT_COLUMNS, score_expr.alias("score"))
        relation = relation.order("score DESC")
    else:
        needle = request.query
        predicate = (
            _contains_case_insensitive("text", needle)
            | _contains_case_insensitive("name", needle)
            | _contains_case_insensitive("module", needle)
        )
        relation = relation.filter(predicate)
        relation = relation.select(
            *_SEARCH_SELECT_COLUMNS,
            ConstantExpression(None).alias("score"),
        )
        relation = relation.order("kind, name")
    return relation.limit(request.limit + 1, offset=request.offset)


def _fts_score_expr(query: str) -> Expression:
    return FunctionExpression(
        "fts_docs_search_documents.match_bm25",
        ColumnExpression("doc_id"),
        ConstantExpression(query),
    )


def _contains_case_insensitive(column: str, needle: str) -> Expression:
    lowered = FunctionExpression(
        "lower",
        FunctionExpression("coalesce", ColumnExpression(column), ConstantExpression("")),
    )
    return FunctionExpression("contains", lowered, ConstantExpression(needle.lower()))


__all__ = [
    "SEARCH_TABLE_KEY",
    "SEARCH_TABLE_NAME",
    "SEARCH_TABLE_SCHEMA",
    "build_search_relation",
    "is_fts_available",
]
