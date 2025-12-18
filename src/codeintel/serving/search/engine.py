"""Search query planning for serving.

This module owns the selection logic for which SQL template to use (FTS vs LIKE)
based on server capabilities and request inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from codeintel.serving.semantic.templates import DbApiTemplate
from codeintel.storage.serving.search_index import DuckDBConnection, fts_index_available

if TYPE_CHECKING:
    from codeintel.serving.search.models import SearchQueryRequest
    from codeintel.serving.semantic.templates import DbApiQuery

SEARCH_TABLE_SCHEMA: Final[str] = "docs"
SEARCH_TABLE_NAME: Final[str] = "search_documents"
SEARCH_TABLE_KEY: Final[str] = "docs.search_documents"

_SQL_SEARCH_FTS = """
\tSELECT kind, name, module, rel_path, ref_goid_h128, score
\tFROM (
\t    SELECT
        kind,
        name,
        module,
        rel_path,
        ref_goid_h128,
        fts_docs_search_documents.match_bm25(doc_id, ?) AS score
    FROM docs.search_documents
) ranked
\tWHERE score IS NOT NULL
\tORDER BY score DESC
\tLIMIT ? OFFSET ?
"""

_SQL_SEARCH_FTS_KINDS = """
SELECT kind, name, module, rel_path, ref_goid_h128, score
FROM (
    SELECT
        kind,
        name,
        module,
        rel_path,
        ref_goid_h128,
        fts_docs_search_documents.match_bm25(doc_id, ?) AS score
    FROM docs.search_documents
    WHERE kind = ANY(?)
) ranked
WHERE score IS NOT NULL
ORDER BY score DESC
LIMIT ? OFFSET ?
"""

_SQL_SEARCH_LIKE = """
SELECT kind, name, module, rel_path, ref_goid_h128, NULL AS score
FROM docs.search_documents
WHERE (
    COALESCE(text, '') ILIKE '%' || ? || '%'
    OR COALESCE(name, '') ILIKE '%' || ? || '%'
    OR COALESCE(module, '') ILIKE '%' || ? || '%'
)
ORDER BY kind, name
LIMIT ? OFFSET ?
"""

_SQL_SEARCH_LIKE_KINDS = """
SELECT kind, name, module, rel_path, ref_goid_h128, NULL AS score
FROM docs.search_documents
WHERE (
    COALESCE(text, '') ILIKE '%' || ? || '%'
    OR COALESCE(name, '') ILIKE '%' || ? || '%'
    OR COALESCE(module, '') ILIKE '%' || ? || '%'
)
AND kind = ANY(?)
ORDER BY kind, name
LIMIT ? OFFSET ?
"""

_SEARCH_QUERY_FTS = DbApiTemplate(sql=_SQL_SEARCH_FTS)
_SEARCH_QUERY_FTS_KINDS = DbApiTemplate(sql=_SQL_SEARCH_FTS_KINDS)
_SEARCH_QUERY_LIKE = DbApiTemplate(sql=_SQL_SEARCH_LIKE)
_SEARCH_QUERY_LIKE_KINDS = DbApiTemplate(sql=_SQL_SEARCH_LIKE_KINDS)


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


def build_search_query(request: SearchQueryRequest, *, fts_available: bool) -> DbApiQuery:
    """Return a bound DB-API query for a search request.

    Parameters
    ----------
    request
        Search request model.
    fts_available
        Whether the DuckDB FTS extension and search index are available.

    Returns
    -------
    DbApiQuery
        Query template + bound parameters ready for execution.
    """
    query_limit = request.limit + 1
    if fts_available and request.kinds:
        return _SEARCH_QUERY_FTS_KINDS.bind(
            [request.query, request.kinds, query_limit, request.offset]
        )
    if fts_available:
        return _SEARCH_QUERY_FTS.bind([request.query, query_limit, request.offset])
    if request.kinds:
        return _SEARCH_QUERY_LIKE_KINDS.bind(
            [
                request.query,
                request.query,
                request.query,
                request.kinds,
                query_limit,
                request.offset,
            ]
        )
    return _SEARCH_QUERY_LIKE.bind(
        [request.query, request.query, request.query, query_limit, request.offset]
    )


__all__ = [
    "SEARCH_TABLE_KEY",
    "SEARCH_TABLE_NAME",
    "SEARCH_TABLE_SCHEMA",
    "build_search_query",
    "is_fts_available",
]
