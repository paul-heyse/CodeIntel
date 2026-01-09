"""QuerySpec helpers for ingestion scoping and projection."""

from __future__ import annotations

from collections.abc import Sequence

from codeintel.core.columnar.expr_vocab import E, Expression
from codeintel.core.columnar.queryspec import ProjectionSpec, QuerySpec
from codeintel.core.schemas.service import get_schema_service


def build_ingest_query_spec(
    table_key: str,
    *,
    columns: Sequence[str] | None = None,
    repo: str | None = None,
    commit: str | None = None,
    rel_path: str | None = None,
) -> QuerySpec:
    """Build an ingestion-friendly QuerySpec for repo/commit/rel_path scoping.

    Returns
    -------
    QuerySpec
        Query specification with optional repo/commit/path filtering.
    """
    resolved_columns = _resolve_query_columns(table_key, columns)
    predicate = _ingest_scope_predicate(
        column_names=set(resolved_columns),
        repo=repo,
        commit=commit,
        rel_path=rel_path,
    )
    projection = ProjectionSpec(base_cols=tuple(resolved_columns))
    return QuerySpec(
        predicate=predicate,
        pushdown_predicate=predicate,
        projection=projection,
    )


def _resolve_query_columns(table_key: str, columns: Sequence[str] | None) -> list[str]:
    if columns is not None:
        return list(columns)
    schema = get_schema_service().get_table_schema(table_key)
    if schema is None:
        return []
    return list(schema.column_names())


def _ingest_scope_predicate(
    *,
    column_names: set[str],
    repo: str | None,
    commit: str | None,
    rel_path: str | None,
) -> Expression | None:
    exprs: list[Expression] = []
    if repo is not None and "repo" in column_names:
        exprs.append(E.field("repo") == E.scalar(repo))
    if commit is not None and "commit" in column_names:
        exprs.append(E.field("commit") == E.scalar(commit))
    if rel_path is not None and "rel_path" in column_names:
        exprs.append(E.field("rel_path") == E.scalar(rel_path))
    if not exprs:
        return None
    return E.and_(*exprs)


__all__ = ["build_ingest_query_spec"]
