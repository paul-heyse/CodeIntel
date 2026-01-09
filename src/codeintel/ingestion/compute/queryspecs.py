"""QuerySpec helpers for ingestion scoping and projection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from codeintel.core.columnar.expr_vocab import E, Expression
from codeintel.core.columnar.queryspec import QuerySpec, projection_spec_from_schema_defaults
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.service import get_schema_service


@dataclass(frozen=True, slots=True)
class IngestQuerySpecRequest:
    """Request inputs for ingestion QuerySpec construction."""

    columns: Sequence[str] | None = None
    repo: str | None = None
    commit: str | None = None
    rel_path: str | None = None
    available_columns: Sequence[str] | None = None


def build_ingest_query_spec(table_key: str, request: IngestQuerySpecRequest) -> QuerySpec:
    """Build an ingestion-friendly QuerySpec for repo/commit/rel_path scoping.

    Returns
    -------
    QuerySpec
        Query specification with optional repo/commit/path filtering.
    """
    table_schema = get_schema_service().get_table_schema(table_key)
    resolved_available = _resolve_available_columns(
        table_schema,
        available_columns=request.available_columns,
        columns=request.columns,
    )
    predicate = _ingest_scope_predicate(
        column_names=set(resolved_available),
        repo=request.repo,
        commit=request.commit,
        rel_path=request.rel_path,
    )
    projection = projection_spec_from_schema_defaults(
        request.columns,
        table_schema=table_schema,
        available_columns=resolved_available,
    )
    return QuerySpec(
        predicate=predicate,
        pushdown_predicate=predicate,
        projection=projection,
    )


def _resolve_available_columns(
    table_schema: TableSchema | None,
    *,
    available_columns: Sequence[str] | None,
    columns: Sequence[str] | None,
) -> list[str]:
    if available_columns is not None:
        return list(available_columns)
    if table_schema is None:
        return list(columns) if columns is not None else []
    return list(table_schema.column_names())


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


__all__ = ["IngestQuerySpecRequest", "build_ingest_query_spec"]
