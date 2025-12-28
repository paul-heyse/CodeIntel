"""SQLGlot-backed query AST helpers for semantic serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.serving.semantic.specs import SemanticQuerySpec
from codeintel.serving.semantic.sqlglot_query_builder import build_sqlglot_query

if TYPE_CHECKING:
    from sqlglot import exp


@dataclass(frozen=True, slots=True)
class ServingQuery:
    """Semantic query bundle with SQLGlot AST."""

    spec: SemanticQuerySpec
    ast: exp.Select


def build_serving_query(*, spec: SemanticQuerySpec) -> ServingQuery:
    """Build a ServingQuery from a semantic spec.

    Returns
    -------
    ServingQuery
        Serving query bundle with SQLGlot AST.
    """
    ast = build_sqlglot_query(
        spec=spec,
        allowed_columns=spec.allowed_columns,
        column_types=spec.column_types,
    )
    return ServingQuery(spec=spec, ast=ast)


__all__ = ["ServingQuery", "build_serving_query"]
