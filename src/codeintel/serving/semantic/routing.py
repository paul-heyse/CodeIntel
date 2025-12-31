"""Routing helpers for semantic query engines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlglot import exp

from codeintel.serving.semantic.query_ast import ServingQuery

if TYPE_CHECKING:
    from codeintel.serving.semantic.engines.protocol import EngineContext

_POLARS_ALLOWED_FUNCTIONS = frozenset({"coalesce", "contains", "lower", "starts_with", "upper"})
_POLARS_UNSUPPORTED_NODES: tuple[type[exp.Expression], ...] = (
    exp.Join,
    exp.Subquery,
    exp.CTE,
    exp.Union,
    exp.Intersect,
    exp.Except,
    exp.Window,
    exp.WindowSpec,
    exp.Group,
    exp.Having,
    exp.Distinct,
    exp.Qualify,
)


def ast_supports_polars(ast: exp.Expression) -> bool:
    """Return True when the AST fits the Polars execution envelope.

    Returns
    -------
    bool
        True when the AST stays within the Polars feature envelope.
    """
    for node_type in _POLARS_UNSUPPORTED_NODES:
        if ast.find(node_type) is not None:
            return False
    for func in ast.find_all(exp.Func):
        func_name = func.sql_name().lower()
        if func_name not in _POLARS_ALLOWED_FUNCTIONS:
            return False
    return True


def auto_preference(query: ServingQuery, *, ctx: EngineContext) -> tuple[str, ...]:
    """Return engine ordering for auto mode based on AST and view capabilities.

    Returns
    -------
    tuple[str, ...]
        Engine names ordered by preference.
    """
    if not ast_supports_polars(query.ast):
        return ("duckdb",)
    table_key = query.spec.table_key
    has_polars_source = (
        ctx.view_registry.get(table_key) is not None
        or ctx.dataset_manifests.get(table_key) is not None
    )
    if not has_polars_source:
        return ("duckdb",)
    return ("duckdb", "polars")


__all__ = ["ast_supports_polars", "auto_preference"]
