"""Routing helpers for semantic query engines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlglot import exp

from codeintel.serving.semantic.query_ast import ServingQuery

if TYPE_CHECKING:
    from codeintel.serving.semantic.engines.protocol import EngineContext

_POLARS_ALLOWED_FUNCTIONS = frozenset({"contains", "starts_with"})
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
    for func in ast.find_all(exp.Anonymous):
        func_name = func.name or ""
        if func_name.lower() not in _POLARS_ALLOWED_FUNCTIONS:
            return False
    return True


def auto_preference(query: ServingQuery, *, ctx: EngineContext) -> tuple[str, ...]:
    """Return engine ordering for auto mode based on AST and view capabilities.

    Returns
    -------
    tuple[str, ...]
        Engine names ordered by preference.
    """
    spec = query.spec
    try:
        view = ctx.registry.by_id(spec.view_id)
    except KeyError:
        view = None

    if view is not None and view.kind == "view" and ctx.view_registry.get(spec.table_key) is None:
        return ("duckdb", "polars")

    if not ast_supports_polars(query.ast):
        return ("duckdb", "polars")

    return ("polars", "duckdb")


__all__ = ["ast_supports_polars", "auto_preference"]
