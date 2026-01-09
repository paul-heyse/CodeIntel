"""Routing helpers for semantic query engines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.serving.semantic.query_ast import ServingQuery

if TYPE_CHECKING:
    from codeintel.serving.semantic.engines.protocol import EngineContext


def auto_preference(query: ServingQuery, *, ctx: EngineContext) -> tuple[str, ...]:
    """Return engine ordering for auto mode based on AST and view capabilities.

    Returns
    -------
    tuple[str, ...]
        Engine names ordered by preference.
    """
    _ = query.spec.table_key
    _ = ctx.settings.query_engine
    if query.arrow_plan is not None:
        return ("arrow", "duckdb")
    return ("duckdb",)


__all__ = ["auto_preference"]
