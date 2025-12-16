"""Protocols for Ibis-based view builders.

These protocols define the minimal interface required by view builder functions.
They intentionally avoid coupling to the concrete gateway implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import ibis.expr.types as it
    from ibis.backends.duckdb import Backend as DuckDBBackend


class IbisViewGateway(Protocol):
    """Protocol for objects that provide Ibis access for view building."""

    @property
    def con(self) -> DuckDBBackend:
        """Return an Ibis backend bound to a DuckDB connection."""
        ...

    def table(self, table_name: str) -> it.Table:
        """Return an Ibis table expression for a fully qualified table."""
        ...


class ViewBuilder(Protocol):
    """Protocol for view builder functions."""

    def __call__(self, ibis_gw: IbisViewGateway) -> it.Table:
        """Build and return an Ibis table expression for a view."""
        ...


__all__ = ["IbisViewGateway", "ViewBuilder"]
