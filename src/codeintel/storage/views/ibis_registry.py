"""Ibis View Registry for centralized view definition management.

This module provides a registry for Ibis-defined views that allows:
- Decorator-based view registration
- Centralized view discovery
- Consistent view creation via the policy backend

Example
-------
>>> from codeintel.storage.views.ibis_registry import IbisViewGateway, register_view, VIEW_BUILDERS
>>>
>>> @register_view("analytics.v_function_summary")
>>> def build_function_summary(ibis_gw: IbisViewGateway) -> it.Table:
...     fm = ibis_gw.table("analytics.function_metrics")
...     ft = ibis_gw.table("analytics.function_types")
...     return fm.left_join(ft, ["function_goid_h128"])
>>>
>>> # All registered views
>>> for name, builder in VIEW_BUILDERS.items():
...     print(f"View: {name}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    import ibis.expr.types as it
    from ibis.backends.duckdb import Backend as DuckDBBackend

__all__ = [
    "VIEW_BUILDERS",
    "IbisViewGateway",
    "ViewBuilder",
    "get_registered_views",
    "register_view",
]


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
    """Protocol for view builder functions.

    A ViewBuilder takes an IbisViewGateway and returns an Ibis table expression
    that defines the view's query.
    """

    def __call__(self, ibis_gw: IbisViewGateway) -> it.Table:
        """Build an Ibis table expression for the view.

        Parameters
        ----------
        ibis_gw
            Ibis gateway for accessing tables and building expressions.

        Returns
        -------
        it.Table
            Ibis table expression defining the view.
        """
        ...


# Global registry mapping view names to builder functions
VIEW_BUILDERS: dict[str, ViewBuilder] = {}


def register_view(table_key: str) -> Callable[[ViewBuilder], ViewBuilder]:
    """Register a view builder function in the global registry.

    Parameters
    ----------
    table_key
        Fully qualified view name (e.g., "analytics.v_function_summary").

    Returns
    -------
    Callable[[ViewBuilder], ViewBuilder]
        Decorator that registers the function and returns it unchanged.

    Example
    -------
    >>> @register_view("analytics.v_my_view")
    >>> def build_my_view(ibis_gw: IbisViewGateway) -> it.Table:
    ...     return ibis_gw.table("analytics.source_table").select("col1", "col2")
    """

    def decorator(func: ViewBuilder) -> ViewBuilder:
        VIEW_BUILDERS[table_key] = func
        return func

    return decorator


def get_registered_views() -> dict[str, ViewBuilder]:
    """Return a copy of all registered view builders.

    Returns
    -------
    dict[str, ViewBuilder]
        Mapping of view names to their builder functions.
    """
    return dict(VIEW_BUILDERS)
