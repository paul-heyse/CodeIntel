"""View creation helpers using the Ibis-based VIEW_BUILDERS registry.

This module is separate from __init__.py to avoid circular imports when
config.datasets imports from storage.view_names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from duckdb import DuckDBPyConnection

from codeintel.storage.gateway.minimal import MinimalStorageGateway
from codeintel.storage.views.ibis_registry import VIEW_BUILDERS
from codeintel.storage.views.ibis_views import _create_view

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import StorageGateway
    from codeintel.storage.views.ibis_registry import IbisViewGateway

__all__ = ["create_all_views"]


def _get_ibis_gateway(
    con_or_gateway: DuckDBPyConnection | StorageGateway,
) -> IbisViewGateway:
    """Extract or create IbisGateway from connection or gateway.

    Parameters
    ----------
    con_or_gateway
        Either a DuckDB connection or a StorageGateway.

    Returns
    -------
    IbisViewGateway
        Ibis gateway-like object for building expressions.
    """
    if isinstance(con_or_gateway, DuckDBPyConnection):
        return MinimalStorageGateway(con_or_gateway).ibis

    return con_or_gateway.ibis


def create_all_views(
    con_or_gateway: DuckDBPyConnection | StorageGateway,
) -> None:
    """Create or replace all docs.* views using Ibis expressions.

    This function iterates through all registered view builders in VIEW_BUILDERS
    and creates each view using the Ibis gateway.

    Parameters
    ----------
    con_or_gateway
        Either a DuckDB connection or a StorageGateway. For backward
        compatibility, raw connections are wrapped in an IbisGateway.
    """
    ibis_gw = _get_ibis_gateway(con_or_gateway)
    for view_name, builder in VIEW_BUILDERS.items():
        expr = builder(ibis_gw)
        _create_view(ibis_gw.con, view_name, expr)
