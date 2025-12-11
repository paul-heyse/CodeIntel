"""Docs view registry and creation helpers.

This module provides Ibis-based view creation using the VIEW_BUILDERS registry.
All views are now defined as Ibis expressions in ibis_views.py.

The legacy SQL-based view creation functions have been removed in favor of
the unified Ibis approach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ibis
from duckdb import DuckDBPyConnection

# Import ibis_views to ensure view builders are registered
import codeintel.storage.views.ibis_views as _ibis_views
from codeintel.storage.ibis_adapter import IbisGateway
from codeintel.storage.views.data_model_views import DATA_MODEL_VIEW_NAMES
from codeintel.storage.views.function_views import FUNCTION_VIEW_NAMES
from codeintel.storage.views.graph_views import GRAPH_VIEW_NAMES
from codeintel.storage.views.ibis_registry import VIEW_BUILDERS, ViewBuilder, get_registered_views
from codeintel.storage.views.ibis_views import _create_view
from codeintel.storage.views.ide_views import IDE_VIEW_NAMES
from codeintel.storage.views.module_views import MODULE_VIEW_NAMES
from codeintel.storage.views.subsystem_views import SUBSYSTEM_VIEW_NAMES
from codeintel.storage.views.test_views import TEST_VIEW_NAMES

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import StorageGateway

ALIAS_DOCS_VIEWS: dict[str, str] = {
    "docs.v_function_profile": "analytics.function_profile",
    "docs.v_file_profile": "analytics.file_profile",
    "docs.v_module_profile": "analytics.module_profile",
    "docs.v_config_graph_metrics_keys": "analytics.config_graph_metrics_keys",
    "docs.v_config_graph_metrics_modules": "analytics.config_graph_metrics_modules",
    "docs.v_config_projection_key_edges": "analytics.config_projection_key_edges",
    "docs.v_config_projection_module_edges": "analytics.config_projection_module_edges",
}

DOCS_VIEWS: tuple[str, ...] = (
    *FUNCTION_VIEW_NAMES,
    *MODULE_VIEW_NAMES,
    *TEST_VIEW_NAMES,
    *SUBSYSTEM_VIEW_NAMES,
    *GRAPH_VIEW_NAMES,
    *IDE_VIEW_NAMES,
    *DATA_MODEL_VIEW_NAMES,
)

DERIVED_DOCS_VIEWS: tuple[str, ...] = tuple(
    view for view in DOCS_VIEWS if view not in ALIAS_DOCS_VIEWS
)


def _get_ibis_gateway(
    con_or_gateway: DuckDBPyConnection | StorageGateway,
) -> IbisGateway:
    """Extract or create IbisGateway from connection or gateway.

    Parameters
    ----------
    con_or_gateway
        Either a DuckDB connection or a StorageGateway.

    Returns
    -------
    IbisGateway
        Ibis gateway for building expressions.

    Raises
    ------
    TypeError
        If the provided gateway does not expose an IbisGateway.
    """
    if isinstance(con_or_gateway, DuckDBPyConnection):
        ibis_con = ibis.duckdb.from_connection(con_or_gateway)
        return IbisGateway(ibis_con)

    ibis_gateway = getattr(con_or_gateway, "ibis", None)
    if isinstance(ibis_gateway, IbisGateway):
        return ibis_gateway

    message = "StorageGateway must expose an IbisGateway via `ibis`"
    raise TypeError(message)


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


__all__ = [
    "ALIAS_DOCS_VIEWS",
    "DERIVED_DOCS_VIEWS",
    "DOCS_VIEWS",
    "VIEW_BUILDERS",
    "ViewBuilder",
    "_ibis_views",
    "create_all_views",
    "get_registered_views",
]
