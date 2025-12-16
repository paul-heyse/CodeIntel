"""View creation helpers using the Ibis-based VIEW_BUILDERS registry.

This module is separate from __init__.py to avoid circular imports when
config.datasets imports from storage.view_names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from duckdb import DuckDBPyConnection

from codeintel.storage.gateway.minimal import MinimalStorageGateway

if TYPE_CHECKING:
    from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
    from codeintel.storage.gateway.protocol import StorageGateway

__all__ = ["create_all_views"]


def _get_policy_backend(
    con_or_gateway: DuckDBPyConnection | StorageGateway,
) -> DuckDBPolicyBackend:
    """Extract or create a policy backend from connection or gateway.

    Parameters
    ----------
    con_or_gateway
        Either a DuckDB connection or a StorageGateway.

    Returns
    -------
    DuckDBPolicyBackend
        Policy backend for view materialization.
    """
    if isinstance(con_or_gateway, DuckDBPyConnection):
        return MinimalStorageGateway(con_or_gateway).policy
    return con_or_gateway.policy


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
    policy = _get_policy_backend(con_or_gateway)
    policy.ensure_all_views(overwrite=True, strict=True)
