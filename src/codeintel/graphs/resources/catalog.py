"""Catalog resource provider.

This module provides backward-compatible re-exports of the unified CatalogService.

.. deprecated:: 5.0.0
    Use CatalogService from ``codeintel.graphs.catalog`` directly.
    CatalogResource is retained as an alias for backward compatibility.

Migration
---------
Old::

    from codeintel.graphs.resources.catalog import CatalogResource

    resource = CatalogResource(catalog)

New::

    from codeintel.graphs.catalog import CatalogService

    service = CatalogService(catalog)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from codeintel.graphs.catalog import CatalogService, FunctionSpan

if TYPE_CHECKING:
    from codeintel.graphs.catalog import FunctionCatalog


def CatalogResource(catalog: FunctionCatalog) -> CatalogService:  # noqa: N802
    """Create a CatalogService from a FunctionCatalog.

    .. deprecated:: 5.0.0
        Use CatalogService from ``codeintel.graphs.catalog`` directly.

    Parameters
    ----------
    catalog
        Function catalog to wrap.

    Returns
    -------
    CatalogService
        Unified catalog service.
    """
    warnings.warn(
        "CatalogResource is deprecated. Use CatalogService from codeintel.graphs.catalog directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return CatalogService(catalog)


# Type alias for backward compatibility
CatalogResourceType = CatalogService


__all__ = [
    "CatalogResource",
    "CatalogResourceType",
    "CatalogService",  # Re-export the canonical class
    "FunctionSpan",  # Re-export for convenience
]
