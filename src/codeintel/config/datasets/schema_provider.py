"""Schema provider for dataset contracts.

This module centralizes access to TABLE_SCHEMAS and COMPOSITE_SCHEMAS to avoid
in-function imports and keep import order explicit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.config.datasets.schemas import COMPOSITE_SCHEMAS, TABLE_SCHEMAS

if TYPE_CHECKING:
    from codeintel.config.datasets.primitives import CompositeSchema, TableSchema


def table_schemas() -> dict[str, TableSchema]:
    """Return all registered table schemas.

    Returns
    -------
    dict[str, TableSchema]
        Mapping of fully qualified table keys to schema definitions.
    """
    return TABLE_SCHEMAS


def composite_schemas() -> dict[str, CompositeSchema]:
    """Return all registered composite schemas.

    Returns
    -------
    dict[str, CompositeSchema]
        Mapping of dataset names to composite profile metadata.
    """
    return COMPOSITE_SCHEMAS


__all__ = ["composite_schemas", "table_schemas"]
