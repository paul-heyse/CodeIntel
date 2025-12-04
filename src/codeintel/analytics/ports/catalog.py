"""Catalog port interface re-export for analytics.

This module re-exports the CatalogPort protocol from graphs.ports,
providing a consistent import path for analytics modules.
"""

from __future__ import annotations

from codeintel.graphs.ports.catalog import CatalogPort, FunctionSpanData

__all__ = [
    "CatalogPort",
    "FunctionSpanData",
]
