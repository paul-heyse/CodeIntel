"""Catalog data types for function catalog access.

This module re-exports FunctionSpan from the catalog module for convenience.

New code should import directly from ``codeintel.graphs.catalog``.

See Also
--------
codeintel.graphs.catalog : Canonical catalog types
"""

from __future__ import annotations

from codeintel.graphs.catalog import FunctionSpan

__all__ = [
    "FunctionSpan",
]
