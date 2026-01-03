"""Deprecated wrapper for lazy import helpers.

Use codeintel.core.imports.lazy instead.
"""

from __future__ import annotations

from codeintel.core.imports.lazy import LazyAttrMap, lazy_callable, make_lazy_getattr

__all__ = [
    "LazyAttrMap",
    "lazy_callable",
    "make_lazy_getattr",
]
