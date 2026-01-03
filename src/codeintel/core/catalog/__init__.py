"""Unified function catalog primitives for CodeIntel.

This module provides the canonical function span and catalog protocols
shared across graphs and analytics modules. Storage-backed catalog
implementations live in ``codeintel.storage.catalog``.

Examples
--------
>>> from codeintel.core.catalog import FunctionSpan, SpanIndex
>>> span = FunctionSpan(
...     goid=123, rel_path="src/main.py", qualname="main", start_line=1, end_line=10
... )
>>> index = SpanIndex([span])
>>> index.lookup("src/main.py", 5)
123
"""

from __future__ import annotations

from codeintel.core.catalog.function_span import FunctionSpan
from codeintel.core.catalog.protocol import CatalogProtocol, CatalogProviderProtocol
from codeintel.core.catalog.span_index import SpanIndex

__all__ = [
    "CatalogProtocol",
    "CatalogProviderProtocol",
    "FunctionSpan",
    "SpanIndex",
]
