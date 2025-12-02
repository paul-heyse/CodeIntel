"""Persistence adapters for analytics data access.

This package provides adapter classes that handle all database I/O for
analytics modules. Adapters encapsulate:
- Loading data from DuckDB tables
- Persisting computed results
- Managing transaction boundaries

By separating I/O into adapters, the computation layer remains pure
and easily testable.

Modules
-------
functions
    Adapters for function metrics and types tables.
graphs
    Adapters for graph metrics tables.
base
    Base adapter classes and protocols.
"""

from __future__ import annotations

from codeintel.analytics.adapters.base import (
    AnalyticsAdapter,
    DeleteScope,
)
from codeintel.analytics.adapters.functions import (
    FunctionMetricsAdapter,
    FunctionTypesAdapter,
)

__all__ = [
    "AnalyticsAdapter",
    "DeleteScope",
    "FunctionMetricsAdapter",
    "FunctionTypesAdapter",
]
