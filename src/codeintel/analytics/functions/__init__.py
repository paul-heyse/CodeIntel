"""Function-level analytics public API.

This module exposes type definitions used across analytics and Hamilton targets.
Execution entrypoints live in Hamilton-native modules and are not re-exported
from this package to enforce DAG-first execution.
"""

from __future__ import annotations

from codeintel.analytics.functions.config import FunctionAnalyticsOptions
from codeintel.analytics.functions.metrics import FunctionAnalyticsResult

__all__ = [
    "FunctionAnalyticsOptions",
    "FunctionAnalyticsResult",
]
