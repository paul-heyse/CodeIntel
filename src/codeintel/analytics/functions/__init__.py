"""
Function-level analytics: metrics, typedness, and validation helpers.

The original monolithic `analytics.functions` module has been split into
focused submodules. Import `compute_function_metrics_and_types` from this
package to run the analytics pipeline.
"""

from __future__ import annotations

from codeintel.analytics.functions.config import FunctionAnalyticsOptions
from codeintel.analytics.functions.metrics import (
    FunctionAnalyticsResult,
    compute_function_metrics_and_types,
)
from codeintel.analytics.functions.validation import ValidationReporter

__all__ = [
    "FunctionAnalyticsOptions",
    "FunctionAnalyticsResult",
    "ValidationReporter",
    "compute_function_metrics_and_types",
]
