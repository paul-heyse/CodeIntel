"""Coverage computation module.

This package provides pure computation functions for analyzing code coverage.

For Hamilton native execution, use `build_coverage_functions_expr` to get an
Ibis expression, then materialize with `materialize_table`.
"""

from __future__ import annotations

from codeintel.analytics.compute.coverage.compute import build_coverage_functions_expr
from codeintel.analytics.compute.coverage.functions import compute_coverage_functions

__all__ = [
    "build_coverage_functions_expr",
    "compute_coverage_functions",
]
