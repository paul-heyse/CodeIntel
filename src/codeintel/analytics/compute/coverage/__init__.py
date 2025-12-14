"""Coverage computation module.

This package provides pure computation functions for analyzing code coverage.

For Hamilton native execution, use ``build_coverage_functions_expr`` to get an
Ibis expression, then materialize with ``materialize_table``.

The Hamilton native module is at:
``codeintel.build.hamilton.native.analytics.coverage_functions``
"""

from __future__ import annotations

from codeintel.analytics.compute.coverage.compute import build_coverage_functions_expr

__all__ = [
    "build_coverage_functions_expr",
]
