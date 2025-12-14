"""Coverage analytics plugins using the new protocol.

This module provides coverage-related analytics plugins migrated
to the new unified plugin protocol.

Note: CoverageFunctionsPlugin has been removed; use the Hamilton native module
``codeintel.build.hamilton.native.analytics.coverage_functions`` instead.
"""

from __future__ import annotations

from codeintel.build.plugins.analytics.coverage.test_edges import CoverageTestEdgesPlugin

__all__ = [
    "CoverageTestEdgesPlugin",
]
