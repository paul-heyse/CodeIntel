"""Coverage analytics plugins using the new protocol.

This module provides coverage-related analytics plugins migrated
to the new unified plugin protocol.
"""

from __future__ import annotations

from codeintel.analytics.plugins.coverage.functions import CoverageFunctionsPlugin
from codeintel.analytics.plugins.coverage.test_edges import CoverageTestEdgesPlugin

__all__ = [
    "CoverageFunctionsPlugin",
    "CoverageTestEdgesPlugin",
]
