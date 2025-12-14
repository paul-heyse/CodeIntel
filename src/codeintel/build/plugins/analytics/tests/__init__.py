"""Test analytics plugins using the new protocol.

This module provides test-related analytics plugins migrated
to the new unified plugin protocol.

Note: TestGraphMetricsPlugin has been removed; use the Hamilton native module
``codeintel.build.hamilton.native.analytics.test_graph_metrics`` instead.
"""

from __future__ import annotations

from codeintel.build.plugins.analytics.tests.behavioral_coverage import (
    BehavioralCoveragePlugin,
)
from codeintel.build.plugins.analytics.tests.profile import TestProfilePlugin

__all__ = [
    "BehavioralCoveragePlugin",
    "TestProfilePlugin",
]
