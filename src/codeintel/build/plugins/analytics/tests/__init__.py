"""Test analytics plugins using the new protocol.

This module provides test-related analytics plugins migrated
to the new unified plugin protocol.
"""

from __future__ import annotations

from codeintel.build.plugins.analytics.tests.behavioral_coverage import (
    BehavioralCoveragePlugin,
)
from codeintel.build.plugins.analytics.tests.graph_metrics import TestGraphMetricsPlugin
from codeintel.build.plugins.analytics.tests.profile import TestProfilePlugin

__all__ = [
    "BehavioralCoveragePlugin",
    "TestGraphMetricsPlugin",
    "TestProfilePlugin",
]
