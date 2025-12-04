"""Test analytics plugins using the new protocol.

This module provides test-related analytics plugins migrated
to the new unified plugin protocol.
"""

from __future__ import annotations

from codeintel.analytics.plugins.tests.behavioral_coverage import (
    BehavioralCoveragePlugin,
)
from codeintel.analytics.plugins.tests.profile import TestProfilePlugin

__all__ = [
    "BehavioralCoveragePlugin",
    "TestProfilePlugin",
]
