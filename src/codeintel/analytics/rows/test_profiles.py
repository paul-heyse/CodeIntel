"""Typed row re-exports for test profile analytics tables."""

from __future__ import annotations

from codeintel.storage.rows import (
    BehavioralCoverageRowModel,
    ProfileRowModel,
    behavioral_coverage_row_to_tuple,
    serialize_test_profile_row,
)

TestProfileRow = ProfileRowModel
BehavioralCoverageRow = BehavioralCoverageRowModel

__all__ = [
    "BehavioralCoverageRow",
    "TestProfileRow",
    "behavioral_coverage_row_to_tuple",
    "serialize_test_profile_row",
]
