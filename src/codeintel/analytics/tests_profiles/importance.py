"""Importance scoring - re-exports from testing/.

This module provides backward-compatible imports. New code should import
directly from ``codeintel.analytics.testing.behavioral.importance``.
"""

from __future__ import annotations

from codeintel.analytics.testing.behavioral.importance import (
    compute_flakiness_score,
    compute_importance_score,
)
from codeintel.analytics.testing.profiles.types import ImportanceInputs, IoFlags

__all__ = [
    "ImportanceInputs",
    "IoFlags",
    "compute_flakiness_score",
    "compute_importance_score",
]
