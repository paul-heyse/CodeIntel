"""Flakiness and importance scoring helpers for tests."""

from __future__ import annotations

from codeintel.analytics.tests import profiles as legacy
from codeintel.analytics.tests_profiles.types import ImportanceInputs, IoFlags


def compute_flakiness_score(
    *,
    status: str | None,
    markers: list[str],
    duration_ms: float | None,
    io_flags: IoFlags,
    slow_test_threshold_ms: float,
) -> float:
    """
    Delegate to legacy flakiness scoring used by test_profile.

    Returns
    -------
    float
        Flakiness score between 0.0 and 1.0.
    """
    return legacy.compute_flakiness_score(
        status=status,
        markers=markers,
        duration_ms=duration_ms,
        io_flags=io_flags,
        slow_test_threshold_ms=slow_test_threshold_ms,
    )


def compute_importance_score(inputs: ImportanceInputs) -> float | None:
    """
    Delegate to legacy importance scoring used by test_profile.

    Returns
    -------
    float | None
        Importance score in [0, 1] or None when no signals present.
    """
    return legacy.compute_importance_score(inputs)
