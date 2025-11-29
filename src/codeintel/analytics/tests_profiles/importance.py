"""Flakiness and importance scoring helpers for tests."""

from __future__ import annotations

from collections.abc import Iterable

from codeintel.analytics.tests_profiles.types import ImportanceInputs, IoFlags


def compute_flakiness_score(
    *,
    status: str | None,
    markers: Iterable[str],
    duration_ms: float | None,
    io_flags: IoFlags,
    slow_test_threshold_ms: float,
) -> float:
    """
    Derive a heuristic flakiness score in the range [0.0, 1.0].

    Returns
    -------
    float
        Flakiness score between 0.0 and 1.0.
    """
    score = 0.0
    markers_lower = [marker.lower() for marker in markers]
    if any("flaky" in marker for marker in markers_lower):
        score += 0.6
    if status is not None and status.lower() in {"xfail", "xpass"}:
        score += 0.2
    if io_flags.uses_network:
        score += 0.15
    if io_flags.uses_db or io_flags.uses_subprocess:
        score += 0.1
    if io_flags.uses_filesystem:
        score += 0.05
    if duration_ms is not None and duration_ms > slow_test_threshold_ms:
        score += 0.1
    return min(score, 1.0)


def compute_importance_score(inputs: ImportanceInputs) -> float | None:
    """
    Estimate relative importance using coverage breadth and graph metrics.

    Returns
    -------
    float | None
        Normalized importance in [0, 1] or None when no signals are present.
    """
    scores: list[float] = []
    if inputs.functions_covered_count > 0 and inputs.max_function_count > 0:
        scores.append(inputs.functions_covered_count / inputs.max_function_count)
    if inputs.weighted_degree is not None and inputs.max_weighted_degree > 0:
        scores.append(inputs.weighted_degree / inputs.max_weighted_degree)
    if inputs.subsystem_risk is not None and inputs.max_subsystem_risk > 0:
        scores.append(inputs.subsystem_risk / inputs.max_subsystem_risk)
    if not scores:
        return None
    return min(sum(scores) / len(scores), 1.0)
