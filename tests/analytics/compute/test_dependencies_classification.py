"""Tests for dependency classification helpers."""

from __future__ import annotations

from codeintel.analytics.compute.dependencies import (
    DependencyModePattern,
    LibraryPattern,
    classify_modes,
    risk_level,
    risk_score,
    severity_score,
)


def test_classify_modes_prioritizes_specific_matchers() -> None:
    """Ensure matcher ordering and fallback semantics behave as expected."""
    pattern = LibraryPattern(
        library="requests",
        service_name="HTTP",
        category="http",
        matchers=[
            DependencyModePattern(modes=["read"], method="get"),
            DependencyModePattern(modes=["write"], method_prefix="post"),
            DependencyModePattern(modes=["admin"], match="delete"),
        ],
        severity="medium",
        criticality=2.0,
    )

    modes, matched = classify_modes(pattern, "get", "requests.get")
    assert modes == ["read"]
    assert matched is not None
    assert matched.modes == ["read"]

    modes_with_prefix, matched_prefix = classify_modes(pattern, "post_json", "requests.post_json")
    assert "write" in modes_with_prefix
    assert matched_prefix is not None

    modes_unknown, matched_unknown = classify_modes(pattern, "head", "requests.head")
    assert modes_unknown == ["unknown"]
    assert matched_unknown is None


def test_severity_and_risk_scores() -> None:
    """Map severities to scores and derive risk scores."""
    assert severity_score("high") == 3.0
    assert severity_score("unknown") is None
    assert risk_score("high", 2.0) == 6.0
    assert risk_score(None, 2.0) is None


def test_risk_level_balances_modes_and_frequency() -> None:
    """Derive risk level from usage modes and callsite frequency."""
    assert risk_level({"write"}, 1) == "high"
    assert risk_level({"read"}, 15) == "medium"
    assert risk_level({"read"}, 5) == "low"
