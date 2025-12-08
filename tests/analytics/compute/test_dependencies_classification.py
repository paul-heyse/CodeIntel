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
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_is_none,
    expect_is_not_none,
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
    expect_equal(modes, ["read"], label="direct mode")
    expect_is_not_none(matched, label="matcher present")
    if matched is not None:
        expect_equal(matched.modes, ["read"], label="matcher modes")

    modes_with_prefix, matched_prefix = classify_modes(pattern, "post_json", "requests.post_json")
    expect_in("write", modes_with_prefix, label="write mode present")
    expect_is_not_none(matched_prefix, label="prefix matcher")

    modes_unknown, matched_unknown = classify_modes(pattern, "head", "requests.head")
    expect_equal(modes_unknown, ["unknown"], label="unknown modes")
    expect_is_none(matched_unknown, label="unknown matcher")


def test_severity_and_risk_scores() -> None:
    """Map severities to scores and derive risk scores."""
    expect_equal(severity_score("high"), 3.0, label="high severity")
    expect_is_none(severity_score("unknown"), label="unknown severity")
    expect_equal(risk_score("high", 2.0), 6.0, label="risk score")
    expect_is_none(risk_score(None, 2.0), label="risk score none severity")


def test_risk_level_balances_modes_and_frequency() -> None:
    """Derive risk level from usage modes and callsite frequency."""
    expect_equal(risk_level({"write"}, 1), "high", label="write single call high risk")
    expect_equal(risk_level({"read"}, 15), "medium", label="read many medium risk")
    expect_equal(risk_level({"read"}, 5), "low", label="read few low risk")
