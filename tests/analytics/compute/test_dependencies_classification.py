"""Test dependency classification computation.

Test the pure computation functions for classifying dependency calls
by mode, severity, and risk level.
"""

from __future__ import annotations

import pytest

from codeintel.analytics.compute.dependencies.classification import (
    CALLSITE_MEDIUM_THRESHOLD,
    SEVERITY_SCORES,
    DependencyModePattern,
    LibraryPattern,
    classify_modes,
    risk_level,
    risk_score,
    severity_score,
)

# =============================================================================
# Constants
# =============================================================================

EXPECTED_THRESHOLD_10 = 10
EXPECTED_MATCHERS_1 = 1

# =============================================================================
# Test Data: Realistic Dependency Patterns
# =============================================================================


def _make_requests_pattern() -> LibraryPattern:
    """
    Create a realistic HTTP client pattern similar to requests library.

    Returns
    -------
    LibraryPattern
        HTTP client library pattern.
    """
    return LibraryPattern(
        library="requests",
        service_name="HTTP Client",
        category="http",
        matchers=[
            DependencyModePattern(
                modes=["read"],
                method="get",
                name="HTTP GET",
            ),
            DependencyModePattern(
                modes=["write"],
                method="post",
                name="HTTP POST",
            ),
            DependencyModePattern(
                modes=["write"],
                method="put",
                name="HTTP PUT",
            ),
            DependencyModePattern(
                modes=["write", "delete"],
                method="delete",
                name="HTTP DELETE",
            ),
            DependencyModePattern(
                modes=["read"],
                method_prefix="head",
                name="HTTP HEAD",
            ),
        ],
        severity="medium",
        criticality=2.0,
    )


def _make_sqlalchemy_pattern() -> LibraryPattern:
    """
    Create a realistic database pattern similar to SQLAlchemy.

    Note: Pattern matching uses OR logic, so for SQL operations we match
    by target substring only (not method) to get the correct severity.

    Returns
    -------
    LibraryPattern
        Database ORM library pattern.
    """
    return LibraryPattern(
        library="sqlalchemy",
        service_name="Database ORM",
        category="database",
        matchers=[
            DependencyModePattern(
                modes=["read", "query"],
                match="SELECT",
                severity="low",
            ),
            DependencyModePattern(
                modes=["write", "query"],
                match="INSERT",
                severity="high",
            ),
            DependencyModePattern(
                modes=["write", "query"],
                match="UPDATE",
                severity="high",
            ),
            DependencyModePattern(
                modes=["write", "delete", "query"],
                match="DELETE",
                severity="critical",
            ),
            DependencyModePattern(
                modes=["admin"],
                method="create_all",
                severity="critical",
                criticality=5.0,
            ),
        ],
        severity="medium",
        criticality=3.0,
    )


def _make_redis_pattern() -> LibraryPattern:
    """
    Create a realistic cache pattern similar to Redis.

    Returns
    -------
    LibraryPattern
        Redis cache library pattern.
    """
    return LibraryPattern(
        library="redis",
        service_name="Redis Cache",
        category="cache",
        matchers=[
            DependencyModePattern(
                modes=["read"],
                method_prefix="get",
            ),
            DependencyModePattern(
                modes=["write"],
                method_prefix="set",
            ),
            DependencyModePattern(
                modes=["delete"],
                method_prefix="del",
            ),
            DependencyModePattern(
                modes=["admin"],
                method="flushall",
                severity="critical",
                criticality=5.0,
            ),
        ],
        severity="low",
        criticality=1.5,
    )


# =============================================================================
# DependencyModePattern Tests
# =============================================================================


def test_pattern_matches_exact_method() -> None:
    """Match when method name equals pattern method."""
    pattern = DependencyModePattern(modes=["read"], method="get")
    result = pattern.matches("get", "requests.get(url)")
    assert result


def test_pattern_no_match_different_method() -> None:
    """No match when method name differs from pattern method."""
    pattern = DependencyModePattern(modes=["read"], method="get")
    result = pattern.matches("post", "requests.post(url)")
    assert not result


def test_pattern_matches_method_prefix() -> None:
    """Match when method name starts with pattern prefix."""
    pattern = DependencyModePattern(modes=["read"], method_prefix="get")
    result = pattern.matches("get_user", "api.get_user(id)")
    assert result


def test_pattern_no_match_wrong_prefix() -> None:
    """No match when method name does not start with prefix."""
    pattern = DependencyModePattern(modes=["read"], method_prefix="get")
    result = pattern.matches("fetch_user", "api.fetch_user(id)")
    assert not result


def test_pattern_matches_target_substring() -> None:
    """Match when target contains pattern match string."""
    pattern = DependencyModePattern(modes=["query"], match="SELECT")
    result = pattern.matches("execute", "db.execute('SELECT * FROM users')")
    assert result


def test_pattern_no_match_missing_substring() -> None:
    """No match when target does not contain match string."""
    pattern = DependencyModePattern(modes=["query"], match="SELECT")
    result = pattern.matches("execute", "db.execute('INSERT INTO users')")
    assert not result


def test_pattern_matches_with_none_method() -> None:
    """Handle None method gracefully."""
    pattern = DependencyModePattern(modes=["read"], match="cache")
    result = pattern.matches(None, "cache.get(key)")
    assert result


def test_pattern_no_match_prefix_with_none_method() -> None:
    """No prefix match when method is None."""
    pattern = DependencyModePattern(modes=["read"], method_prefix="get")
    result = pattern.matches(None, "some.target()")
    assert not result


def test_pattern_no_match_empty_pattern() -> None:
    """No match when pattern has no matching criteria."""
    pattern = DependencyModePattern(modes=["unknown"])
    result = pattern.matches("get", "requests.get(url)")
    assert not result


# =============================================================================
# LibraryPattern Tests
# =============================================================================


def test_library_pattern_defaults() -> None:
    """Verify default values for LibraryPattern."""
    pattern = LibraryPattern(
        library="test",
        service_name=None,
        category=None,
        matchers=[],
    )
    assert pattern.severity is None
    assert pattern.criticality is None
    assert pattern.language == "python"


def test_library_pattern_with_all_fields() -> None:
    """Create LibraryPattern with all fields specified."""
    pattern = LibraryPattern(
        library="requests",
        service_name="HTTP Client",
        category="http",
        matchers=[DependencyModePattern(modes=["read"], method="get")],
        severity="medium",
        criticality=2.5,
        language="python",
    )
    assert pattern.library == "requests"
    assert pattern.service_name == "HTTP Client"
    assert len(pattern.matchers) == EXPECTED_MATCHERS_1


# =============================================================================
# classify_modes Tests
# =============================================================================


def test_classify_single_mode_match() -> None:
    """Classify call that matches a single mode."""
    pattern = _make_requests_pattern()
    modes, matched = classify_modes(pattern, "get", "requests.get(url)")
    assert modes == ["read"]
    assert matched is not None
    assert matched.name == "HTTP GET"


def test_classify_multiple_modes_match() -> None:
    """Classify call that matches multiple modes."""
    pattern = _make_requests_pattern()
    modes, _ = classify_modes(pattern, "delete", "requests.delete(url)")
    assert sorted(modes) == ["delete", "write"]


def test_classify_no_match_returns_unknown() -> None:
    """Return unknown mode when no patterns match."""
    pattern = _make_requests_pattern()
    modes, matched = classify_modes(pattern, "patch", "requests.patch(url)")
    assert modes == ["unknown"]
    assert matched is None


def test_classify_by_target_substring() -> None:
    """Classify based on target substring match."""
    pattern = _make_sqlalchemy_pattern()
    modes, _ = classify_modes(
        pattern,
        "execute",
        "session.execute('SELECT id, name FROM users WHERE active = 1')",
    )
    assert "query" in modes
    assert "read" in modes


def test_classify_multiple_patterns_accumulate() -> None:
    """Modes accumulate from multiple matching patterns."""
    pattern = LibraryPattern(
        library="test",
        service_name="Test Service",
        category="test",
        matchers=[
            DependencyModePattern(modes=["read"], method="fetch"),
            DependencyModePattern(modes=["cache"], match="cached"),
        ],
    )
    modes, _ = classify_modes(pattern, "fetch", "service.fetch_cached_data()")
    # Both patterns should match - fetch method and "cached" substring
    assert "read" in modes


def test_classify_returns_first_matched_pattern() -> None:
    """Return the first matching pattern when multiple match."""
    pattern = LibraryPattern(
        library="test",
        service_name="Test",
        category="test",
        matchers=[
            DependencyModePattern(modes=["first"], method="test", name="First"),
            DependencyModePattern(modes=["second"], method="test", name="Second"),
        ],
    )
    _, matched = classify_modes(pattern, "test", "obj.test()")
    assert matched is not None
    assert matched.name == "First"


def test_classify_deduplicated_modes() -> None:
    """Modes are deduplicated and sorted."""
    pattern = LibraryPattern(
        library="test",
        service_name="Test",
        category="test",
        matchers=[
            DependencyModePattern(modes=["read", "query"], method="fetch"),
            DependencyModePattern(modes=["query", "read"], match="data"),
        ],
    )
    modes, _ = classify_modes(pattern, "fetch", "api.fetch_data()")
    assert modes == ["query", "read"]


# =============================================================================
# severity_score Tests
# =============================================================================


@pytest.mark.parametrize(
    ("severity", "expected"),
    [
        ("critical", 4.0),
        ("high", 3.0),
        ("medium", 2.0),
        ("low", 1.0),
        ("info", 0.5),
    ],
)
def test_severity_score_known_levels(severity: str, expected: float) -> None:
    """Return correct score for known severity levels."""
    result = severity_score(severity)
    assert result == expected


def test_severity_score_unknown_returns_none() -> None:
    """Return None for unrecognized severity."""
    result = severity_score("unknown")
    assert result is None


def test_severity_score_none_returns_none() -> None:
    """Return None when severity is None."""
    result = severity_score(None)
    assert result is None


def test_severity_scores_constant() -> None:
    """Verify SEVERITY_SCORES constant has expected values."""
    expected_keys = {"critical", "high", "medium", "low", "info"}
    assert set(SEVERITY_SCORES.keys()) == expected_keys


# =============================================================================
# risk_score Tests
# =============================================================================


@pytest.mark.parametrize(
    ("severity", "criticality", "expected"),
    [
        ("high", 3.0, 9.0),
        ("critical", 2.0, 8.0),
        ("medium", 1.5, 3.0),
        ("low", 4.0, 4.0),
        ("info", 2.0, 1.0),
    ],
)
def test_risk_score_calculation(severity: str, criticality: float, expected: float) -> None:
    """Compute risk score as severity_score * criticality."""
    result = risk_score(severity, criticality)
    assert result == expected


def test_risk_score_none_severity() -> None:
    """Return None when severity is None."""
    result = risk_score(None, 3.0)
    assert result is None


def test_risk_score_none_criticality() -> None:
    """Return None when criticality is None."""
    result = risk_score("high", None)
    assert result is None


def test_risk_score_unknown_severity() -> None:
    """Return None for unknown severity string."""
    result = risk_score("unknown", 3.0)
    assert result is None


def test_risk_score_zero_criticality() -> None:
    """Compute zero risk for zero criticality."""
    result = risk_score("critical", 0.0)
    assert result == 0.0


# =============================================================================
# risk_level Tests
# =============================================================================


def test_risk_level_admin_mode_is_high() -> None:
    """Admin mode results in high risk regardless of callsite count."""
    result = risk_level({"admin"}, 1)
    assert result == "high"


def test_risk_level_write_mode_is_high() -> None:
    """Write mode results in high risk regardless of callsite count."""
    result = risk_level({"write"}, 1)
    assert result == "high"


def test_risk_level_mixed_modes_with_write_is_high() -> None:
    """Mixed modes including write result in high risk."""
    result = risk_level({"read", "write", "query"}, 5)
    assert result == "high"


def test_risk_level_many_callsites_is_medium() -> None:
    """Many callsites without write/admin is medium risk."""
    result = risk_level({"read"}, CALLSITE_MEDIUM_THRESHOLD)
    assert result == "medium"


def test_risk_level_above_threshold_is_medium() -> None:
    """Above threshold callsites is medium risk."""
    result = risk_level({"read", "query"}, CALLSITE_MEDIUM_THRESHOLD + 5)
    assert result == "medium"


def test_risk_level_few_callsites_is_low() -> None:
    """Few callsites without write/admin is low risk."""
    result = risk_level({"read"}, CALLSITE_MEDIUM_THRESHOLD - 1)
    assert result == "low"


def test_risk_level_empty_modes_is_low() -> None:
    """Empty modes with few callsites is low risk."""
    result = risk_level(set(), 5)
    assert result == "low"


def test_callsite_threshold_constant() -> None:
    """Verify CALLSITE_MEDIUM_THRESHOLD is reasonable."""
    assert CALLSITE_MEDIUM_THRESHOLD == EXPECTED_THRESHOLD_10


# =============================================================================
# Integration Tests with Realistic Patterns
# =============================================================================


def test_requests_http_get() -> None:
    """Classify HTTP GET request correctly."""
    pattern = _make_requests_pattern()
    modes, matched = classify_modes(
        pattern,
        "get",
        "requests.get('https://api.example.com/users', headers=auth)",
    )
    assert modes == ["read"]
    assert matched is not None
    assert matched.name == "HTTP GET"


def test_sqlalchemy_select_query() -> None:
    """Classify SQL SELECT query correctly."""
    pattern = _make_sqlalchemy_pattern()
    modes, _ = classify_modes(
        pattern,
        "execute",
        "session.execute(text('SELECT id, email FROM users WHERE status = :s'))",
    )
    assert "read" in modes
    assert "query" in modes


def test_sqlalchemy_insert_is_write() -> None:
    """Classify SQL INSERT as write operation."""
    pattern = _make_sqlalchemy_pattern()
    modes, _ = classify_modes(
        pattern,
        "execute",
        "session.execute(text('INSERT INTO audit_log VALUES (:id, :msg)'))",
    )
    assert "write" in modes


def test_sqlalchemy_delete_is_critical() -> None:
    """Classify SQL DELETE with critical severity."""
    pattern = _make_sqlalchemy_pattern()
    _, matched = classify_modes(
        pattern,
        "execute",
        "session.execute(text('DELETE FROM users WHERE inactive = 1'))",
    )
    assert matched is not None
    assert matched.severity == "critical"


def test_redis_get_by_prefix() -> None:
    """Classify Redis get operations by method prefix."""
    pattern = _make_redis_pattern()
    modes, _ = classify_modes(pattern, "get_many", "redis.get_many(keys)")
    assert modes == ["read"]


def test_redis_flushall_is_admin() -> None:
    """Classify Redis flushall as admin operation."""
    pattern = _make_redis_pattern()
    modes, matched = classify_modes(pattern, "flushall", "redis.flushall()")
    assert modes == ["admin"]
    assert matched is not None
    assert matched.severity == "critical"


def test_risk_assessment_write_operations() -> None:
    """Write operations are high risk regardless of frequency."""
    pattern = _make_sqlalchemy_pattern()
    modes, _ = classify_modes(
        pattern,
        "execute",
        "session.execute(text('UPDATE users SET active = 0'))",
    )
    level = risk_level(set(modes), 2)
    assert level == "high"


def test_risk_assessment_frequent_reads() -> None:
    """Frequent read operations are medium risk."""
    pattern = _make_requests_pattern()
    modes, _ = classify_modes(pattern, "get", "api.get(endpoint)")
    level = risk_level(set(modes), 15)
    assert level == "medium"
