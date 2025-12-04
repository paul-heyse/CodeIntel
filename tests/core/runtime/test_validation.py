"""Test validation utilities from codeintel.core.runtime.validation.

This module tests:
- BaseValidationOptions defaults
- apply_severity_overrides() with wildcards
- cap_findings() per-rule limits
- has_error_findings() detection
- filter_by_severity() threshold filtering
- group_findings_by_key() grouping
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from codeintel.core.runtime.validation import (
    BaseValidationOptions,
    SeverityLevel,
    apply_severity_overrides,
    cap_findings,
    filter_by_severity,
    group_findings_by_key,
    has_error_findings,
)

# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


@dataclass
class TestFinding:
    """Simple finding dataclass for testing."""

    rule: str
    message: str
    severity: str = "warning"


def get_rule(finding: TestFinding) -> str:
    """Extract rule from finding."""
    return finding.rule


def get_severity(finding: TestFinding) -> str:
    """Extract severity from finding."""
    return finding.severity


def set_severity(finding: TestFinding, severity: SeverityLevel) -> TestFinding:
    """Create new finding with updated severity."""
    return replace(finding, severity=severity)


# =============================================================================
# BaseValidationOptions Tests
# =============================================================================


def test_base_validation_options_defaults() -> None:
    """Verify BaseValidationOptions default values."""
    options = BaseValidationOptions()

    assert options.severity_overrides is None
    assert options.hard_fail is False
    assert options.max_findings_per_rule is None


def test_base_validation_options_custom() -> None:
    """Verify BaseValidationOptions accepts custom values."""
    options = BaseValidationOptions(
        severity_overrides={"table_a": "error", "*": "warning"},
        hard_fail=True,
        max_findings_per_rule=10,
    )

    assert options.severity_overrides == {"table_a": "error", "*": "warning"}
    assert options.hard_fail is True
    assert options.max_findings_per_rule == 10


def test_base_validation_options_is_frozen() -> None:
    """Verify BaseValidationOptions is immutable."""
    options = BaseValidationOptions()

    with pytest.raises(AttributeError):
        options.hard_fail = True  # type: ignore[misc]


# =============================================================================
# apply_severity_overrides Tests
# =============================================================================


def test_apply_severity_overrides_no_overrides() -> None:
    """Verify apply_severity_overrides returns copy when no overrides."""
    findings = [
        TestFinding(rule="A", message="msg1", severity="info"),
        TestFinding(rule="B", message="msg2", severity="warning"),
    ]

    result = apply_severity_overrides(
        findings,
        overrides=None,
        get_key=get_rule,
        set_severity=set_severity,
    )

    assert len(result) == 2
    assert result[0].severity == "info"
    assert result[1].severity == "warning"


def test_apply_severity_overrides_empty_overrides() -> None:
    """Verify apply_severity_overrides handles empty overrides dict."""
    findings = [TestFinding(rule="A", message="msg", severity="info")]

    result = apply_severity_overrides(
        findings,
        overrides={},
        get_key=get_rule,
        set_severity=set_severity,
    )

    assert result[0].severity == "info"


def test_apply_severity_overrides_specific_rule() -> None:
    """Verify apply_severity_overrides applies specific rule override."""
    findings = [
        TestFinding(rule="A", message="msg1", severity="info"),
        TestFinding(rule="B", message="msg2", severity="info"),
    ]

    result = apply_severity_overrides(
        findings,
        overrides={"A": "error"},
        get_key=get_rule,
        set_severity=set_severity,
    )

    assert result[0].severity == "error"  # A overridden
    assert result[1].severity == "info"  # B unchanged


def test_apply_severity_overrides_wildcard() -> None:
    """Verify apply_severity_overrides applies wildcard override."""
    findings = [
        TestFinding(rule="A", message="msg1", severity="info"),
        TestFinding(rule="B", message="msg2", severity="info"),
    ]

    result = apply_severity_overrides(
        findings,
        overrides={"*": "warning"},
        get_key=get_rule,
        set_severity=set_severity,
    )

    assert result[0].severity == "warning"
    assert result[1].severity == "warning"


def test_apply_severity_overrides_specific_over_wildcard() -> None:
    """Verify specific override takes precedence over wildcard."""
    findings = [
        TestFinding(rule="A", message="msg1", severity="info"),
        TestFinding(rule="B", message="msg2", severity="info"),
    ]

    result = apply_severity_overrides(
        findings,
        overrides={"A": "error", "*": "warning"},
        get_key=get_rule,
        set_severity=set_severity,
    )

    assert result[0].severity == "error"  # Specific override
    assert result[1].severity == "warning"  # Wildcard override


def test_apply_severity_overrides_preserves_message() -> None:
    """Verify apply_severity_overrides preserves other fields."""
    findings = [TestFinding(rule="A", message="original_message", severity="info")]

    result = apply_severity_overrides(
        findings,
        overrides={"A": "error"},
        get_key=get_rule,
        set_severity=set_severity,
    )

    assert result[0].message == "original_message"


# =============================================================================
# cap_findings Tests
# =============================================================================


def test_cap_findings_no_limit() -> None:
    """Verify cap_findings returns all when no limit set."""
    findings = [
        TestFinding(rule="A", message=f"msg{i}")
        for i in range(10)
    ]

    result = cap_findings(findings, max_per_rule=None, get_key=get_rule)

    assert len(result) == 10


def test_cap_findings_zero_limit() -> None:
    """Verify cap_findings returns all when limit is zero."""
    findings = [
        TestFinding(rule="A", message=f"msg{i}")
        for i in range(5)
    ]

    result = cap_findings(findings, max_per_rule=0, get_key=get_rule)

    assert len(result) == 5


def test_cap_findings_negative_limit() -> None:
    """Verify cap_findings returns all when limit is negative."""
    findings = [
        TestFinding(rule="A", message=f"msg{i}")
        for i in range(5)
    ]

    result = cap_findings(findings, max_per_rule=-1, get_key=get_rule)

    assert len(result) == 5


def test_cap_findings_caps_single_rule() -> None:
    """Verify cap_findings limits findings for a single rule."""
    findings = [
        TestFinding(rule="A", message=f"msg{i}")
        for i in range(10)
    ]

    result = cap_findings(findings, max_per_rule=3, get_key=get_rule)

    assert len(result) == 3
    assert all(f.rule == "A" for f in result)


def test_cap_findings_caps_per_rule() -> None:
    """Verify cap_findings limits findings per rule independently."""
    findings = [
        TestFinding(rule="A", message="a1"),
        TestFinding(rule="A", message="a2"),
        TestFinding(rule="A", message="a3"),
        TestFinding(rule="B", message="b1"),
        TestFinding(rule="B", message="b2"),
        TestFinding(rule="B", message="b3"),
    ]

    result = cap_findings(findings, max_per_rule=2, get_key=get_rule)

    a_findings = [f for f in result if f.rule == "A"]
    b_findings = [f for f in result if f.rule == "B"]

    assert len(a_findings) == 2
    assert len(b_findings) == 2


def test_cap_findings_preserves_order() -> None:
    """Verify cap_findings preserves finding order."""
    findings = [
        TestFinding(rule="A", message="first"),
        TestFinding(rule="A", message="second"),
        TestFinding(rule="A", message="third"),
    ]

    result = cap_findings(findings, max_per_rule=2, get_key=get_rule)

    assert result[0].message == "first"
    assert result[1].message == "second"


# =============================================================================
# has_error_findings Tests
# =============================================================================


def test_has_error_findings_empty() -> None:
    """Verify has_error_findings returns False for empty list."""
    result = has_error_findings([], get_severity)
    assert result is False


def test_has_error_findings_no_errors() -> None:
    """Verify has_error_findings returns False when no errors present."""
    findings = [
        TestFinding(rule="A", message="msg", severity="info"),
        TestFinding(rule="B", message="msg", severity="warning"),
    ]

    result = has_error_findings(findings, get_severity)
    assert result is False


def test_has_error_findings_with_error() -> None:
    """Verify has_error_findings returns True when error present."""
    findings = [
        TestFinding(rule="A", message="msg", severity="info"),
        TestFinding(rule="B", message="msg", severity="error"),
    ]

    result = has_error_findings(findings, get_severity)
    assert result is True


def test_has_error_findings_all_errors() -> None:
    """Verify has_error_findings returns True when all are errors."""
    findings = [
        TestFinding(rule="A", message="msg", severity="error"),
        TestFinding(rule="B", message="msg", severity="error"),
    ]

    result = has_error_findings(findings, get_severity)
    assert result is True


# =============================================================================
# filter_by_severity Tests
# =============================================================================


def test_filter_by_severity_info_threshold() -> None:
    """Verify filter_by_severity with 'info' includes all."""
    findings = [
        TestFinding(rule="A", message="msg", severity="info"),
        TestFinding(rule="B", message="msg", severity="warning"),
        TestFinding(rule="C", message="msg", severity="error"),
    ]

    result = filter_by_severity(findings, "info", get_severity)
    assert len(result) == 3


def test_filter_by_severity_warning_threshold() -> None:
    """Verify filter_by_severity with 'warning' excludes info."""
    findings = [
        TestFinding(rule="A", message="msg", severity="info"),
        TestFinding(rule="B", message="msg", severity="warning"),
        TestFinding(rule="C", message="msg", severity="error"),
    ]

    result = filter_by_severity(findings, "warning", get_severity)
    assert len(result) == 2
    assert all(f.severity in {"warning", "error"} for f in result)


def test_filter_by_severity_error_threshold() -> None:
    """Verify filter_by_severity with 'error' only includes errors."""
    findings = [
        TestFinding(rule="A", message="msg", severity="info"),
        TestFinding(rule="B", message="msg", severity="warning"),
        TestFinding(rule="C", message="msg", severity="error"),
    ]

    result = filter_by_severity(findings, "error", get_severity)
    assert len(result) == 1
    assert result[0].severity == "error"


def test_filter_by_severity_empty_list() -> None:
    """Verify filter_by_severity handles empty list."""
    result = filter_by_severity([], "warning", get_severity)
    assert result == []


def test_filter_by_severity_preserves_order() -> None:
    """Verify filter_by_severity preserves finding order."""
    findings = [
        TestFinding(rule="A", message="first", severity="warning"),
        TestFinding(rule="B", message="second", severity="error"),
        TestFinding(rule="C", message="third", severity="warning"),
    ]

    result = filter_by_severity(findings, "warning", get_severity)

    assert result[0].message == "first"
    assert result[1].message == "second"
    assert result[2].message == "third"


# =============================================================================
# group_findings_by_key Tests
# =============================================================================


def test_group_findings_by_key_empty() -> None:
    """Verify group_findings_by_key handles empty list."""
    result = group_findings_by_key([], get_rule)
    assert result == {}


def test_group_findings_by_key_single_group() -> None:
    """Verify group_findings_by_key with single key."""
    findings = [
        TestFinding(rule="A", message="msg1"),
        TestFinding(rule="A", message="msg2"),
    ]

    result = group_findings_by_key(findings, get_rule)

    assert len(result) == 1
    assert "A" in result
    assert len(result["A"]) == 2


def test_group_findings_by_key_multiple_groups() -> None:
    """Verify group_findings_by_key with multiple keys."""
    findings = [
        TestFinding(rule="A", message="a1"),
        TestFinding(rule="B", message="b1"),
        TestFinding(rule="A", message="a2"),
        TestFinding(rule="C", message="c1"),
        TestFinding(rule="B", message="b2"),
    ]

    result = group_findings_by_key(findings, get_rule)

    assert len(result) == 3
    assert len(result["A"]) == 2
    assert len(result["B"]) == 2
    assert len(result["C"]) == 1


def test_group_findings_by_key_preserves_order() -> None:
    """Verify group_findings_by_key preserves order within groups."""
    findings = [
        TestFinding(rule="A", message="first"),
        TestFinding(rule="A", message="second"),
        TestFinding(rule="A", message="third"),
    ]

    result = group_findings_by_key(findings, get_rule)

    assert result["A"][0].message == "first"
    assert result["A"][1].message == "second"
    assert result["A"][2].message == "third"


def test_group_findings_by_key_with_severity() -> None:
    """Verify group_findings_by_key can group by severity."""
    findings = [
        TestFinding(rule="A", message="msg1", severity="info"),
        TestFinding(rule="B", message="msg2", severity="error"),
        TestFinding(rule="C", message="msg3", severity="info"),
    ]

    result = group_findings_by_key(findings, get_severity)

    assert len(result) == 2
    assert len(result["info"]) == 2
    assert len(result["error"]) == 1


# =============================================================================
# Integration Tests
# =============================================================================


def test_combined_override_and_filter() -> None:
    """Verify overrides and filtering work together."""
    findings = [
        TestFinding(rule="important", message="msg1", severity="warning"),
        TestFinding(rule="minor", message="msg2", severity="info"),
        TestFinding(rule="important", message="msg3", severity="warning"),
    ]

    # First apply overrides
    overridden = apply_severity_overrides(
        findings,
        overrides={"important": "error"},
        get_key=get_rule,
        set_severity=set_severity,
    )

    # Then filter by severity
    filtered = filter_by_severity(overridden, "warning", get_severity)

    assert len(filtered) == 2  # Both "important" findings (now error level)
    assert all(f.rule == "important" for f in filtered)


def test_combined_cap_and_group() -> None:
    """Verify capping and grouping work together."""
    findings = [
        TestFinding(rule="A", message=f"a{i}")
        for i in range(5)
    ] + [
        TestFinding(rule="B", message=f"b{i}")
        for i in range(5)
    ]

    # First cap findings
    capped = cap_findings(findings, max_per_rule=2, get_key=get_rule)

    # Then group by key
    grouped = group_findings_by_key(capped, get_rule)

    assert len(grouped["A"]) == 2
    assert len(grouped["B"]) == 2


def test_full_pipeline() -> None:
    """Verify full validation pipeline: override -> cap -> filter -> check errors."""
    findings = [
        TestFinding(rule="critical", message="c1", severity="warning"),
        TestFinding(rule="critical", message="c2", severity="warning"),
        TestFinding(rule="critical", message="c3", severity="warning"),
        TestFinding(rule="minor", message="m1", severity="info"),
        TestFinding(rule="minor", message="m2", severity="info"),
    ]

    # 1. Apply severity overrides (make critical -> error)
    step1 = apply_severity_overrides(
        findings,
        overrides={"critical": "error"},
        get_key=get_rule,
        set_severity=set_severity,
    )

    # 2. Cap findings per rule
    step2 = cap_findings(step1, max_per_rule=2, get_key=get_rule)

    # 3. Filter to warning+
    step3 = filter_by_severity(step2, "warning", get_severity)

    # 4. Check for errors
    has_errors = has_error_findings(step3, get_severity)

    # Verify pipeline results
    assert len(step3) == 2  # Only critical findings (capped to 2)
    assert all(f.severity == "error" for f in step3)
    assert has_errors is True
