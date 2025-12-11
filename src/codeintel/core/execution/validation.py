"""Generic validation finding utilities.

This module provides type-parameterized utilities for working with
validation findings, including severity overrides, capping findings
per rule, and checking for errors. These utilities are used across
graphs, ingestion, and analytics validation modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

T = TypeVar("T")
SeverityLevel = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class BaseValidationOptions:
    """Base validation options shared across domains.

    Provide common configuration for controlling validation behavior
    including severity overrides and result limits.

    Attributes
    ----------
    severity_overrides
        Mapping of rule/table names to severity levels.
        Use "*" as a key to override all rules.
    hard_fail
        Whether to raise an exception on error-level findings.
    max_findings_per_rule
        Maximum findings to collect per rule (None for unlimited).

    Examples
    --------
    >>> options = BaseValidationOptions(
    ...     severity_overrides={"*": "error"},
    ...     hard_fail=True,
    ... )
    """

    severity_overrides: Mapping[str, SeverityLevel] | None = None
    hard_fail: bool = False
    max_findings_per_rule: int | None = None


def apply_severity_overrides[T](
    findings: Sequence[T],
    overrides: Mapping[str, SeverityLevel] | None,
    get_key: Callable[[T], str],
    set_severity: Callable[[T, SeverityLevel], T],
) -> list[T]:
    """Apply severity overrides to findings.

    Process a sequence of findings and apply severity overrides based on
    a key extracted from each finding. Supports wildcard override via "*".

    Parameters
    ----------
    findings
        Sequence of finding objects to process.
    overrides
        Mapping of keys to severity levels, or None to skip.
        Use "*" as a key to override all findings.
    get_key
        Function to extract the key from a finding (e.g., table name).
    set_severity
        Function to create a new finding with updated severity.

    Returns
    -------
    list[T]
        List of findings with severity overrides applied.

    Examples
    --------
    >>> @dataclass
    ... class Finding:
    ...     table: str
    ...     severity: str
    >>> findings = [Finding("users", "warning"), Finding("orders", "info")]
    >>> result = apply_severity_overrides(
    ...     findings,
    ...     {"users": "error", "*": "warning"},
    ...     get_key=lambda f: f.table,
    ...     set_severity=lambda f, s: Finding(f.table, s),
    ... )
    """
    if not overrides:
        return list(findings)

    result: list[T] = []
    for finding in findings:
        key = get_key(finding)
        # Check specific override first, then wildcard
        override = overrides.get(key) or overrides.get("*")
        if override is None:
            result.append(finding)
        else:
            result.append(set_severity(finding, override))
    return result


def cap_findings[T](
    findings: Sequence[T],
    max_per_rule: int | None,
    get_key: Callable[[T], str],
) -> list[T]:
    """Cap the number of findings per rule/category.

    Limit the number of findings returned per unique key to prevent
    overwhelming output with many similar findings.

    Parameters
    ----------
    findings
        Sequence of finding objects to cap.
    max_per_rule
        Maximum findings per key (None or <= 0 for unlimited).
    get_key
        Function to extract the grouping key from a finding.

    Returns
    -------
    list[T]
        List of findings with counts capped per key.

    Examples
    --------
    >>> findings = [
    ...     {"rule": "A", "msg": "1"},
    ...     {"rule": "A", "msg": "2"},
    ...     {"rule": "A", "msg": "3"},
    ...     {"rule": "B", "msg": "1"},
    ... ]
    >>> capped = cap_findings(findings, 2, lambda f: f["rule"])
    >>> len([f for f in capped if f["rule"] == "A"])
    2
    """
    if max_per_rule is None or max_per_rule <= 0:
        return list(findings)

    counts: dict[str, int] = {}
    result: list[T] = []

    for finding in findings:
        key = get_key(finding)
        current = counts.get(key, 0)
        if current < max_per_rule:
            result.append(finding)
            counts[key] = current + 1

    return result


def has_error_findings[T](
    findings: Sequence[T],
    get_severity: Callable[[T], str],
) -> bool:
    """Check if any findings have error severity.

    Parameters
    ----------
    findings
        Sequence of finding objects to check.
    get_severity
        Function to get the severity from a finding.

    Returns
    -------
    bool
        True if any finding has error severity.

    Examples
    --------
    >>> findings = [{"severity": "warning"}, {"severity": "error"}]
    >>> has_error_findings(findings, lambda f: f["severity"])
    True
    """
    return any(get_severity(f) == "error" for f in findings)


def filter_by_severity[T](
    findings: Sequence[T],
    min_severity: SeverityLevel,
    get_severity: Callable[[T], str],
) -> list[T]:
    """Filter findings to only include those at or above a minimum severity.

    Parameters
    ----------
    findings
        Sequence of finding objects to filter.
    min_severity
        Minimum severity level to include.
    get_severity
        Function to get the severity from a finding.

    Returns
    -------
    list[T]
        Findings at or above the minimum severity level.

    Examples
    --------
    >>> findings = [
    ...     {"severity": "info"},
    ...     {"severity": "warning"},
    ...     {"severity": "error"},
    ... ]
    >>> filtered = filter_by_severity(findings, "warning", lambda f: f["severity"])
    >>> len(filtered)
    2
    """
    severity_order = {"info": 0, "warning": 1, "error": 2}
    min_level = severity_order.get(min_severity, 0)

    return [f for f in findings if severity_order.get(get_severity(f), 0) >= min_level]


def group_findings_by_key[T](
    findings: Sequence[T],
    get_key: Callable[[T], str],
) -> dict[str, list[T]]:
    """Group findings by a key.

    Parameters
    ----------
    findings
        Sequence of finding objects to group.
    get_key
        Function to extract the grouping key from a finding.

    Returns
    -------
    dict[str, list[T]]
        Dictionary mapping keys to lists of findings.

    Examples
    --------
    >>> findings = [
    ...     {"rule": "A", "msg": "1"},
    ...     {"rule": "B", "msg": "2"},
    ...     {"rule": "A", "msg": "3"},
    ... ]
    >>> grouped = group_findings_by_key(findings, lambda f: f["rule"])
    >>> len(grouped["A"])
    2
    """
    result: dict[str, list[T]] = {}
    for finding in findings:
        key = get_key(finding)
        if key not in result:
            result[key] = []
        result[key].append(finding)
    return result


__all__ = [
    "BaseValidationOptions",
    "SeverityLevel",
    "apply_severity_overrides",
    "cap_findings",
    "filter_by_severity",
    "group_findings_by_key",
    "has_error_findings",
]
