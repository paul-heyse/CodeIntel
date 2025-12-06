"""Generic validation finding types and helper functions.

This module provides the common validation finding structure and helper
functions used by both graph validation and ingestion validation frameworks.

The helpers are designed to work with generic finding dictionaries that have
at minimum a ``check_name`` (or similar identifier) and ``severity`` field.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from codeintel.core.validation.options import ValidationSeverity


def apply_severity_overrides[T: Mapping[str, object]](
    findings: Sequence[T],
    overrides: Mapping[str, ValidationSeverity] | None,
    *,
    check_key: str = "check_name",
) -> list[T]:
    """Apply severity overrides to findings.

    Update the severity of findings based on the provided overrides mapping.
    Supports both specific rule overrides and a wildcard "*" to override all.

    Parameters
    ----------
    findings
        Sequence of findings to process. Each finding must be a mapping
        with at least a check identifier and severity field.
    overrides
        Mapping of rule names to severity levels. Use "*" as a key
        to override all rules.
    check_key
        Key used to identify the check/rule name in findings.

    Returns
    -------
    list[T]
        Findings with severity overrides applied. Original findings
        are not modified; new dict copies are returned where needed.

    Examples
    --------
    >>> findings = [{"check_name": "null_check", "severity": "warning"}]
    >>> apply_severity_overrides(findings, {"null_check": "error"})
    [{'check_name': 'null_check', 'severity': 'error'}]
    >>> apply_severity_overrides(findings, {"*": "error"})
    [{'check_name': 'null_check', 'severity': 'error'}]
    """
    if not overrides:
        return list(findings)

    normalized: list[T] = []
    for finding in findings:
        check = str(finding.get(check_key) or "")
        override = overrides.get(check)
        if override is None:
            override = overrides.get("*")
        if override is None:
            normalized.append(finding)
            continue
        # Create updated copy
        updated = dict(finding)
        updated["severity"] = override
        # Type checker sees dict, but the structure matches T
        normalized.append(updated)  # type: ignore[arg-type]
    return normalized


def cap_findings[T: Mapping[str, object]](
    findings: Sequence[T],
    max_per_rule: int | None,
    *,
    check_key: str = "check_name",
) -> list[T]:
    """Cap the number of findings per rule.

    Limit the number of findings returned for each distinct check/rule
    to avoid overwhelming output when many violations occur.

    Parameters
    ----------
    findings
        Sequence of findings to cap.
    max_per_rule
        Maximum findings per check name. None or <= 0 for unlimited.
    check_key
        Key used to identify the check/rule name in findings.

    Returns
    -------
    list[T]
        Capped list of findings, preserving original order.

    Examples
    --------
    >>> findings = [
    ...     {"check_name": "null", "id": 1},
    ...     {"check_name": "null", "id": 2},
    ...     {"check_name": "null", "id": 3},
    ... ]
    >>> cap_findings(findings, 2)
    [{'check_name': 'null', 'id': 1}, {'check_name': 'null', 'id': 2}]
    """
    if max_per_rule is None or max_per_rule <= 0:
        return list(findings)

    counts: dict[str, int] = {}
    capped: list[T] = []
    for finding in findings:
        check = str(finding.get(check_key) or "")
        seen = counts.get(check, 0)
        if seen >= max_per_rule:
            continue
        counts[check] = seen + 1
        capped.append(finding)
    return capped


def has_error_findings(
    findings: Sequence[Mapping[str, object]],
    *,
    severity_key: str = "severity",
    error_value: str = "error",
) -> bool:
    """Check if any findings have error severity.

    Parameters
    ----------
    findings
        Sequence of findings to check.
    severity_key
        Key used to identify the severity field in findings.
    error_value
        Value that indicates error severity.

    Returns
    -------
    bool
        True if any finding has error severity.

    Examples
    --------
    >>> findings = [{"severity": "warning"}, {"severity": "error"}]
    >>> has_error_findings(findings)
    True
    >>> has_error_findings([{"severity": "warning"}])
    False
    """
    return any(finding.get(severity_key) == error_value for finding in findings)


__all__ = [
    "apply_severity_overrides",
    "cap_findings",
    "has_error_findings",
]
