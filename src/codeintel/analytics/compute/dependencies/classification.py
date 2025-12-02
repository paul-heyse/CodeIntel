"""Pure computation for dependency classification.

This module provides functions to classify dependency calls by mode,
severity, and risk level. All functions are pure and side-effect-free.

Examples
--------
>>> from codeintel.analytics.compute.dependencies.classification import risk_score
>>> risk_score("high", 3.0)
9.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

SEVERITY_SCORES: Final[dict[str, float]] = {
    "critical": 4.0,
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
    "info": 0.5,
}

CALLSITE_MEDIUM_THRESHOLD: Final[int] = 10


@dataclass(frozen=True)
class DependencyModePattern:
    """Classification rule for a dependency call.

    Attributes
    ----------
    modes
        List of modes this pattern matches (e.g., ["read", "query"]).
    method
        Exact method name to match.
    method_prefix
        Method name prefix to match.
    match
        Substring to match in the call target.
    severity
        Override severity for matched calls.
    criticality
        Override criticality for matched calls.
    name
        Human-readable name for the pattern.
    """

    modes: list[str]
    method: str | None = None
    method_prefix: str | None = None
    match: str | None = None
    severity: str | None = None
    criticality: float | None = None
    name: str | None = None

    def matches(self, method: str | None, target: str) -> bool:
        """Check if this pattern matches the given method and target.

        Parameters
        ----------
        method
            Method name being called.
        target
            Full call target string.

        Returns
        -------
        bool
            True if pattern matches.
        """
        method_matches = self.method is not None and method == self.method
        prefix_matches = (
            self.method_prefix is not None
            and method is not None
            and method.startswith(self.method_prefix)
        )
        target_matches = self.match is not None and self.match in target

        return method_matches or prefix_matches or target_matches


@dataclass(frozen=True)
class LibraryPattern:
    """Pattern bundle for a specific library.

    Attributes
    ----------
    library
        Library identifier (e.g., "requests", "sqlalchemy").
    service_name
        Human-readable service name.
    category
        Dependency category (e.g., "http", "database").
    matchers
        List of mode patterns for this library.
    severity
        Default severity for unmatched calls.
    criticality
        Default criticality for unmatched calls.
    language
        Programming language (default: "python").
    """

    library: str
    service_name: str | None
    category: str | None
    matchers: list[DependencyModePattern]
    severity: str | None = None
    criticality: float | None = None
    language: str = "python"


def classify_modes(
    pattern: LibraryPattern,
    method: str | None,
    target: str,
) -> tuple[list[str], DependencyModePattern | None]:
    """Classify the usage modes for a dependency call.

    Match the method name and target string against the library's
    matchers to determine which modes apply.

    Parameters
    ----------
    pattern
        Library pattern with matchers.
    method
        Method name being called (e.g., "get", "execute").
    target
        Full call target string.

    Returns
    -------
    tuple[list[str], DependencyModePattern | None]
        List of matched modes and the first matching pattern (if any).

    Examples
    --------
    >>> pattern = LibraryPattern(
    ...     library="requests",
    ...     service_name="HTTP Client",
    ...     category="http",
    ...     matchers=[
    ...         DependencyModePattern(modes=["read"], method="get"),
    ...         DependencyModePattern(modes=["write"], method="post"),
    ...     ],
    ... )
    >>> modes, matched = classify_modes(pattern, "get", "requests.get(url)")
    >>> modes
    ['read']
    """
    modes: list[str] = []
    matched_pattern: DependencyModePattern | None = None

    for matcher in pattern.matchers:
        if matcher.matches(method, target):
            modes.extend(matcher.modes)
            matched_pattern = matched_pattern or matcher

    return sorted(set(modes)) if modes else ["unknown"], matched_pattern


def severity_score(severity: str | None) -> float | None:
    """Convert severity string to numeric score.

    Parameters
    ----------
    severity
        Severity level string (critical, high, medium, low, info).

    Returns
    -------
    float | None
        Numeric score or None if severity is not recognized.

    Examples
    --------
    >>> severity_score("high")
    3.0
    >>> severity_score("unknown") is None
    True
    """
    if severity is None:
        return None
    return SEVERITY_SCORES.get(severity)


def risk_score(severity: str | None, criticality: float | None) -> float | None:
    """Compute risk score from severity and criticality.

    Parameters
    ----------
    severity
        Severity level string.
    criticality
        Criticality multiplier (typically 1.0-5.0).

    Returns
    -------
    float | None
        Risk score (severity_score * criticality) or None if either is missing.

    Examples
    --------
    >>> risk_score("high", 3.0)
    9.0
    >>> risk_score("low", None) is None
    True
    """
    base = severity_score(severity)
    if base is None:
        return None
    if criticality is None:
        return None
    return base * criticality


def risk_level(modes: set[str], callsite_count: int) -> str:
    """Determine risk level from usage patterns.

    Parameters
    ----------
    modes
        Set of usage modes for the dependency.
    callsite_count
        Number of call sites using this dependency.

    Returns
    -------
    str
        Risk level: "high", "medium", or "low".

    Examples
    --------
    >>> risk_level({"write", "query"}, 5)
    'high'
    >>> risk_level({"read"}, 5)
    'low'
    >>> risk_level({"read"}, 15)
    'medium'
    """
    if "admin" in modes or "write" in modes:
        return "high"
    if callsite_count >= CALLSITE_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


__all__ = [
    "CALLSITE_MEDIUM_THRESHOLD",
    "SEVERITY_SCORES",
    "DependencyModePattern",
    "LibraryPattern",
    "classify_modes",
    "risk_level",
    "risk_score",
    "severity_score",
]
