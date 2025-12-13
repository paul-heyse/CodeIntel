"""Dependency analysis row types and utilities.

This module provides data types for external dependency tracking tables:
- DependencyCallRow for analytics.external_dependency_calls
- DependencyAggregateRow for analytics.external_dependencies

These types were originally in analytics.adapters.dependencies and were
extracted to support direct usage without the deprecated adapter layer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True)
class DependencyCallRow:
    """Row for external_dependency_calls table.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    dep_id
        Unique dependency identifier.
    library
        Library name.
    service_name
        Human-readable service name.
    function_goid_h128
        Function global ID (as Decimal for DuckDB hugeint).
    function_urn
        Function URN.
    rel_path
        Relative source file path.
    module
        Module name.
    qualname
        Fully qualified function name.
    callsite_count
        Number of call sites.
    modes
        List of usage modes.
    evidence_json
        Evidence data as list of dicts.
    created_at
        Row creation timestamp.
    """

    repo: str
    commit: str
    dep_id: str
    library: str
    service_name: str
    function_goid_h128: Decimal
    function_urn: str
    rel_path: str
    module: str
    qualname: str
    callsite_count: int
    modes: list[str]
    evidence_json: list[dict[str, object]]
    created_at: datetime


@dataclass(frozen=True)
class DependencyAggregateRow:
    """Row for external_dependencies table.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    dep_id
        Unique dependency identifier.
    library
        Library name.
    service_name
        Human-readable service name.
    category
        Dependency category.
    language
        Programming language.
    severity
        Severity level.
    criticality
        Criticality score.
    risk_score
        Computed risk score.
    function_count
        Number of functions using this dependency.
    callsite_count
        Total call sites.
    modules_json
        List of modules using this dependency.
    usage_modes
        List of usage modes.
    config_keys
        List of related config keys.
    risk_level
        Risk level classification.
    created_at
        Row creation timestamp.
    """

    repo: str
    commit: str
    dep_id: str
    library: str
    service_name: str
    category: str | None
    language: str
    severity: str | None
    criticality: float | None
    risk_score: float | None
    function_count: int
    callsite_count: int
    modules_json: list[str]
    usage_modes: list[str]
    config_keys: list[str]
    risk_level: str
    created_at: datetime


def compute_dep_id(repo: str, commit: str, library: str) -> str:
    """Compute unique dependency identifier.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    library
        Library name.

    Returns
    -------
    str
        SHA-1 hash prefix as dependency ID.
    """
    raw = f"{repo}:{commit}:{library}"
    return hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def to_decimal(value: int) -> Decimal:
    """Convert integer to Decimal for DuckDB hugeint.

    Parameters
    ----------
    value
        Integer value.

    Returns
    -------
    Decimal
        Decimal representation.
    """
    return Decimal(value)


__all__ = [
    "DependencyAggregateRow",
    "DependencyCallRow",
    "compute_dep_id",
    "to_decimal",
]
