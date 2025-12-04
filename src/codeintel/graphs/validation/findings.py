"""Finding types, persistence, and severity handling for graph validation.

This module provides the core data structures and utilities for working
with validation findings, including persistence to the analytics schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from codeintel.analytics.parsing.validation import GraphValidationReporter
from codeintel.analytics.runtime import GraphRuntime, GraphRuntimeOptions
from codeintel.storage.sql_builder import ensure_schema

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# =============================================================================
# Constants
# =============================================================================

SAMPLE_LIMIT = 5
SYMBOL_COMMUNITY_MIN = 2
CONFIG_KEY_MIN_THRESHOLD = 2
HUB_MIN_DEGREE_FLOOR = 10
HUB_DEGREE_RATIO = 0.1
CALL_SCC_MIN = 5


# =============================================================================
# Options
# =============================================================================


@dataclass(frozen=True)
class GraphValidationOptions:
    """Optional controls for graph validation behavior."""

    severity_overrides: Mapping[str, Literal["info", "warning", "error"]] | None = None
    hard_fail: bool = False
    max_findings_per_rule: int | None = None


def resolve_validation_options(
    runtime: GraphRuntime | GraphRuntimeOptions,
    options: GraphValidationOptions | None,
) -> GraphValidationOptions:
    """
    Determine effective validation options, applying runtime feature flags.

    Parameters
    ----------
    runtime : GraphRuntime | GraphRuntimeOptions
        Runtime or options containing feature flags.
    options : GraphValidationOptions | None
        Explicit options to use if provided.

    Returns
    -------
    GraphValidationOptions
        Options merged with any feature flag overrides.
    """
    if options is not None:
        return options
    features = runtime.options.features if isinstance(runtime, GraphRuntime) else runtime.features
    strict = features.validation_strict if features is not None else None
    if strict:
        return GraphValidationOptions(severity_overrides={"*": "error"}, hard_fail=True)
    return GraphValidationOptions()


# =============================================================================
# Finding Helpers
# =============================================================================


def hub_threshold(node_count: int) -> int:
    """
    Compute a hub threshold that scales with graph size.

    Parameters
    ----------
    node_count : int
        Number of nodes in the graph.

    Returns
    -------
    int
        Degree threshold used to flag hubs.
    """
    return max(HUB_MIN_DEGREE_FLOOR, int(node_count * HUB_DEGREE_RATIO))


def apply_severity_overrides(
    findings: list[dict[str, object]],
    overrides: Mapping[str, Literal["info", "warning", "error"]] | None,
) -> list[dict[str, object]]:
    """
    Apply severity overrides to findings.

    Parameters
    ----------
    findings : list[dict[str, object]]
        List of findings to process.
    overrides : Mapping[str, Literal["info", "warning", "error"]] | None
        Severity overrides by check name, or "*" for all.

    Returns
    -------
    list[dict[str, object]]
        Findings with severity overrides applied.
    """
    if not overrides:
        return findings
    normalized: list[dict[str, object]] = []
    for finding in findings:
        check = str(finding.get("check_name") or "")
        override = overrides.get(check) if overrides else None
        if override is None:
            override = overrides.get("*") if overrides else None
        if override is None:
            normalized.append(finding)
            continue
        updated = dict(finding)
        updated["severity"] = override
        normalized.append(updated)
    return normalized


def cap_findings(
    findings: list[dict[str, object]], max_per_rule: int | None
) -> list[dict[str, object]]:
    """
    Cap the number of findings per rule.

    Parameters
    ----------
    findings : list[dict[str, object]]
        List of findings to cap.
    max_per_rule : int | None
        Maximum findings per check name. None for unlimited.

    Returns
    -------
    list[dict[str, object]]
        Capped list of findings.
    """
    if max_per_rule is None or max_per_rule <= 0:
        return findings
    counts: dict[str, int] = {}
    capped: list[dict[str, object]] = []
    for finding in findings:
        check = str(finding.get("check_name") or "")
        seen = counts.get(check, 0)
        if seen >= max_per_rule:
            continue
        counts[check] = seen + 1
        capped.append(finding)
    return capped


def has_error_findings(findings: list[dict[str, object]]) -> bool:
    """
    Check if any findings have error severity.

    Parameters
    ----------
    findings : list[dict[str, object]]
        List of findings to check.

    Returns
    -------
    bool
        True if any finding has error severity.
    """
    return any(finding.get("severity") == "error" for finding in findings)


# =============================================================================
# Persistence
# =============================================================================


def persist_findings(
    gateway: StorageGateway, findings: list[dict[str, object]], repo: str, commit: str
) -> None:
    """
    Persist validation findings to the analytics schema.

    Parameters
    ----------
    gateway : StorageGateway
        Storage gateway for database access.
    findings : list[dict[str, object]]
        List of findings to persist.
    repo : str
        Repository identifier.
    commit : str
        Commit identifier.
    """
    if not findings:
        return
    con = gateway.con
    ensure_schema(con, "analytics.graph_validation")
    con.execute(
        "DELETE FROM analytics.graph_validation WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    reporter = GraphValidationReporter(repo=repo, commit=commit)
    for finding in findings:
        graph_name = str(finding.get("check_name") or "graph_validation")
        entity_ref = finding.get("path") or finding.get("entity_id") or finding.get("graph_name")
        entity_id = str(entity_ref) if entity_ref is not None else graph_name
        issue = str(finding.get("issue") or finding.get("severity") or graph_name)
        severity = str(finding.get("severity") or "info")
        rel_path = finding.get("path")
        detail = str(finding.get("detail") or "")
        metadata = finding.get("context")
        extras = {
            "severity": severity,
            "rel_path": str(rel_path) if rel_path is not None else None,
            "metadata": metadata,
        }
        reporter.record(
            graph_name=graph_name,
            entity_id=entity_id,
            issue=issue,
            detail=detail,
            extras=extras,
        )
    reporter.flush(gateway)


__all__ = [
    # Constants
    "CALL_SCC_MIN",
    "CONFIG_KEY_MIN_THRESHOLD",
    "HUB_DEGREE_RATIO",
    "HUB_MIN_DEGREE_FLOOR",
    "SAMPLE_LIMIT",
    "SYMBOL_COMMUNITY_MIN",
    # Types
    "GraphValidationOptions",
    # Functions
    "apply_severity_overrides",
    "cap_findings",
    "has_error_findings",
    "hub_threshold",
    "persist_findings",
    "resolve_validation_options",
]
