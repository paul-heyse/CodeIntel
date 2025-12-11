"""Finding types, persistence, and severity handling for graph validation.

This module provides graph-specific validation types and utilities,
extending the core validation infrastructure with graph-specific features.

The helper functions are re-exported from core to maintain a consistent API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.analytics.parsing.validation import GraphValidationReporter
from codeintel.analytics.runtime import GraphRuntime, GraphRuntimeOptions
from codeintel.core.validation import (
    BaseValidationOptions,
    ValidationSeverity,
    apply_severity_overrides,
    cap_findings,
    has_error_findings,
)
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.ibis_types import and_predicates

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
class GraphValidationOptions(BaseValidationOptions):
    """Options for controlling graph validation behavior.

    Extend ``BaseValidationOptions`` with graph-specific options.
    Currently no additional fields, but this allows for future extension.

    Attributes
    ----------
    severity_overrides
        Mapping of rule names to severity levels. Use "*" for all.
    hard_fail
        Whether to raise an exception on error-level findings.
    max_findings_per_rule
        Maximum findings to collect per rule.
    """


def resolve_validation_options(
    runtime: GraphRuntime | GraphRuntimeOptions,
    options: GraphValidationOptions | None,
) -> GraphValidationOptions:
    """Determine effective validation options from runtime feature flags.

    Parameters
    ----------
    runtime
        Runtime or options containing feature flags.
    options
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
# Graph-Specific Helpers
# =============================================================================


def hub_threshold(node_count: int) -> int:
    """Compute a hub threshold that scales with graph size.

    Parameters
    ----------
    node_count
        Number of nodes in the graph.

    Returns
    -------
    int
        Degree threshold used to flag hubs.
    """
    return max(HUB_MIN_DEGREE_FLOOR, int(node_count * HUB_DEGREE_RATIO))


# =============================================================================
# Persistence
# =============================================================================


def persist_findings(
    gateway: StorageGateway, findings: list[dict[str, object]], repo: str, commit: str
) -> None:
    """Persist validation findings to the analytics schema.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    findings
        List of findings to persist.
    repo
        Repository identifier.
    commit
        Commit identifier.
    """
    if not findings:
        return
    try:
        table = gateway.ibis.table("analytics.graph_validation")
        gateway.ibis.delete(
            "analytics.graph_validation",
            where=and_predicates(table.repo == repo, table.commit == commit),
        )
    except DuckDBError:
        return
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
    "ValidationSeverity",
    # Core re-exports
    "apply_severity_overrides",
    "cap_findings",
    "has_error_findings",
    # Graph-specific functions
    "hub_threshold",
    "persist_findings",
    "resolve_validation_options",
]
