"""Anomaly detection validation checks.

This module contains validation checks that detect community and subsystem
level anomalies in the codebase.

Check classes implement CheckProtocol from core/validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.build.graphs.validation.base import GraphCheckBase
from codeintel.build.graphs.validation.findings import (
    SAMPLE_LIMIT,
    SYMBOL_COMMUNITY_MIN,
)
from codeintel.storage.duckdb_types import ColumnExpression, ConstantExpression
from codeintel.storage.gateway import DuckDBError

if TYPE_CHECKING:
    import logging

    from codeintel.build.graphs.validation.context import GraphValidationContext
    from codeintel.core.validation import ValidationSeverity
    from codeintel.storage.gateway import StorageGateway


# =============================================================================
# Check Classes (CheckProtocol-compliant)
# =============================================================================


class SymbolCommunityCheck(GraphCheckBase):
    """Check for large symbol communities."""

    check_name: ClassVar[str] = "symbol_communities"
    check_description: ClassVar[str] = "Detect large symbol communities"
    default_severity: ClassVar[ValidationSeverity] = "info"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute symbol community check.

        Parameters
        ----------
        ctx
            Graph validation context with gateway.

        Returns
        -------
        list[dict[str, object]]
            Findings for symbol community anomalies.
        """
        _ = self  # Instance method required for CheckProtocol
        if ctx.gateway is None:
            return []
        return _symbol_community_findings_impl(ctx.gateway, ctx.repo, ctx.commit, ctx.logger)


class SubsystemDisagreementCheck(GraphCheckBase):
    """Check for subsystem vs import community disagreements."""

    check_name: ClassVar[str] = "subsystem_disagreement"
    check_description: ClassVar[str] = "Detect subsystem vs import community mismatches"
    default_severity: ClassVar[ValidationSeverity] = "warning"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute subsystem disagreement check.

        Parameters
        ----------
        ctx
            Graph validation context with gateway.

        Returns
        -------
        list[dict[str, object]]
            Findings for subsystem disagreement anomalies.
        """
        _ = self  # Instance method required for CheckProtocol
        if ctx.gateway is None:
            return []
        return _subsystem_disagreement_findings_impl(ctx.gateway, ctx.repo, ctx.commit, ctx.logger)


# =============================================================================
# Implementation Functions (internal)
# =============================================================================


def _symbol_community_findings_impl(
    gateway: StorageGateway, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for large symbol communities (implementation).

    Returns
    -------
    list[dict[str, object]]
        Findings for symbol community anomalies.
    """
    try:
        predicate = (ColumnExpression("repo") == ConstantExpression(repo)) & (
            ColumnExpression("commit") == ConstantExpression(commit)
        )
        relation = (
            gateway.relation_from_table_key("analytics.symbol_graph_metrics_modules")
            .filter(predicate)
            .filter(~ColumnExpression("symbol_community_id").isnull())
            .aggregate("count(*) as sym_count", "symbol_community_id")
            .filter(ColumnExpression("sym_count") > ConstantExpression(SYMBOL_COMMUNITY_MIN))
        )
        comm_counts = relation.fetchall()
    except DuckDBError:
        return []

    if not comm_counts:
        return []
    largest = max(comm_counts, key=lambda row: row[1])
    log.warning("Validation: large symbol communities detected (largest size %d)", largest[1])
    return [
        {
            "repo": repo,
            "commit": commit,
            "check_name": "symbol_graph_large_community",
            "severity": "info",
            "path": None,
            "detail": f"{len(comm_counts)} communities exceed threshold; largest {largest[1]}",
            "context": {"communities": comm_counts[: SAMPLE_LIMIT * 4]},
        }
    ]


def _subsystem_disagreement_findings_impl(
    gateway: StorageGateway, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for subsystem vs import community disagreements (implementation).

    Returns
    -------
    list[dict[str, object]]
        Findings for subsystem disagreement anomalies.
    """
    try:
        agrees_value = False
        predicate = (
            (ColumnExpression("repo") == ConstantExpression(repo))
            & (ColumnExpression("commit") == ConstantExpression(commit))
            & (ColumnExpression("agrees") == ConstantExpression(agrees_value))
        )
        relation = (
            gateway.relation_from_table_key("analytics.subsystem_agreement")
            .filter(predicate)
            .select("module", "subsystem_id", "import_community_id")
        )
        disagreements = relation.fetchall()
    except DuckDBError:
        return []
    if not disagreements:
        return []
    sample = ", ".join(str(row[0]) for row in disagreements[:SAMPLE_LIMIT])
    log.warning(
        "Validation: %d module(s) disagree on subsystem vs import community (sample: %s)",
        len(disagreements),
        sample,
    )
    return [
        {
            "repo": repo,
            "commit": commit,
            "check_name": "subsystem_community_disagreement",
            "severity": "warning",
            "path": None,
            "detail": f"{len(disagreements)} modules disagree (sample: {sample})",
            "context": {"modules": disagreements[: SAMPLE_LIMIT * 4]},
        }
    ]


# =============================================================================
# All Check Classes (for runner registration)
# =============================================================================

ALL_ANOMALY_CHECKS: tuple[type[GraphCheckBase], ...] = (
    SymbolCommunityCheck,
    SubsystemDisagreementCheck,
)

__all__ = [
    # Check classes
    "ALL_ANOMALY_CHECKS",
    "SubsystemDisagreementCheck",
    "SymbolCommunityCheck",
]
