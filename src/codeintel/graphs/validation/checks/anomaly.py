"""Anomaly detection validation checks.

This module contains validation checks that detect community and subsystem
level anomalies in the codebase.

Check classes implement CheckProtocol from core/validation; legacy
function wrappers are provided for backward compatibility.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

from codeintel.graphs.validation.base import GraphCheckBase
from codeintel.graphs.validation.findings import (
    SAMPLE_LIMIT,
    SYMBOL_COMMUNITY_MIN,
)
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.ibis_types import bool_not, filter_by, ibis_bool

if TYPE_CHECKING:
    from codeintel.core.validation import ValidationSeverity
    from codeintel.graphs.validation.context import GraphValidationContext
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
        metrics = cast("Any", gateway.ibis.table("analytics.symbol_graph_metrics_modules"))
        filtered = filter_by(
            metrics,
            ibis_bool(metrics.repo == repo),
            ibis_bool(metrics.commit == commit),
            ibis_bool(metrics.symbol_community_id.notnull()),
        )
        grouped = filtered.group_by(metrics.symbol_community_id).aggregate(
            sym_count=metrics.symbol_community_id.count()
        )
        expr = grouped.filter(ibis_bool(grouped["sym_count"] > SYMBOL_COMMUNITY_MIN))
        comm_counts_df = expr.execute()
    except DuckDBError:
        return []

    if getattr(comm_counts_df, "empty", True):
        return []
    comm_counts = list(comm_counts_df.itertuples(index=False, name=None))
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
        agreement = cast("Any", gateway.ibis.table("analytics.subsystem_agreement"))
        filtered = filter_by(
            agreement,
            agreement.repo == repo,
            agreement.commit == commit,
            bool_not(agreement.agrees),
        )
        disagreements_df = filtered.select(
            agreement.module, agreement.subsystem_id, agreement.import_community_id
        ).execute()
    except DuckDBError:
        return []

    disagreements = (
        list(disagreements_df.itertuples(index=False, name=None))
        if not getattr(disagreements_df, "empty", True)
        else []
    )
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
# Backward-Compatible Function Wrappers
# =============================================================================


def symbol_community_findings(
    gateway: StorageGateway, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for large symbol communities.

    Returns
    -------
    list[dict[str, object]]
        Findings for symbol community anomalies.
    """
    return _symbol_community_findings_impl(gateway, repo, commit, log)


def subsystem_disagreement_findings(
    gateway: StorageGateway, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for subsystem vs import community disagreements.

    Returns
    -------
    list[dict[str, object]]
        Findings for subsystem disagreement anomalies.
    """
    return _subsystem_disagreement_findings_impl(gateway, repo, commit, log)


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
    # Backward-compatible functions
    "subsystem_disagreement_findings",
    "symbol_community_findings",
]
