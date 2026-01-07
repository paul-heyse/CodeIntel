"""Anomaly detection validation checks.

This module contains validation checks that detect community and subsystem
level anomalies in the codebase.

Check classes implement CheckProtocol from core/validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pyarrow as pa

from codeintel.build.graphs.engine.datasets import SnapshotScanRequest, scan_snapshot_table
from codeintel.build.graphs.validation.base import GraphCheckBase
from codeintel.build.graphs.validation.findings import (
    SAMPLE_LIMIT,
    SYMBOL_COMMUNITY_MIN,
)
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.core.query_results import coerce_int, coerce_str

if TYPE_CHECKING:
    import logging

    from codeintel.build.graphs.validation.context import GraphValidationContext
    from codeintel.core.validation import ValidationSeverity


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
        return _symbol_community_findings_impl(
            ctx.dataset_root_dir,
            ctx.repo,
            ctx.commit,
            ctx.logger,
        )


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
        return _subsystem_disagreement_findings_impl(
            ctx.dataset_root_dir,
            ctx.repo,
            ctx.commit,
            ctx.logger,
        )


# =============================================================================
# Implementation Functions (internal)
# =============================================================================


def _scan_snapshot_table(request: SnapshotScanRequest) -> pa.Table | None:
    return scan_snapshot_table(request)


def _symbol_community_findings_impl(
    dataset_root_dir: Path | None,
    repo: str,
    commit: str,
    log: logging.Logger,
) -> list[dict[str, object]]:
    """Check for large symbol communities (implementation).

    Returns
    -------
    list[dict[str, object]]
        Findings for symbol community anomalies.
    """
    if dataset_root_dir is None:
        return []
    table = _scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="analytics.symbol_graph_metrics_modules",
            snapshot_id=commit,
            columns=("symbol_community_id", "repo", "commit"),
            repo=repo,
            commit=commit,
        )
    )
    if table is None:
        return []
    counts: dict[object, int] = {}
    for row in iter_rows(table):
        community_id = row.get("symbol_community_id")
        if community_id is None:
            continue
        counts[community_id] = counts.get(community_id, 0) + 1
    comm_counts = [
        (
            coerce_str(community_id, ctx="symbol_community_id"),
            coerce_int(count, ctx="symbol_community_count"),
        )
        for community_id, count in counts.items()
        if count > SYMBOL_COMMUNITY_MIN
    ]

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
    dataset_root_dir: Path | None,
    repo: str,
    commit: str,
    log: logging.Logger,
) -> list[dict[str, object]]:
    """Check for subsystem vs import community disagreements (implementation).

    Returns
    -------
    list[dict[str, object]]
        Findings for subsystem disagreement anomalies.
    """
    if dataset_root_dir is None:
        return []
    table = _scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="analytics.subsystem_agreement",
            snapshot_id=commit,
            columns=("module", "subsystem_id", "import_community_id", "agrees", "repo", "commit"),
            repo=repo,
            commit=commit,
        )
    )
    if table is None:
        return []
    disagreements = [
        (
            coerce_str(row.get("module"), ctx="subsystem_agreement.module"),
            coerce_str(row.get("subsystem_id"), ctx="subsystem_agreement.subsystem_id"),
            coerce_str(
                row.get("import_community_id"),
                ctx="subsystem_agreement.import_community_id",
            ),
        )
        for row in iter_rows(table)
        if row.get("agrees") is False
    ]
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
