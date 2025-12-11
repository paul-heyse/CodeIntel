"""Anomaly detection validation checks.

This module contains validation checks that detect community and subsystem
level anomalies in the codebase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from codeintel.graphs.validation.findings import (
    SAMPLE_LIMIT,
    SYMBOL_COMMUNITY_MIN,
)
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.ibis_types import bool_not, filter_by, ibis_bool

if TYPE_CHECKING:
    import logging

    from codeintel.storage.gateway import StorageGateway


def symbol_community_findings(
    gateway: StorageGateway, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for large symbol communities.

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


def subsystem_disagreement_findings(
    gateway: StorageGateway, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for subsystem vs import community disagreements.

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


__all__ = [
    "subsystem_disagreement_findings",
    "symbol_community_findings",
]
