"""Anomaly detection validation checks.

This module contains validation checks that detect community and subsystem
level anomalies in the codebase.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.graphs.validation.findings import (
    SAMPLE_LIMIT,
    SYMBOL_COMMUNITY_MIN,
)

if TYPE_CHECKING:
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
    comm_counts = gateway.con.execute(
        """
        SELECT symbol_community_id, COUNT(*) AS count
        FROM analytics.symbol_graph_metrics_modules
        WHERE repo = ? AND commit = ? AND symbol_community_id IS NOT NULL
        GROUP BY symbol_community_id
        HAVING count > ?
        """,
        [repo, commit, SYMBOL_COMMUNITY_MIN],
    ).fetchall()
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


def subsystem_disagreement_findings(
    gateway: StorageGateway, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for subsystem vs import community disagreements.

    Returns
    -------
    list[dict[str, object]]
        Findings for subsystem disagreement anomalies.
    """
    disagreements = gateway.con.execute(
        """
        SELECT module, subsystem_id, import_community_id
        FROM analytics.subsystem_agreement
        WHERE repo = ? AND commit = ? AND agrees = false
        """,
        [repo, commit],
    ).fetchall()
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
