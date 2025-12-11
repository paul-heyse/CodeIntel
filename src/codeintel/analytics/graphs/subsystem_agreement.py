"""Agreement checks between subsystems and import communities."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.sql.builder import ensure_schema

log = logging.getLogger(__name__)

SUBSYSTEM_AGREEMENT_COLS = [
    "repo", "commit", "module", "subsystem_id", "import_community_id", "agrees", "created_at",
]


def compute_subsystem_agreement(gateway: StorageGateway, *, repo: str, commit: str) -> None:
    """Compare subsystem assignments with import community labels."""
    con = gateway.con
    ensure_schema(con, "analytics.subsystem_agreement")
    rows = con.execute(
        """
        SELECT sm.module,
               sm.subsystem_id,
               gmx.import_community_id
        FROM analytics.subsystem_modules sm
        LEFT JOIN analytics.graph_metrics_modules_ext gmx
          ON gmx.module = sm.module
         AND gmx.repo = sm.repo
         AND gmx.commit = sm.commit
        WHERE sm.repo = ? AND sm.commit = ?
        """,
        [repo, commit],
    ).fetchall()
    now = datetime.now(UTC)
    inserts = []
    for module, subsystem_id, community_id in rows:
        agrees = True
        if subsystem_id is not None and community_id is not None:
            agrees = str(subsystem_id) == str(community_id)
        inserts.append((repo, commit, str(module), subsystem_id, community_id, agrees, now))

    backend = DuckDBPolicyBackend(gateway)
    backend.delete_for_snapshot("analytics.subsystem_agreement", repo=repo, commit=commit)
    if inserts:
        gateway.ibis.write(
            "analytics.subsystem_agreement",
            inserts,
            columns=SUBSYSTEM_AGREEMENT_COLS,
        )
    disagreeing = [row for row in inserts if not row[5]]
    if disagreeing:
        sample = ", ".join(row[2] for row in disagreeing[:5])
        log.warning(
            "Subsystem/import community disagreement: %d modules (sample: %s) for %s@%s",
            len(disagreeing),
            sample,
            repo,
            commit,
        )
