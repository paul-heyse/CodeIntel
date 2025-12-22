"""Agreement checks between subsystems and import communities."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.analytics.utilities.datasets import write_analytics_tuple_rows
from codeintel.analytics.utilities.persistence import DeleteScope

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def compute_subsystem_agreement(gateway: StorageGateway, *, repo: str, commit: str) -> None:
    """Compare subsystem assignments with import community labels."""
    con = gateway.con
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
    inserts: list[tuple[object, ...]] = []
    for module, subsystem_id, community_id in rows:
        agrees = True
        if subsystem_id is not None and community_id is not None:
            agrees = str(subsystem_id) == str(community_id)
        subsystem_value = str(subsystem_id) if subsystem_id is not None else None
        inserts.append((repo, commit, str(module), subsystem_value, community_id, agrees, now))

    delete_scope = DeleteScope(repo=repo, commit=commit)
    write_analytics_tuple_rows(
        gateway,
        "analytics.subsystem_agreement",
        inserts,
        delete_scope=delete_scope,
    )
    disagreeing = [row for row in inserts if not row[5]]
    if disagreeing:
        sample = ", ".join(str(row[2]) for row in disagreeing[:5])
        log.warning(
            "Subsystem/import community disagreement: %d modules (sample: %s) for %s@%s",
            len(disagreeing),
            sample,
            repo,
            commit,
        )
