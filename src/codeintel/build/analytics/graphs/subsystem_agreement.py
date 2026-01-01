"""Agreement checks between subsystems and import communities."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime

log = logging.getLogger(__name__)


def _matches_optional_scope(value: object, expected: str) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return str(value) == expected


def build_subsystem_agreement_rows(
    *,
    repo: str,
    commit: str,
    subsystem_module_rows: Iterable[Mapping[str, object]],
    graph_metrics_module_rows: Iterable[Mapping[str, object]],
) -> list[tuple[object, ...]]:
    """Compare subsystem assignments with import community labels.

    Returns
    -------
    list[tuple[object, ...]]
        Row tuples for analytics.subsystem_agreement.
    """
    community_by_module: dict[str, object] = {}
    for row in graph_metrics_module_rows:
        if not _matches_optional_scope(row.get("repo"), repo):
            continue
        if not _matches_optional_scope(row.get("commit"), commit):
            continue
        module = row.get("module")
        if module is None:
            continue
        community_by_module[str(module)] = row.get("import_community_id")

    now = datetime.now(UTC)
    inserts: list[tuple[object, ...]] = []
    for row in subsystem_module_rows:
        if not _matches_optional_scope(row.get("repo"), repo):
            continue
        if not _matches_optional_scope(row.get("commit"), commit):
            continue
        module = row.get("module")
        subsystem_id = row.get("subsystem_id")
        if module is None:
            continue
        community_id = community_by_module.get(str(module))
        agrees = True
        if subsystem_id is not None and community_id is not None:
            agrees = str(subsystem_id) == str(community_id)
        subsystem_value = str(subsystem_id) if subsystem_id is not None else None
        inserts.append((repo, commit, str(module), subsystem_value, community_id, agrees, now))

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
    return inserts
