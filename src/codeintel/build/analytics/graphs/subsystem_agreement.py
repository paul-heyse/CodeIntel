"""Agreement checks between subsystems and import communities."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from codeintel.core.columnar.rows import ColumnarRowBuffer

log = logging.getLogger(__name__)


def _matches_optional_scope(value: object, expected: str) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return str(value) == expected


@dataclass(frozen=True)
class SubsystemAgreementInputs:
    """Inputs required to compute subsystem agreement rows."""

    repo: str
    commit: str
    subsystem_module_rows: Iterable[Mapping[str, object]] | ColumnarRowBuffer
    graph_metrics_module_rows: Iterable[Mapping[str, object]] | ColumnarRowBuffer


def build_subsystem_agreement_rows(inputs: SubsystemAgreementInputs) -> list[tuple[object, ...]]:
    """Compare subsystem assignments with import community labels.

    Returns
    -------
    list[tuple[object, ...]]
        Row tuples for analytics.subsystem_agreement.
    """
    community_by_module = _community_by_module(
        inputs.graph_metrics_module_rows,
        repo=inputs.repo,
        commit=inputs.commit,
    )
    now = datetime.now(UTC)
    inserts = _agreement_rows(
        inputs.subsystem_module_rows,
        repo=inputs.repo,
        commit=inputs.commit,
        community_by_module=community_by_module,
        now=now,
    )
    _log_disagreements(inserts, repo=inputs.repo, commit=inputs.commit)
    return inserts


def _community_by_module(
    rows: Iterable[Mapping[str, object]] | ColumnarRowBuffer,
    *,
    repo: str,
    commit: str,
) -> dict[str, object]:
    community_by_module: dict[str, object] = {}
    for row in _iter_row_mappings(rows):
        if not _matches_optional_scope(row.get("repo"), repo):
            continue
        if not _matches_optional_scope(row.get("commit"), commit):
            continue
        module = row.get("module")
        if module is None:
            continue
        community_by_module[str(module)] = row.get("import_community_id")
    return community_by_module


def _agreement_rows(
    rows: Iterable[Mapping[str, object]] | ColumnarRowBuffer,
    *,
    repo: str,
    commit: str,
    community_by_module: Mapping[str, object],
    now: datetime,
) -> list[tuple[object, ...]]:
    inserts: list[tuple[object, ...]] = []
    for row in _iter_row_mappings(rows):
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
    return inserts


def _iter_row_mappings(
    rows: Iterable[Mapping[str, object]] | ColumnarRowBuffer,
) -> Iterable[Mapping[str, object]]:
    yield from rows


def _log_disagreements(
    rows: Iterable[tuple[object, ...]],
    *,
    repo: str,
    commit: str,
) -> None:
    disagreeing = [row for row in rows if not row[5]]
    if not disagreeing:
        return
    sample = ", ".join(str(row[2]) for row in disagreeing[:5])
    log.warning(
        "Subsystem/import community disagreement: %d modules (sample: %s) for %s@%s",
        len(disagreeing),
        sample,
        repo,
        commit,
    )
