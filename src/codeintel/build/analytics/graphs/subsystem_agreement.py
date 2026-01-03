"""Agreement checks between subsystems and import communities."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl

log = logging.getLogger(__name__)


def _matches_optional_scope(value: object, expected: str) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return str(value) == expected


def _matches_optional_scope_expr(column: str, expected: str) -> pl.Expr:
    col = pl.col(column)
    col_str = col.cast(pl.Utf8, strict=False)
    stripped = col_str.str.strip_chars()
    return col.is_null() | (stripped.str.len_chars() == 0) | (col_str == expected)


@dataclass(frozen=True)
class SubsystemAgreementInputs:
    """Inputs required to compute subsystem agreement rows."""

    repo: str
    commit: str
    subsystem_module_rows: Iterable[Mapping[str, object]]
    graph_metrics_module_rows: Iterable[Mapping[str, object]]


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


def build_subsystem_agreement_frame(
    *,
    repo: str,
    commit: str,
    subsystem_modules: pl.LazyFrame,
    graph_metrics_modules: pl.LazyFrame,
) -> pl.LazyFrame:
    """Build subsystem agreement rows as a lazy Polars frame.

    Returns
    -------
    pl.LazyFrame
        Lazy frame for analytics.subsystem_agreement rows.
    """
    filtered_subsystems = subsystem_modules.filter(
        _matches_optional_scope_expr("repo", repo) & _matches_optional_scope_expr("commit", commit)
    ).filter(pl.col("module").is_not_null())
    filtered_metrics = graph_metrics_modules.filter(
        _matches_optional_scope_expr("repo", repo) & _matches_optional_scope_expr("commit", commit)
    ).filter(pl.col("module").is_not_null())

    filtered_subsystems = filtered_subsystems.with_columns(
        pl.col("module").cast(pl.Utf8, strict=False).alias("module"),
        pl.col("subsystem_id").cast(pl.Utf8, strict=False).alias("subsystem_id"),
    )
    filtered_metrics = filtered_metrics.with_columns(
        pl.col("module").cast(pl.Utf8, strict=False).alias("module")
    )
    joined = filtered_subsystems.join(
        filtered_metrics.select(["module", "import_community_id"]),
        on="module",
        how="left",
    )
    agree_expr = (
        pl.when(pl.col("subsystem_id").is_not_null() & pl.col("import_community_id").is_not_null())
        .then(pl.col("subsystem_id") == pl.col("import_community_id").cast(pl.Utf8, strict=False))
        .otherwise(pl.lit(value=True))
    )
    now = datetime.now(UTC)
    frame = joined.with_columns(
        pl.lit(repo).alias("repo"),
        pl.lit(commit).alias("commit"),
        agree_expr.alias("agrees"),
        pl.lit(now).alias("created_at"),
    ).select(
        [
            "repo",
            "commit",
            "module",
            "subsystem_id",
            "import_community_id",
            "agrees",
            "created_at",
        ]
    )
    _log_disagreements_frame(frame, repo=repo, commit=commit)
    return frame


def _community_by_module(
    rows: Iterable[Mapping[str, object]],
    *,
    repo: str,
    commit: str,
) -> dict[str, object]:
    community_by_module: dict[str, object] = {}
    for row in rows:
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
    rows: Iterable[Mapping[str, object]],
    *,
    repo: str,
    commit: str,
    community_by_module: Mapping[str, object],
    now: datetime,
) -> list[tuple[object, ...]]:
    inserts: list[tuple[object, ...]] = []
    for row in rows:
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


def _log_disagreements_frame(frame: pl.LazyFrame, *, repo: str, commit: str) -> None:
    disagreeing = frame.filter(~pl.col("agrees"))
    count = disagreeing.select(pl.len()).collect().item()
    if not isinstance(count, int) or count <= 0:
        return
    sample = disagreeing.select("module").limit(5).collect().get_column("module").to_list()
    sample_str = ", ".join(str(value) for value in sample)
    log.warning(
        "Subsystem/import community disagreement: %d modules (sample: %s) for %s@%s",
        count,
        sample_str,
        repo,
        commit,
    )
