"""Aggregate per-function git history and churn metrics.

For new code, use ``build_function_history_rows`` with Hamilton materializers.

The Hamilton native module is at:
``codeintel.build.hamilton.native.analytics.function_history``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast

from codeintel.analytics.history.git_history import iter_file_history
from codeintel.core.ibis_typing import and_predicates

if TYPE_CHECKING:
    import pandas as pd

    from codeintel.analytics.history.git_history import FileCommitDelta
    from codeintel.config.primitives import SnapshotRef
    from codeintel.ingestion.engine.infrastructure import ToolRunner
    from codeintel.storage.gateway import StorageGateway

FUNCTION_HISTORY_COLS = [
    "repo",
    "commit",
    "function_goid_h128",
    "urn",
    "rel_path",
    "module",
    "qualname",
    "created_in_commit",
    "created_at",
    "last_modified_commit",
    "last_modified_at",
    "age_days",
    "commit_count",
    "author_count",
    "lines_added",
    "lines_deleted",
    "churn_score",
    "stability_bucket",
    "history_window_start",
    "history_window_end",
    "created_at_row",
]

log = logging.getLogger(__name__)

NEW_DAYS_THRESHOLD = 30
HOT_CHURN_THRESHOLD = 0.5
HOT_COMMIT_THRESHOLD = 5
STABLE_COMMIT_THRESHOLD = 2
CHURNING_THRESHOLD = 0.2


@dataclass(frozen=True)
class FuncSpan:
    """Function span anchored to a source path."""

    repo: str
    commit: str
    goid: int
    urn: str
    module: str
    qualname: str
    rel_path: str
    start: int
    end: int
    loc: int


@dataclass
class FuncHistoryAgg:
    """Mutable accumulator for per-function history."""

    first_commit: str | None = None
    first_ts: datetime | None = None
    last_commit: str | None = None
    last_ts: datetime | None = None
    commit_count: int = 0
    authors: set[str] = field(default_factory=set)
    lines_added: int = 0
    lines_deleted: int = 0


def build_function_history_rows(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    runner: ToolRunner | None = None,
    min_lines_threshold: int = 2,
) -> tuple[tuple[object, ...], ...]:
    """Build function_history rows without writing to database.

    Compute per-function git history and churn metrics, returning row tuples
    suitable for materialization via Hamilton materializers.

    Parameters
    ----------
    gateway
        StorageGateway bound to the CodeIntel DuckDB database.
    snapshot
        Repository and commit identifiers.
    runner
        Optional shared ToolRunner for git invocations.
    min_lines_threshold
        Minimum number of overlapping line edits required to count a commit towards churn.

    Returns
    -------
    tuple[tuple[object, ...], ...]
        Row tuples matching FUNCTION_HISTORY_COLS schema, ready for bulk insert.
    """
    max_history_days = 365
    default_branch = "main"

    spans_by_path = _load_function_spans(gateway, snapshot.repo, snapshot.commit)
    if not spans_by_path:
        log.info(
            "No function spans found for %s@%s; skipping history.", snapshot.repo, snapshot.commit
        )
        return ()

    now = datetime.now(tz=UTC)
    window_start = _history_window_start(now, max_history_days)
    aggregates: dict[int, FuncHistoryAgg] = {}

    for rel_path, spans in spans_by_path.items():
        deltas = iter_file_history(
            snapshot.repo_root,
            rel_path,
            max_history_days=max_history_days,
            default_branch=default_branch,
            runner=runner,
        )
        for delta in deltas:
            _update_aggregates(aggregates, spans, delta, min_lines_threshold)

    insert_rows = [
        _build_insert_row(
            span,
            aggregates.get(span.goid, FuncHistoryAgg()),
            now=now,
            window_start=window_start,
        )
        for spans in spans_by_path.values()
        for span in spans
    ]

    log.info(
        "function_history computed: %s rows for %s@%s",
        len(insert_rows),
        snapshot.repo,
        snapshot.commit,
    )
    return tuple(insert_rows)


def _history_window_start(now: datetime, max_history_days: int | None) -> datetime | None:
    if max_history_days is None:
        return None
    return now - timedelta(days=max_history_days)


def _compute_churn_score(lines_added: int, lines_deleted: int, loc: int) -> float:
    total = lines_added + lines_deleted
    safe_loc = max(loc, 1)
    raw = total / safe_loc
    return min(raw / 10.0, 1.0)


def _classify_stability(
    *,
    age_days: int | None,
    commit_count: int,
    churn_score: float,
    window_days: int | None,
) -> str:
    if age_days is None:
        return "new_hot" if churn_score > 0 else "unknown"

    recent_window_days = window_days or 365
    is_new = age_days <= NEW_DAYS_THRESHOLD
    is_hot = churn_score >= HOT_CHURN_THRESHOLD or commit_count >= HOT_COMMIT_THRESHOLD

    if is_new and is_hot:
        return "new_hot"
    if not is_new and not is_hot and commit_count <= STABLE_COMMIT_THRESHOLD:
        return "stable"
    if churn_score >= CHURNING_THRESHOLD:
        return "churning"
    return "legacy_hot" if age_days > recent_window_days and is_hot else "stable"


def _update_aggregates(
    aggregates: dict[int, FuncHistoryAgg],
    spans: list[FuncSpan],
    delta: FileCommitDelta,
    min_lines_threshold: int,
) -> None:
    for span in spans:
        added = _sum_overlap(span.start, span.end, delta.added_spans)
        deleted = _sum_overlap(span.start, span.end, delta.deleted_spans)
        if added + deleted < min_lines_threshold:
            continue
        agg = aggregates.setdefault(span.goid, FuncHistoryAgg())
        agg.commit_count += 1
        agg.authors.add(delta.author_email)
        agg.lines_added += added
        agg.lines_deleted += deleted
        if delta.author_ts is not None:
            if agg.first_ts is None or delta.author_ts < agg.first_ts:
                agg.first_ts = delta.author_ts
                agg.first_commit = delta.commit_hash
            if agg.last_ts is None or delta.author_ts > agg.last_ts:
                agg.last_ts = delta.author_ts
                agg.last_commit = delta.commit_hash


def _sum_overlap(start: int, end: int, spans: list[tuple[int, int]]) -> int:
    total = 0
    for s, e in spans:
        overlap_start = max(start, s)
        overlap_end = min(end, e)
        if overlap_start <= overlap_end:
            total += overlap_end - overlap_start + 1
    return total


def _build_insert_row(
    span: FuncSpan,
    agg: FuncHistoryAgg,
    *,
    now: datetime,
    window_start: datetime | None,
) -> tuple[object, ...]:
    age_days = (now - agg.first_ts).days if agg.first_ts is not None else None
    churn_score = _compute_churn_score(agg.lines_added, agg.lines_deleted, span.loc)
    stability = _classify_stability(
        age_days=age_days,
        commit_count=agg.commit_count,
        churn_score=churn_score,
        window_days=None if window_start is None else int((now - window_start).days),
    )
    return (
        span.repo,
        span.commit,
        span.goid,
        span.urn,
        span.rel_path,
        span.module,
        span.qualname,
        agg.first_commit,
        agg.first_ts,
        agg.last_commit,
        agg.last_ts,
        age_days,
        agg.commit_count,
        len(agg.authors),
        agg.lines_added,
        agg.lines_deleted,
        churn_score,
        stability,
        window_start,
        now,
        now,
    )


def _load_function_spans(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, list[FuncSpan]]:
    metrics = gateway.ibis.table("analytics.function_metrics")
    modules = gateway.ibis.table("core.modules")
    join_expr = metrics.left_join(
        modules,
        [
            metrics.repo == modules.repo,
            metrics.commit == modules.commit,
            metrics.rel_path == modules.path,
        ],
    )
    expr = join_expr.filter(and_predicates(metrics.repo == repo, metrics.commit == commit)).select(
        metrics.repo.name("repo"),
        metrics.commit.name("commit"),
        metrics.function_goid_h128,
        metrics.urn,
        metrics.rel_path,
        modules.module,
        metrics.qualname,
        metrics.start_line,
        metrics.end_line,
        metrics.loc,
    )
    df = cast("pd.DataFrame", expr.execute())
    rows = df.to_dict(orient="records")

    spans_by_path: dict[str, list[FuncSpan]] = {}
    for row in rows:
        goid = int(row["function_goid_h128"])
        module_name = row.get("module") or ""
        rel_path = str(row["rel_path"])
        spans_by_path.setdefault(rel_path, []).append(
            FuncSpan(
                goid=goid,
                urn=str(row["urn"]),
                module=str(module_name),
                qualname=str(row["qualname"]),
                rel_path=rel_path,
                start=int(row["start_line"] or 0),
                end=int(row["end_line"] or 0),
                loc=int(row["loc"] or 0),
                repo=str(row["repo"]),
                commit=str(row["commit"]),
            )
        )
    return spans_by_path
