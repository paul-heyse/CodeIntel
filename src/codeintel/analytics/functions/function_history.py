"""Aggregate per-function git history and churn metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.history.git_history import FileCommitDelta, iter_file_history
from codeintel.config import ConfigBuilder, FunctionHistoryStepConfig
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.storage.gateway import DuckDBConnection, StorageGateway

if TYPE_CHECKING:
    from codeintel.config.models import FunctionHistoryConfig

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


def compute_function_history(
    gateway: StorageGateway,
    cfg: FunctionHistoryStepConfig | FunctionHistoryConfig,
    *,
    runner: ToolRunner | None = None,
    context: AnalyticsContext | None = None,
) -> None:
    """
    Populate `analytics.function_history` for the given repo/commit snapshot.

    Parameters
    ----------
    gateway:
        StorageGateway bound to the CodeIntel DuckDB database.
    cfg:
        Function history configuration.
    runner:
        Optional shared ToolRunner for git invocations.
    context:
        Optional shared analytics context to enforce snapshot consistency.
    """
    if not isinstance(cfg, FunctionHistoryStepConfig):
        builder = ConfigBuilder.from_snapshot(
            repo=cfg.repo,
            commit=cfg.commit,
            repo_root=cfg.repo_root,
        )
        cfg = builder.function_history(
            max_history_days=cfg.max_history_days,
            min_lines_threshold=cfg.min_lines_threshold,
            default_branch=cfg.default_branch,
        )

    con = gateway.con
    if context is not None and (context.repo != cfg.repo or context.commit != cfg.commit):
        log.warning(
            "function_history context mismatch: context=%s@%s cfg=%s@%s",
            context.repo,
            context.commit,
            cfg.repo,
            cfg.commit,
        )

    ensure_schema(con, "analytics.function_history")
    con.execute(
        "DELETE FROM analytics.function_history WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    spans_by_path = _load_function_spans(con, cfg.repo, cfg.commit)
    if not spans_by_path:
        log.info("No function spans found for %s@%s; skipping history.", cfg.repo, cfg.commit)
        return

    now = datetime.now(tz=UTC)
    window_start = _history_window_start(now, cfg.max_history_days)
    aggregates: dict[int, FuncHistoryAgg] = {}

    for rel_path, spans in spans_by_path.items():
        deltas = iter_file_history(
            cfg.repo_root,
            rel_path,
            max_history_days=cfg.max_history_days,
            default_branch=cfg.default_branch,
            runner=runner,
        )
        for delta in deltas:
            _update_aggregates(aggregates, spans, delta, cfg.min_lines_threshold)

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
    con.executemany(
        """
        INSERT INTO analytics.function_history (
            repo,
            commit,
            function_goid_h128,
            urn,
            rel_path,
            module,
            qualname,
            created_in_commit,
            created_at,
            last_modified_commit,
            last_modified_at,
            age_days,
            commit_count,
            author_count,
            lines_added,
            lines_deleted,
            churn_score,
            stability_bucket,
            history_window_start,
            history_window_end,
            created_at_row
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        insert_rows,
    )
    log.info(
        "function_history populated: %s rows for %s@%s",
        len(insert_rows),
        cfg.repo,
        cfg.commit,
    )


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
    con: DuckDBConnection,
    repo: str,
    commit: str,
) -> dict[str, list[FuncSpan]]:
    rows = con.execute(
        """
        SELECT
            fm.repo,
            fm.commit,
            fm.function_goid_h128,
            fm.urn,
            fm.rel_path,
            m.module,
            fm.qualname,
            fm.start_line,
            fm.end_line,
            fm.loc
        FROM analytics.function_metrics fm
        LEFT JOIN core.modules m
          ON m.repo = fm.repo
         AND m.commit = fm.commit
         AND m.path = fm.rel_path
        WHERE fm.repo = ? AND fm.commit = ?
        """,
        [repo, commit],
    ).fetchall()

    spans_by_path: dict[str, list[FuncSpan]] = {}
    for row in rows:
        repo_val, commit_val, goid_raw, urn, rel_path, module, qualname, start, end, loc = row
        goid = int(goid_raw)
        module_name = module or ""
        spans_by_path.setdefault(rel_path, []).append(
            FuncSpan(
                goid=goid,
                urn=str(urn),
                module=str(module_name),
                qualname=str(qualname),
                rel_path=str(rel_path),
                start=int(start or 0),
                end=int(end or 0),
                loc=int(loc or 0),
                repo=str(repo_val),
                commit=str(commit_val),
            )
        )
    return spans_by_path
