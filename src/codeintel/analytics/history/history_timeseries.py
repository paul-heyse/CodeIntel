"""Cross-commit history aggregation for functions and modules.

For new code, use ``build_history_timeseries_rows`` with Hamilton materializers.

The Hamilton native module is at:
``codeintel.build.hamilton.native.analytics.history_timeseries``
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, SupportsFloat, SupportsIndex

import polars as pl

from codeintel.build.hamilton.native.ingestion.frame_utils import (
    empty_lazyframe_for_table,
    lazyframe_for_table_columns,
)
from codeintel.core.columnar.rows import ColumnarRowBuffer, columnar_buffer_for_table_key
from codeintel.core.hashing import sha256_short
from codeintel.ingestion.engine.infrastructure import ToolRunner, ToolRunOptions
from codeintel.storage.duckdb_types import ColumnExpression, ConstantExpression

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class HistoryTimeseriesOptions:
    """Configuration options for history timeseries computation.

    Parameters
    ----------
    commits
        Tuple of commit SHAs to include in the timeseries.
    entity_kind
        Kind of entity to track: "function", "module", or "both".
    max_entities
        Maximum number of entities to include per kind.
    selection_strategy
        Strategy for selecting top entities: "risk_score" or others.
    """

    commits: tuple[str, ...]
    entity_kind: str = "function"
    max_entities: int = 500
    selection_strategy: str = "risk_score"


HISTORY_TIMESERIES_TABLE_KEY = "analytics.history_timeseries"

log = logging.getLogger(__name__)

GatewayResolver = Callable[[str], "StorageGateway"]
type NumericLike = SupportsFloat | SupportsIndex | str | bytes | bytearray | int | float | Decimal


@dataclass(frozen=True)
class EntitySelection:
    """Selected entities to include in the timeseries."""

    functions: set[str]
    modules: set[str]


@dataclass(frozen=True)
class CommitContext:
    """Commit metadata shared across entity collectors."""

    commit: str
    commit_ts: datetime
    created_at: datetime


def make_entity_stable_id(
    *,
    repo: str,
    rel_path: str,
    language: str,
    kind: str,
    qualname: str,
) -> str:
    """
    Build a stable identifier independent of commit-specific GOIDs.

    Parameters
    ----------
    repo:
        Repository slug.
    rel_path:
        Repository-relative path.
    language:
        Language of the entity.
    kind:
        Entity kind (e.g., ``function`` or ``module``).
    qualname:
        Qualified name for functions; empty for modules.

    Returns
    -------
    str
        Stable identifier hashed from entity coordinates.
    """
    raw = f"{repo}:{rel_path}:{language}:{kind}:{qualname}"
    return sha256_short(raw, length=20, used_for_security=False)


def _safe_number(value: NumericLike | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_history_timeseries_rows(
    snapshot: SnapshotRef,
    gateway_resolver: GatewayResolver,
    *,
    options: HistoryTimeseriesOptions,
    runner: ToolRunner | None = None,
) -> pl.LazyFrame:
    """Build history_timeseries rows without writing to database.

    Compute cross-commit history aggregation for functions and modules,
    returning a columnar LazyFrame suitable for materialization.

    Parameters
    ----------
    snapshot
        Snapshot reference with repo, commit, and repo_root.
    gateway_resolver
        Callable returning a storage gateway for a given commit.
    options
        History timeseries configuration options.
    runner
        Optional ToolRunner for git timestamp lookups. When omitted, commit timestamps
        fall back to the current time.

    Returns
    -------
    pl.LazyFrame
        Lazy frame matching the schema order, ready for materialization.
    """
    if not options.commits:
        log.info("No commits provided for history_timeseries; skipping.")
        return empty_lazyframe_for_table(HISTORY_TIMESERIES_TABLE_KEY)

    selection = _select_entities(snapshot, options, gateway_resolver)
    if not selection.functions and not selection.modules:
        log.info("No entities selected for history_timeseries; skipping.")
        return empty_lazyframe_for_table(HISTORY_TIMESERIES_TABLE_KEY)

    now = datetime.now(tz=UTC)
    buffer = columnar_buffer_for_table_key(HISTORY_TIMESERIES_TABLE_KEY)
    for commit in options.commits:
        snapshot_gateway = gateway_resolver(commit)
        commit_ts = _fetch_commit_timestamp(snapshot.repo_root, commit, runner) or now
        commit_ctx = CommitContext(commit=commit, commit_ts=commit_ts, created_at=now)

        if options.entity_kind in {"function", "both"}:
            _append_rows(
                buffer,
                _collect_function_rows_for_commit(
                    snapshot,
                    snapshot_gateway,
                    commit_ctx=commit_ctx,
                    selection=selection.functions,
                ),
            )
        if options.entity_kind in {"module", "both"}:
            _append_rows(
                buffer,
                _collect_module_rows_for_commit(
                    snapshot,
                    snapshot_gateway,
                    commit_ctx=commit_ctx,
                    selection=selection.modules,
                ),
            )

    log.info(
        "history_timeseries computed: %s rows for %s commits",
        buffer.row_count,
        len(options.commits),
    )
    return lazyframe_for_table_columns(HISTORY_TIMESERIES_TABLE_KEY, buffer.data)


def _append_rows(buffer: ColumnarRowBuffer, rows: Iterable[dict[str, object]]) -> None:
    for row in rows:
        buffer.append(row)


def _select_entities(
    snapshot: SnapshotRef,
    options: HistoryTimeseriesOptions,
    gateway_resolver: GatewayResolver,
) -> EntitySelection:
    """Select top entities from the first commit for tracking.

    Parameters
    ----------
    snapshot
        Snapshot reference.
    options
        Timeseries options.
    gateway_resolver
        Callable returning a storage gateway for a given commit.

    Returns
    -------
    EntitySelection
        Selected function and module stable IDs.
    """
    base_commit = options.commits[0]
    gateway = gateway_resolver(base_commit)
    functions = _select_top_functions(gateway, snapshot, options, base_commit)
    modules = _select_top_modules(gateway, snapshot, options, base_commit)
    return EntitySelection(functions=functions, modules=modules)


def _select_top_functions(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    options: HistoryTimeseriesOptions,
    commit: str,
) -> set[str]:
    """Select top functions by risk score.

    Parameters
    ----------
    gateway
        Storage gateway.
    snapshot
        Snapshot reference.
    options
        Timeseries options.
    commit
        Commit SHA.

    Returns
    -------
    set[str]
        Set of stable entity IDs.
    """
    relation = gateway.relation_from_table_key("analytics.function_profile")
    predicate = (ColumnExpression("repo") == ConstantExpression(snapshot.repo)) & (
        ColumnExpression("commit") == ConstantExpression(commit)
    )
    rows = (
        relation.filter(predicate)
        .order("risk_score DESC")
        .select("rel_path", "language", "qualname")
        .limit(options.max_entities)
        .fetchall()
    )
    return {
        make_entity_stable_id(
            repo=snapshot.repo,
            rel_path=str(rel_path),
            language=str(language),
            kind="function",
            qualname=str(qualname),
        )
        for rel_path, language, qualname in rows
    }


def _select_top_modules(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    options: HistoryTimeseriesOptions,
    commit: str,
) -> set[str]:
    """Select top modules by max risk score.

    Parameters
    ----------
    gateway
        Storage gateway.
    snapshot
        Snapshot reference.
    options
        Timeseries options.
    commit
        Commit SHA.

    Returns
    -------
    set[str]
        Set of stable entity IDs.
    """
    relation = gateway.relation_from_table_key("analytics.module_profile")
    predicate = (ColumnExpression("repo") == ConstantExpression(snapshot.repo)) & (
        ColumnExpression("commit") == ConstantExpression(commit)
    )
    rows = (
        relation.filter(predicate)
        .order("max_risk_score DESC")
        .select("path", "language", "module")
        .limit(options.max_entities)
        .fetchall()
    )
    return {
        make_entity_stable_id(
            repo=snapshot.repo,
            rel_path=str(path),
            language=str(language),
            kind="module",
            qualname="",
        )
        for path, language, _ in rows
    }


def _collect_function_rows_for_commit(
    snapshot: SnapshotRef,
    gateway: StorageGateway,
    *,
    commit_ctx: CommitContext,
    selection: set[str],
) -> Iterable[dict[str, object]]:
    """Collect function timeseries rows for a single commit.

    Parameters
    ----------
    snapshot
        Snapshot reference.
    gateway
        Storage gateway for the commit.
    commit_ctx
        Commit context with timestamps.
    selection
        Set of stable IDs to include.

    Yields
    ------
    dict[str, object]
        Row mappings for analytics.history_timeseries.
    """
    relation = gateway.relation_from_table_key("analytics.function_profile")
    predicate = (ColumnExpression("repo") == ConstantExpression(snapshot.repo)) & (
        ColumnExpression("commit") == ConstantExpression(commit_ctx.commit)
    )
    rows = (
        relation.filter(predicate)
        .select(
            "function_goid_h128",
            "rel_path",
            "module",
            "language",
            "qualname",
            "loc",
            "cyclomatic_complexity",
            "coverage_ratio",
            "static_error_count",
            "typedness_bucket",
            "risk_score",
            "risk_level",
        )
        .fetchall()
    )

    for (
        goid,
        rel_path,
        module,
        language,
        qualname,
        loc,
        cyclomatic_complexity,
        coverage_ratio,
        static_error_count,
        typedness_bucket,
        risk_score,
        risk_level,
    ) in rows:
        stable_id = make_entity_stable_id(
            repo=snapshot.repo,
            rel_path=str(rel_path),
            language=str(language),
            kind="function",
            qualname=str(qualname),
        )
        if stable_id not in selection:
            continue
        goid_val = int(goid)
        yield {
            "repo": snapshot.repo,
            "entity_kind": "function",
            "entity_stable_id": stable_id,
            "function_goid_h128": goid_val,
            "module": str(module),
            "rel_path": str(rel_path),
            "language": str(language),
            "qualname": str(qualname),
            "commit": commit_ctx.commit,
            "commit_ts": commit_ctx.commit_ts,
            "loc": _safe_number(loc),
            "cyclomatic_complexity": _safe_number(cyclomatic_complexity),
            "coverage_ratio": _safe_number(coverage_ratio),
            "static_error_count": _safe_number(static_error_count),
            "typedness_bucket": typedness_bucket,
            "risk_score": _safe_number(risk_score),
            "risk_level": risk_level,
            "bucket_label": commit_ctx.commit_ts.date().isoformat(),
            "created_at_row": commit_ctx.created_at,
        }


def _collect_module_rows_for_commit(
    snapshot: SnapshotRef,
    gateway: StorageGateway,
    *,
    commit_ctx: CommitContext,
    selection: set[str],
) -> Iterable[dict[str, object]]:
    """Collect module timeseries rows for a single commit.

    Parameters
    ----------
    snapshot
        Snapshot reference.
    gateway
        Storage gateway for the commit.
    commit_ctx
        Commit context with timestamps.
    selection
        Set of stable IDs to include.

    Yields
    ------
    dict[str, object]
        Row mappings for analytics.history_timeseries.
    """
    relation = gateway.relation_from_table_key("analytics.module_profile")
    predicate = (ColumnExpression("repo") == ConstantExpression(snapshot.repo)) & (
        ColumnExpression("commit") == ConstantExpression(commit_ctx.commit)
    )
    rows = (
        relation.filter(predicate)
        .select(
            "module",
            "path",
            "language",
            "module_coverage_ratio",
            "max_risk_score",
            "avg_risk_score",
            "role",
            "role_confidence",
        )
        .fetchall()
    )

    for (
        module,
        path,
        language,
        module_coverage_ratio,
        max_risk_score,
        _avg_risk_score,
        _role,
        _role_confidence,
    ) in rows:
        stable_id = make_entity_stable_id(
            repo=snapshot.repo,
            rel_path=str(path),
            language=str(language),
            kind="module",
            qualname="",
        )
        if stable_id not in selection:
            continue
        yield {
            "repo": snapshot.repo,
            "entity_kind": "module",
            "entity_stable_id": stable_id,
            "function_goid_h128": None,
            "module": str(module),
            "rel_path": str(path),
            "language": str(language),
            "qualname": None,
            "commit": commit_ctx.commit,
            "commit_ts": commit_ctx.commit_ts,
            "loc": None,
            "cyclomatic_complexity": None,
            "coverage_ratio": _safe_number(module_coverage_ratio),
            "static_error_count": None,
            "typedness_bucket": None,
            "risk_score": _safe_number(max_risk_score),
            "risk_level": None,
            "bucket_label": commit_ctx.commit_ts.date().isoformat(),
            "created_at_row": commit_ctx.created_at,
        }


def _fetch_commit_timestamp(
    repo_root: Path,
    commit: str,
    runner: ToolRunner | None = None,
) -> datetime | None:
    args = ["git", "show", "-s", "--format=%cI", commit]
    if runner is None:
        log.warning("ToolRunner not provided for git timestamps; using fallback for %s", commit)
        return None
    result = runner.run(
        "git",
        args,
        options=ToolRunOptions(cwd=repo_root),
    )
    if result.returncode != 0:
        log.warning(
            "git show failed for %s: code=%s stderr=%s",
            commit,
            result.returncode,
            result.stderr[:500],
        )
        return None
    ts_raw = result.stdout.strip()
    try:
        dt = datetime.fromisoformat(ts_raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)
