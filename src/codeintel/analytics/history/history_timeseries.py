"""Cross-commit history aggregation for functions and modules.

For new code, use ``build_history_timeseries_rows`` with Hamilton materializers.

The Hamilton native module is at:
``codeintel.build.hamilton.native.analytics.history_timeseries``
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, SupportsFloat, SupportsIndex

import ibis

from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.storage.gateway import (
    DuckDBConnection,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import (
        StorageGateway,
    )


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


HISTORY_TIMESERIES_COLS = [
    "repo",
    "entity_kind",
    "entity_stable_id",
    "function_goid_h128",
    "module",
    "rel_path",
    "language",
    "qualname",
    "commit",
    "commit_ts",
    "loc",
    "cyclomatic_complexity",
    "coverage_ratio",
    "static_error_count",
    "typedness_bucket",
    "risk_score",
    "risk_level",
    "bucket_label",
    "created_at_row",
]

log = logging.getLogger(__name__)

DBResolver = Callable[[str], DuckDBConnection]
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
    return hashlib.sha256(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:20]


def _safe_number(value: NumericLike | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_history_timeseries_rows(
    history_gateway: StorageGateway,
    snapshot: SnapshotRef,
    db_resolver: DBResolver,
    *,
    options: HistoryTimeseriesOptions,
    runner: ToolRunner | None = None,
) -> tuple[tuple[object, ...], ...]:
    """Build history_timeseries rows without writing to database.

    Compute cross-commit history aggregation for functions and modules,
    returning row tuples suitable for materialization via Hamilton materializers.

    Parameters
    ----------
    history_gateway
        Gateway to the database (used for entity selection queries).
    snapshot
        Snapshot reference with repo, commit, and repo_root.
    db_resolver
        Callable returning a DuckDB connection for a given commit.
    options
        History timeseries configuration options.
    runner
        Optional ToolRunner for git timestamp lookups.

    Returns
    -------
    tuple[tuple[object, ...], ...]
        Row tuples matching HISTORY_TIMESERIES_COLS schema, ready for bulk insert.
    """
    # history_gateway is kept for API consistency but not used directly;
    # entity selection uses db_resolver to resolve per-commit connections
    _ = history_gateway

    if not options.commits:
        log.info("No commits provided for history_timeseries; skipping.")
        return ()

    selection = _select_entities(snapshot, options, db_resolver)
    if not selection.functions and not selection.modules:
        log.info("No entities selected for history_timeseries; skipping.")
        return ()

    now = datetime.now(tz=UTC)
    rows: list[tuple[object, ...]] = []
    for commit in options.commits:
        con_ci = db_resolver(commit)
        commit_ts = _fetch_commit_timestamp(snapshot.repo_root, commit, runner) or now
        commit_ctx = CommitContext(commit=commit, commit_ts=commit_ts, created_at=now)

        if options.entity_kind in {"function", "both"}:
            rows.extend(
                _collect_function_rows_for_commit(
                    snapshot,
                    options,
                    con_ci,
                    commit_ctx=commit_ctx,
                    selection=selection.functions,
                )
            )
        if options.entity_kind in {"module", "both"}:
            rows.extend(
                _collect_module_rows_for_commit(
                    snapshot,
                    options,
                    con_ci,
                    commit_ctx=commit_ctx,
                    selection=selection.modules,
                )
            )

    log.info(
        "history_timeseries computed: %s rows for %s commits",
        len(rows),
        len(options.commits),
    )
    return tuple(rows)


def _select_entities(
    snapshot: SnapshotRef,
    options: HistoryTimeseriesOptions,
    db_resolver: DBResolver,
) -> EntitySelection:
    """Select top entities from the first commit for tracking.

    Parameters
    ----------
    snapshot
        Snapshot reference.
    options
        Timeseries options.
    db_resolver
        Callable to resolve DuckDB connection by commit.

    Returns
    -------
    EntitySelection
        Selected function and module stable IDs.
    """
    base_commit = options.commits[0]
    con = db_resolver(base_commit)
    functions = _select_top_functions(con, snapshot, options, base_commit)
    modules = _select_top_modules(con, snapshot, options, base_commit)
    return EntitySelection(functions=functions, modules=modules)


def _select_top_functions(
    con: DuckDBConnection,
    snapshot: SnapshotRef,
    options: HistoryTimeseriesOptions,
    commit: str,
) -> set[str]:
    """Select top functions by risk score.

    Parameters
    ----------
    con
        DuckDB connection.
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
    conn = ibis.duckdb.from_connection(con)
    table = conn.table("function_profile", database="analytics")
    rows_df = (
        table.filter((table.repo == snapshot.repo) & (table.commit == commit))
        .order_by(ibis.desc(table.risk_score))
        .select("rel_path", "language", "qualname")
        .limit(options.max_entities)
        .execute()
    )
    rows = rows_df.itertuples(index=False, name=None)
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
    con: DuckDBConnection,
    snapshot: SnapshotRef,
    options: HistoryTimeseriesOptions,
    commit: str,
) -> set[str]:
    """Select top modules by max risk score.

    Parameters
    ----------
    con
        DuckDB connection.
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
    conn = ibis.duckdb.from_connection(con)
    table = conn.table("module_profile", database="analytics")
    rows_df = (
        table.filter((table.repo == snapshot.repo) & (table.commit == commit))
        .order_by(ibis.desc(table.max_risk_score))
        .select("path", "language", "module")
        .limit(options.max_entities)
        .execute()
    )
    rows = rows_df.itertuples(index=False, name=None)
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
    options: HistoryTimeseriesOptions,
    con_ci: DuckDBConnection,
    *,
    commit_ctx: CommitContext,
    selection: set[str],
) -> Iterable[tuple[object, ...]]:
    """Collect function timeseries rows for a single commit.

    Parameters
    ----------
    snapshot
        Snapshot reference.
    options
        Timeseries options (unused but kept for API consistency).
    con_ci
        DuckDB connection for the commit.
    commit_ctx
        Commit context with timestamps.
    selection
        Set of stable IDs to include.

    Yields
    ------
    tuple[object, ...]
        Row tuples for analytics.history_timeseries.
    """
    _ = options  # Unused, kept for signature consistency
    conn = ibis.duckdb.from_connection(con_ci)
    table = conn.table("function_profile", database="analytics")
    rows_df = (
        table.filter((table.repo == snapshot.repo) & (table.commit == commit_ctx.commit))
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
        .execute()
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
    ) in rows_df.itertuples(index=False, name=None):
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
        yield (
            snapshot.repo,
            "function",
            stable_id,
            goid_val,
            str(module),
            str(rel_path),
            str(language),
            str(qualname),
            commit_ctx.commit,
            commit_ctx.commit_ts,
            _safe_number(loc),
            _safe_number(cyclomatic_complexity),
            _safe_number(coverage_ratio),
            _safe_number(static_error_count),
            typedness_bucket,
            _safe_number(risk_score),
            risk_level,
            commit_ctx.commit_ts.date().isoformat(),
            commit_ctx.created_at,
        )


def _collect_module_rows_for_commit(
    snapshot: SnapshotRef,
    options: HistoryTimeseriesOptions,
    con_ci: DuckDBConnection,
    *,
    commit_ctx: CommitContext,
    selection: set[str],
) -> Iterable[tuple[object, ...]]:
    """Collect module timeseries rows for a single commit.

    Parameters
    ----------
    snapshot
        Snapshot reference.
    options
        Timeseries options (unused but kept for API consistency).
    con_ci
        DuckDB connection for the commit.
    commit_ctx
        Commit context with timestamps.
    selection
        Set of stable IDs to include.

    Yields
    ------
    tuple[object, ...]
        Row tuples for analytics.history_timeseries.
    """
    _ = options  # Unused, kept for signature consistency
    conn = ibis.duckdb.from_connection(con_ci)
    table = conn.table("module_profile", database="analytics")
    rows_df = (
        table.filter((table.repo == snapshot.repo) & (table.commit == commit_ctx.commit))
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
        .execute()
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
    ) in rows_df.itertuples(index=False, name=None):
        stable_id = make_entity_stable_id(
            repo=snapshot.repo,
            rel_path=str(path),
            language=str(language),
            kind="module",
            qualname="",
        )
        if stable_id not in selection:
            continue
        yield (
            snapshot.repo,
            "module",
            stable_id,
            None,
            str(module),
            str(path),
            str(language),
            None,
            commit_ctx.commit,
            commit_ctx.commit_ts,
            None,
            None,
            _safe_number(module_coverage_ratio),
            None,
            None,
            _safe_number(max_risk_score),
            None,
            commit_ctx.commit_ts.date().isoformat(),
            commit_ctx.created_at,
        )


def _fetch_commit_timestamp(
    repo_root: Path,
    commit: str,
    runner: ToolRunner | None = None,
) -> datetime | None:
    args = ["git", "show", "-s", "--format=%cI", commit]
    active_runner = runner or ToolRunner(cache_dir=repo_root / "build" / ".tool_cache")
    result = active_runner.run("git", args, cwd=repo_root)
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
