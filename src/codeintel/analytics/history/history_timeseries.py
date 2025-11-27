"""Cross-commit history aggregation for functions and modules."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import SupportsFloat, SupportsIndex

from codeintel.config import HistoryTimeseriesStepConfig
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.storage.gateway import (
    DuckDBConnection,
    SnapshotGatewayResolver,
    StorageGateway,
)
from codeintel.storage.sql_helpers import ensure_schema

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


def compute_history_timeseries(
    history_con: DuckDBConnection,
    cfg: HistoryTimeseriesStepConfig,
    db_resolver: DBResolver,
    *,
    runner: ToolRunner | None = None,
) -> None:
    """
    Populate `analytics.history_timeseries` across multiple commits.

    Parameters
    ----------
    history_con:
        Connection to the database where history rows will be written.
    cfg:
        History aggregation configuration.
    db_resolver:
        Callable returning a DuckDB connection for a given commit.
    runner:
        Optional ToolRunner for git timestamp lookups.
    """
    if not cfg.commits:
        log.info("No commits provided for history_timeseries; skipping.")
        return

    ensure_schema(history_con, "analytics.history_timeseries")
    history_con.execute(
        "DELETE FROM analytics.history_timeseries WHERE repo = ?",
        [cfg.repo],
    )

    selection = _select_entities(cfg, db_resolver)
    if not selection.functions and not selection.modules:
        log.info("No entities selected for history_timeseries; skipping.")
        return

    now = datetime.now(tz=UTC)
    rows: list[tuple[object, ...]] = []
    for commit in cfg.commits:
        con_ci = db_resolver(commit)
        commit_ts = _fetch_commit_timestamp(cfg.repo_root, commit, runner) or now
        commit_ctx = CommitContext(commit=commit, commit_ts=commit_ts, created_at=now)

        if cfg.entity_kind in {"function", "both"}:
            rows.extend(
                _collect_function_rows_for_commit(
                    cfg,
                    con_ci,
                    commit_ctx=commit_ctx,
                    selection=selection.functions,
                )
            )
        if cfg.entity_kind in {"module", "both"}:
            rows.extend(
                _collect_module_rows_for_commit(
                    cfg,
                    con_ci,
                    commit_ctx=commit_ctx,
                    selection=selection.modules,
                )
            )

    history_con.executemany(
        """
        INSERT INTO analytics.history_timeseries (
            repo,
            entity_kind,
            entity_stable_id,
            function_goid_h128,
            module,
            rel_path,
            language,
            qualname,
            commit,
            commit_ts,
            loc,
            cyclomatic_complexity,
            coverage_ratio,
            static_error_count,
            typedness_bucket,
            risk_score,
            risk_level,
            bucket_label,
            created_at_row
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    log.info(
        "history_timeseries populated: %s rows for %s commits",
        len(rows),
        len(cfg.commits),
    )


def compute_history_timeseries_gateways(
    history_gateway: StorageGateway,
    cfg: HistoryTimeseriesStepConfig,
    snapshot_resolver: SnapshotGatewayResolver,
    *,
    runner: ToolRunner | None = None,
) -> None:
    """
    Gateway-based wrapper around compute_history_timeseries.

    Parameters
    ----------
    history_gateway:
        StorageGateway for the destination history DuckDB database.
    cfg:
        History aggregation configuration.
    snapshot_resolver:
        Callable returning a StorageGateway bound to the per-commit snapshot DB.
    runner:
        Optional ToolRunner for git timestamp lookups.
    """
    snapshot_gateways: dict[str, StorageGateway] = {}

    def _db_resolver(commit: str) -> DuckDBConnection:
        cached_gateway = snapshot_gateways.get(commit)
        if cached_gateway is not None:
            return history_gateway.con if cached_gateway is history_gateway else cached_gateway.con

        snapshot_gateway = snapshot_resolver(commit)
        if snapshot_gateway.config.db_path.resolve() == history_gateway.config.db_path.resolve():
            if snapshot_gateway is not history_gateway:
                snapshot_gateway.close()
            snapshot_gateways[commit] = history_gateway
            return history_gateway.con

        snapshot_gateways[commit] = snapshot_gateway
        return snapshot_gateway.con

    try:
        compute_history_timeseries(
            history_gateway.con,
            cfg,
            _db_resolver,
            runner=runner,
        )
    finally:
        for gateway in snapshot_gateways.values():
            if gateway is history_gateway:
                continue
            gateway.close()


def _select_entities(cfg: HistoryTimeseriesStepConfig, db_resolver: DBResolver) -> EntitySelection:
    base_commit = cfg.commits[0]
    con = db_resolver(base_commit)
    functions = _select_top_functions(con, cfg, base_commit)
    modules = _select_top_modules(con, cfg, base_commit)
    return EntitySelection(functions=functions, modules=modules)


def _select_top_functions(
    con: DuckDBConnection,
    cfg: HistoryTimeseriesStepConfig,
    commit: str,
) -> set[str]:
    rows = con.execute(
        """
        SELECT
            rel_path,
            language,
            qualname
        FROM analytics.function_profile
        WHERE repo = ? AND commit = ?
        ORDER BY risk_score DESC NULLS LAST
        LIMIT ?
        """,
        [cfg.repo, commit, cfg.max_entities],
    ).fetchall()
    return {
        make_entity_stable_id(
            repo=cfg.repo,
            rel_path=str(rel_path),
            language=str(language),
            kind="function",
            qualname=str(qualname),
        )
        for rel_path, language, qualname in rows
    }


def _select_top_modules(
    con: DuckDBConnection,
    cfg: HistoryTimeseriesStepConfig,
    commit: str,
) -> set[str]:
    rows = con.execute(
        """
        SELECT
            path,
            language,
            module
        FROM analytics.module_profile
        WHERE repo = ? AND commit = ?
        ORDER BY max_risk_score DESC NULLS LAST
        LIMIT ?
        """,
        [cfg.repo, commit, cfg.max_entities],
    ).fetchall()
    return {
        make_entity_stable_id(
            repo=cfg.repo,
            rel_path=str(path),
            language=str(language),
            kind="module",
            qualname="",
        )
        for path, language, _ in rows
    }


def _collect_function_rows_for_commit(
    cfg: HistoryTimeseriesStepConfig,
    con_ci: DuckDBConnection,
    *,
    commit_ctx: CommitContext,
    selection: set[str],
) -> Iterable[tuple[object, ...]]:
    rows = con_ci.execute(
        """
        SELECT
            function_goid_h128,
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
            risk_level
        FROM analytics.function_profile
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, commit_ctx.commit],
    ).fetchall()

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
            repo=cfg.repo,
            rel_path=str(rel_path),
            language=str(language),
            kind="function",
            qualname=str(qualname),
        )
        if stable_id not in selection:
            continue
        goid_val = int(goid)
        yield (
            cfg.repo,
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
    cfg: HistoryTimeseriesStepConfig,
    con_ci: DuckDBConnection,
    *,
    commit_ctx: CommitContext,
    selection: set[str],
) -> Iterable[tuple[object, ...]]:
    rows = con_ci.execute(
        """
        SELECT
            module,
            path,
            language,
            module_coverage_ratio,
            max_risk_score,
            avg_risk_score,
            role,
            role_confidence
        FROM analytics.module_profile
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, commit_ctx.commit],
    ).fetchall()

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
            repo=cfg.repo,
            rel_path=str(path),
            language=str(language),
            kind="module",
            qualname="",
        )
        if stable_id not in selection:
            continue
        yield (
            cfg.repo,
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
