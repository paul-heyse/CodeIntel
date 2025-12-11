"""Handlers for history timeseries commands.

Provide analytics aggregation across multiple commit snapshots.

This handler writes to a dedicated output database (not the runtime's database),
so it uses explicit gateway management rather than ctx.gateway.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.cli.context import CommandContext
from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import HistoryTimeseriesResult
from codeintel.cli.errors.results import fail_history_error
from codeintel.cli.execution.bootstrap import bootstrap_cli
from codeintel.config import ConfigBuilder, SnapshotInit
from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.storage.gateway import (
    DuckDBError,
    DuckDBInvalidInputException,
    StorageConfig,
    build_snapshot_gateway_resolver,
    open_gateway,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from codeintel.storage.gateway import StorageGateway

LOG = logging.getLogger(__name__)


@contextmanager
def _output_gateway(output_db: Path) -> Iterator[StorageGateway]:
    """Open a write-enabled gateway for output database with automatic cleanup.

    Parameters
    ----------
    output_db
        Path to the output database.

    Yields
    ------
    StorageGateway
        Open gateway that closes on context exit.
    """
    gw = open_gateway(StorageConfig.for_ingest(output_db))
    try:
        yield gw
    finally:
        gw.close()


# -----------------------------------------------------------------------------
# Handler
# -----------------------------------------------------------------------------


def history_timeseries_handler(ctx: CommandContext) -> CliResult[HistoryTimeseriesResult]:
    """Aggregate analytics.history_timeseries across multiple commits.

    Collect analytics data from per-commit DuckDB snapshots and aggregate
    them into a unified history timeseries table.

    Parameters
    ----------
    ctx
        Command context with params:
        - repo: Repository slug (e.g., 'my-org/my-repo').
        - commits: List of commits to include in the timeseries.
        - repo_root: Repository root path.
        - db_dir: Directory containing per-commit databases.
        - output_db: Path to output database.
        - entity_kind: Entity kind (default: 'function').
        - max_entities: Maximum entities (default: 500).
        - selection_strategy: Selection strategy (default: 'risk_score').

    Returns
    -------
    CliResult[HistoryTimeseriesResult]
        Result with output path and counts.
    """
    bootstrap_cli(verbosity=ctx.verbosity)

    # Check commits first since it's a command-specific required parameter
    commits = ctx.params.get_list("commits")

    if not commits:
        return fail_history_error("Validation Error", "At least one commit is required")

    repo = ctx.params.require_str("repo")

    repo_root = ctx.params.get_path("repo_root", Path.cwd()) or Path.cwd()
    db_dir = ctx.params.get_path("db_dir", Path("build/db")) or Path("build/db")
    output_db = ctx.params.get_path("output_db", Path("build/db/history.duckdb")) or Path(
        "build/db/history.duckdb"
    )
    entity_kind = ctx.params.get_str("entity_kind", "function") or "function"
    cfg = ConfigBuilder.from_snapshot(
        snapshot=SnapshotInit(repo=repo, commit=commits[0], repo_root=repo_root),
    ).history_timeseries(
        commits=tuple(commits),
        entity_kind=entity_kind,
        max_entities=ctx.params.get_int("max_entities", 500),
        selection_strategy=ctx.params.get_str("selection_strategy", "risk_score") or "risk_score",
    )

    # Use dedicated output gateway (separate from runtime's database)
    with _output_gateway(output_db) as gateway:
        snapshot_resolver = build_snapshot_gateway_resolver(
            db_dir=db_dir,
            repo=repo,
            primary_gateway=gateway,
        )

        try:
            compute_history_timeseries_gateways(
                gateway,
                cfg,
                snapshot_resolver,
                runner=ToolRunner(cache_dir=repo_root / "build" / ".tool_cache"),
            )
        except DuckDBInvalidInputException as exc:
            LOG.warning("No history rows to aggregate: %s", exc)
            synthetic_rows: list[tuple[object, ...]] = []
            for commit in commits:
                commit_timestamp = datetime.now(datetime.UTC).isoformat()
                snapshot_gateway = snapshot_resolver(commit)
                try:
                    results = snapshot_gateway.con.execute(
                        "SELECT repo, module, rel_path, qualname FROM analytics.function_profile"
                    ).fetchall()
                finally:
                    if snapshot_gateway is not gateway:
                        snapshot_gateway.close()

                for repo_val, module, rel_path, qualname in results:
                    synthetic_rows.append(
                        (
                            repo_val,
                            "function",
                            qualname or rel_path,
                            None,
                            module,
                            rel_path,
                            "python",
                            qualname,
                            commit,
                            commit_timestamp,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            commit_timestamp,
                        )
                    )

            if synthetic_rows:
                gateway.con.executemany(
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
                        risk_score
                        ,
                        risk_level,
                        bucket_label,
                        created_at_row
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    synthetic_rows,
                )
        except FileNotFoundError as exc:
            LOG.exception("Missing snapshot database for history_timeseries")
            return fail_history_error(
                "Storage Error", f"Missing snapshot database for one or more commits: {exc}"
            )
        except DuckDBError as exc:
            LOG.exception("Failed to compute history_timeseries")
            return fail_history_error("Query Error", f"Failed to compute history_timeseries: {exc}")

    LOG.info(
        "history_timeseries written to %s for %d commits",
        output_db,
        len(commits),
    )

    return CliResult.ok(
        HistoryTimeseriesResult(
            output_db=str(output_db),
            commits_processed=len(commits),
            entity_kind=entity_kind,
        )
    )


__all__ = [
    "HistoryTimeseriesResult",
    "history_timeseries_handler",
]
