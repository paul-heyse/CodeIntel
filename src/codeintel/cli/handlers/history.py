"""Handlers for history timeseries commands.

Provide analytics aggregation across multiple commit snapshots.

This handler writes to a dedicated output database (not the runtime's database),
so it uses explicit gateway management rather than ctx.gateway.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ibis.common.exceptions import IbisError

from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.analytics.utilities.datasets import get_analytics_dataset_contract
from codeintel.cli.context import CommandContext
from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import HistoryTimeseriesResult
from codeintel.cli.errors.results import fail_history_error
from codeintel.cli.execution.bootstrap import bootstrap_cli
from codeintel.config import ConfigBuilder, SnapshotInit
from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.gateway import (
    DuckDBError,
    DuckDBInvalidInputException,
    StorageConfig,
    build_snapshot_gateway_resolver,
    open_gateway,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

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


def _write_synthetic_history_rows(
    *,
    repo: str,
    commits: list[str],
    gateway: StorageGateway,
    snapshot_resolver: Callable[[str], StorageGateway],
) -> None:
    """Backfill history_timeseries when no rows exist by projecting profiles."""
    contract = get_analytics_dataset_contract(gateway, "analytics.history_timeseries")
    columns = contract.schema.column_names() if contract.schema is not None else ()
    backend = DuckDBPolicyBackend(gateway)
    synthetic_rows: list[dict[str, object]] = []

    for commit in commits:
        commit_timestamp = datetime.now(UTC)
        snapshot_gateway = snapshot_resolver(commit)
        try:
            profile = snapshot_gateway.ibis.table("analytics.function_profile")
            df = profile.select("repo", "module", "rel_path", "qualname").execute()
        except (DuckDBError, IbisError, RuntimeError, ValueError, TypeError) as load_exc:
            LOG.warning(
                "Failed to synthesize history rows from function_profile for %s@%s: %s",
                repo,
                commit,
                load_exc,
            )
            df = None
        finally:
            if snapshot_gateway is not gateway:
                snapshot_gateway.close()

        if df is None:
            continue

        records = cast("list[dict[str, object]]", df.to_dict("records"))
        for record in records:
            rel_path = record["rel_path"]
            qualname = record["qualname"]
            synthetic_rows.append(
                {
                    "repo": record["repo"],
                    "entity_kind": "function",
                    "entity_stable_id": qualname or rel_path,
                    "function_goid_h128": None,
                    "module": record["module"],
                    "rel_path": rel_path,
                    "language": "python",
                    "qualname": qualname,
                    "commit": commit,
                    "commit_ts": commit_timestamp,
                    "loc": None,
                    "cyclomatic_complexity": None,
                    "coverage_ratio": None,
                    "static_error_count": None,
                    "typedness_bucket": None,
                    "risk_score": None,
                    "risk_level": None,
                    "bucket_label": None,
                    "created_at_row": commit_timestamp,
                }
            )

    if not synthetic_rows:
        return

    for commit in commits:
        backend.delete_for_snapshot(
            contract.table_key,
            repo=repo,
            commit=commit,
        )
    backend.bulk_insert(
        contract.table_key,
        [contract.to_tuple(row) for row in synthetic_rows],
        columns=columns,
    )


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
            _write_synthetic_history_rows(
                repo=repo,
                commits=commits,
                gateway=gateway,
                snapshot_resolver=snapshot_resolver,
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
