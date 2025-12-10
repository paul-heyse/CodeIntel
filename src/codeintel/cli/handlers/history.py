"""Handlers for history timeseries commands.

Migrate to use HandlerContext and return CliResult.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.cli.core import CliResult
from codeintel.cli.errors import ProblemDetail
from codeintel.cli.handlers.base import setup_logging
from codeintel.cli.handlers.context import HandlerContext
from codeintel.config import ConfigBuilder, SnapshotInit
from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.storage.gateway import (
    DuckDBError,
    StorageConfig,
    build_snapshot_gateway_resolver,
    open_gateway,
)

LOG = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Result Types
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryTimeseriesResult:
    """Result from history timeseries aggregation.

    Parameters
    ----------
    output_db
        Path to the output database.
    commits_processed
        Number of commits processed.
    entity_kind
        Entity kind used for aggregation.
    """

    output_db: str
    commits_processed: int
    entity_kind: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "output_db": self.output_db,
            "commits_processed": self.commits_processed,
            "entity_kind": self.entity_kind,
        }


def _make_error(title: str, detail: str, status: int = 1) -> ProblemDetail:
    """Create a ProblemDetail for an error.

    Parameters
    ----------
    title
        Short, human-readable summary of the problem.
    detail
        Human-readable explanation specific to this occurrence.
    status
        Exit code for this error.

    Returns
    -------
    ProblemDetail
        Structured error.
    """
    return ProblemDetail(
        type=f"urn:codeintel:cli:history:{title.lower().replace(' ', '-')}",
        title=title,
        status=status,
        detail=detail,
    )


# -----------------------------------------------------------------------------
# Handler
# -----------------------------------------------------------------------------


def history_timeseries_handler(ctx: HandlerContext) -> CliResult[HistoryTimeseriesResult]:
    """Aggregate analytics.history_timeseries across multiple commits.

    Collects analytics data from per-commit DuckDB snapshots and aggregates
    them into a unified history timeseries table.

    Parameters
    ----------
    ctx
        Handler context with params:
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
    setup_logging(ctx.verbosity)

    # Check commits first since it's a command-specific required parameter
    commits = ctx.param_list("commits")

    if not commits:
        return CliResult.fail(_make_error("Validation Error", "At least one commit is required"))

    repo = ctx.require_str("repo")

    repo_root = ctx.param_path("repo_root", Path.cwd()) or Path.cwd()
    db_dir = ctx.param_path("db_dir", Path("build/db")) or Path("build/db")
    output_db = ctx.param_path("output_db", Path("build/db/history.duckdb")) or Path(
        "build/db/history.duckdb"
    )
    entity_kind = ctx.param_str("entity_kind", "function") or "function"
    max_entities = ctx.param_int("max_entities", 500)
    selection_strategy = ctx.param_str("selection_strategy", "risk_score") or "risk_score"

    runner = ToolRunner(cache_dir=repo_root / "build" / ".tool_cache")
    builder = ConfigBuilder.from_snapshot(
        snapshot=SnapshotInit(repo=repo, commit=commits[0], repo_root=repo_root),
    )
    cfg = builder.history_timeseries(
        commits=tuple(commits),
        entity_kind=entity_kind,
        max_entities=max_entities,
        selection_strategy=selection_strategy,
    )

    storage_cfg = StorageConfig.for_ingest(output_db)
    gateway = open_gateway(storage_cfg)
    snapshot_resolver = build_snapshot_gateway_resolver(
        db_dir=db_dir,
        repo=repo,
        primary_gateway=gateway,
    )

    try:
        compute_history_timeseries_gateways(gateway, cfg, snapshot_resolver, runner=runner)
    except FileNotFoundError as exc:
        LOG.exception("Missing snapshot database for history_timeseries")
        gateway.close()
        return CliResult.fail(
            _make_error(
                "Storage Error", f"Missing snapshot database for one or more commits: {exc}"
            )
        )
    except DuckDBError as exc:
        LOG.exception("Failed to compute history_timeseries")
        gateway.close()
        return CliResult.fail(
            _make_error("Query Error", f"Failed to compute history_timeseries: {exc}")
        )

    gateway.close()

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
