"""History timeseries commands for the CodeIntel CLI.

This module provides Typer commands for aggregating analytics data
across multiple commits to build historical timeseries.

Commands
--------
- **timeseries**: Aggregate analytics.history_timeseries across commits
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.cli.commands._common import VerboseOpt, setup_logging
from codeintel.config import ConfigBuilder
from codeintel.ingestion.infrastructure_utilities.tool_runner import ToolRunner
from codeintel.storage.config import StorageConfig
from codeintel.storage.gateway import DuckDBError, build_snapshot_gateway_resolver, open_gateway

LOG = logging.getLogger(__name__)

history_app = typer.Typer(
    name="history",
    help="Historical timeseries aggregation commands.",
    no_args_is_help=True,
)


# -----------------------------------------------------------------------------
# Option Type Aliases
# -----------------------------------------------------------------------------

RepoRootOpt = Annotated[
    Path,
    typer.Option(
        "--repo-root",
        help="Path to the repository root (default: current directory).",
    ),
]

RepoArg = Annotated[
    str,
    typer.Option(
        "--repo",
        help="Repository slug (e.g., 'my-org/my-repo').",
    ),
]

CommitsArg = Annotated[
    list[str],
    typer.Option(
        "--commits",
        help="Commits to include in the timeseries (latest first).",
    ),
]

DbDirOpt = Annotated[
    Path,
    typer.Option(
        "--db-dir",
        help="Directory with per-commit DuckDB snapshots (codeintel-<commit>.duckdb).",
    ),
]

OutputDbOpt = Annotated[
    Path,
    typer.Option(
        "--output-db",
        help="Destination DuckDB for history_timeseries (will be created if missing).",
    ),
]

EntityKindOpt = Annotated[
    str,
    typer.Option(
        "--entity-kind",
        help="Entity kind to include: function, module, or both.",
    ),
]

MaxEntitiesOpt = Annotated[
    int,
    typer.Option(
        "--max-entities",
        help="Maximum entities to track (top-N by selection strategy).",
    ),
]

SelectionStrategyOpt = Annotated[
    str,
    typer.Option(
        "--selection-strategy",
        help="Selection strategy for picking entities (default: risk_score).",
    ),
]


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


@history_app.command("timeseries")
def history_timeseries(
    repo: RepoArg,
    commits: CommitsArg,
    repo_root: RepoRootOpt = Path(),
    db_dir: DbDirOpt = Path("build/db"),
    output_db: OutputDbOpt = Path("build/db/history.duckdb"),
    entity_kind: EntityKindOpt = "function",
    max_entities: MaxEntitiesOpt = 500,
    selection_strategy: SelectionStrategyOpt = "risk_score",
    verbose: VerboseOpt = 0,
) -> None:
    r"""Aggregate analytics.history_timeseries across multiple commits.

    Collects analytics data from per-commit DuckDB snapshots and aggregates
    them into a unified history timeseries table.

    Examples
    --------
    .. code-block:: bash

        # Build history for multiple commits
        codeintel history timeseries --repo my-org/repo \
            --commits abc123 --commits def456 --commits ghi789

        # Customize entity tracking
        codeintel history timeseries --repo my-org/repo \
            --commits abc123 --commits def456 \
            --entity-kind both --max-entities 1000
    """
    setup_logging(verbose)

    commit_list = list(commits)
    if not commit_list:
        typer.secho("Error: At least one commit is required", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    runner = ToolRunner(cache_dir=repo_root / "build" / ".tool_cache")
    builder = ConfigBuilder.from_snapshot(
        repo=repo,
        commit=commit_list[0],
        repo_root=repo_root,
    )
    cfg = builder.history_timeseries(
        commits=tuple(commit_list),
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
        compute_history_timeseries_gateways(
            gateway,
            cfg,
            snapshot_resolver,
            runner=runner,
        )
    except FileNotFoundError:
        LOG.exception("Missing snapshot database for history_timeseries")
        typer.secho(
            "Error: Missing snapshot database for one or more commits",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None
    except DuckDBError:
        LOG.exception("Failed to compute history_timeseries")
        typer.secho(
            "Error: Failed to compute history_timeseries",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None

    LOG.info(
        "history_timeseries written to %s for %d commits",
        output_db,
        len(commit_list),
    )
    typer.secho(
        f"History timeseries written to {output_db} for {len(commit_list)} commits.",
        fg=typer.colors.GREEN,
    )


__all__ = ["history_app"]
