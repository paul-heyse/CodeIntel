"""History timeseries commands for the CodeIntel CLI.

This module provides Typer commands for aggregating analytics data
across multiple commits to build historical timeseries.

Commands
--------
- **timeseries**: Aggregate analytics.history_timeseries across commits
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import typer

from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.cli.commands._common import VerboseOpt, setup_logging
from codeintel.cli.commands._option_shim import OptionSpec, wrap_command
from codeintel.config import ConfigBuilder, SnapshotInit
from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.storage.gateway import (
    DuckDBError,
    StorageConfig,
    build_snapshot_gateway_resolver,
    open_gateway,
)

LOG = logging.getLogger(__name__)

history_app = typer.Typer(
    name="history",
    help="Historical timeseries aggregation commands.",
    no_args_is_help=True,
)


# -----------------------------------------------------------------------------
# Option Type Aliases
# -----------------------------------------------------------------------------

RepoOpt = typer.Option(
    ...,
    "--repo",
    help="Repository slug (e.g., 'my-org/my-repo').",
)

RepoArg = Annotated[str, RepoOpt]

CommitsOpt = typer.Option(
    ...,
    "--commits",
    help="Commits to include in the timeseries (latest first).",
)

CommitsArg = Annotated[list[str], CommitsOpt]

RepoRootOptInfo = typer.Option(
    Path(),
    "--repo-root",
    help="Path to the repository root (default: current directory).",
)
RepoRootOpt = Annotated[Path, RepoRootOptInfo]

DbDirOptInfo = typer.Option(
    Path("build/db"),
    "--db-dir",
    help="Directory with per-commit DuckDB snapshots (codeintel-<commit>.duckdb).",
)
DbDirOpt = Annotated[Path, DbDirOptInfo]

OutputDbOptInfo = typer.Option(
    Path("build/db/history.duckdb"),
    "--output-db",
    help="Destination DuckDB for history_timeseries (will be created if missing).",
)
OutputDbOpt = Annotated[Path, OutputDbOptInfo]

EntityKindOptInfo = typer.Option(
    "function",
    "--entity-kind",
    help="Entity kind to include: function, module, or both.",
)
EntityKindOpt = Annotated[str, EntityKindOptInfo]

MaxEntitiesOptInfo = typer.Option(
    500,
    "--max-entities",
    help="Maximum entities to track (top-N by selection strategy).",
)
MaxEntitiesOpt = Annotated[int, MaxEntitiesOptInfo]

SelectionStrategyOptInfo = typer.Option(
    "risk_score",
    "--selection-strategy",
    help="Selection strategy for picking entities (default: risk_score).",
)
SelectionStrategyOpt = Annotated[str, SelectionStrategyOptInfo]


@dataclass(frozen=True)
class HistoryOptions:
    """Selection and storage options for history aggregation."""

    repo_root: Path
    db_dir: Path
    output_db: Path
    entity_kind: str
    max_entities: int
    selection_strategy: str


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


def history_timeseries_handler(
    repo: str,
    commits: list[str],
    options: HistoryOptions,
    verbose: int,
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

    Raises
    ------
    typer.Exit
        If required inputs are missing or history aggregation fails.
    """
    setup_logging(verbose)

    commit_list = list(commits)
    if not commit_list:
        typer.secho("Error: At least one commit is required", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    runner = ToolRunner(cache_dir=options.repo_root / "build" / ".tool_cache")
    builder = ConfigBuilder.from_snapshot(
        snapshot=SnapshotInit(repo=repo, commit=commit_list[0], repo_root=options.repo_root),
    )
    cfg = builder.history_timeseries(
        commits=tuple(commit_list),
        entity_kind=options.entity_kind,
        max_entities=options.max_entities,
        selection_strategy=options.selection_strategy,
    )

    storage_cfg = StorageConfig.for_ingest(options.output_db)
    gateway = open_gateway(storage_cfg)
    snapshot_resolver = build_snapshot_gateway_resolver(
        db_dir=options.db_dir,
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
        options.output_db,
        len(commit_list),
    )
    typer.secho(
        f"History timeseries written to {options.output_db} for {len(commit_list)} commits.",
        fg=typer.colors.GREEN,
    )


def _bundle_history(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "repo": cast("str", cli_kwargs["repo"]),
        "commits": cast("list[str]", cli_kwargs["commits"]),
        "options": HistoryOptions(
            repo_root=cast("Path", cli_kwargs.get("repo_root", Path())),
            db_dir=cast("Path", cli_kwargs.get("db_dir", Path("build/db"))),
            output_db=cast("Path", cli_kwargs.get("output_db", Path("build/db/history.duckdb"))),
            entity_kind=cast("str", cli_kwargs.get("entity_kind", "function")),
            max_entities=int(cast("int | str", cli_kwargs.get("max_entities", 500)) or 500),
            selection_strategy=cast("str", cli_kwargs.get("selection_strategy", "risk_score")),
        ),
        "verbose": int(cast("int | str", cli_kwargs.get("verbose", 0)) or 0),
    }


_HISTORY_SPECS = [
    OptionSpec("repo", str, RepoOpt),
    OptionSpec("commits", list[str], CommitsOpt),
    OptionSpec("repo_root", Path, RepoRootOptInfo),
    OptionSpec("db_dir", Path, DbDirOptInfo),
    OptionSpec("output_db", Path, OutputDbOptInfo),
    OptionSpec("entity_kind", str, EntityKindOptInfo),
    OptionSpec("max_entities", int, MaxEntitiesOptInfo),
    OptionSpec("selection_strategy", str, SelectionStrategyOptInfo),
    OptionSpec("verbose", int, VerboseOpt),
]

history_timeseries = history_app.command("timeseries")(
    wrap_command(
        history_timeseries_handler,
        _HISTORY_SPECS,
        bundle=_bundle_history,
        name="history_timeseries",
    )
)


__all__ = ["history_app"]
