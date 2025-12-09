"""History timeseries commands for the CodeIntel CLI.

This module provides Typer commands for aggregating analytics data
across multiple commits to build historical timeseries.

Commands
--------
- **timeseries**: Aggregate analytics.history_timeseries across commits
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import typer

from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.cli.commands._common import VerboseOpt, setup_logging
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
CommitsOpt = typer.Option(
    None,
    "--commits",
    help="Commits to include in the timeseries (latest first).",
)

RepoRootOptInfo = typer.Option(
    Path(),
    "--repo-root",
    help="Path to the repository root (default: current directory).",
)

DbDirOptInfo = typer.Option(
    Path("build/db"),
    "--db-dir",
    help="Directory with per-commit DuckDB snapshots (codeintel-<commit>.duckdb).",
)

OutputDbOptInfo = typer.Option(
    Path("build/db/history.duckdb"),
    "--output-db",
    help="Destination DuckDB for history_timeseries (will be created if missing).",
)

EntityKindOptInfo = typer.Option(
    "function",
    "--entity-kind",
    help="Entity kind to include: function, module, or both.",
)

MaxEntitiesOptInfo = typer.Option(
    500,
    "--max-entities",
    help="Maximum entities to track (top-N by selection strategy).",
)

SelectionStrategyOptInfo = typer.Option(
    "risk_score",
    "--selection-strategy",
    help="Selection strategy for picking entities (default: risk_score).",
)

REPO_OPT = RepoOpt
COMMITS_OPT = CommitsOpt
REPO_ROOT_OPT = RepoRootOptInfo
DB_DIR_OPT = DbDirOptInfo
OUTPUT_DB_OPT = OutputDbOptInfo
ENTITY_KIND_OPT = EntityKindOptInfo
MAX_ENTITIES_OPT = MaxEntitiesOptInfo
SELECTION_STRATEGY_OPT = SelectionStrategyOptInfo
VERBOSE_OPT = VerboseOpt


@dataclass(frozen=True)
class HistoryOptions:
    """Selection and storage options for history aggregation."""

    repo_root: Path
    db_dir: Path
    output_db: Path
    entity_kind: str
    max_entities: int
    selection_strategy: str


@dataclass(frozen=True)
class HistoryOptionsInput:
    """Raw CLI values before normalization."""

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


def _normalize_commits(commits: tuple[str, ...] | list[str] | None) -> list[str]:
    """Flatten repeatable commit arguments into a list of strings.

    Returns
    -------
    list[str]
        Normalized commit identifiers.
    """
    if commits is None:
        return []
    return [str(value) for value in commits]


def _build_history_options(values: HistoryOptionsInput) -> HistoryOptions:
    """Construct HistoryOptions with normalized types.

    Returns
    -------
    HistoryOptions
        Aggregated history command options.
    """
    return HistoryOptions(
        repo_root=Path(values.repo_root),
        db_dir=Path(values.db_dir),
        output_db=Path(values.output_db),
        entity_kind=str(values.entity_kind),
        max_entities=int(values.max_entities),
        selection_strategy=str(values.selection_strategy),
    )


def history_timeseries(**cli_kwargs: object) -> None:
    """CLI entrypoint for history_timeseries."""
    commits_raw = cast("tuple[str, ...] | list[str] | None", cli_kwargs.get("commits"))
    repo_root_raw = cast("Path | str | None", cli_kwargs.get("repo_root"))
    db_dir_raw = cast("Path | str | None", cli_kwargs.get("db_dir"))
    output_db_raw = cast("Path | str | None", cli_kwargs.get("output_db"))
    entity_kind_raw = cast("str | None", cli_kwargs.get("entity_kind"))
    max_entities_raw = cast("int | str | None", cli_kwargs.get("max_entities"))
    selection_strategy_raw = cast("str | None", cli_kwargs.get("selection_strategy"))
    verbose_raw = cast("int | str | None", cli_kwargs.get("verbose"))

    history_timeseries_handler(
        repo=str(cli_kwargs["repo"]),
        commits=_normalize_commits(list(commits_raw) if commits_raw is not None else None),
        options=_build_history_options(
            HistoryOptionsInput(
                repo_root=Path(repo_root_raw or Path()),
                db_dir=Path(db_dir_raw or Path("build/db")),
                output_db=Path(output_db_raw or Path("build/db/history.duckdb")),
                entity_kind=str(entity_kind_raw or "function"),
                max_entities=int(max_entities_raw or 500),
                selection_strategy=str(selection_strategy_raw or "risk_score"),
            )
        ),
        verbose=int(verbose_raw or 0),
    )


_HISTORY_PARAMETERS = [
    inspect.Parameter(
        "repo",
        inspect.Parameter.KEYWORD_ONLY,
        default=REPO_OPT,
        annotation=str,
    ),
    inspect.Parameter(
        "commits",
        inspect.Parameter.KEYWORD_ONLY,
        default=COMMITS_OPT,
        annotation=list[str],
    ),
    inspect.Parameter(
        "repo_root",
        inspect.Parameter.KEYWORD_ONLY,
        default=REPO_ROOT_OPT,
        annotation=Path,
    ),
    inspect.Parameter(
        "db_dir",
        inspect.Parameter.KEYWORD_ONLY,
        default=DB_DIR_OPT,
        annotation=Path,
    ),
    inspect.Parameter(
        "output_db",
        inspect.Parameter.KEYWORD_ONLY,
        default=OUTPUT_DB_OPT,
        annotation=Path,
    ),
    inspect.Parameter(
        "entity_kind",
        inspect.Parameter.KEYWORD_ONLY,
        default=ENTITY_KIND_OPT,
        annotation=str,
    ),
    inspect.Parameter(
        "max_entities",
        inspect.Parameter.KEYWORD_ONLY,
        default=MAX_ENTITIES_OPT,
        annotation=int,
    ),
    inspect.Parameter(
        "selection_strategy",
        inspect.Parameter.KEYWORD_ONLY,
        default=SELECTION_STRATEGY_OPT,
        annotation=str,
    ),
    inspect.Parameter(
        "verbose",
        inspect.Parameter.KEYWORD_ONLY,
        default=VERBOSE_OPT,
        annotation=int,
    ),
]

history_timeseries = history_app.command("timeseries")(history_timeseries)


__all__ = ["history_app"]
