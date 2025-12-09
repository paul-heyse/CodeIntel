"""Typer-free handlers for history timeseries commands.

These helpers keep operational logic while allowing Cyclopts to invoke
them without importing Typer. All user-facing errors surface as
:class:`~codeintel.cli.cli_errors.ValidationError`.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.cli.cli_errors import ValidationError

# Import consolidated setup_logging from handlers.base
from codeintel.cli.handlers.base import setup_logging as _setup_logging_impl
from codeintel.cli.results import CliResult
from codeintel.config import ConfigBuilder, SnapshotInit
from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.storage.gateway import (
    DuckDBError,
    StorageConfig,
    build_snapshot_gateway_resolver,
    open_gateway,
)

if TYPE_CHECKING:
    from codeintel.cli.execution.context import ExecutionContext

LOG = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

# Use consolidated setup_logging from handlers.base
setup_logging = _setup_logging_impl


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


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
# Helper Functions
# -----------------------------------------------------------------------------


def _normalize_commits(commits: tuple[str, ...] | list[str] | None) -> list[str]:
    """Flatten repeatable commit arguments into a list of strings.

    Parameters
    ----------
    commits
        Raw commit arguments.

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

    Parameters
    ----------
    values
        Raw input values.

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


# -----------------------------------------------------------------------------
# Bundle Function
# -----------------------------------------------------------------------------


def bundle_history_timeseries(cli_kwargs: dict[str, object]) -> dict[str, object]:
    """Bundle CLI arguments for history timeseries command.

    Parameters
    ----------
    cli_kwargs
        Raw CLI keyword arguments.

    Returns
    -------
    dict[str, object]
        Bundled arguments.
    """
    commits_raw = cast("tuple[str, ...] | list[str] | None", cli_kwargs.get("commits"))
    repo_root_raw = cast("Path | str | None", cli_kwargs.get("repo_root"))
    db_dir_raw = cast("Path | str | None", cli_kwargs.get("db_dir"))
    output_db_raw = cast("Path | str | None", cli_kwargs.get("output_db"))
    entity_kind_raw = cast("str | None", cli_kwargs.get("entity_kind"))
    max_entities_raw = cast("int | str | None", cli_kwargs.get("max_entities"))
    selection_strategy_raw = cast("str | None", cli_kwargs.get("selection_strategy"))
    verbose_raw = cast("int | str | None", cli_kwargs.get("verbose"))

    return {
        "repo": str(cli_kwargs["repo"]),
        "commits": _normalize_commits(list(commits_raw) if commits_raw is not None else None),
        "options": _build_history_options(
            HistoryOptionsInput(
                repo_root=Path(repo_root_raw or Path()),
                db_dir=Path(db_dir_raw or Path("build/db")),
                output_db=Path(output_db_raw or Path("build/db/history.duckdb")),
                entity_kind=str(entity_kind_raw or "function"),
                max_entities=int(max_entities_raw or 500),
                selection_strategy=str(selection_strategy_raw or "risk_score"),
            )
        ),
        "verbose": int(verbose_raw or 0),
    }


# -----------------------------------------------------------------------------
# Handler
# -----------------------------------------------------------------------------


def history_timeseries_handler(
    repo: str,
    commits: list[str],
    options: HistoryOptions,
    verbose: int,
) -> None:
    """Aggregate analytics.history_timeseries across multiple commits.

    Collects analytics data from per-commit DuckDB snapshots and aggregates
    them into a unified history timeseries table.

    Parameters
    ----------
    repo
        Repository slug (e.g., 'my-org/my-repo').
    commits
        Commits to include in the timeseries (latest first).
    options
        History options.
    verbose
        Verbosity level.

    Raises
    ------
    ValidationError
        If required inputs are missing or history aggregation fails.
    """
    setup_logging(verbose)

    commit_list = list(commits)
    if not commit_list:
        msg = "At least one commit is required"
        raise ValidationError(msg)

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
        msg = "Missing snapshot database for one or more commits"
        raise ValidationError(msg) from None
    except DuckDBError:
        LOG.exception("Failed to compute history_timeseries")
        msg = "Failed to compute history_timeseries"
        raise ValidationError(msg) from None

    LOG.info(
        "history_timeseries written to %s for %d commits",
        options.output_db,
        len(commit_list),
    )
    sys.stdout.write(
        f"History timeseries written to {options.output_db} for {len(commit_list)} commits.\n"
    )


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


# -----------------------------------------------------------------------------
# Structured Handler Helper
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _HistoryParams:
    """Internal holder for parsed history params."""

    repo: str
    commits: list[str]
    repo_root: Path
    db_dir: Path
    output_db: Path
    entity_kind: str
    max_entities: int
    selection_strategy: str


def _parse_history_params(ctx: ExecutionContext) -> _HistoryParams:
    """Parse history params from execution context.

    Parameters
    ----------
    ctx
        Execution context with params.

    Returns
    -------
    _HistoryParams
        Parsed parameters.

    Raises
    ------
    RuntimeError
        If required params are missing.
    """
    repo = ctx.require_str_param("repo")
    commits_raw = ctx.params.get("commits")
    commits = list(commits_raw) if commits_raw else []

    if not commits:
        msg = "At least one commit is required"
        raise RuntimeError(msg)

    repo_root_raw = ctx.params.get("repo_root")
    db_dir_raw = ctx.params.get("db_dir")
    output_db_raw = ctx.params.get("output_db")

    return _HistoryParams(
        repo=repo,
        commits=commits,
        repo_root=Path(repo_root_raw) if repo_root_raw else Path(),
        db_dir=Path(db_dir_raw) if db_dir_raw else Path("build/db"),
        output_db=Path(output_db_raw) if output_db_raw else Path("build/db/history.duckdb"),
        entity_kind=ctx.get_str_param("entity_kind", "function") or "function",
        max_entities=ctx.get_int_param("max_entities", 500),
        selection_strategy=ctx.get_str_param("selection_strategy", "risk_score") or "risk_score",
    )


# -----------------------------------------------------------------------------
# Structured Handler (accepts ExecutionContext)
# -----------------------------------------------------------------------------


def history_timeseries_ctx(ctx: ExecutionContext) -> CliResult[HistoryTimeseriesResult]:
    """Aggregate analytics.history_timeseries across multiple commits.

    Collects analytics data from per-commit DuckDB snapshots and aggregates
    them into a unified history timeseries table.

    Parameters
    ----------
    ctx
        Execution context with params:
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

    Raises
    ------
    RuntimeError
        If no commits are provided or history aggregation fails.
    """
    setup_logging(ctx.verbosity)
    params = _parse_history_params(ctx)

    runner = ToolRunner(cache_dir=params.repo_root / "build" / ".tool_cache")
    builder = ConfigBuilder.from_snapshot(
        snapshot=SnapshotInit(
            repo=params.repo, commit=params.commits[0], repo_root=params.repo_root
        ),
    )
    cfg = builder.history_timeseries(
        commits=tuple(params.commits),
        entity_kind=params.entity_kind,
        max_entities=params.max_entities,
        selection_strategy=params.selection_strategy,
    )

    storage_cfg = StorageConfig.for_ingest(params.output_db)
    gateway = open_gateway(storage_cfg)
    snapshot_resolver = build_snapshot_gateway_resolver(
        db_dir=params.db_dir,
        repo=params.repo,
        primary_gateway=gateway,
    )

    try:
        compute_history_timeseries_gateways(gateway, cfg, snapshot_resolver, runner=runner)
    except FileNotFoundError as exc:
        LOG.exception("Missing snapshot database for history_timeseries")
        msg = "Missing snapshot database for one or more commits"
        raise RuntimeError(msg) from exc
    except DuckDBError as exc:
        LOG.exception("Failed to compute history_timeseries")
        msg = "Failed to compute history_timeseries"
        raise RuntimeError(msg) from exc
    finally:
        gateway.close()

    LOG.info(
        "history_timeseries written to %s for %d commits",
        params.output_db,
        len(params.commits),
    )

    return CliResult.ok(
        HistoryTimeseriesResult(
            output_db=str(params.output_db),
            commits_processed=len(params.commits),
            entity_kind=params.entity_kind,
        )
    )


__all__ = [
    "HistoryOptions",
    "HistoryOptionsInput",
    "HistoryTimeseriesResult",
    "bundle_history_timeseries",
    "history_timeseries_ctx",
    "history_timeseries_handler",
    "setup_logging",
]
