"""Cyclopts wiring for history commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from cyclopts import App, Parameter

from codeintel.cli.commands.history import HistoryOptions, history_timeseries_handler
from codeintel.cli.cyclopts_common import Verbose

history_app = App(
    name="history",
    help="Historical timeseries aggregation.",
)


@dataclass
class HistoryTimeseriesCli:
    """CLI surface for `codeintel history timeseries`."""

    repo: Annotated[
        str,
        Parameter(
            name="--repo",
            help="Repository slug (e.g., 'my-org/my-repo').",
        ),
    ] = ""
    commits: Annotated[
        list[str] | None,
        Parameter(
            name="--commits",
            help="Commits to include in the timeseries (latest first).",
        ),
    ] = None
    repo_root: Annotated[
        Path,
        Parameter(
            name="--repo-root",
            help="Path to the repository root (default: current directory).",
        ),
    ] = Path()
    db_dir: Annotated[
        Path,
        Parameter(
            name="--db-dir",
            help="Directory with per-commit DuckDB snapshots.",
        ),
    ] = Path("build/db")
    output_db: Annotated[
        Path,
        Parameter(
            name="--output-db",
            help="Destination DuckDB for history_timeseries.",
        ),
    ] = Path("build/db/history.duckdb")
    entity_kind: Annotated[
        str,
        Parameter(
            name="--entity-kind",
            help="Entity kind to include: function, module, or both.",
        ),
    ] = "function"
    max_entities: Annotated[
        int,
        Parameter(
            name="--max-entities",
            help="Maximum entities to track (top-N by selection strategy).",
        ),
    ] = 500
    selection_strategy: Annotated[
        str,
        Parameter(
            name="--selection-strategy",
            help="Selection strategy for picking entities (default: risk_score).",
        ),
    ] = "risk_score"
    verbose: Verbose = 0


@history_app.command(name="timeseries")
def timeseries(
    cfg: Annotated[HistoryTimeseriesCli, Parameter(name="*")] | None = None,
) -> None:
    """Aggregate analytics.history_timeseries across commits.

    Raises
    ------
    SystemExit
        When required arguments are missing or the handler exits.
    """
    cfg = cfg or HistoryTimeseriesCli()
    if not cfg.repo:
        raise SystemExit(2)
    options = HistoryOptions(
        repo_root=cfg.repo_root,
        db_dir=cfg.db_dir,
        output_db=cfg.output_db,
        entity_kind=cfg.entity_kind,
        max_entities=cfg.max_entities,
        selection_strategy=cfg.selection_strategy,
    )
    commits = list(cfg.commits) if cfg.commits is not None else []
    try:
        history_timeseries_handler(
            repo=cfg.repo,
            commits=commits,
            options=options,
            verbose=cfg.verbose,
        )
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from exc


__all__ = ["history_app"]
