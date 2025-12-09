"""Cyclopts wiring for history commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cli_errors import ValidationError, run_handler
from codeintel.cli.cyclopts_common import (
    ExistingDir,
    OutputPath,
    RuntimeCLI,
    get_verbose,
    runtime_cli_to_options,
)
from codeintel.cli.history_handlers import HistoryOptions, history_timeseries_handler

history_app = App(
    name="history",
    help="Historical timeseries aggregation.",
)


class EntityKind(Enum):
    """Entity categories to include in history aggregation."""

    FUNCTION = "function"
    MODULE = "module"
    BOTH = "both"


class SelectionStrategy(Enum):
    """Strategies for selecting top entities."""

    RISK_SCORE = "risk_score"
    CALL_PAGERANK = "call_pagerank"
    HOTSPOT_SCORE = "hotspot_score"


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
    runtime: Annotated[RuntimeCLI | None, Parameter(name="*")] = None
    db_dir: Annotated[
        ExistingDir,
        Parameter(
            name="--db-dir",
            help="Directory with per-commit DuckDB snapshots.",
        ),
    ] = Path("build/db")
    output_db: Annotated[
        OutputPath,
        Parameter(
            name="--output-db",
            help="Destination DuckDB for history_timeseries.",
        ),
    ] = Path("build/db/history.duckdb")
    entity_kind: Annotated[
        EntityKind,
        Parameter(
            name="--entity-kind",
            help="Entity kind to include: function, module, or both.",
        ),
    ] = EntityKind.FUNCTION
    max_entities: Annotated[
        int,
        Parameter(
            name="--max-entities",
            help="Maximum entities to track (top-N by selection strategy).",
        ),
    ] = 500
    selection_strategy: Annotated[
        SelectionStrategy,
        Parameter(
            name="--selection-strategy",
            help="Selection strategy for picking entities (default: risk_score).",
        ),
    ] = SelectionStrategy.RISK_SCORE


@history_app.command(name="timeseries")
@dataclass
class HistoryTimeseriesCommand:
    """Aggregate analytics.history_timeseries across commits."""

    cfg: Annotated[HistoryTimeseriesCli, Parameter(name="*")] = field(
        default_factory=HistoryTimeseriesCli
    )

    def __call__(self) -> None:
        if not self.cfg.repo:
            message = "Repository slug is required."
            raise ValidationError(message)
        if self.cfg.commits is None or not list(self.cfg.commits):
            message = "At least one commit is required."
            raise ValidationError(message)
        runtime = self.cfg.runtime or RuntimeCLI()
        runtime_options = runtime_cli_to_options(runtime)
        options = HistoryOptions(
            repo_root=runtime_options.repo_root or Path(),
            db_dir=self.cfg.db_dir,
            output_db=self.cfg.output_db,
            entity_kind=self.cfg.entity_kind.value,
            max_entities=self.cfg.max_entities,
            selection_strategy=self.cfg.selection_strategy.value,
        )
        commits = list(self.cfg.commits)
        verbose = get_verbose(runtime)
        run_handler(
            history_timeseries_handler,
            repo=self.cfg.repo,
            commits=commits,
            options=options,
            verbose=verbose,
        )


__all__ = ["history_app"]
