"""Cyclopts wiring for history commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cli_errors import invoke_with_typer_translation
from codeintel.cli.commands.history import HistoryOptions, history_timeseries_handler
from codeintel.cli.cyclopts_common import RuntimeCLI, runtime_cli_to_options

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
    runtime: Annotated[RuntimeCLI, Parameter(name="*")] = field(default_factory=RuntimeCLI)
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
def timeseries(
    cfg: Annotated[HistoryTimeseriesCli, Parameter(name="*")] | None = None,
) -> None:
    """Aggregate analytics.history_timeseries across commits.

    Raises
    ------
    SystemExit
        If required arguments are missing or a delegated handler exits.
    """
    cfg = cfg or HistoryTimeseriesCli()  # type: ignore[call-arg]
    if not cfg.repo:
        raise SystemExit(2)
    runtime_options = runtime_cli_to_options(cfg.runtime)
    options = HistoryOptions(
        repo_root=runtime_options.repo_root or Path(),
        db_dir=cfg.db_dir,
        output_db=cfg.output_db,
        entity_kind=cfg.entity_kind.value,
        max_entities=cfg.max_entities,
        selection_strategy=cfg.selection_strategy.value,
    )
    commits = list(cfg.commits) if cfg.commits is not None else []
    invoke_with_typer_translation(
        history_timeseries_handler,
        repo=cfg.repo,
        commits=commits,
        options=options,
        verbose=cfg.runtime.verbose,
    )


__all__ = ["history_app"]
