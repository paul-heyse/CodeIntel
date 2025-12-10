"""Cyclopts wiring for history commands.

This module wires Cyclopts command classes to unified ExecutionContext handlers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.execution.adapter import CycloptsAdapter
from codeintel.cli.history_handlers import history_timeseries_ctx

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


@history_app.command(name="timeseries")
@dataclass
class HistoryTimeseriesCommand:
    """Aggregate analytics.history_timeseries across commits."""

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
    repo_root: Annotated[
        Path | None,
        Parameter(
            name="--repo-root",
            help="Repository root directory.",
        ),
    ] = None
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the history timeseries command."""
        CycloptsAdapter("history.timeseries", history_timeseries_ctx)(self)


__all__ = ["history_app"]
