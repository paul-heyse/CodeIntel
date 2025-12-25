"""History commands for snapshot management.

Note: History commands require runtime/gateway access via handler pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from cyclopts import App

from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.history import history_timeseries_handler
from codeintel.cli.options.registry import (
    HISTORY_COMMITS,
    HISTORY_DB_DIR,
    HISTORY_ENTITY_KIND,
    HISTORY_MAX_ENTITIES,
    HISTORY_OUTPUT_DB,
    HISTORY_REPO,
    HISTORY_REPO_ROOT,
    HISTORY_SELECTION_STRATEGY,
)
from codeintel.cli.options.shared_flags import SharedFlagsProtocol, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param

history_app = App(
    name="history",
    help="Historical timeseries aggregation.",
)


class EntityKind(StrEnum):
    """Entity categories to include in history aggregation."""

    FUNCTION = "function"
    MODULE = "module"
    BOTH = "both"


class SelectionStrategy(StrEnum):
    """Strategies for selecting top entities."""

    RISK_SCORE = "risk_score"
    CALL_PAGERANK = "call_pagerank"
    HOTSPOT_SCORE = "hotspot_score"


_HISTORY_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)

HISTORY_TIMESERIES_PATH: CommandPath = ("history", "timeseries")

_HISTORY_TIMESERIES_FLAGS_FIELD = shared_flags_field(HISTORY_TIMESERIES_PATH)


@cli_command("history.timeseries", handler=history_timeseries_handler, config=_HISTORY_CONFIG)
@history_app.command(name="timeseries")
@dataclass
class HistoryTimeseriesCommand:
    """Aggregate analytics.history_timeseries across commits."""

    repo: Annotated[
        str,
        option_param(HISTORY_REPO, command_path=HISTORY_TIMESERIES_PATH),
    ] = ""
    commits: Annotated[
        list[str] | None,
        option_param(HISTORY_COMMITS, command_path=HISTORY_TIMESERIES_PATH),
    ] = None
    db_dir: Annotated[
        Path,
        option_param(HISTORY_DB_DIR, command_path=HISTORY_TIMESERIES_PATH),
    ] = Path("build/db")
    output_db: Annotated[
        Path,
        option_param(HISTORY_OUTPUT_DB, command_path=HISTORY_TIMESERIES_PATH),
    ] = Path("build/db/history.duckdb")
    entity_kind: Annotated[
        EntityKind,
        option_param(HISTORY_ENTITY_KIND, command_path=HISTORY_TIMESERIES_PATH),
    ] = EntityKind.FUNCTION
    max_entities: Annotated[
        int,
        option_param(HISTORY_MAX_ENTITIES, command_path=HISTORY_TIMESERIES_PATH),
    ] = 500
    selection_strategy: Annotated[
        SelectionStrategy,
        option_param(HISTORY_SELECTION_STRATEGY, command_path=HISTORY_TIMESERIES_PATH),
    ] = SelectionStrategy.RISK_SCORE
    repo_root: Annotated[
        Path | None,
        option_param(HISTORY_REPO_ROOT, command_path=HISTORY_TIMESERIES_PATH),
    ] = None
    flags: SharedFlagsProtocol = _HISTORY_TIMESERIES_FLAGS_FIELD


__all__ = ["history_app"]
