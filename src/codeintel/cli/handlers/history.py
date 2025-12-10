"""Handlers for history timeseries commands.

These handlers follow the EnhancedHandlerContext pattern and return CliResult.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.cli.errors import ProblemDetail
from codeintel.cli.handlers.base import setup_logging
from codeintel.cli.core import CliResult
from codeintel.config import ConfigBuilder, SnapshotInit
from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.storage.gateway import (
    DuckDBError,
    StorageConfig,
    build_snapshot_gateway_resolver,
    open_gateway,
)

if TYPE_CHECKING:
    from codeintel.cli.handlers.protocol import EnhancedHandlerContext

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


# -----------------------------------------------------------------------------
# Parameter Helpers
# -----------------------------------------------------------------------------


def _get_str_param(ctx: EnhancedHandlerContext, key: str, default: str = "") -> str:
    """Get string parameter with default.

    Parameters
    ----------
    ctx
        Handler context.
    key
        Parameter key.
    default
        Default value.

    Returns
    -------
    str
        Parameter value or default.
    """
    value = ctx.params.get(key)
    if value is None:
        return default
    return str(value)


def _require_str_param(ctx: EnhancedHandlerContext, key: str) -> str:
    """Require string parameter.

    Parameters
    ----------
    ctx
        Handler context.
    key
        Parameter key.

    Returns
    -------
    str
        Parameter value.

    Raises
    ------
    RuntimeError
        If parameter is missing or empty.
    """
    value = ctx.params.get(key)
    if value is None or (isinstance(value, str) and not value):
        msg = f"Missing required parameter: {key}"
        raise RuntimeError(msg)
    return str(value)


def _get_int_param(ctx: EnhancedHandlerContext, key: str, default: int = 0) -> int:
    """Get integer parameter with default.

    Parameters
    ----------
    ctx
        Handler context.
    key
        Parameter key.
    default
        Default value.

    Returns
    -------
    int
        Parameter value or default.
    """
    value = ctx.params.get(key)
    if value is None:
        return default
    if isinstance(value, int):
        return value
    return int(str(value))


def _get_path_param(ctx: EnhancedHandlerContext, key: str, default: Path) -> Path:
    """Get path parameter with default.

    Parameters
    ----------
    ctx
        Handler context.
    key
        Parameter key.
    default
        Default value.

    Returns
    -------
    Path
        Parameter value or default.
    """
    value = ctx.params.get(key)
    if value is None:
        return default
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _get_enum_str_param(ctx: EnhancedHandlerContext, key: str, default: str = "") -> str:
    """Get enum parameter as string.

    Parameters
    ----------
    ctx
        Handler context.
    key
        Parameter key.
    default
        Default value.

    Returns
    -------
    str
        Parameter value as string.
    """
    value = ctx.params.get(key)
    if value is None:
        return default
    # Handle enum values
    value_attr = getattr(value, "value", None)
    if value_attr is not None:
        return str(value_attr)
    return str(value)


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


def history_timeseries_handler(ctx: EnhancedHandlerContext) -> CliResult[HistoryTimeseriesResult]:
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
    commits_raw = ctx.params.get("commits")
    commits: list[str] = []
    if commits_raw is not None:
        if isinstance(commits_raw, list):
            commits = [str(c) for c in commits_raw]
        else:
            commits = [str(commits_raw)]

    if not commits:
        return CliResult.fail(_make_error("Validation Error", "At least one commit is required"))

    repo = _require_str_param(ctx, "repo")

    repo_root = _get_path_param(ctx, "repo_root", Path.cwd())
    db_dir = _get_path_param(ctx, "db_dir", Path("build/db"))
    output_db = _get_path_param(ctx, "output_db", Path("build/db/history.duckdb"))
    entity_kind = _get_enum_str_param(ctx, "entity_kind", "function")
    max_entities = _get_int_param(ctx, "max_entities", 500)
    selection_strategy = _get_enum_str_param(ctx, "selection_strategy", "risk_score")

    runner = ToolRunner(cache_dir=repo_root / "build" / ".tool_cache")
    builder = ConfigBuilder.from_snapshot(
        snapshot=SnapshotInit(
            repo=repo, commit=commits[0], repo_root=repo_root
        ),
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
            _make_error("Storage Error", f"Missing snapshot database for one or more commits: {exc}")
        )
    except DuckDBError as exc:
        LOG.exception("Failed to compute history_timeseries")
        gateway.close()
        return CliResult.fail(_make_error("Query Error", f"Failed to compute history_timeseries: {exc}"))

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
