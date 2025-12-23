"""Handlers for history timeseries commands.

Provide analytics aggregation across multiple commit snapshots.

This handler writes to a dedicated output database (not the runtime's database),
so it uses explicit gateway management rather than ctx.gateway.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.analytics.history.history_timeseries import HistoryTimeseriesOptions
from codeintel.build.config import load_build_config
from codeintel.build.hamilton import HamiltonBuildExecutor
from codeintel.build.providers import create_default_providers
from codeintel.build.run_context import BuildRunContext, BuildRunContextOverrides
from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import HistoryTimeseriesResult
from codeintel.cli.errors.results import fail_history_error
from codeintel.cli.execution.bootstrap import bootstrap_cli
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import (
    BuildPaths,
    GraphBackendConfig,
    GraphFeatureFlags,
    SnapshotRef,
)
from codeintel.core.execution import ExecutionContext, new_run_context
from codeintel.core.runtime.loader import (
    RuntimeInputs,
    build_runtime_primitives,
    load_execution_context,
)
from codeintel.storage.gateway import (
    DuckDBError,
    DuckDBInvalidInputException,
    StorageConfig,
    build_snapshot_gateway_resolver,
    open_gateway,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.cli.context import CommandContext
    from codeintel.storage.gateway import StorageGateway

LOG = logging.getLogger(__name__)


@contextmanager
def _output_gateway(output_db: Path) -> Iterator[StorageGateway]:
    """Open a write-enabled gateway for output database with automatic cleanup.

    Parameters
    ----------
    output_db
        Path to the output database.

    Yields
    ------
    StorageGateway
        Open gateway that closes on context exit.
    """
    gw = open_gateway(StorageConfig.for_ingest(output_db))
    try:
        yield gw
    finally:
        gw.close()


def _coerce_enum_param(value: object | None, *, default: str) -> str:
    """Coerce cyclopts enum parameters to their string value.

    Cyclopts may supply Enum instances for typed parameters; downstream config
    expects the Enum's `.value` string, not `EnumClass.MEMBER`.

    Returns
    -------
    str
        String representation of the enum or default when None.
    """
    if value is None:
        return default
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _build_execution_context(
    *,
    snapshot: SnapshotRef,
    paths: BuildPaths,
    tools: ToolsConfig,
) -> ExecutionContext:
    primitives = build_runtime_primitives(
        RuntimeInputs(
            snapshot=snapshot,
            paths=paths,
            tools=tools.to_binaries(),
            graph_backend=GraphBackendConfig(),
            graph_features=GraphFeatureFlags(),
            profiles=None,
        )
    )
    run_context = new_run_context(snapshot=snapshot, kind="analytics", trigger="cli")
    return load_execution_context(primitives=primitives, run=run_context)


def _build_history_env(
    ctx: CommandContext,
    snapshot: SnapshotRef,
    gateway: StorageGateway,
    *,
    options: HistoryTimeseriesOptions,
    gateway_resolver: Callable[[str], StorageGateway],
) -> BuildEnv:
    tools = ctx.runtime.tools if ctx.has_runtime else ToolsConfig.default()
    providers = create_default_providers(tools)
    config = load_build_config(snapshot.repo_root)
    paths = ctx.runtime.paths if ctx.has_runtime else BuildPaths.from_repo_root(snapshot.repo_root)
    execution_context = _build_execution_context(snapshot=snapshot, paths=paths, tools=tools)
    overrides = BuildRunContextOverrides(
        history_options=options,
        history_db_resolver=gateway_resolver,
    )
    context = BuildRunContext.from_execution_context(
        execution_context=execution_context,
        gateway=gateway,
        providers=providers,
        config=config,
        overrides=overrides,
    )
    return context.build_env()


def _build_gateway_resolver(
    gateway: StorageGateway,
    snapshot_resolver: Callable[[str], StorageGateway],
    snapshot_gateways: dict[str, StorageGateway],
) -> Callable[[str], StorageGateway]:
    """Build a gateway resolver from a snapshot_resolver.

    Parameters
    ----------
    gateway
        Primary output gateway.
    snapshot_resolver
        Callable returning a StorageGateway for a given commit.
    snapshot_gateways
        Cache dict for storing opened gateways (modified in place).

    Returns
    -------
    Callable[[str], StorageGateway]
        Storage gateway resolver for each commit.
    """

    def _gateway_resolver(commit: str) -> StorageGateway:
        cached_gateway = snapshot_gateways.get(commit)
        if cached_gateway is not None:
            return cached_gateway

        snapshot_gateway = snapshot_resolver(commit)
        if snapshot_gateway.config.db_path.resolve() == gateway.config.db_path.resolve():
            if snapshot_gateway is not gateway:
                snapshot_gateway.close()
            snapshot_gateways[commit] = gateway
            return gateway

        snapshot_gateways[commit] = snapshot_gateway
        return snapshot_gateway

    return _gateway_resolver


def history_timeseries_handler(ctx: CommandContext) -> CliResult[HistoryTimeseriesResult]:
    """Aggregate analytics.history_timeseries across multiple commits.

    Collect analytics data from per-commit DuckDB snapshots and aggregate
    them into a unified history timeseries table.

    Parameters
    ----------
    ctx
        Command context with params:
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
    bootstrap_cli(verbosity=ctx.verbosity)

    commits = ctx.params.get_list("commits")

    if not commits:
        return fail_history_error("Validation Error", "At least one commit is required")

    repo = ctx.params.require_str("repo")

    repo_root = ctx.params.get_path("repo_root", Path.cwd()) or Path.cwd()
    db_dir = ctx.params.get_path("db_dir", Path("build/db")) or Path("build/db")
    output_db = ctx.params.get_path("output_db", Path("build/db/history.duckdb")) or Path(
        "build/db/history.duckdb"
    )
    entity_kind = _coerce_enum_param(ctx.params.raw.get("entity_kind"), default="function")

    snapshot = SnapshotRef(repo=repo, commit=commits[0], repo_root=repo_root)
    options = HistoryTimeseriesOptions(
        commits=tuple(commits),
        entity_kind=entity_kind,
        max_entities=ctx.params.get_int("max_entities", 500),
        selection_strategy=_coerce_enum_param(
            ctx.params.raw.get("selection_strategy"),
            default="risk_score",
        ),
    )

    with _output_gateway(output_db) as gateway:
        snapshot_resolver = build_snapshot_gateway_resolver(
            db_dir=db_dir,
            repo=repo,
            primary_gateway=gateway,
        )

        snapshot_gateways: dict[str, StorageGateway] = {}
        gateway_resolver = _build_gateway_resolver(
            gateway,
            snapshot_resolver,
            snapshot_gateways,
        )

        try:
            env = _build_history_env(
                ctx,
                snapshot,
                gateway,
                options=options,
                gateway_resolver=gateway_resolver,
            )
            executor = HamiltonBuildExecutor(
                parallel_backend=env.execution_settings.parallel_backend,
                max_workers=env.execution_settings.max_workers,
            )
            result = executor.run(env=env, targets=["history_timeseries"])
        except DuckDBInvalidInputException as exc:
            LOG.warning("No history rows to aggregate: %s", exc)
        except FileNotFoundError as exc:
            LOG.exception("Missing snapshot database for history_timeseries")
            return fail_history_error(
                "Storage Error", f"Missing snapshot database for one or more commits: {exc}"
            )
        except DuckDBError as exc:
            LOG.exception("Failed to compute history_timeseries")
            return fail_history_error("Query Error", f"Failed to compute history_timeseries: {exc}")
        else:
            if not result.success:
                error_msg = result.error or "History timeseries execution failed"
                if "Invalid Input" in error_msg or "invalid input" in error_msg:
                    LOG.warning("history_timeseries returned no rows: %s", error_msg)
                else:
                    return fail_history_error("Execution Error", error_msg)
        finally:
            for gw in snapshot_gateways.values():
                if gw is not gateway:
                    gw.close()

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
