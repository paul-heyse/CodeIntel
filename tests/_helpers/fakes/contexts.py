"""Unified builders for plugin and target execution contexts.

All contexts are production-backed (real StorageGateway, SnapshotRef, BuildPaths) with
optional SQL call recording. This replaces fragmented builders in ingestion and plugin
helpers while keeping the wiring consistent with runtime code.

This module provides the canonical entry point for building execution contexts in tests:

- ``ExecutionContextBuilder`` - Fluent builder for PluginExecutionContext and
  TargetExecutionContext
- ``RecordingGateway`` - SQL recording wrapper around real gateways
- ``make_test_output_target`` - Create minimal OutputTarget for plugin testing
- ``execute_plugin`` / ``execute_plugin_async`` - Execute TargetPlugin with builder

Migration Guide
---------------
**From plugin_execution.py:**

Before::

    from tests._helpers.plugin_execution import PluginTestContext, execute_target_plugin


    def test_plugin(plugin_ctx: PluginTestContext) -> None:
        result = execute_target_plugin(MyPlugin(), plugin_ctx)

After::

    from tests._helpers.fakes.contexts import ExecutionContextBuilder


    def test_plugin(tmp_path: Path) -> None:
        builder = ExecutionContextBuilder.create(tmp_path)
        result = builder.execute_plugin(MyPlugin())

**From ingestion_context.py:**

Before::

    from tests._helpers.fakes.ingestion_context import make_target_context

    ctx = make_target_context(repo_root, modules=("a.py",))

After::

    from tests._helpers.fakes.contexts import ExecutionContextBuilder

    builder = ExecutionContextBuilder.create(tmp_path, repo_root=repo_root)
    ctx = builder.build_target_context(target, resources=TargetResourceOverrides(modules=("a.py",)))
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast

from codeintel.build.context import ContextResources, TargetExecutionContext
from codeintel.build.contracts import EMPTY_CONTRACT
from codeintel.build.parameters import EMPTY_PARAMETERS
from codeintel.build.targets import OutputTarget
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.core.plugins.execution.context import (
    ConfigProvider,
    PluginExecutionContext,
    PluginScratch,
)
from codeintel.core.resources import ResourceRegistry
from tests._helpers.defaults import DEFAULT_COMMIT, DEFAULT_REPO, DEFAULT_RUN_ID
from tests._helpers.env_options import EnvOptions

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from codeintel.analytics.resources.catalog import FunctionCatalogProvider
    from codeintel.build.context import TargetResult
    from codeintel.build.parameters import TargetParameters
    from codeintel.build.plugin import TargetPlugin
    from codeintel.build.providers import Providers
    from codeintel.core.execution import RunContext
    from codeintel.graphs.runtime import GraphRuntime
    from codeintel.ingestion.tracker import ChangeTracker
    from codeintel.storage.gateway import DuckDBConnection, StorageGateway

T = TypeVar("T")


def make_test_output_target(plugin: TargetPlugin) -> OutputTarget:
    """Create a minimal OutputTarget for testing a plugin.

    Parameters
    ----------
    plugin
        Plugin instance to create target for.

    Returns
    -------
    OutputTarget
        Minimal target suitable for test execution.
    """
    return OutputTarget(
        name=plugin.plugin_name,
        module="analytics",
        plugin=plugin.plugin_name,
        contract=EMPTY_CONTRACT,
        dependencies=(),
        description=plugin.plugin_description,
    )


@dataclass(frozen=True)
class SqlCall:
    """Record of a SQL execution against the gateway."""

    sql: str
    params: tuple[object, ...]


class RecordingGateway:
    """Thin wrapper that records execute calls while delegating to the real gateway.

    This is the canonical recording gateway implementation that wraps a real
    StorageGateway and records all SQL executions for test assertions.

    Attributes
    ----------
    records : list[SqlCall]
        List of recorded SQL executions (preferred).
    executions : list[tuple[str, list[object]]]
        Legacy-compatible list of (sql, params) tuples.
    """

    def __init__(self, gateway: StorageGateway) -> None:
        self._gateway = gateway
        self.records: list[SqlCall] = []

    @property
    def con(self) -> DuckDBConnection:
        """Expose the underlying connection.

        Returns
        -------
        DuckDBConnection
            Underlying DuckDB connection.
        """
        return self._gateway.con

    @property
    def executions(self) -> list[tuple[str, list[object]]]:
        """Provide legacy-compatible access to recorded SQL.

        This property converts the internal `records` list to the legacy
        (sql, params_list) format for backward compatibility.

        Returns
        -------
        list[tuple[str, list[object]]]
            List of (sql, params) tuples.
        """
        return [(r.sql, list(r.params)) for r in self.records]

    def close(self) -> None:
        """Close the underlying gateway."""
        self._gateway.close()

    def execute(self, sql: str, params: Iterable[object] | None = None) -> DuckDBConnection:
        """Record and forward SQL execution.

        Returns
        -------
        DuckDBConnection
            Underlying connection after execution.
        """
        params_tuple = tuple(params or ())
        self.records.append(SqlCall(sql=sql, params=params_tuple))
        return self._gateway.execute(sql, params_tuple)

    def table(self, name: str) -> object:
        """Forward relation lookup.

        Returns
        -------
        object
            Relation for the requested name.
        """
        return self._gateway.table(name)

    def __getattr__(self, item: str) -> object:
        """Delegate unknown attributes to the real gateway.

        Returns
        -------
        object
            Attribute from the underlying gateway.
        """
        return getattr(self._gateway, item)


@dataclass(frozen=True)
class BuilderOptions:
    """Options for creating execution context builders."""

    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    run_id: str = DEFAULT_RUN_ID
    file_backed: bool = False
    record_sql: bool = False


@dataclass(frozen=True)
class EnvOverrides:
    """Environment overrides for execution contexts."""

    snapshot: SnapshotRef | tuple[str, str] | None = None
    gateway: StorageGateway | RecordingGateway | None = None
    tmp_path: Path | None = None


@dataclass(frozen=True)
class TargetResourceOverrides:
    """Optional resource overrides for target execution contexts."""

    providers: Providers | None = None
    modules: tuple[str, ...] = ()
    change_tracker: ChangeTracker | None = None
    graph_runtime: GraphRuntime | None = None
    catalog: FunctionCatalogProvider | None = None


@dataclass
class ExecutionContextBuilder:
    """Fluent builder for plugin and target execution contexts."""

    gateway: StorageGateway | RecordingGateway
    snapshot: SnapshotRef
    paths: BuildPaths | None = None
    run_id: str = DEFAULT_RUN_ID
    record_sql: bool = False

    def __post_init__(self) -> None:
        self._resources: ResourceRegistry = ResourceRegistry()
        self._configs: dict[type[Any], object] = {}
        self._extra: dict[str, object] = {}
        self._paths: BuildPaths | None = self.paths
        self._options: object | None = None
        self._plugin_name: str | None = None
        self._run_context: RunContext | None = None
        self._scratch: PluginScratch | None = None
        self._gateway: StorageGateway | RecordingGateway = self.gateway
        self.sql_records: list[SqlCall] | None = None
        if self.record_sql:
            recorder = RecordingGateway(cast("StorageGateway", self.gateway))
            self._gateway = recorder
            self.sql_records = recorder.records

    @classmethod
    def create(
        cls,
        tmp_path: Path,
        options: BuilderOptions | None = None,
        env_overrides: EnvOverrides | None = None,
        *,
        repo_root: Path | None = None,
        build_dir: Path | None = None,
    ) -> Self:
        """Create a builder with a fresh gateway and snapshot.

        Parameters
        ----------
        tmp_path
            Temporary directory for test artifacts (required for test isolation).
        options
            Builder options for repo/commit/file_backed configuration.
        env_overrides
            Optional environment overrides for gateway/snapshot.
        repo_root
            Optional override for the repository root; defaults to ``tmp_path / "repo"``.
        build_dir
            Optional override for the build directory; defaults to ``tmp_path / "build"``.

        Returns
        -------
        Self
            Configured builder.
        """
        opts = options or BuilderOptions()
        overrides = env_overrides or EnvOverrides()
        snapshot_override = overrides.snapshot
        if isinstance(snapshot_override, SnapshotRef):
            repo = snapshot_override.repo
            commit = snapshot_override.commit
        else:
            repo, commit = snapshot_override or (opts.repo, opts.commit)

        effective_repo_root = repo_root or overrides.tmp_path
        if effective_repo_root is None and isinstance(snapshot_override, SnapshotRef):
            effective_repo_root = snapshot_override.repo_root
        if effective_repo_root is None:
            effective_repo_root = tmp_path / "repo"

        effective_build_dir = build_dir or (tmp_path / "build")

        effective_repo_root.mkdir(parents=True, exist_ok=True)
        effective_build_dir.mkdir(parents=True, exist_ok=True)

        gateway = overrides.gateway
        snapshot: SnapshotRef
        build_paths: BuildPaths
        if gateway is None:
            create_test_context = import_module("tests._helpers.context").create_test_context

            env_ctx = create_test_context(
                tmp_path,
                options=EnvOptions(
                    repo=repo,
                    commit=commit,
                    file_backed=opts.file_backed,
                    repo_root=effective_repo_root,
                    build_dir=effective_build_dir,
                ),
            )
            gateway = env_ctx.gateway
            snapshot = env_ctx.snapshot
            build_paths = env_ctx.build_paths
        else:
            snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=effective_repo_root)
            build_paths = BuildPaths.from_repo_root(
                effective_repo_root,
                build_dir=effective_build_dir,
            )
        return cls(
            gateway=gateway,
            snapshot=snapshot,
            paths=build_paths,
            run_id=opts.run_id,
            record_sql=opts.record_sql,
        )

    def with_config(self, config_type: type[T], config: T) -> Self:
        """Register a config object for the context.

        Returns
        -------
        Self
            Builder for chaining.
        """
        self._configs[config_type] = config
        return self

    def with_resource(self, resource_type: type[T], provider: object) -> Self:
        """Register a resource provider.

        Returns
        -------
        Self
            Builder for chaining.
        """
        self._resources.register(resource_type, provider)
        return self

    def with_extra(self, key: str, value: object) -> Self:
        """Add extra metadata to the context.

        Returns
        -------
        Self
            Builder for chaining.
        """
        self._extra[key] = value
        return self

    def with_paths(self, paths: BuildPaths) -> Self:
        """Override build paths.

        Returns
        -------
        Self
            Builder for chaining.
        """
        self._paths = paths
        return self

    def with_options(self, options: object) -> Self:
        """Attach plugin options.

        Returns
        -------
        Self
            Builder for chaining.
        """
        self._options = options
        return self

    def with_plugin_name(self, name: str) -> Self:
        """Set plugin name for PluginExecutionContext.

        Returns
        -------
        Self
            Builder for chaining.
        """
        self._plugin_name = name
        return self

    def with_scratch(self, scratch: PluginScratch) -> Self:
        """Use a shared scratch store.

        Returns
        -------
        Self
            Builder for chaining.
        """
        self._scratch = scratch
        return self

    def with_run_context(self, run_context: RunContext) -> Self:
        """Set the unified run context.

        Returns
        -------
        Self
            Builder for chaining.
        """
        self._run_context = run_context
        return self

    def build_plugin_context(self) -> PluginExecutionContext:
        """Build a production PluginExecutionContext.

        Returns
        -------
        PluginExecutionContext
            Configured execution context.
        """
        return PluginExecutionContext(
            gateway=cast("StorageGateway", self._gateway),
            snapshot=self.snapshot,
            run_id=self.run_id,
            resources=self._resources,
            configs=ConfigProvider(self._configs),
            scratch=self._scratch or PluginScratch(),
            paths=self._paths,
            options=self._options,
            plugin_name=self._plugin_name,
            extra=dict(self._extra),
            run_context=self._run_context,
        )

    def build_target_context(
        self,
        target: OutputTarget,
        *,
        parameters: TargetParameters | None = None,
        resources: TargetResourceOverrides | None = None,
    ) -> TargetExecutionContext:
        """Build a production TargetExecutionContext.

        Returns
        -------
        TargetExecutionContext
            Configured target execution context.
        """
        overrides = resources or TargetResourceOverrides()
        ctx_resources = ContextResources(
            providers=overrides.providers,
            gateway=cast("StorageGateway", self._gateway),
            modules=overrides.modules,
            change_tracker=overrides.change_tracker,
            graph_runtime=overrides.graph_runtime,
            catalog=overrides.catalog,
        )
        return TargetExecutionContext(
            target=target,
            snapshot=self.snapshot,
            paths=self._paths or BuildPaths.from_repo_root(self.snapshot.repo_root),
            resources=ctx_resources,
            parameters=parameters or EMPTY_PARAMETERS,
        )

    def build_target_context_for_plugin(
        self,
        plugin: TargetPlugin,
        *,
        parameters: TargetParameters | None = None,
        resources: TargetResourceOverrides | None = None,
    ) -> TargetExecutionContext:
        """Build a TargetExecutionContext using plugin metadata for target.

        This convenience method creates an OutputTarget from the plugin's
        metadata and builds the context in one step.

        Parameters
        ----------
        plugin
            Plugin to create context for.
        parameters
            Optional target parameters.
        resources
            Optional resource overrides.

        Returns
        -------
        TargetExecutionContext
            Configured target execution context.
        """
        target = make_test_output_target(plugin)
        return self.build_target_context(
            target=target,
            parameters=parameters,
            resources=resources,
        )

    async def execute_plugin_async(
        self,
        plugin: TargetPlugin,
        *,
        parameters: TargetParameters | None = None,
        resources: TargetResourceOverrides | None = None,
    ) -> TargetResult:
        """Execute a TargetPlugin asynchronously.

        This method builds the execution context and runs the plugin.
        Use this in async test functions.

        Parameters
        ----------
        plugin
            Plugin instance to execute.
        parameters
            Optional target parameters.
        resources
            Optional resource overrides.

        Returns
        -------
        TargetResult
            Result of plugin execution.
        """
        ctx = self.build_target_context_for_plugin(
            plugin,
            parameters=parameters,
            resources=resources,
        )
        return await plugin.execute(ctx)

    def execute_plugin(
        self,
        plugin: TargetPlugin,
        *,
        parameters: TargetParameters | None = None,
        resources: TargetResourceOverrides | None = None,
    ) -> TargetResult:
        """Execute a TargetPlugin synchronously.

        This method wraps the async plugin.execute() and runs it using
        asyncio.run() for use in synchronous test code.

        Parameters
        ----------
        plugin
            Plugin instance to execute.
        parameters
            Optional target parameters.
        resources
            Optional resource overrides.

        Returns
        -------
        TargetResult
            Result of plugin execution.
        """
        return asyncio.run(
            self.execute_plugin_async(
                plugin,
                parameters=parameters,
                resources=resources,
            )
        )

    @property
    def effective_gateway(self) -> StorageGateway | RecordingGateway:
        """Return the effective gateway (possibly with recording).

        Returns
        -------
        StorageGateway | RecordingGateway
            The gateway being used for context building.
        """
        return self._gateway

    @property
    def build_paths(self) -> BuildPaths:
        """Return effective build paths.

        Returns
        -------
        BuildPaths
            Build paths for the context.
        """
        return self._paths or BuildPaths.from_repo_root(self.snapshot.repo_root)


def build_plugin_execution_context(
    tmp_path: Path,
    *,
    options: BuilderOptions | None = None,
    configs: dict[type[Any], object] | None = None,
    resources: dict[type[Any], object] | None = None,
) -> PluginExecutionContext:
    """Build a production PluginExecutionContext.

    Parameters
    ----------
    tmp_path
        Temporary directory for test isolation (required).
    options
        Optional builder configuration.
    configs
        Optional config objects to register.
    resources
        Optional resource providers to register.

    Returns
    -------
    PluginExecutionContext
        Configured execution context.
    """
    builder = ExecutionContextBuilder.create(
        tmp_path,
        options=options,
    )
    if configs:
        for cfg_type, cfg in configs.items():
            builder.with_config(cfg_type, cfg)
    if resources:
        for res_type, provider in resources.items():
            builder.with_resource(res_type, provider)
    return builder.build_plugin_context()


def build_target_execution_context(
    target: OutputTarget,
    tmp_path: Path,
    *,
    options: BuilderOptions | None = None,
    parameters: TargetParameters | None = None,
    resources: TargetResourceOverrides | None = None,
) -> TargetExecutionContext:
    """Build a production TargetExecutionContext.

    Parameters
    ----------
    target
        Output target for the context.
    tmp_path
        Temporary directory for test isolation (required).
    options
        Optional builder configuration.
    parameters
        Optional target parameters.
    resources
        Optional resource overrides.

    Returns
    -------
    TargetExecutionContext
        Configured target execution context.
    """
    overrides = resources or TargetResourceOverrides()
    builder = ExecutionContextBuilder.create(
        tmp_path,
        options=options,
    )
    return builder.build_target_context(
        target=target,
        parameters=parameters,
        resources=overrides,
    )


__all__ = [
    "BuilderOptions",
    "EnvOverrides",
    "ExecutionContextBuilder",
    "RecordingGateway",
    "SqlCall",
    "TargetResourceOverrides",
    "build_plugin_execution_context",
    "build_target_execution_context",
    "make_test_output_target",
]
