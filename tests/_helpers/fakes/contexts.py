"""Unified builders for plugin and target execution contexts.

All contexts are production-backed (real StorageGateway, SnapshotRef, BuildPaths) with
optional SQL call recording. This replaces fragmented builders in ingestion and plugin
helpers while keeping the wiring consistent with runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast

from codeintel.build.context import ContextResources, TargetExecutionContext
from codeintel.build.parameters import EMPTY_PARAMETERS, TargetParameters
from codeintel.build.targets import OutputTarget
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.core.execution import RunContext
from codeintel.core.plugins.execution.context import (
    ConfigProvider,
    PluginExecutionContext,
    PluginScratch,
)
from codeintel.core.resources import ResourceRegistry
from codeintel.storage.gateway import DuckDBConnection, StorageGateway
from tests._helpers.env import DEFAULT_COMMIT, DEFAULT_REPO, DEFAULT_RUN_ID, create_test_env

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.analytics.runtime import GraphRuntime
    from codeintel.build.protocols import (
        CoverageCollector,
        GitHistoryProvider,
        ScipIndexer,
        ToolRunner,
        TypeChecker,
    )
    from codeintel.build.providers import Providers, RealTestReporter

T = TypeVar("T")


@dataclass(frozen=True)
class SqlCall:
    """Record of a SQL execution against the gateway."""

    sql: str
    params: tuple[object, ...]


class RecordingGateway:
    """Thin wrapper that records execute calls while delegating to the real gateway."""

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
class TargetResourceOverrides:
    """Optional resource overrides for target execution contexts."""

    providers: object | None = None
    modules: tuple[str, ...] = ()
    change_tracker: object | None = None
    graph_runtime: object | None = None
    catalog: object | None = None
    test_reporter: object | None = None
    coverage_collector: object | None = None
    scip_indexer: object | None = None
    type_checker: object | None = None
    tool_runner: object | None = None
    git_history: object | None = None


@dataclass
class ExecutionContextBuilder:
    """Fluent builder for plugin and target execution contexts."""

    gateway: StorageGateway
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
        self._gateway = self.gateway
        self.sql_records: list[SqlCall] | None = None
        if self.record_sql:
            recorder = RecordingGateway(self.gateway)
            self._gateway = cast("StorageGateway", recorder)
            self.sql_records = recorder.records

    @classmethod
    def create(
        cls,
        tmp_path: Path | None = None,
        options: BuilderOptions | None = None,
    ) -> Self:
        """Create a builder with a fresh gateway and snapshot.

        Returns
        -------
        Self
            Configured builder.
        """
        opts = options or BuilderOptions()
        base_path = tmp_path if tmp_path is not None else Path("/mock/repo")
        env_ctx = create_test_env(
            base_path,
            repo=opts.repo,
            commit=opts.commit,
            file_backed=opts.file_backed,
            repo_root=base_path,
        )
        return cls(
            gateway=env_ctx.gateway,
            snapshot=env_ctx.snapshot,
            paths=env_ctx.build_paths,
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
            gateway=self._gateway,
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
        resources = ContextResources(
            providers=cast("Providers | None", overrides.providers),
            gateway=self._gateway,
            modules=overrides.modules,
            change_tracker=overrides.change_tracker,
            graph_runtime=cast("GraphRuntime | None", overrides.graph_runtime),
            catalog=overrides.catalog,
            test_reporter=cast("RealTestReporter | None", overrides.test_reporter),
            coverage_collector=cast("CoverageCollector | None", overrides.coverage_collector),
            scip_indexer=cast("ScipIndexer | None", overrides.scip_indexer),
            type_checker=cast("TypeChecker | None", overrides.type_checker),
            tool_runner=cast("ToolRunner | None", overrides.tool_runner),
            git_history=cast("GitHistoryProvider | None", overrides.git_history),
        )
        return TargetExecutionContext(
            target=target,
            snapshot=self.snapshot,
            paths=self._paths or BuildPaths.from_repo_root(self.snapshot.repo_root),
            resources=resources,
            parameters=parameters or EMPTY_PARAMETERS,
        )


def build_plugin_execution_context(
    *,
    tmp_path: Path | None = None,
    options: BuilderOptions | None = None,
    configs: dict[type[Any], object] | None = None,
    resources: dict[type[Any], object] | None = None,
) -> PluginExecutionContext:
    """Build a production PluginExecutionContext.

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
    *,
    tmp_path: Path | None = None,
    options: BuilderOptions | None = None,
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
    builder = ExecutionContextBuilder.create(
        tmp_path,
        options=options,
    )
    return builder.build_target_context(
        target=target,
        parameters=parameters,
        providers=cast("Providers | None", overrides.providers),
        modules=overrides.modules,
        change_tracker=overrides.change_tracker,
        graph_runtime=cast("GraphRuntime | None", overrides.graph_runtime),
        catalog=overrides.catalog,
        test_reporter=cast("RealTestReporter | None", overrides.test_reporter),
        coverage_collector=cast("CoverageCollector | None", overrides.coverage_collector),
        scip_indexer=cast("ScipIndexer | None", overrides.scip_indexer),
        type_checker=cast("TypeChecker | None", overrides.type_checker),
        tool_runner=cast("ToolRunner | None", overrides.tool_runner),
        git_history=cast("GitHistoryProvider | None", overrides.git_history),
    )


__all__ = [
    "BuilderOptions",
    "ExecutionContextBuilder",
    "RecordingGateway",
    "SqlCall",
    "TargetResourceOverrides",
    "build_plugin_execution_context",
    "build_target_execution_context",
]
