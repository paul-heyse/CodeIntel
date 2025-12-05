"""Test execution context builders using real production types.

This module provides fluent builders for creating real `PluginExecutionContext`
instances for testing. Per the Testing Charter, no mocking is allowed - these
builders use the actual production context classes and resource providers.

Example
-------
>>> from tests._helpers.fakes import TestExecutionContextBuilder
>>> from tests._helpers.fakes import create_graph_gateway, create_test_snapshot
>>> gateway = create_graph_gateway()
>>> snapshot = create_test_snapshot(tmp_path)
>>> ctx = TestExecutionContextBuilder(gateway, snapshot).with_config(MyStepConfig, config).build()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypeVar

from codeintel.analytics.core.context import (
    PluginExecutionContext,
)
from codeintel.analytics.runtime.manifest import AnalyticsScope
from codeintel.config.primitives import SnapshotRef
from codeintel.core.plugins.context import ConfigProvider, PluginScratch
from codeintel.core.resources import ResourceRegistry
from codeintel.storage.gateway import StorageGateway, open_memory_gateway
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO, DEFAULT_RUN_ID

if TYPE_CHECKING:
    from codeintel.config.primitives import BuildPaths
    from codeintel.runtime import RunContext

T = TypeVar("T")


class TestExecutionContextBuilder:
    """Fluent builder for real plugin execution contexts.

    Creates production-grade `PluginExecutionContext` instances for testing,
    using real types throughout. No mocking is involved.

    This builder provides test-friendly defaults:
    - In-memory gateway with schema applied
    - Standard test snapshot (DEFAULT_REPO, DEFAULT_COMMIT)
    - Standard run ID

    Example
    -------
    >>> builder = TestExecutionContextBuilder.create(tmp_path)
    >>> ctx = builder.with_config(MyStepConfig, config).build()
    >>> assert ctx.has_config(MyStepConfig)
    """

    def __init__(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        run_id: str = DEFAULT_RUN_ID,
    ) -> None:
        """Initialize the builder with required dependencies.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        snapshot
            Repository snapshot reference.
        run_id
            Unique run identifier.
        """
        self._gateway = gateway
        self._snapshot = snapshot
        self._run_id = run_id
        self._configs: dict[type[Any], object] = {}
        self._resources: ResourceRegistry = ResourceRegistry()
        self._extra: dict[str, Any] = {}
        self._paths: BuildPaths | None = None
        self._options: object | None = None
        self._plugin_name: str | None = None
        self._run_context: RunContext | None = None
        self._scope: AnalyticsScope = AnalyticsScope()
        self._scratch: PluginScratch | None = None

    @classmethod
    def create(
        cls,
        tmp_path: Path | None = None,
        *,
        repo: str = DEFAULT_REPO,
        commit: str = DEFAULT_COMMIT,
        run_id: str = DEFAULT_RUN_ID,
    ) -> Self:
        """Create a builder with default test infrastructure.

        Convenience factory that creates a gateway and snapshot automatically.

        Parameters
        ----------
        tmp_path
            Temporary directory for repo root. Uses mock path if None.
        repo
            Repository identifier.
        commit
            Commit hash.
        run_id
            Run identifier.

        Returns
        -------
        Self
            Configured builder instance.
        """
        gateway = open_memory_gateway(
            apply_schema=True,
            ensure_views=True,
            validate_schema=True,
        )
        repo_root = tmp_path if tmp_path is not None else Path("/mock/repo")
        snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=repo_root)
        return cls(gateway, snapshot, run_id)

    def with_config(self, config_type: type[T], config: T) -> Self:
        """Add a configuration to the context.

        Parameters
        ----------
        config_type
            Type of configuration (used as lookup key).
        config
            Configuration instance.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._configs[config_type] = config
        return self

    def with_resource(self, resource_type: type[T], provider: object) -> Self:
        """Register a resource provider.

        Parameters
        ----------
        resource_type
            Type key for the provider.
        provider
            Resource provider instance.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._resources.register(resource_type, provider)
        return self

    def with_extra(self, key: str, value: object) -> Self:
        """Add extra metadata to the context.

        Parameters
        ----------
        key
            Metadata key.
        value
            Metadata value.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._extra[key] = value
        return self

    def with_extras(self, **extras: object) -> Self:
        """Add multiple extra metadata entries.

        Parameters
        ----------
        **extras
            Key-value pairs to add to extra dict.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._extra.update(extras)
        return self

    def with_paths(self, paths: BuildPaths) -> Self:
        """Set the build paths configuration.

        Parameters
        ----------
        paths
            Build paths configuration.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._paths = paths
        return self

    def with_options(self, options: object) -> Self:
        """Set plugin-specific options.

        Parameters
        ----------
        options
            Options object.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._options = options
        return self

    def with_plugin_name(self, name: str) -> Self:
        """Set the current plugin name.

        Parameters
        ----------
        name
            Plugin name.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._plugin_name = name
        return self

    def with_scope(self, scope: AnalyticsScope) -> Self:
        """Set the analytics execution scope.

        Parameters
        ----------
        scope
            Analytics scope for execution.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._scope = scope
        return self

    def with_scratch(self, scratch: PluginScratch) -> Self:
        """Set a shared scratch store.

        Parameters
        ----------
        scratch
            Shared scratch store.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._scratch = scratch
        return self

    def with_run_context(self, run_context: RunContext) -> Self:
        """Set the unified run context.

        Parameters
        ----------
        run_context
            Run context for cross-engine correlation.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._run_context = run_context
        return self

    def build(self) -> PluginExecutionContext:
        """Build the execution context.

        Returns
        -------
        PluginExecutionContext
            Configured execution context with real production types.
        """
        return PluginExecutionContext(
            gateway=self._gateway,
            snapshot=self._snapshot,
            run_id=self._run_id,
            resources=self._resources,
            configs=ConfigProvider(self._configs),
            scratch=self._scratch or PluginScratch(),
            paths=self._paths,
            options=self._options,
            plugin_name=self._plugin_name,
            extra=dict(self._extra),
            run_context=self._run_context,
            scope=self._scope,
        )


def create_test_execution_context(
    tmp_path: Path | None = None,
    *,
    gateway: StorageGateway | None = None,
    snapshot: SnapshotRef | None = None,
    configs: dict[type[Any], object] | None = None,
) -> PluginExecutionContext:
    """Create a test execution context with sensible defaults.

    Convenience function for tests that need a simple context without
    extensive configuration. For more control, use TestExecutionContextBuilder.

    Parameters
    ----------
    tmp_path
        Temporary path for repo root (used if snapshot is None).
    gateway
        Storage gateway. Creates in-memory gateway if None.
    snapshot
        Snapshot reference. Creates default if None.
    configs
        Optional configs to register.

    Returns
    -------
    PluginExecutionContext
        Configured execution context.

    Example
    -------
    >>> ctx = create_test_execution_context(tmp_path=tmp_path)
    >>> assert ctx.repo == DEFAULT_REPO
    """
    if gateway is None:
        gateway = open_memory_gateway(
            apply_schema=True,
            ensure_views=True,
            validate_schema=True,
        )

    if snapshot is None:
        repo_root = tmp_path if tmp_path is not None else Path("/mock/repo")
        snapshot = SnapshotRef(
            repo=DEFAULT_REPO,
            commit=DEFAULT_COMMIT,
            repo_root=repo_root,
        )

    builder = TestExecutionContextBuilder(gateway, snapshot)

    if configs:
        for config_type, config in configs.items():
            builder.with_config(config_type, config)

    return builder.build()


__all__ = [
    "TestExecutionContextBuilder",
    "create_test_execution_context",
]
