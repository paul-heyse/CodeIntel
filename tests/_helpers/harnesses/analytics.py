"""Test harness for analytics plugins.

This module provides a fluent test harness for analytics plugins, inheriting
from the shared base harness to reduce code duplication.

Example
-------
>>> from tests._helpers.harnesses import PluginTestHarness, assert_result
>>> from codeintel.analytics.resources.graphs import GraphProvider
>>> from my_plugin import MyPlugin
>>>
>>> def test_my_plugin(analytics_gateway, graph_provider):
...     result = (
...         PluginTestHarness.for_plugin(MyPlugin())
...         .with_gateway(analytics_gateway)
...         .with_snapshot("test-repo", "abc123")
...         .with_config(MyConfig(enabled=True))
...         .with_graph_provider(graph_provider)
...         .execute()
...     )
...     assert_result(result).succeeded().has_row_count("analytics.my_table", min_rows=1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self, TypeVar

from codeintel.analytics.core.context import (
    ConfigProvider,
    PluginExecutionContext,
    PluginScratch,
)
from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.analytics.resources.graphs import GraphProvider
from codeintel.analytics.resources.registry import ResourceRegistry
from codeintel.analytics.runtime.manifest import AnalyticsScope
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO
from tests._helpers.harnesses.base import BaseResultAssertions, BaseTestHarness

if TYPE_CHECKING:
    from codeintel.analytics.core.protocol import (
        AnalyticsPluginProtocol,
        PluginResult,
        ValidationResult,
    )
    from codeintel.analytics.resources.protocol import ResourceProvider

T = TypeVar("T")


class StepConfigProtocol(Protocol):
    """Protocol for step configs with snapshot properties.

    Any step config class with `repo`, `commit`, and `repo_root` properties
    (derived from a `snapshot` attribute) satisfies this protocol.
    """

    @property
    def repo(self) -> str:
        """Repository identifier."""
        ...

    @property
    def commit(self) -> str:
        """Commit identifier."""
        ...

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        ...


@dataclass
class PluginTestHarness(BaseTestHarness["AnalyticsPluginProtocol", PluginExecutionContext]):
    """Fluent test harness for analytics plugins.

    Provide a clean API for setting up plugin tests with minimal
    boilerplate. All resource access uses the ResourceRegistry pattern.

    Attributes
    ----------
    _plugin : AnalyticsPluginProtocol
        The plugin being tested.
    _configs : dict[type, object]
        Configuration objects keyed by type.
    _resources : ResourceRegistry | None
        Resource registry for provider access.
    _options : object | None
        Plugin-specific options.
    _extra : dict[str, Any]
        Additional context metadata.
    """

    _configs: dict[type[object], object] = field(default_factory=dict)
    _resources: ResourceRegistry | None = None
    _options: object | None = None
    _extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_plugin(cls, plugin: AnalyticsPluginProtocol) -> PluginTestHarness:
        """Create a harness for testing a specific plugin.

        Parameters
        ----------
        plugin
            The plugin instance to test.

        Returns
        -------
        PluginTestHarness
            A new harness configured for the plugin.
        """
        return cls(_plugin=plugin)

    @classmethod
    def for_analytics_plugin(
        cls,
        plugin: AnalyticsPluginProtocol,
        gateway: StorageGateway,
        repo_root: Path,
        *,
        repo: str = DEFAULT_REPO,
        commit: str = DEFAULT_COMMIT,
    ) -> PluginTestHarness:
        """Create a harness with standard analytics setup.

        Convenience factory that configures common analytics plugin
        test environment in one call.

        Parameters
        ----------
        plugin
            The plugin instance to test.
        gateway
            Storage gateway with schema applied.
        repo_root
            Repository root path for the snapshot.
        repo
            Repository identifier (defaults to DEFAULT_REPO).
        commit
            Commit identifier (defaults to DEFAULT_COMMIT).

        Returns
        -------
        PluginTestHarness
            Harness configured with gateway and snapshot.

        Example
        -------
        >>> harness = PluginTestHarness.for_analytics_plugin(
        ...     MyPlugin(),
        ...     gateway,
        ...     tmp_path,
        ... )
        """
        return cls.for_plugin(plugin).with_gateway(gateway).with_snapshot(repo, commit, repo_root)

    def with_snapshot(
        self,
        repo: str,
        commit: str,
        repo_root: Path | None = None,
        *,
        _keyword_only: None = None,
    ) -> Self:
        """Set the repository snapshot.

        Parameters
        ----------
        repo
            Repository identifier.
        commit
            Commit identifier.
        repo_root
            Optional repository root path.
        _keyword_only
            Unused, forces keyword-only arguments after repo_root.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._repo = repo
        self._commit = commit
        if repo_root is not None:
            self._repo_root = repo_root
        return self

    def with_config(self, config: object) -> Self:
        """Add a configuration object.

        The config type is used as the key for lookup.

        Parameters
        ----------
        config
            Configuration object.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._configs[type(config)] = config
        return self

    def with_step_config(self, config: StepConfigProtocol) -> Self:
        """Add a step config and extract snapshot from it.

        Convenience method for step configs that have embedded snapshot
        information. Registers the config and sets repo/commit/repo_root
        from the config's properties.

        Parameters
        ----------
        config
            Step config with repo, commit, and repo_root properties.

        Returns
        -------
        Self
            Self for chaining.

        Example
        -------
        >>> from codeintel.config import FunctionContractsStepConfig
        >>> harness.with_step_config(FunctionContractsStepConfig(snapshot=snapshot))
        """
        self._configs[type(config)] = config
        self._repo = config.repo
        self._commit = config.commit
        self._repo_root = config.repo_root
        return self

    def with_configs(self, *configs: object) -> Self:
        """Add multiple configuration objects.

        Parameters
        ----------
        configs
            Configuration objects to add.

        Returns
        -------
        Self
            Self for chaining.
        """
        for cfg in configs:
            self.with_config(cfg)
        return self

    def with_resources(self, registry: ResourceRegistry) -> Self:
        """Set the resource registry.

        Parameters
        ----------
        registry
            Resource registry with providers.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._resources = registry
        return self

    def with_resource(
        self,
        resource_type: type[T],
        provider: ResourceProvider[T],
    ) -> Self:
        """Register a resource provider.

        Parameters
        ----------
        resource_type
            Type key for the resource.
        provider
            Provider instance.

        Returns
        -------
        Self
            Self for chaining.
        """
        if self._resources is None:
            self._resources = ResourceRegistry()
        self._resources.register(resource_type, provider)
        return self

    def with_graph_provider(self, provider: GraphProvider) -> Self:
        """Register a graph provider.

        Parameters
        ----------
        provider
            Graph provider instance.

        Returns
        -------
        Self
            Self for chaining.
        """
        if self._resources is None:
            self._resources = ResourceRegistry()
        self._resources.register(GraphProvider, provider)
        return self

    def with_catalog_provider(self, provider: CatalogProvider) -> Self:
        """Register a catalog provider.

        Parameters
        ----------
        provider
            Catalog provider instance.

        Returns
        -------
        Self
            Self for chaining.
        """
        if self._resources is None:
            self._resources = ResourceRegistry()
        self._resources.register(CatalogProvider, provider)
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

    def with_extra(self, key: str, value: object) -> Self:
        """Add extra context metadata.

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

    def build_context(self) -> PluginExecutionContext:
        """Build the execution context for testing.

        Returns
        -------
        PluginExecutionContext
            Configured execution context.

        Raises
        ------
        ValueError
            If gateway is not set.
        """
        if self._gateway is None:
            message = "Gateway must be set before building context"
            raise ValueError(message)

        repo_root = self._repo_root or Path.cwd()
        snapshot = SnapshotRef(
            repo=self._repo,
            commit=self._commit,
            repo_root=repo_root,
        )

        scope = AnalyticsScope()

        # Build scratch with pre-populated data
        scratch = PluginScratch()
        for key, value in self._scratch_data.items():
            scratch.declare(key, value)

        # Use provided resources or create empty registry
        resources = self._resources if self._resources is not None else ResourceRegistry()

        return PluginExecutionContext(
            gateway=self._gateway,
            snapshot=snapshot,
            run_id=self._run_id,
            scope=scope,
            configs=ConfigProvider(self._configs),
            scratch=scratch,
            resources=resources,
            options=self._options,
            plugin_name=self._plugin.metadata.name,
            extra=dict(self._extra),
        )

    def validate(self) -> ValidationResult:
        """Run input validation.

        Returns
        -------
        ValidationResult
            Validation result.
        """
        ctx = self.build_context()
        return self._plugin.validate_inputs(ctx)

    def execute(self) -> PluginResult:
        """Execute the plugin.

        Returns
        -------
        PluginResult
            Execution result.
        """
        ctx = self.build_context()
        return self._plugin.execute(ctx)

    def execute_with_context(self) -> tuple[PluginResult, PluginExecutionContext]:
        """Execute and return both result and context.

        Useful when you need to inspect scratch store after execution.

        Returns
        -------
        tuple[PluginResult, PluginExecutionContext]
            Result and context.
        """
        ctx = self.build_context()
        result = self._plugin.execute(ctx)
        return result, ctx


# =============================================================================
# Assertion Helpers
# =============================================================================


@dataclass
class PluginResultAssertions(BaseResultAssertions["PluginResult"]):
    """Fluent assertions for analytics plugin results.

    Extends base assertions with analytics-specific checks.

    Example
    -------
    >>> result = harness.execute()
    >>> (
    ...     PluginResultAssertions(_result=result)
    ...     .succeeded()
    ...     .has_row_count("analytics.my_table", min_rows=1)
    ...     .has_no_error()
    ... )
    """

    def has_meta(self, key: str, value: object | None = None) -> Self:
        """Assert that metadata contains a key.

        Parameters
        ----------
        key
            Metadata key to check.
        value
            Optional expected value.

        Returns
        -------
        Self
            Self for chaining.

        Raises
        ------
        AssertionError
            If key missing or value doesn't match.
        """
        if key not in self._result.meta:
            msg = f"{self._message_prefix}Expected meta to contain '{key}'"
            raise AssertionError(msg.strip())

        if value is not None and self._result.meta[key] != value:
            msg = (
                f"{self._message_prefix}Expected meta['{key}'] to be {value!r}, "
                f"got {self._result.meta[key]!r}"
            )
            raise AssertionError(msg.strip())

        return self


def assert_result(result: PluginResult) -> PluginResultAssertions:
    """Start fluent assertions on a plugin result.

    Parameters
    ----------
    result
        Result to assert on.

    Returns
    -------
    PluginResultAssertions
        Fluent assertion builder.

    Example
    -------
    >>> result = harness.execute()
    >>> assert_result(result).succeeded().has_row_count("analytics.my_table", min_rows=1)
    """
    return PluginResultAssertions(_result=result)


# =============================================================================
# Validation Assertions
# =============================================================================


@dataclass
class ValidationResultAssertions:
    """Fluent assertions for validation results.

    Example
    -------
    >>> result = harness.validate()
    >>> assert_validation(result).is_valid()
    """

    _result: ValidationResult
    _message_prefix: str = ""

    def with_message(self, prefix: str) -> Self:
        """Set assertion message prefix.

        Parameters
        ----------
        prefix
            Message prefix.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._message_prefix = prefix
        return self

    def is_valid(self) -> Self:
        """Assert that validation passed.

        Returns
        -------
        Self
            Self for chaining.

        Raises
        ------
        AssertionError
            If validation failed.
        """
        if not self._result.valid:
            msg = f"{self._message_prefix}Expected valid but got errors: {self._result.errors}"
            raise AssertionError(msg.strip())
        return self

    def is_invalid(self) -> Self:
        """Assert that validation failed.

        Returns
        -------
        Self
            Self for chaining.

        Raises
        ------
        AssertionError
            If validation passed.
        """
        if self._result.valid:
            msg = f"{self._message_prefix}Expected invalid but validation passed"
            raise AssertionError(msg.strip())
        return self

    def has_error(self, containing: str) -> Self:
        """Assert that there is an error containing text.

        Parameters
        ----------
        containing
            Substring to find in errors.

        Returns
        -------
        Self
            Self for chaining.

        Raises
        ------
        AssertionError
            If no error contains the text.
        """
        for error in self._result.errors:
            if containing in error:
                return self

        msg = f"{self._message_prefix}Expected error containing '{containing}' but got: {self._result.errors}"
        raise AssertionError(msg.strip())


def assert_validation(result: ValidationResult) -> ValidationResultAssertions:
    """Start fluent assertions on a validation result.

    Parameters
    ----------
    result
        Validation result to assert on.

    Returns
    -------
    ValidationResultAssertions
        Fluent assertion builder.
    """
    return ValidationResultAssertions(_result=result)


__all__ = [
    "PluginResultAssertions",
    "PluginTestHarness",
    "ValidationResultAssertions",
    "assert_result",
    "assert_validation",
]
