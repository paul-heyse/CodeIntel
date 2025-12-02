"""Standardized test infrastructure for analytics plugins.

This module provides a fluent test harness that makes it easy to test
plugins with minimal boilerplate. The harness handles context setup,
execution, and assertion patterns.

All plugins use the ResourceRegistry pattern for resource access.
Use `with_resource()`, `with_graph_provider()`, or `with_catalog_provider()`
to configure resources.

Example
-------
>>> from tests._helpers.plugin_harness import PluginTestHarness
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
...     assert result.success
...     assert result.row_counts.get("analytics.my_table", 0) > 0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypeVar
from uuid import uuid4

from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.analytics.resources.graphs import GraphProvider
from codeintel.analytics.resources.registry import ResourceRegistry

if TYPE_CHECKING:
    from codeintel.analytics.core.execution_context import PluginExecutionContext
    from codeintel.analytics.core.plugin_protocol import (
        AnalyticsPluginProtocol,
        PluginResult,
        ValidationResult,
    )
    from codeintel.analytics.resources.protocol import ResourceProvider
    from codeintel.storage.gateway import StorageGateway

T = TypeVar("T")


@dataclass
class PluginTestHarness:
    """Fluent test harness for analytics plugins.

    Provides a clean API for setting up plugin tests with minimal
    boilerplate. All resource access uses the ResourceRegistry pattern.

    Attributes
    ----------
    _plugin : AnalyticsPluginProtocol
        The plugin being tested.
    _gateway : StorageGateway | None
        Storage gateway for database access.
    _repo : str
        Repository identifier.
    _commit : str
        Commit identifier.
    _repo_root : Path | None
        Repository root path.
    _configs : dict[type, object]
        Configuration objects keyed by type.
    _resources : ResourceRegistry | None
        Resource registry for provider access.
    _options : object | None
        Plugin-specific options.
    _extra : dict[str, Any]
        Additional context metadata.
    _scratch_data : dict[str, object]
        Pre-populated scratch store data.
    """

    _plugin: AnalyticsPluginProtocol
    _gateway: StorageGateway | None = None
    _repo: str = "test-repo"
    _commit: str = "test-commit"
    _repo_root: Path | None = None
    _configs: dict[type[object], object] = field(default_factory=dict)
    _resources: ResourceRegistry | None = None
    _options: object | None = None
    _extra: dict[str, Any] = field(default_factory=dict)
    _scratch_data: dict[str, object] = field(default_factory=dict)
    _run_id: str = field(default_factory=lambda: uuid4().hex)

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

    def with_gateway(self, gateway: StorageGateway) -> Self:
        """Set the storage gateway.

        Parameters
        ----------
        gateway
            Storage gateway for database access.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._gateway = gateway
        return self

    def with_snapshot(
        self,
        repo: str,
        commit: str,
        *,
        repo_root: Path | None = None,
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

        Returns
        -------
        Self
            Self for chaining.
        """
        self._repo = repo
        self._commit = commit
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
        return self.with_resource(GraphProvider, provider)

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
        return self.with_resource(CatalogProvider, provider)

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

    def with_scratch(self, key: str, value: object) -> Self:
        """Pre-populate scratch store with data.

        Useful for testing plugins that consume data from upstream plugins.

        Parameters
        ----------
        key
            Scratch key.
        value
            Value to store.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._scratch_data[key] = value
        return self

    def with_run_id(self, run_id: str) -> Self:
        """Set the run identifier.

        Parameters
        ----------
        run_id
            Unique run identifier.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._run_id = run_id
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
        from pathlib import Path  # noqa: PLC0415

        from codeintel.analytics.core.execution_context import (  # noqa: PLC0415
            ConfigProvider,
            PluginExecutionContext,
            PluginScratch,
        )
        from codeintel.analytics.runtime_manifest import AnalyticsScope  # noqa: PLC0415
        from codeintel.config.primitives import SnapshotRef  # noqa: PLC0415

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

        from codeintel.analytics.resources.registry import (  # noqa: PLC0415
            ResourceRegistry,
        )

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
class PluginResultAssertions:
    """Fluent assertions for plugin results.

    Provides chainable assertion methods for validating plugin results.

    Example
    -------
    >>> result = harness.execute()
    >>> (
    ...     PluginResultAssertions(result)
    ...     .succeeded()
    ...     .has_row_count("analytics.my_table", min_rows=1)
    ...     .has_no_error()
    ... )
    """

    _result: PluginResult
    _message_prefix: str = ""

    def with_message(self, prefix: str) -> Self:
        """Set a prefix for assertion messages.

        Parameters
        ----------
        prefix
            Prefix to add to assertion messages.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._message_prefix = prefix
        return self

    def succeeded(self) -> Self:
        """Assert that execution succeeded.

        Returns
        -------
        Self
            Self for chaining.

        Raises
        ------
        AssertionError
            If execution failed.
        """
        msg = f"{self._message_prefix}Expected success but got failure: {self._result.error}"
        if not self._result.success:
            raise AssertionError(msg.strip())
        return self

    def failed(self) -> Self:
        """Assert that execution failed.

        Returns
        -------
        Self
            Self for chaining.

        Raises
        ------
        AssertionError
            If execution succeeded.
        """
        msg = f"{self._message_prefix}Expected failure but got success"
        if self._result.success:
            raise AssertionError(msg.strip())
        return self

    def has_error(self, containing: str | None = None) -> Self:
        """Assert that there is an error message.

        Parameters
        ----------
        containing
            Optional substring the error must contain.

        Returns
        -------
        Self
            Self for chaining.

        Raises
        ------
        AssertionError
            If no error or substring not found.
        """
        msg = f"{self._message_prefix}Expected error but got none"
        if self._result.error is None:
            raise AssertionError(msg.strip())

        if containing is not None and containing not in self._result.error:
            msg = f"{self._message_prefix}Expected error containing '{containing}' but got: {self._result.error}"
            raise AssertionError(msg.strip())

        return self

    def has_no_error(self) -> Self:
        """Assert that there is no error message.

        Returns
        -------
        Self
            Self for chaining.

        Raises
        ------
        AssertionError
            If there is an error.
        """
        if self._result.error is not None:
            msg = f"{self._message_prefix}Expected no error but got: {self._result.error}"
            raise AssertionError(msg.strip())
        return self

    def has_row_count(
        self,
        table: str,
        *,
        min_rows: int | None = None,
        max_rows: int | None = None,
        exact: int | None = None,
    ) -> Self:
        """Assert row count for a table.

        Parameters
        ----------
        table
            Table name to check.
        min_rows
            Minimum expected rows.
        max_rows
            Maximum expected rows.
        exact
            Exact expected row count.

        Returns
        -------
        Self
            Self for chaining.

        Raises
        ------
        AssertionError
            If row count doesn't match expectations.
        """
        actual = self._result.row_counts.get(table, 0)

        if exact is not None:
            if actual != exact:
                msg = f"{self._message_prefix}Expected {table} to have {exact} rows, got {actual}"
                raise AssertionError(msg.strip())
            return self

        if min_rows is not None and actual < min_rows:
            msg = f"{self._message_prefix}Expected {table} to have at least {min_rows} rows, got {actual}"
            raise AssertionError(msg.strip())

        if max_rows is not None and actual > max_rows:
            msg = f"{self._message_prefix}Expected {table} to have at most {max_rows} rows, got {actual}"
            raise AssertionError(msg.strip())

        return self

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
