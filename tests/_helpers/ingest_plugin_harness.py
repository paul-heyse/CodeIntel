"""Standardized test infrastructure for ingestion plugins.

This module provides a fluent test harness that makes it easy to test
class-based ingestion plugins with minimal boilerplate. The harness
handles context setup, execution, and assertion patterns.

All plugins use the ResourceRegistry pattern for resource access.
Use `with_resource()`, `with_module_provider()`, or `with_tracker_provider()`
to configure resources.

Example
-------
>>> from tests._helpers.ingest_plugin_harness import IngestPluginTestHarness
>>> from codeintel.ingestion.resources.modules import ModuleProvider
>>> from codeintel.ingestion.plugins.ast_extract import AstExtractPlugin
>>>
>>> def test_ast_extract_plugin(storage_gateway, tmp_path):
...     result = (
...         IngestPluginTestHarness.for_plugin(AstExtractPlugin())
...         .with_gateway(storage_gateway)
...         .with_snapshot("test-repo", "abc123", tmp_path)
...         .execute()
...     )
...     assert result.success
...     assert result.row_counts.get("core.ast_nodes", 0) >= 0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypeVar
from uuid import uuid4

if TYPE_CHECKING:
    from codeintel.config.models import ToolsConfig
    from codeintel.ingestion.core.base import BaseIngestPlugin
    from codeintel.ingestion.core.execution_context import IngestExecutionContext
    from codeintel.ingestion.infrastructure_utilities.source_scanner import ScanProfile
    from codeintel.ingestion.plugins.protocol import (
        IngestPluginResult,
    )
    from codeintel.ingestion.resources.protocol import ResourceProvider
    from codeintel.ingestion.resources.registry import ResourceRegistry
    from codeintel.storage.gateway import StorageGateway

T = TypeVar("T")


@dataclass
class IngestPluginTestHarness:
    """Fluent test harness for class-based ingestion plugins.

    Provide a clean API for setting up plugin tests with minimal
    boilerplate. All resource access uses the ResourceRegistry pattern.

    Attributes
    ----------
    _plugin : BaseIngestPlugin
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
    _scratch_data : dict[str, object]
        Pre-populated scratch store data.
    """

    _plugin: BaseIngestPlugin
    _gateway: StorageGateway | None = None
    _repo: str = "test-repo"
    _commit: str = "test-commit"
    _repo_root: Path | None = None
    _build_dir: Path | None = None
    _configs: dict[type[object], object] = field(default_factory=dict)
    _resources: ResourceRegistry | None = None
    _code_profile: ScanProfile | None = None
    _config_profile: ScanProfile | None = None
    _tools_config: ToolsConfig | None = None
    _scratch_data: dict[str, object] = field(default_factory=dict)
    _run_id: str = field(default_factory=lambda: uuid4().hex)

    @classmethod
    def for_plugin(cls, plugin: BaseIngestPlugin) -> IngestPluginTestHarness:
        """Create a harness for testing a specific plugin.

        Parameters
        ----------
        plugin
            The plugin instance to test.

        Returns
        -------
        IngestPluginTestHarness
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

    def with_build_dir(self, build_dir: Path) -> Self:
        """Set the build directory.

        Parameters
        ----------
        build_dir
            Build directory path.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._build_dir = build_dir
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
            from codeintel.ingestion.resources.registry import ResourceRegistry

            self._resources = ResourceRegistry()
        self._resources.register(resource_type, provider)
        return self

    def with_module_provider(self, provider: ResourceProvider[Any]) -> Self:
        """Register a module provider.

        Parameters
        ----------
        provider
            Module provider instance.

        Returns
        -------
        Self
            Self for chaining.
        """
        from codeintel.ingestion.resources.modules import ModuleProvider

        return self.with_resource(ModuleProvider, provider)

    def with_tracker_provider(self, provider: ResourceProvider[Any]) -> Self:
        """Register a tracker provider.

        Parameters
        ----------
        provider
            Tracker provider instance.

        Returns
        -------
        Self
            Self for chaining.
        """
        from codeintel.ingestion.resources.tracker import TrackerProvider

        return self.with_resource(TrackerProvider, provider)

    def with_tools_provider(self, provider: ResourceProvider[Any]) -> Self:
        """Register a tools provider.

        Parameters
        ----------
        provider
            Tools provider instance.

        Returns
        -------
        Self
            Self for chaining.
        """
        from codeintel.ingestion.resources.tools import ToolsProvider

        return self.with_resource(ToolsProvider, provider)

    def with_code_profile(self, profile: ScanProfile) -> Self:
        """Set the code scan profile.

        Parameters
        ----------
        profile
            Scan profile for code files.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._code_profile = profile
        return self

    def with_config_profile(self, profile: ScanProfile) -> Self:
        """Set the config scan profile.

        Parameters
        ----------
        profile
            Scan profile for config files.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._config_profile = profile
        return self

    def with_tools_config(self, config: ToolsConfig) -> Self:
        """Set the tools configuration.

        Parameters
        ----------
        config
            Tools configuration.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._tools_config = config
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

    def build_context(self) -> IngestExecutionContext:
        """Build the execution context for testing.

        Returns
        -------
        IngestExecutionContext
            Configured execution context.

        Raises
        ------
        ValueError
            If gateway is not set.
        """
        from codeintel.config.models import ToolsConfig
        from codeintel.config.primitives import BuildPaths, SnapshotRef
        from codeintel.ingestion.core.execution_context import IngestExecutionContext
        from codeintel.ingestion.infrastructure_utilities.source_scanner import (
            default_code_profile,
            default_config_profile,
        )
        from codeintel.ingestion.plugins.protocol import IngestRuntimeScratch
        from codeintel.ingestion.resources.registry import ResourceRegistry

        if self._gateway is None:
            message = "Gateway must be set before building context"
            raise ValueError(message)

        repo_root = self._repo_root or Path.cwd()
        build_dir = self._build_dir or repo_root / "build"

        snapshot = SnapshotRef(
            repo=self._repo,
            commit=self._commit,
            repo_root=repo_root,
        )

        paths = BuildPaths(
            build_dir=build_dir,
            artifacts_dir=build_dir / "artifacts",
            coverage_json=build_dir / "coverage.json",
            pytest_report=build_dir / "pytest-report.json",
        )

        # Use provided tools config or create default
        tools = self._tools_config or ToolsConfig()

        # Use provided profiles or defaults
        code_profile = self._code_profile or default_code_profile(repo_root)
        config_profile = self._config_profile or default_config_profile(repo_root)

        # Build scratch with pre-populated data
        scratch = IngestRuntimeScratch()
        for key, value in self._scratch_data.items():
            scratch.declare(key, value)

        # Use provided resources or create empty registry
        resources = self._resources if self._resources is not None else ResourceRegistry()

        return IngestExecutionContext(
            gateway=self._gateway,
            snapshot=snapshot,
            paths=paths,
            tools=tools,
            code_profile=code_profile,
            config_profile=config_profile,
            resources=resources,
            scratch=scratch,
            configs=dict(self._configs),
            plugin_name=self._plugin.metadata.name,
            run_id=self._run_id,
        )

    def validate(self) -> Any:
        """Run input validation.

        Returns
        -------
        ValidationResult
            Validation result.
        """
        ctx = self.build_context()
        return self._plugin.validate_inputs(ctx)

    def execute(self) -> IngestPluginResult:
        """Execute the plugin.

        Returns
        -------
        IngestPluginResult
            Execution result.
        """
        ctx = self.build_context()
        return self._plugin.execute(ctx)

    def execute_with_context(self) -> tuple[IngestPluginResult, IngestExecutionContext]:
        """Execute and return both result and context.

        Useful when you need to inspect scratch store after execution.

        Returns
        -------
        tuple[IngestPluginResult, IngestExecutionContext]
            Result and context.
        """
        ctx = self.build_context()
        result = self._plugin.execute(ctx)
        return result, ctx


# =============================================================================
# Assertion Helpers
# =============================================================================


@dataclass
class IngestPluginResultAssertions:
    """Fluent assertions for ingestion plugin results.

    Provide chainable assertion methods for validating plugin results.

    Example
    -------
    >>> result = harness.execute()
    >>> (
    ...     IngestPluginResultAssertions(result)
    ...     .succeeded()
    ...     .has_row_count("core.ast_nodes", min_rows=1)
    ...     .has_no_error()
    ... )
    """

    _result: IngestPluginResult
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

    def skipped(self) -> Self:
        """Assert that execution was skipped.

        Returns
        -------
        Self
            Self for chaining.

        Raises
        ------
        AssertionError
            If execution was not skipped.
        """
        msg = f"{self._message_prefix}Expected skip but got result"
        if not self._result.skipped:
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
            msg = (
                f"{self._message_prefix}Expected error containing '{containing}' "
                f"but got: {self._result.error}"
            )
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
            msg = (
                f"{self._message_prefix}Expected {table} to have at least "
                f"{min_rows} rows, got {actual}"
            )
            raise AssertionError(msg.strip())

        if max_rows is not None and actual > max_rows:
            msg = (
                f"{self._message_prefix}Expected {table} to have at most "
                f"{max_rows} rows, got {actual}"
            )
            raise AssertionError(msg.strip())

        return self


def assert_ingest_result(result: IngestPluginResult) -> IngestPluginResultAssertions:
    """Start fluent assertions on an ingest plugin result.

    Parameters
    ----------
    result
        Result to assert on.

    Returns
    -------
    IngestPluginResultAssertions
        Fluent assertion builder.

    Example
    -------
    >>> result = harness.execute()
    >>> assert_ingest_result(result).succeeded().has_row_count("core.ast_nodes", min_rows=1)
    """
    return IngestPluginResultAssertions(_result=result)


__all__ = [
    "IngestPluginResultAssertions",
    "IngestPluginTestHarness",
    "assert_ingest_result",
]
