"""Test harness for ingestion plugins.

This module provides a fluent test harness for ingestion plugins, inheriting
from the shared base harness to reduce code duplication.

Example
-------
>>> from tests._helpers.harnesses import IngestPluginTestHarness, assert_ingest_result
>>> from codeintel.ingestion.plugins.ast_extract import AstExtractPlugin
>>>
>>> def test_ast_extract_plugin(storage_gateway, tmp_path):
...     result = (
...         IngestPluginTestHarness.for_plugin(AstExtractPlugin())
...         .with_gateway(storage_gateway)
...         .with_snapshot("test-repo", "abc123", tmp_path)
...         .execute()
...     )
...     assert_ingest_result(result).succeeded().has_row_count("core.ast_nodes", min_rows=1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, TypeVar

from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.core.config.registry import ConfigRegistry
from codeintel.ingestion.core.base import BaseIngestPlugin, ValidationResult
from codeintel.ingestion.core.execution_context import IngestExecutionContext
from codeintel.ingestion.infrastructure.scanning import (
    ScanProfile,
    default_code_profile,
    default_config_profile,
)
from codeintel.ingestion.plugins.protocol import (
    IngestPluginResult,
    IngestRuntimeScratch,
)
from codeintel.ingestion.resources.modules import ModuleProvider
from codeintel.ingestion.resources.protocol import ResourceProvider
from codeintel.ingestion.resources.registry import ResourceRegistry
from codeintel.ingestion.resources.tools import ToolsProvider
from codeintel.ingestion.resources.tracker import TrackerProvider
from tests._helpers.harnesses.base import BaseResultAssertions, BaseTestHarness

T = TypeVar("T")


@dataclass
class IngestPluginTestHarness(BaseTestHarness[BaseIngestPlugin, IngestExecutionContext]):
    """Fluent test harness for class-based ingestion plugins.

    Provide a clean API for setting up plugin tests with minimal
    boilerplate. All resource access uses the ResourceRegistry pattern.

    Attributes
    ----------
    _plugin : BaseIngestPlugin
        The plugin being tested.
    _build_dir : Path | None
        Build directory path.
    _configs : ConfigRegistry
        Configuration registry for plugin configs.
    _resources : ResourceRegistry | None
        Resource registry for provider access.
    _code_profile : ScanProfile | None
        Code file scan profile.
    _config_profile : ScanProfile | None
        Config file scan profile.
    _tools_config : ToolsConfig | None
        Tools configuration.
    """

    _build_dir: Path | None = None
    _configs: ConfigRegistry = field(default_factory=ConfigRegistry)
    _resources: ResourceRegistry | None = None
    _code_profile: ScanProfile | None = None
    _config_profile: ScanProfile | None = None
    _tools_config: ToolsConfig | None = None

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
        self._configs.register(type(config), config)
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

        paths = BuildPaths.from_repo_root(repo_root, build_dir=build_dir)

        # Use provided tools config or create default
        tools = self._tools_config or ToolsConfig.default()

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
            configs=self._configs.copy(),
            plugin_name=self._plugin.metadata.name,
            run_id=self._run_id,
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
class IngestPluginResultAssertions(BaseResultAssertions[IngestPluginResult]):
    """Fluent assertions for ingestion plugin results.

    Extends base assertions with ingestion-specific checks.

    Example
    -------
    >>> result = harness.execute()
    >>> (
    ...     IngestPluginResultAssertions(_result=result)
    ...     .succeeded()
    ...     .has_row_count("core.ast_nodes", min_rows=1)
    ...     .has_no_error()
    ... )
    """

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
        if not self._result.skipped:
            msg = f"{self._message_prefix}Expected skip but got result"
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
