"""Base plugin contracts and registry for ingestion tooling.

This module defines the plugin protocol for external tool execution along with
the registry for managing tool plugins. It is the canonical location for
tool-related types including ``ToolStatus``.

Architecture Note
-----------------
Tool plugins wrap external CLI tools (pyright, ruff, coverage, scip, pytest)
and return ``ToolPluginResult`` objects. These results contain a ``parsed``
field holding a rich domain object (DiagnosticReport, CoverageReport, etc.)
from ``tools/results.py``.

The ``ToolService`` facade orchestrates tool plugins and extracts the data
needed by ingestion steps. The ``ToolRunnerAdapter`` then converts these
rich results into simpler port interface types for clean architectural
boundaries.

Integration with Core
---------------------
This module integrates with the core plugin system:
- ``ToolPlugin`` protocol is compatible with ``AsyncPluginProtocol``
- ``ToolPluginMetadata`` can be converted to core ``PluginMetadata``
- Use ``to_core_metadata()`` for unified plugin introspection

See Also
--------
codeintel.core.plugins.types : Unified plugin types
codeintel.ingestion.engine.results : Rich parsed result types
codeintel.ingestion.tool_service : Facade for tool orchestration
codeintel.ingestion.ports.tools : Port interface types
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

# Core plugin types for interoperability
from codeintel.core.plugins.types import (
    AsyncPluginProtocol,
)
from codeintel.core.plugins.types import (
    PluginMetadata as CorePluginMetadata,
)
from codeintel.ingestion.engine.infrastructure import (
    ToolNotFoundError,
)
from codeintel.ingestion.engine.results import DiagnosticReport
from codeintel.ingestion.engine.status import ToolStatus

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping
    from pathlib import Path

    from codeintel.config.models import ToolsConfig
    from codeintel.ingestion.engine.infrastructure import (
        ToolName,
        ToolRunner,
        ToolRunResult,
    )
    from codeintel.ingestion.engine.results import ParsedToolResult

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolPluginResult:
    """High-level result from a tool plugin execution.

    Attributes
    ----------
    tool
        Logical tool identifier (from ToolName).
    status
        Normalized status code describing the outcome.
    artifacts
        Logical name -> on-disk artifact path (e.g., "json_report" -> Path).
    run
        Underlying ToolRunResult when a subprocess ran, otherwise None.
    error
        Exception captured by the plugin, if any.
    parsed
        Parsed domain object (DiagnosticReport, CoverageReport, etc.).
    """

    tool: ToolName
    status: ToolStatus
    artifacts: Mapping[str, Path]
    run: ToolRunResult | None
    error: Exception | None = None
    parsed: ParsedToolResult | None = None

    @property
    def ok(self) -> bool:
        """Return True when the plugin completed successfully."""
        return self.status is ToolStatus.OK


@dataclass(frozen=True)
class ToolPluginMetadata:
    """Declarative metadata for a tool plugin.

    This class provides tool-specific metadata that can be converted to
    core ``PluginMetadata`` for unified plugin introspection.

    Attributes
    ----------
    name
        Registry name (e.g., "pyright").
    produces_artifacts
        Logical artifact names exposed by this plugin.
    consumes_configs
        ToolConfig fields this plugin depends on (e.g., "pyright_bin").
    datasets
        Datasets (table keys) that conceptually rely on this tool.
    tool_binary
        Optional explicit binary name for the tool.
    description
        Human-readable description of the plugin.
    """

    name: str
    produces_artifacts: tuple[str, ...]
    consumes_configs: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    tool_binary: str | None = None
    description: str | None = None

    def to_core_metadata(self) -> CorePluginMetadata:
        """Convert to core PluginMetadata for unified introspection.

        Returns
        -------
        CorePluginMetadata
            Core plugin metadata with tool-specific fields populated.
        """
        return CorePluginMetadata(
            name=self.name,
            description=self.description or f"Tool plugin: {self.name}",
            kind="tool",
            stage="pipeline_ingestion",
            tool_binary=self.tool_binary,
            produces_artifacts=self.produces_artifacts,
            consumes_configs=self.consumes_configs,
            produces_tables=self.datasets,
        )


@dataclass(frozen=True)
class ToolDependencies:
    """Declare configuration and dataset dependencies for a tool plugin.

    Attributes
    ----------
    consumes_configs
        ToolConfig fields this tool depends on (e.g., "pyright_bin").
    datasets
        Dataset table keys this tool conceptually supports.
    """

    consumes_configs: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()


def tool_metadata(
    name: str,
    produces_artifacts: tuple[str, ...],
    *,
    dependencies: ToolDependencies | None = None,
    tool_binary: str | None = None,
    description: str | None = None,
) -> ToolPluginMetadata:
    """Create tool plugin metadata with sensible defaults.

    Parameters
    ----------
    name
        Registry name for the plugin.
    produces_artifacts
        Logical artifact names exposed by this plugin.
    dependencies
        Optional dependency metadata for tool configuration keys and datasets.
    tool_binary
        Optional explicit binary name for the tool.
    description
        Optional human-readable description of the plugin.

    Returns
    -------
    ToolPluginMetadata
        Configured metadata instance.

    Examples
    --------
    >>> meta = tool_metadata(
    ...     "pyright",
    ...     ("pyright_json",),
    ...     dependencies=ToolDependencies(consumes_configs=("pyright_bin",)),
    ... )
    >>> meta.name
    'pyright'
    """
    active_dependencies = dependencies or ToolDependencies()

    return ToolPluginMetadata(
        name=name,
        produces_artifacts=produces_artifacts,
        consumes_configs=active_dependencies.consumes_configs,
        datasets=active_dependencies.datasets,
        tool_binary=tool_binary,
        description=description,
    )


@runtime_checkable
class ToolPlugin(Protocol):
    """Protocol implemented by all tool plugins.

    This protocol is compatible with the core ``AsyncPluginProtocol`` and
    can be used interchangeably where async execution is expected.

    Notes
    -----
    Tool plugins have their own metadata type (``ToolPluginMetadata``) but
    can be converted to core metadata via ``metadata.to_core_metadata()``.
    """

    metadata: ToolPluginMetadata
    runner: ToolRunner
    tools_config: ToolsConfig

    async def run(self, *, repo_root: Path, **kwargs: object) -> ToolPluginResult:
        """Execute the tool for the given repository root.

        Parameters
        ----------
        repo_root
            Repository root path for tool execution.
        **kwargs
            Tool-specific arguments (e.g., coverage_file, json_output_path).

        Returns
        -------
        ToolPluginResult
            Result containing status, artifacts, and parsed output.
        """
        ...


@dataclass
class DiagnosticToolPlugin:
    """Base class for diagnostic tool plugins (pyright, pyrefly, ruff).

    Provides standardized NOT_FOUND handling for diagnostic tools. Subclasses
    must define ``tool_name`` as a class variable and implement the ``run``
    method.

    Attributes
    ----------
    runner
        Shared ToolRunner for subprocess execution.
    tools_config
        Effective tool configuration.
    metadata
        Plugin metadata for registry integration.
    tool_name
        Class variable specifying the ToolName for this plugin.
    """

    tool_name: ClassVar[ToolName]
    runner: ToolRunner
    tools_config: ToolsConfig
    metadata: ToolPluginMetadata

    def _not_found_result(self) -> ToolPluginResult:
        """Return standard NOT_FOUND response for this tool.

        Returns
        -------
        ToolPluginResult
            Result with NOT_FOUND status and empty diagnostics.
        """
        return ToolPluginResult(
            tool=self.tool_name,
            status=ToolStatus.NOT_FOUND,
            artifacts={},
            run=None,
            error=ToolNotFoundError(self.tool_name, configured_path="(not found)"),
            parsed=DiagnosticReport.empty(self.tool_name.value),
        )


@dataclass
class ToolPluginRegistry:
    """Registry of tool plugins keyed by logical name.

    Parameters
    ----------
    runner
        Shared ToolRunner used by plugins.
    tools_config
        Effective ToolsConfig configuration.
    plugins
        Initial mapping of name -> plugin instance (optional).
    """

    runner: ToolRunner
    tools_config: ToolsConfig
    _plugins: MutableMapping[str, ToolPlugin] = field(default_factory=dict)

    def register(self, plugin: ToolPlugin) -> None:
        """Register or overwrite a plugin.

        Parameters
        ----------
        plugin
            Plugin instance to register.
        """
        name = plugin.metadata.name
        self._plugins[name] = plugin
        log.debug("Registered tool plugin %s", name)

    def get(self, name: str) -> ToolPlugin:
        """Return a plugin by name or raise KeyError.

        Parameters
        ----------
        name
            Plugin registry name.

        Returns
        -------
        ToolPlugin
            Registered plugin instance.

        Raises
        ------
        KeyError
            If no plugin exists for the given name.
        """
        try:
            return self._plugins[name]
        except KeyError as exc:
            message = f"Unknown tool plugin: {name!r}"
            raise KeyError(message) from exc

    def names(self) -> tuple[str, ...]:
        """Return all registered plugin names.

        Returns
        -------
        tuple[str, ...]
            Plugin names in registration order.
        """
        return tuple(self._plugins.keys())

    def items(self) -> Mapping[str, ToolPlugin]:
        """Return an immutable view of registered plugins.

        Returns
        -------
        Mapping[str, ToolPlugin]
            Copy of the registry contents.
        """
        return dict(self._plugins)

    def get_core_metadata(self, name: str) -> CorePluginMetadata:
        """Return core-compatible metadata for a plugin.

        Parameters
        ----------
        name
            Plugin registry name.

        Returns
        -------
        CorePluginMetadata
            Core plugin metadata for unified introspection.
        """
        plugin = self.get(name)
        return plugin.metadata.to_core_metadata()

    def all_core_metadata(self) -> Mapping[str, CorePluginMetadata]:
        """Return core-compatible metadata for all plugins.

        Returns
        -------
        Mapping[str, CorePluginMetadata]
            Mapping of plugin names to core metadata.
        """
        return {name: p.metadata.to_core_metadata() for name, p in self._plugins.items()}


def build_default_registry(runner: ToolRunner, tools_config: ToolsConfig) -> ToolPluginRegistry:
    """Construct a registry with all built-in tool plugins.

    Parameters
    ----------
    runner
        Shared ToolRunner for plugin invocations.
    tools_config
        Effective tool configuration.

    Returns
    -------
    ToolPluginRegistry
        Registry populated with built-in plugins.
    """
    pyright_plugin = import_module("codeintel.ingestion.engine.pyright").PyrightPlugin
    pyrefly_plugin = import_module("codeintel.ingestion.engine.pyrefly").PyreflyPlugin
    ruff_plugin = import_module("codeintel.ingestion.engine.ruff").RuffPlugin
    coverage_plugin = import_module("codeintel.ingestion.engine.coverage").CoveragePlugin
    pytest_plugin = import_module("codeintel.ingestion.engine.pytest").PytestPlugin
    scip_plugin = import_module("codeintel.ingestion.engine.scip").ScipPlugin

    registry = ToolPluginRegistry(runner=runner, tools_config=tools_config)

    registry.register(pyright_plugin(runner=runner, tools_config=tools_config))
    registry.register(pyrefly_plugin(runner=runner, tools_config=tools_config))
    registry.register(ruff_plugin(runner=runner, tools_config=tools_config))
    registry.register(coverage_plugin(runner=runner, tools_config=tools_config))
    registry.register(pytest_plugin(runner=runner, tools_config=tools_config))
    registry.register(scip_plugin(runner=runner, tools_config=tools_config))

    return registry


__all__ = [
    "AsyncPluginProtocol",
    "CorePluginMetadata",
    "DiagnosticToolPlugin",
    "ToolPlugin",
    "ToolPluginMetadata",
    "ToolPluginRegistry",
    "ToolPluginResult",
    "ToolStatus",
    "build_default_registry",
    "tool_metadata",
]
