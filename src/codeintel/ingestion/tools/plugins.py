"""Base plugin contracts and registry for ingestion tooling."""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.infrastructure_utilities.tool_runner import (
    ToolName,
    ToolRunner,
    ToolRunResult,
)

if TYPE_CHECKING:
    from codeintel.ingestion.tools.results import ParsedToolResult

log = logging.getLogger(__name__)


class ToolStatus(StrEnum):
    """Normalized status for plugin invocations."""

    OK = "ok"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True)
class ToolPluginResult:
    """
    High-level result from a tool plugin execution.

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
    """
    Declarative metadata for a plugin.

    Attributes
    ----------
    name:
        Registry name (e.g., "pyright").
    produces_artifacts:
        Logical artifact names exposed by this plugin.
    consumes_configs:
        ToolConfig fields this plugin depends on (e.g., "pyright_bin").
    datasets:
        Datasets (table keys) that conceptually rely on this tool.
    """

    name: str
    produces_artifacts: tuple[str, ...]
    consumes_configs: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()


@runtime_checkable
class ToolPlugin(Protocol):
    """Protocol implemented by all tool plugins."""

    metadata: ToolPluginMetadata
    runner: ToolRunner
    tools_config: ToolsConfig

    async def run(self, *, repo_root: Path, **kwargs: object) -> ToolPluginResult:
        """
        Execute the tool for the given repository root.

        Additional keyword arguments are tool-specific (for example,
        coverage_file, json_output_path, rel_paths for sharded SCIP).
        """
        ...


@dataclass
class ToolPluginRegistry:
    """
    Registry of tool plugins keyed by logical name.

    Parameters
    ----------
    runner:
        Shared ToolRunner used by plugins.
    tools_config:
        Effective ToolsConfig configuration.
    plugins:
        Initial mapping of name -> plugin instance (optional).
    """

    runner: ToolRunner
    tools_config: ToolsConfig
    _plugins: MutableMapping[str, ToolPlugin] = field(default_factory=dict)

    def register(self, plugin: ToolPlugin) -> None:
        """
        Register or overwrite a plugin.

        Parameters
        ----------
        plugin
            Plugin instance to register.
        """
        name = plugin.metadata.name
        self._plugins[name] = plugin
        log.debug("Registered tool plugin %s", name)

    def get(self, name: str) -> ToolPlugin:
        """
        Return a plugin by name or raise KeyError.

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
        """
        Return all registered plugin names.

        Returns
        -------
        tuple[str, ...]
            Plugin names in registration order.
        """
        return tuple(self._plugins.keys())

    def items(self) -> Mapping[str, ToolPlugin]:
        """
        Return an immutable view of registered plugins.

        Returns
        -------
        Mapping[str, ToolPlugin]
            Copy of the registry contents.
        """
        return dict(self._plugins)


def build_default_registry(runner: ToolRunner, tools_config: ToolsConfig) -> ToolPluginRegistry:
    """
    Construct a registry with all built-in tool plugins.

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
    pyright_plugin = import_module("codeintel.ingestion.tools.pyright").PyrightPlugin
    pyrefly_plugin = import_module("codeintel.ingestion.tools.pyrefly").PyreflyPlugin
    ruff_plugin = import_module("codeintel.ingestion.tools.ruff").RuffPlugin
    coverage_plugin = import_module("codeintel.ingestion.tools.coverage").CoveragePlugin
    pytest_plugin = import_module("codeintel.ingestion.tools.pytest").PytestPlugin
    scip_plugin = import_module("codeintel.ingestion.tools.scip").ScipPlugin

    registry = ToolPluginRegistry(runner=runner, tools_config=tools_config)

    registry.register(pyright_plugin(runner=runner, tools_config=tools_config))
    registry.register(pyrefly_plugin(runner=runner, tools_config=tools_config))
    registry.register(ruff_plugin(runner=runner, tools_config=tools_config))
    registry.register(coverage_plugin(runner=runner, tools_config=tools_config))
    registry.register(pytest_plugin(runner=runner, tools_config=tools_config))
    registry.register(scip_plugin(runner=runner, tools_config=tools_config))

    return registry
