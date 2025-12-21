"""Production implementations of DI protocols.

This module provides the Providers container used by BuildEnv and a factory
for constructing production-ready providers.

Example
-------
>>> from codeintel.build.providers import create_default_providers
>>> from codeintel.config.models import ToolsConfig
>>> providers = create_default_providers(ToolsConfig.default())
>>> result = await providers.tool_service.run_pyright(Path("."))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.analytics_resources import AnalyticsResourceRegistryProvider
from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.ingestion.engine.service import ToolService

if TYPE_CHECKING:
    from codeintel.config.models import ToolsConfig

__all__ = ["Providers", "create_default_providers"]


@dataclass
class Providers:
    """Container for all DI providers.

    Attributes
    ----------
    tool_runner
        Canonical tool runner for external tooling.
    tool_service
        Canonical tool service for tool plugin execution.
    resources
        Analytics resource registry factory for BuildEnv usage.
    """

    tool_runner: ToolRunner
    tool_service: ToolService
    resources: AnalyticsResourceRegistryProvider


def create_default_providers(tools_config: ToolsConfig) -> Providers:
    """Create a complete set of production providers.

    Parameters
    ----------
    tools_config
        Tool configuration with binary paths.

    Returns
    -------
    Providers
        Container with all providers wired together.
    """
    tool_runner = ToolRunner(tools_config=tools_config)
    tool_service = ToolService(tool_runner, tools_config=tools_config)
    resources = AnalyticsResourceRegistryProvider()

    return Providers(
        tool_runner=tool_runner,
        tool_service=tool_service,
        resources=resources,
    )
