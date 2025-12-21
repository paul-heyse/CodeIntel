"""Tests for build providers wiring."""

from __future__ import annotations

from codeintel.build.analytics_resources import AnalyticsResourceRegistryProvider
from codeintel.build.providers import create_default_providers
from codeintel.config.models import ToolsConfig
from tests._helpers.assertions import expect_equal, expect_true


def test_create_default_providers_wires_tooling() -> None:
    """Ensure ToolRunner and ToolService share the configured ToolsConfig."""
    tools_config = ToolsConfig.default()
    providers = create_default_providers(tools_config)

    expect_equal(providers.tool_runner.tools_config, tools_config, label="tools_config")
    expect_true(
        providers.tool_service.runner is providers.tool_runner,
        message="ToolService should reuse ToolRunner",
    )
    expect_true(
        providers.tool_service.tools_config is tools_config,
        message="ToolService should use the provided ToolsConfig",
    )


def test_create_default_providers_registers_resources() -> None:
    """Ensure resources registry provider is initialized."""
    providers = create_default_providers(ToolsConfig.default())
    expect_true(
        isinstance(providers.resources, AnalyticsResourceRegistryProvider),
        message="Providers.resources should be an AnalyticsResourceRegistryProvider",
    )
