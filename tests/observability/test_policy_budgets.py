"""Observability policy budget tests."""

from __future__ import annotations

from codeintel.core.config.settings import ObservabilitySettings
from codeintel.observability.policy import policy_from_settings

CLI_ARG_NAMES_MAX = 3
HTTP_ROUTE_MAX_LEN = 7
MCP_TOOL_NAME_MAX_LEN = 9


def test_policy_from_settings_applies_budget_limits() -> None:
    """Policy should reflect configured budget limits."""
    settings = ObservabilitySettings(
        cli_arg_names_max=CLI_ARG_NAMES_MAX,
        http_route_max_len=HTTP_ROUTE_MAX_LEN,
        mcp_tool_name_max_len=MCP_TOOL_NAME_MAX_LEN,
    )
    policy = policy_from_settings(settings)
    assert policy.cli_arg_names_max == CLI_ARG_NAMES_MAX
    assert policy.http_route_max_len == HTTP_ROUTE_MAX_LEN
    assert policy.mcp_tool_name_max_len == MCP_TOOL_NAME_MAX_LEN


def test_operation_allowlist_override_precedence() -> None:
    """Specific overrides should win over component-level defaults."""
    settings = ObservabilitySettings(
        operation_attribute_allowlist_overrides=(
            ("cli", ("http.method",)),
            ("cli.health", ("codeintel.output_format",)),
        )
    )
    policy = policy_from_settings(settings)
    assert policy.operation_allowlist_for("cli", "health") == frozenset({"codeintel.output_format"})
    assert policy.operation_allowlist_for("cli", "status") == frozenset({"http.method"})
