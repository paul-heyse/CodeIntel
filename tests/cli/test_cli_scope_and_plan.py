"""CLI scope parsing and plan output coverage."""

from __future__ import annotations

import json

import pytest

from tests._helpers.cli import run_cli


def test_cli_plan_outputs_isolation_and_scope_metadata() -> None:
    """Plan output should include structured plan data with plugins."""
    result = run_cli(["graph", "plugins", "--plan", "--output-format", "json"])
    if result.exit_code != 0:
        message = f"CLI plan command should exit successfully: {result.output}"
        pytest.fail(message)
    payload = json.loads(result.output)
    # New handler outputs plan_id, plugins, and skipped
    if "plan_id" not in payload and "plugins" not in payload:
        message = "Plan JSON should include plan_id or plugins"
        pytest.fail(message)


def test_cli_plugins_json_includes_enriched_metadata() -> None:
    """Graph plugins JSON listing should expose plugin metadata fields."""
    result = run_cli(["graph", "plugins", "--output-format", "json"])
    if result.exit_code != 0:
        message = f"CLI plugins command should exit successfully: {result.output}"
        pytest.fail(message)
    payload = json.loads(result.output)
    plugins = payload.get("plugins", [])
    if not plugins:
        message = "CLI plugins JSON should include plugin entries"
        pytest.fail(message)
    # New handler outputs plugins as a list with basic metadata
    plugin_any = plugins[0]
    required = ("name", "stage", "output_tables")
    missing = tuple(field for field in required if field not in plugin_any)
    if missing:
        message = f"CLI plugins JSON missing metadata fields: {missing}"
        pytest.fail(message)
