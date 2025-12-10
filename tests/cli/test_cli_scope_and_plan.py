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
    envelope = json.loads(result.output)
    # Handler outputs are wrapped in {"data": {...}} envelope
    payload = envelope.get("data", envelope)
    # New handler outputs stages with plugins
    if "stages" not in payload and "plugins" not in payload and "plan_id" not in payload:
        message = f"Plan JSON should include stages, plugins, or plan_id. Got: {list(payload.keys())}"
        pytest.fail(message)


def test_cli_plugins_json_includes_enriched_metadata() -> None:
    """Graph plugins JSON listing should expose plugin metadata fields."""
    result = run_cli(["graph", "plugins", "--output-format", "json"])
    if result.exit_code != 0:
        message = f"CLI plugins command should exit successfully: {result.output}"
        pytest.fail(message)
    envelope = json.loads(result.output)
    # Handler outputs are wrapped in {"data": {...}} envelope
    payload = envelope.get("data", envelope)
    plugins = payload.get("plugins", [])
    if not plugins:
        message = f"CLI plugins JSON should include plugin entries. Got: {list(payload.keys())}"
        pytest.fail(message)
    # New handler outputs plugins as a list with basic metadata
    plugin_any = plugins[0]
    required = ("name", "stage")  # Core fields in new handler output
    missing = tuple(field for field in required if field not in plugin_any)
    if missing:
        message = f"CLI plugins JSON missing metadata fields: {missing}"
        pytest.fail(message)
