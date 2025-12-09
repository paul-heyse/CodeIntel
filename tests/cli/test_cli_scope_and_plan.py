"""CLI scope parsing and plan output coverage."""

from __future__ import annotations

import json

import pytest

from tests._helpers.cli import run_cli


def test_cli_plan_outputs_isolation_and_scope_metadata() -> None:
    """Plan output should include isolation and scope metadata fields."""
    result = run_cli(["graph", "plugins", "--plan", "--json"])
    if result.exit_code != 0:
        message = f"CLI plan command should exit successfully: {result.output}"
        pytest.fail(message)
    payload = json.loads(result.output)
    if "plugin_metadata" not in payload:
        message = "Plan JSON should include plugin_metadata"
        pytest.fail(message)
    meta_any = next(iter(payload["plugin_metadata"].values()))
    for field in ("requires_isolation", "isolation_kind", "scope_aware", "supported_scopes"):
        if field not in meta_any:
            message = f"Plan metadata should include field '{field}'"
            pytest.fail(message)


def test_cli_plugins_json_includes_enriched_metadata() -> None:
    """Graph plugins JSON listing should expose enriched metadata fields."""
    result = run_cli(["graph", "plugins", "--json"])
    if result.exit_code != 0:
        message = f"CLI plugins command should exit successfully: {result.output}"
        pytest.fail(message)
    payload = json.loads(result.output)
    plugins = payload.get("plugins", {})
    if not plugins:
        message = "CLI plugins JSON should include plugin entries"
        pytest.fail(message)
    meta_any = next(iter(plugins.values()))
    required = (
        "resource_hints",
        "options_model",
        "options_default",
        "version_hash",
        "contract_checkers",
        "row_count_tables",
        "config_schema_ref",
        "depends_on",
        "provides",
        "requires",
        "scope_aware",
        "supported_scopes",
        "requires_isolation",
        "isolation_kind",
        "cache_populates",
        "cache_consumes",
    )
    missing = tuple(field for field in required if field not in meta_any)
    if missing:
        message = f"CLI plugins JSON missing metadata fields: {missing}"
        pytest.fail(message)
