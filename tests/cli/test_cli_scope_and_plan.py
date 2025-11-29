"""CLI scope parsing and plan output coverage."""

from __future__ import annotations

import json

import pytest

import codeintel.cli.main as cli_main


def test_cli_plan_outputs_isolation_and_scope_metadata(capsys: pytest.CaptureFixture[str]) -> None:
    """Plan output should include isolation and scope metadata fields."""
    exit_code = cli_main.main(["graph", "plugins", "--plan", "--json"])
    captured = capsys.readouterr()
    if exit_code != 0:
        message = "CLI plan command should exit successfully"
        pytest.fail(message)
    payload = json.loads(captured.out)
    if "plugin_metadata" not in payload:
        message = "Plan JSON should include plugin_metadata"
        pytest.fail(message)
    meta_any = next(iter(payload["plugin_metadata"].values()))
    for field in ("requires_isolation", "isolation_kind", "scope_aware", "supported_scopes"):
        if field not in meta_any:
            message = f"Plan metadata should include field '{field}'"
            pytest.fail(message)


def test_cli_plugins_json_includes_enriched_metadata(capsys: pytest.CaptureFixture[str]) -> None:
    """Graph plugins JSON listing should expose enriched metadata fields."""
    exit_code = cli_main.main(["graph", "plugins", "--json"])
    captured = capsys.readouterr()
    if exit_code != 0:
        message = "CLI plugins command should exit successfully"
        pytest.fail(message)
    payload = json.loads(captured.out)
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
