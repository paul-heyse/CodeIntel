"""Manifest field coverage for run_id, scope, isolation, and telemetry."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from codeintel.analytics.graph_service_runtime import GraphPluginRunOptions
from codeintel.analytics.graphs.contracts import PluginContractResult
from codeintel.config.steps_graphs import GraphRunScope
from tests.analytics.conftest import PluginTestHarness, make_isolation_plugin


def test_manifest_includes_run_id_scope_and_isolation(
    plugin_harness: PluginTestHarness, tmp_path: Path
) -> None:
    """Manifest should capture run_id, scope, isolation flags, row_counts, and contracts."""
    plugin = make_isolation_plugin(name="manifest_iso_plugin")
    plugin = type(plugin)(**{**plugin.__dict__, "contract_checkers": (_contract_pass,)})
    plugin_harness.register(plugin)
    scope = GraphRunScope(
        paths=("src/demo.py",),
        modules=("pkg.demo",),
        time_window=(datetime.now(tz=UTC), datetime.now(tz=UTC) + timedelta(hours=1)),
    )
    manifest_path = tmp_path / "manifest-iso.json"
    report = plugin_harness.service.run_plugins(
        (plugin.name,),
        cfg=plugin_harness.cfg,
        run_options=GraphPluginRunOptions(scope=scope, manifest_path=manifest_path),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = payload.get("run_id")
    if not run_id or not isinstance(run_id, str):
        message = "Manifest should include top-level run_id"
        pytest.fail(message)
    scope_payload = payload.get("scope")
    if scope_payload is None or scope_payload.get("paths") != list(scope.paths):
        message = "Manifest should include scope paths"
        pytest.fail(message)
    record = payload["records"][0]
    if record.get("requires_isolation") is not True:
        message = "Manifest record should reflect isolation requirement"
        pytest.fail(message)
    if record.get("row_counts") is None:
        message = "Manifest record should include row_counts"
        pytest.fail(message)
    if record.get("contracts") is None:
        message = "Manifest record should include contracts list"
        pytest.fail(message)
    if record.get("run_id") != run_id:
        message = "Record run_id should match manifest run_id"
        pytest.fail(message)
    report_record = report.records[0]
    if report_record.run_id != run_id:
        message = "Report run_id should match manifest run_id"
        pytest.fail(message)


def _contract_pass(_ctx: object) -> object:
    return PluginContractResult(name="iso_manifest_contract", status="passed")
