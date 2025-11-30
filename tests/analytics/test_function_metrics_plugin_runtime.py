"""Runtime test for the function metrics analytics plugin harness integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.analytics.functions.plugins import FUNCTION_METRICS_PLUGIN
from codeintel.analytics.plugin_runtime import (
    AnalyticsPlanRequest,
    AnalyticsRunContext,
    plan_analytics_plugin_run,
    run_analytics_plugins,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_analytics import FunctionAnalyticsStepConfig
from codeintel.config.steps_graphs import GraphPluginPolicy, GraphRunScope
from codeintel.storage.gateway import StorageGateway, open_memory_gateway
from tests._helpers.builders import GoidRow, insert_goids


def _seed_function(
    gateway: StorageGateway,
    repo_root: Path,
    repo: str,
    commit: str,
) -> None:
    rel_path = "mod.py"
    file_path = repo_root / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("def foo(x: int) -> int:\n    return x + 1\n", encoding="utf-8")
    insert_goids(
        gateway,
        [
            GoidRow(
                goid_h128=1,
                urn="urn:demo:foo",
                repo=repo,
                commit=commit,
                rel_path=rel_path,
                kind="function",
                qualname="mod.foo",
                start_line=1,
                end_line=2,
            )
        ],
    )


def test_function_metrics_plugin_executes(tmp_path: Path) -> None:
    """Execute the function metrics plugin end-to-end through the analytics harness."""
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    repo = "demo/repo"
    commit = "abc123"
    _seed_function(gateway, tmp_path, repo=repo, commit=commit)

    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=tmp_path)
    cfg = FunctionAnalyticsStepConfig(snapshot=snapshot)
    policy = GraphPluginPolicy()
    scope = GraphRunScope()

    plan = plan_analytics_plugin_run(
        AnalyticsPlanRequest(
            plugin_names=(FUNCTION_METRICS_PLUGIN.name,),
            policy=policy,
            repo=cfg.repo,
            commit=cfg.commit,
            scope=scope,
            prior_manifest={},
            cfg_options={},
            runtime_options={},
            run_id="test-run",
        )
    )

    report = run_analytics_plugins(
        plan=plan,
        run_context=AnalyticsRunContext(
            gateway=gateway,
            analytics_context=None,
            graph_runtime=None,
            cfgs={"function": cfg},
            extra={},
            catalog_provider=None,
        ),
    )

    if len(report.records) != 1:
        msg = "Expected a single run record for the metrics plugin."
        pytest.fail(msg)
    rec = report.records[0]
    if rec.name != FUNCTION_METRICS_PLUGIN.name:
        msg = "Unexpected plugin name in run record."
        pytest.fail(msg)
    if rec.status != "succeeded":
        msg = f"Plugin did not succeed: {rec.status}"
        pytest.fail(msg)
    summary = rec.meta.get("result")
    if not isinstance(summary, dict):
        msg = "Expected metrics summary dictionary."
        pytest.fail(msg)
    if summary.get("metrics_rows", 0) < 1:
        msg = "Function metrics did not emit rows."
        pytest.fail(msg)
    if summary.get("types_rows", 0) < 1:
        msg = "Function types did not emit rows."
        pytest.fail(msg)
