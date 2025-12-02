"""Runtime test for the tests.profile analytics plugin."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.analytics.core.pipeline_bridge import (
    AnalyticsPlanRequest,
    AnalyticsRunContext,
    plan_analytics_plugin_run,
    run_analytics_plugins,
)
from codeintel.analytics.core.plugins import TEST_PROFILE_PLUGIN
from codeintel.config import ConfigBuilder
from codeintel.config.steps_graphs import GraphPluginPolicy, GraphRunScope
from tests._helpers.fixtures import provisioned_gateway


def test_tests_profile_plugin_runtime(tmp_path: Path) -> None:
    """Execute the test profile plugin through the analytics harness."""
    with provisioned_gateway(tmp_path) as ctx:
        builder = ConfigBuilder.from_snapshot(
            ctx.repo,
            ctx.commit,
            ctx.repo_root,
            build_dir=ctx.build_dir,
            db_path=ctx.db_path,
            document_output_dir=ctx.document_output_dir,
        )
        cfg = builder.test_profile()
        policy = GraphPluginPolicy()
        scope = GraphRunScope()

        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(TEST_PROFILE_PLUGIN.metadata.name,),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest={},
                cfg_options={},
                runtime_options={},
                run_id="test-profile-run",
            )
        )

        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                graph_runtime=None,
                cfgs={"test_profile": cfg},
                extra={},
                catalog_provider=None,
            ),
        )

        if len(report.records) != 1:
            msg = "Expected single run record for test profile plugin."
            pytest.fail(msg)
        rec = report.records[0]
        if rec.name != TEST_PROFILE_PLUGIN.metadata.name:
            msg = "Unexpected plugin recorded."
            pytest.fail(msg)
        if rec.status != "succeeded":
            msg = f"Plugin execution failed with status {rec.status}"
            pytest.fail(msg)
        summary = rec.meta.get("result")
        if not isinstance(summary, dict):
            msg = "Expected summary metadata dictionary."
            pytest.fail(msg)
        if summary.get("profile_rows", 0) < 0:
            msg = "Profile rows count is invalid."
            pytest.fail(msg)
