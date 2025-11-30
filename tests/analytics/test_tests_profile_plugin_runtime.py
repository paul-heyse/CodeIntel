from __future__ import annotations

from pathlib import Path

from codeintel.analytics.plugin_runtime import (
    plan_analytics_plugin_run,
    run_analytics_plugins,
)
from codeintel.analytics.tests.plugins import TEST_PROFILE_PLUGIN
from codeintel.config import ConfigBuilder
from codeintel.config.steps_graphs import GraphPluginPolicy, GraphRunScope
from tests._helpers.fixtures import provisioned_gateway


def test_tests_profile_plugin_runtime(tmp_path: Path) -> None:
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
            plugin_names=(TEST_PROFILE_PLUGIN.name,),
            policy=policy,
            repo=cfg.repo,
            commit=cfg.commit,
            scope=scope,
            prior_manifest={},
            cfg_options={},
            runtime_options={},
            run_id="test-profile-run",
        )

        report = run_analytics_plugins(
            plan=plan,
            gateway=ctx.gateway,
            analytics_context=None,
            graph_runtime=None,
            cfgs={"test_profile": cfg},
            extra={},
        )

        assert len(report.records) == 1
        rec = report.records[0]
        assert rec.name == TEST_PROFILE_PLUGIN.name
        assert rec.status == "succeeded"
        summary = rec.meta.get("result")
        assert isinstance(summary, dict)
        assert summary["profile_rows"] >= 0
