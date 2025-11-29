"""Subsystem agreement plugin coverage."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeintel.analytics.graphs.plugins import (
    GraphMetricExecutionContext,
    GraphMetricPlugin,
    GraphPluginResult,
)
from codeintel.analytics.graphs.subsystem_agreement import compute_subsystem_agreement
from tests.analytics.conftest import PluginTestHarness


def test_subsystem_agreement_plugin_matches_legacy(tmp_path: Path) -> None:
    """
    Ensure subsystem agreement plugin matches legacy shim outputs.

    Raises
    ------
    AssertionError
        If plugin-generated rows differ from legacy computation.
    """
    harness = PluginTestHarness(tmp_path)
    try:
        con = harness.gateway.con
        con.execute(
            """
            INSERT INTO analytics.subsystem_modules (repo, commit, module, subsystem_id)
            VALUES ('demo/repo', 'deadbeef', 'pkg.mod', 1)
            """
        )
        con.execute(
            """
            INSERT INTO analytics.graph_metrics_modules_ext (
                repo, commit, module, import_community_id, created_at
            ) VALUES ('demo/repo', 'deadbeef', 'pkg.mod', 1, ?)
            """,
            [datetime.now(tz=UTC)],
        )
        def _run(ctx: GraphMetricExecutionContext) -> GraphPluginResult:
            compute_subsystem_agreement(ctx.gateway, repo=ctx.repo, commit=ctx.commit)
            row = con.execute(
                """
                SELECT COUNT(*) FROM analytics.subsystem_agreement
                WHERE repo = ? AND commit = ?
                """,
                [ctx.repo, ctx.commit],
            ).fetchone()
            if row is None or row[0] is None:
                message = "Expected subsystem_agreement rows to be present"
                pytest.fail(message)
            return GraphPluginResult(
                row_counts={"analytics.subsystem_agreement": int(row[0])},
            )

        plugin = GraphMetricPlugin(
            name="subsystem_agreement_test",
            description="test subsystem agreement plugin",
            stage="subsystem",
            enabled_by_default=False,
            run=_run,
            row_count_tables=("analytics.subsystem_agreement",),
        )
        harness.register(plugin)
        # Run plugin
        harness.service.run_plugins((plugin.name,), cfg=harness.cfg)
        plugin_rows = con.execute(
            """
            SELECT subsystem_id, import_community_id, agrees
            FROM analytics.subsystem_agreement
            WHERE repo = 'demo/repo' AND commit = 'deadbeef'
            """
        ).fetchall()
        con.execute(
            "DELETE FROM analytics.subsystem_agreement WHERE repo = ? AND commit = ?",
            ["demo/repo", "deadbeef"],
        )
        # Run legacy function for comparison
        compute_subsystem_agreement(harness.gateway, repo="demo/repo", commit="deadbeef")
        legacy_rows = con.execute(
            """
            SELECT subsystem_id, import_community_id, agrees
            FROM analytics.subsystem_agreement
            WHERE repo = 'demo/repo' AND commit = 'deadbeef'
            """
        ).fetchall()
        if plugin_rows != legacy_rows:
            message = "Plugin-generated subsystem agreement should match legacy shim"
            raise AssertionError(message)
    finally:
        harness.cleanup()
