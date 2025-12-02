"""Integration coverage for the graph_metrics pipeline step."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.analytics.context import AnalyticsContextConfig, build_analytics_context
from codeintel.config import BuildPaths, SnapshotRef
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import GraphBackendConfig
from codeintel.graphs.recipes import METRICS_ONLY_RECIPE, RecipeExecutor, RecipeExecutorContext
from codeintel.ingestion.infrastructure_utilities.source_scanner import ScanProfile
from codeintel.pipeline.orchestration.core import ensure_graph_runtime
from codeintel.pipeline.orchestration.steps import PipelineContext
from tests._helpers.architecture import open_seeded_architecture_gateway


def _scan_profile(repo_root: Path) -> ScanProfile:
    return ScanProfile(
        repo_root=repo_root,
        source_roots=(repo_root,),
        include_globs=("*",),
    )


@pytest.mark.integration
def test_graph_metrics_step_runs_plugins(tmp_path: Path) -> None:
    """GraphMetricsStep should execute the plugin pipeline.

    Note: Uses RecipeExecutor directly with force_sequential=True to avoid
    thread-safety issues with shared in-memory DuckDB connections.
    """
    repo = "demo/repo"
    commit = "deadbeef"
    gateway = open_seeded_architecture_gateway(repo=repo, commit=commit)
    build_dir = tmp_path / "build"
    paths = BuildPaths.from_layout(
        repo_root=tmp_path,
        build_dir=build_dir,
        db_path=gateway.config.db_path,
    )
    snapshot = SnapshotRef(repo_root=tmp_path, repo=repo, commit=commit)
    ctx = PipelineContext(
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=ToolsConfig.default(),
        code_profile_cfg=_scan_profile(tmp_path),
        config_profile_cfg=_scan_profile(tmp_path),
        graph_backend_cfg=GraphBackendConfig(),
    )

    # Get runtime context
    acx = build_analytics_context(
        gateway,
        AnalyticsContextConfig(
            repo=repo,
            commit=commit,
            repo_root=tmp_path,
        ),
    )
    runtime = ensure_graph_runtime(ctx, acx=acx)

    # Execute with force_sequential to avoid thread-safety issues
    executor_ctx = RecipeExecutorContext(
        gateway=gateway,
        snapshot=ctx.snapshot,
        engine=runtime.engine,
        catalog_provider=acx.catalog,
        force_sequential=True,
    )
    executor = RecipeExecutor(executor_ctx)
    result = executor.execute(METRICS_ONLY_RECIPE)

    if not result.success:
        pytest.fail(f"Recipe execution failed: {result.failure_count} failures")

    con = gateway.con
    count_row = con.execute(
        "SELECT COUNT(*) FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?",
        [repo, commit],
    ).fetchone()
    if count_row is None:
        pytest.fail("graph_metrics_functions returned no rows")
    if int(count_row[0]) <= 0:
        pytest.fail("graph_metrics_functions should contain computed metrics")

    stats_row = con.execute(
        "SELECT COUNT(*) FROM analytics.graph_stats WHERE repo = ? AND commit = ?",
        [repo, commit],
    ).fetchone()
    if stats_row is None:
        pytest.fail("graph_stats returned no rows")
    if int(stats_row[0]) <= 0:
        pytest.fail("graph_stats should contain computed metrics")
