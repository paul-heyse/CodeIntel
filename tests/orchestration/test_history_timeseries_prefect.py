"""Prefect history_timeseries step wiring coverage."""

from __future__ import annotations

from pathlib import Path

from codeintel.pipeline.orchestration.prefect_flow import (
    HistoryTimeseriesTaskParams,
    close_gateways,
    t_history_timeseries,
)
from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.views import create_all_views
from tests._helpers.assertions import expect_equal
from tests._helpers.history import SnapshotSpec, create_snapshot_db
from tests._helpers.tooling import init_git_repo_with_history


def test_prefect_history_timeseries_step(tmp_path: Path, prefect_quiet_env: None) -> None:
    """Execute Prefect task end-to-end and verify history rows materialize."""
    _ = prefect_quiet_env  # Mark fixture as used
    git_ctx = init_git_repo_with_history(tmp_path)
    repo = "demo/repo"
    commit_new, commit_old = git_ctx.commits
    snapshot_dir = tmp_path / "snapshots"

    # Create snapshot databases for both commits
    create_snapshot_db(
        snapshot_dir,
        SnapshotSpec(
            repo=repo,
            commit=commit_new,
            goid=200,
            rel_path="foo.py",
            module="pkg.foo",
            qualname="foo",
        ),
    )
    create_snapshot_db(
        snapshot_dir,
        SnapshotSpec(
            repo=repo,
            commit=commit_old,
            goid=201,
            rel_path="foo.py",
            module="pkg.foo",
            qualname="foo",
        ),
    )

    # Use a SEPARATE output database (not one of the snapshots)
    # This matches the pattern in test_history_timeseries.py which works correctly
    output_db = tmp_path / "history_output.duckdb"

    params = HistoryTimeseriesTaskParams(
        repo_root=git_ctx.repo_root,
        repo=repo,
        commits=(commit_new, commit_old),
        history_db_dir=snapshot_dir,
        db_path=output_db,
        runner=git_ctx.runner,
    )
    t_history_timeseries.fn(params)
    # Close cached gateways from the task to avoid file handle conflicts
    close_gateways()

    gateway = open_gateway(StorageConfig(db_path=output_db, validate_schema=False))
    create_all_views(gateway.con)
    rows = gateway.con.execute(
        "SELECT commit FROM analytics.history_timeseries WHERE repo = ?",
        [repo],
    ).fetchall()
    gateway.close()
    commits = {row[0] for row in rows}
    expect_equal(commits, {commit_new, commit_old})
