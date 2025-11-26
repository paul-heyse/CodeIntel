"""Prefect history_timeseries step wiring coverage."""

from __future__ import annotations

from pathlib import Path

from codeintel.orchestration.prefect_flow import (
    HistoryTimeseriesTaskParams,
    t_history_timeseries,
)
from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.views import create_all_views
from tests._helpers.assertions import expect_equal
from tests._helpers.history import SnapshotSpec, create_snapshot_db
from tests._helpers.tooling import init_git_repo_with_history


def test_prefect_history_timeseries_step(tmp_path: Path) -> None:
    """Execute Prefect task end-to-end and verify history rows materialize."""
    git_ctx = init_git_repo_with_history(tmp_path)
    repo = "demo/repo"
    commit_new, commit_old = git_ctx.commits
    history_dir = tmp_path / "snapshots"
    db_path = history_dir / f"codeintel-{commit_new}.duckdb"

    # Current commit DB (used as gateway target)
    create_snapshot_db(
        history_dir,
        SnapshotSpec(
            repo=repo,
            commit=commit_new,
            goid=200,
            rel_path="foo.py",
            module="pkg.foo",
            qualname="foo",
        ),
    )
    # Older snapshot
    create_snapshot_db(
        history_dir,
        SnapshotSpec(
            repo=repo,
            commit=commit_old,
            goid=201,
            rel_path="foo.py",
            module="pkg.foo",
            qualname="foo",
        ),
    )

    params = HistoryTimeseriesTaskParams(
        repo_root=git_ctx.repo_root,
        repo=repo,
        commits=(commit_new, commit_old),
        history_db_dir=history_dir,
        db_path=db_path,
        runner=git_ctx.runner,
    )
    t_history_timeseries.fn(params)

    gateway = open_gateway(StorageConfig(db_path=db_path, validate_schema=False))
    create_all_views(gateway.con)
    rows = gateway.con.execute(
        "SELECT commit FROM analytics.history_timeseries WHERE repo = ?",
        [repo],
    ).fetchall()
    gateway.close()
    commits = {row[0] for row in rows}
    expect_equal(commits, {commit_new, commit_old})
