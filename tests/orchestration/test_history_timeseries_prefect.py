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
from tests._helpers.fakes import FakeToolRunner
from tests._helpers.history import SnapshotSpec, create_snapshot_db


def test_prefect_history_timeseries_step(tmp_path: Path) -> None:
    """Execute Prefect task end-to-end and verify history rows materialize."""
    repo = "demo/repo"
    commit_new = "c2"
    commit_old = "c1"
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

    runner = FakeToolRunner(
        cache_dir=tmp_path / ".tool_cache",
        payloads={"git": "2024-01-01T00:00:00+00:00"},
    )
    params = HistoryTimeseriesTaskParams(
        repo_root=tmp_path,
        repo=repo,
        commits=(commit_new, commit_old),
        history_db_dir=history_dir,
        db_path=db_path,
        runner=runner,
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
