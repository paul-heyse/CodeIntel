"""CLI coverage for history-timeseries command."""

from __future__ import annotations

from pathlib import Path

from codeintel.cli.main import main
from codeintel.storage.gateway import StorageConfig, open_gateway
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.history import SnapshotSpec, create_snapshot_db

EXPECTED_HISTORY_ROW_COUNT = 2


def test_history_timeseries_cli_happy_path(tmp_path: Path) -> None:
    """Run CLI end-to-end when both commits have snapshots and expect success."""
    repo = "demo/repo"
    commit_new = "c2"
    commit_old = "c1"
    snapshot_dir = tmp_path / "snapshots"
    create_snapshot_db(
        snapshot_dir,
        SnapshotSpec(
            repo=repo,
            commit=commit_new,
            goid=100,
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
            goid=101,
            rel_path="foo.py",
            module="pkg.foo",
            qualname="foo",
        ),
    )
    output_db = tmp_path / "out.duckdb"
    rc = main(
        [
            "history-timeseries",
            "--repo-root",
            str(tmp_path),
            "--repo",
            repo,
            "--commits",
            commit_new,
            commit_old,
            "--db-dir",
            str(snapshot_dir),
            "--output-db",
            str(output_db),
        ]
    )
    expect_equal(rc, 0)

    gateway = open_gateway(StorageConfig.for_readonly(output_db))
    rows = gateway.con.execute("SELECT COUNT(*) FROM analytics.history_timeseries").fetchone()
    gateway.close()
    expect_true(rows is not None and rows[0] == EXPECTED_HISTORY_ROW_COUNT)


def test_history_timeseries_cli_missing_snapshot(tmp_path: Path) -> None:
    """Return non-zero exit when requested snapshots are missing."""
    repo = "demo/repo"
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    output_db = tmp_path / "out.duckdb"
    rc = main(
        [
            "history-timeseries",
            "--repo-root",
            str(tmp_path),
            "--repo",
            repo,
            "--commits",
            "missing",
            "--db-dir",
            str(snapshot_dir),
            "--output-db",
            str(output_db),
        ]
    )
    expect_equal(rc, 1)
