"""Tests for cross-commit history_timeseries aggregation."""

from __future__ import annotations

from pathlib import Path

from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.config.models import HistoryTimeseriesConfig
from codeintel.storage.gateway import (
    StorageConfig,
    build_snapshot_gateway_resolver,
    open_gateway,
)
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.fakes import FakeToolRunner
from tests._helpers.history import SnapshotSpec, create_snapshot_db


def test_history_timeseries_aggregates_functions(tmp_path: Path) -> None:
    """Aggregate history across commits for function-level metrics."""
    repo = "demo/repo"
    commit_new = "c2"
    commit_old = "c1"
    snapshot_dir = tmp_path / "snapshots"
    create_snapshot_db(
        snapshot_dir,
        SnapshotSpec(
            repo=repo,
            commit=commit_new,
            goid=10,
            rel_path="foo.py",
            module="pkg.foo",
            qualname="foo",
            risk_score=0.9,
            coverage_ratio=0.8,
        ),
    )
    create_snapshot_db(
        snapshot_dir,
        SnapshotSpec(
            repo=repo,
            commit=commit_old,
            goid=11,
            rel_path="foo.py",
            module="pkg.foo",
            qualname="foo",
            risk_score=0.2,
            coverage_ratio=0.6,
        ),
    )

    output_db = tmp_path / "history.duckdb"
    gateway = open_gateway(
        StorageConfig(
            db_path=output_db,
            apply_schema=True,
            ensure_views=True,
            validate_schema=True,
            read_only=False,
        )
    )

    runner = FakeToolRunner(
        cache_dir=tmp_path / ".tool_cache",
        payloads={"git": "2024-01-01T00:00:00+00:00"},
    )
    overrides = HistoryTimeseriesConfig.Overrides(entity_kind="function")
    cfg = HistoryTimeseriesConfig.from_args(
        repo=repo,
        repo_root=tmp_path,
        commits=(commit_new, commit_old),
        overrides=overrides,
    )
    snapshot_resolver = build_snapshot_gateway_resolver(
        db_dir=snapshot_dir,
        repo=repo,
        primary_gateway=gateway,
    )
    compute_history_timeseries_gateways(
        gateway,
        cfg,
        snapshot_resolver,
        runner=runner,
    )

    rows = gateway.con.execute(
        "SELECT entity_kind, commit, risk_score, entity_stable_id FROM analytics.history_timeseries"
    ).fetchall()
    expect_equal(len(rows), 2)
    kinds = {row[0] for row in rows}
    commits = {row[1] for row in rows}
    expect_equal(kinds, {"function"})
    expect_equal(commits, {commit_new, commit_old})
    stable_ids = {row[3] for row in rows}
    expect_true(len(stable_ids) == 1, "Stable ID should be consistent across commits.")
