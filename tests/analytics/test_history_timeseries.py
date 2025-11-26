"""Tests for cross-commit history_timeseries aggregation."""

from __future__ import annotations

from pathlib import Path

from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.config import ConfigBuilder
from codeintel.storage.gateway import (
    StorageConfig,
    build_snapshot_gateway_resolver,
    open_gateway,
)
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.history import SnapshotSpec, create_snapshot_db
from tests._helpers.tooling import init_git_repo_with_history


def test_history_timeseries_aggregates_functions(tmp_path: Path) -> None:
    """Aggregate history across commits for function-level metrics."""
    git_ctx = init_git_repo_with_history(tmp_path)
    commit_new, commit_old = git_ctx.commits
    repo = "demo/repo"
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

    builder = ConfigBuilder.from_snapshot(
        repo=repo,
        commit=commit_new,
        repo_root=git_ctx.repo_root,
    )
    cfg = builder.history_timeseries(
        commits=(commit_new, commit_old),
        entity_kind="function",
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
        runner=git_ctx.runner,
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
