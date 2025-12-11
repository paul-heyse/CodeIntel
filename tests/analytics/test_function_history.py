"""Tests for per-function git history aggregation."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.analytics.functions import compute_function_history
from codeintel.config import SnapshotInit
from codeintel.config.datasets import get_dataset_contracts_by_table_key
from tests._helpers import TestScenario
from tests._helpers.assertions import expect_equal, expect_in, expect_true
from tests._helpers.builders import insert_rows
from tests._helpers.config_factory import function_history_cfg
from tests._helpers.orchestration.tooling import init_git_repo_with_history
from tests._helpers.rows import function_metrics_row, module_row

if TYPE_CHECKING:
    from pathlib import Path

    from tests._helpers import TestContext

# Test constants
EXPECTED_STABILITY_BUCKETS = {"new_hot", "stable", "churning", "legacy_hot"}
MIN_EXPECTED_LINES_ADDED = 2
GOID_TEST_FUNC_1 = 1
GOID_TEST_FUNC_2 = 2


def _seed_function_for_history(
    ctx: TestContext,
    *,
    goid: int,
    urn: str | None = None,
    commit: str,
) -> None:
    """Seed function metrics and module for history testing.

    Parameters
    ----------
    ctx
        Test context with gateway.
    goid
        Global object identifier.
    urn
        Optional URN to assign to the seeded function.
    commit
        Commit hash.
    """
    row = function_metrics_row(
        goid=goid,
        rel_path="pkg/foo.py",
        qualname="pkg.foo",
        snapshot=(ctx.repo, commit),
        metrics={
            "language": "python",
            "kind": "function",
            "start_line": 1,
            "end_line": 3,
            "loc": 3,
            "logical_loc": 3,
            "param_count": 0,
            "positional_params": 0,
            "has_docstring": True,
            "created_at": datetime.now(tz=UTC),
        },
    )
    if urn is not None:
        row = replace(row, urn=urn)

    insert_rows(
        ctx.gateway,
        [row],
    )
    insert_rows(
        ctx.gateway,
        [
            module_row(
                module="pkg.foo",
                path="pkg/foo.py",
                snapshot=(ctx.repo, commit),
            )
        ],
    )


def test_function_history_populates_rows(
    tmp_path: Path,
) -> None:
    """Verify compute_function_history persists metrics for touched functions."""
    # Initialize git repo with history
    git_ctx = init_git_repo_with_history(tmp_path)
    repo_root = git_ctx.repo_root
    commit = git_ctx.commits[0]

    # Create test context with specific repo root
    ctx = TestScenario.minimal().with_repo("demo/repo").with_commit(commit).build(repo_root)

    try:
        _seed_function_for_history(ctx, goid=GOID_TEST_FUNC_1, urn="urn:fn", commit=commit)

        cfg = function_history_cfg(
            SnapshotInit(repo=ctx.repo, commit=commit, repo_root=repo_root),
            min_lines_threshold=None,
        )
        compute_function_history(ctx.gateway, cfg, runner=git_ctx.runner)

        rows = ctx.con.execute("SELECT * FROM analytics.function_history").fetchall()
        expect_equal(len(rows), 1, label="Expected a single function history row.")

        contract = get_dataset_contracts_by_table_key()["analytics.function_history"]
        columns = contract.schema.column_names() if contract.schema else []
        result = dict(zip(columns, rows[0], strict=True))

        expected_commit_count = len(git_ctx.commits)
        expect_equal(result["commit_count"], expected_commit_count)
        expect_equal(result["author_count"], 1)
        expect_true(
            result["lines_added"] >= MIN_EXPECTED_LINES_ADDED,
            message="Expected lines_added to reflect git history",
        )
        expect_in(result["stability_bucket"], EXPECTED_STABILITY_BUCKETS)
        expect_equal(result["function_goid_h128"], GOID_TEST_FUNC_1)
        expect_equal(result["rel_path"], "pkg/foo.py")
        expect_equal(result["module"], "pkg.foo")

    finally:
        ctx.close()


def test_function_history_respects_min_threshold(
    tmp_path: Path,
) -> None:
    """Verify minimum line threshold is respected when populating function history."""
    # Initialize git repo with history
    git_ctx = init_git_repo_with_history(tmp_path)
    repo_root = git_ctx.repo_root
    commit = git_ctx.commits[0]

    # Create test context with specific repo root
    ctx = TestScenario.minimal().with_repo("demo/repo").with_commit(commit).build(repo_root)

    try:
        _seed_function_for_history(ctx, goid=GOID_TEST_FUNC_2, urn="urn:fn2", commit=commit)

        cfg = function_history_cfg(
            SnapshotInit(repo=ctx.repo, commit=commit, repo_root=repo_root),
            min_lines_threshold=10,
        )
        compute_function_history(ctx.gateway, cfg, runner=git_ctx.runner)

        rows = ctx.con.execute(
            "SELECT commit_count, lines_added FROM analytics.function_history"
        ).fetchall()
        expect_equal(len(rows), 1)
        commit_count, lines_added = rows[0]
        expect_equal(commit_count, 0)
        expect_equal(lines_added, 0)

    finally:
        ctx.close()
