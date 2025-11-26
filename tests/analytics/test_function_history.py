"""Tests for per-function git history aggregation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from codeintel.analytics.functions import compute_function_history
from codeintel.config import ConfigBuilder
from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import expect_equal, expect_in, expect_true
from tests._helpers.builders import (
    FunctionMetricsRow,
    ModuleRow,
    insert_function_metrics,
    insert_modules,
)
from tests._helpers.tooling import init_git_repo_with_history

EXPECTED_STABILITY_BUCKETS = {"new_hot", "stable", "churning", "legacy_hot"}
MIN_EXPECTED_LINES_ADDED = 2


def test_function_history_populates_rows(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Compute_function_history should persist metrics for touched functions."""
    git_ctx = init_git_repo_with_history(tmp_path)
    repo_root = git_ctx.repo_root
    repo = "demo/repo"
    commit = git_ctx.commits[0]
    gateway = fresh_gateway
    con = gateway.con
    insert_function_metrics(
        gateway,
        [
            FunctionMetricsRow(
                function_goid_h128=1,
                urn="urn:fn",
                repo=repo,
                commit=commit,
                rel_path="pkg/foo.py",
                language="python",
                kind="function",
                qualname="pkg.foo",
                start_line=1,
                end_line=3,
                loc=3,
                logical_loc=3,
                param_count=0,
                positional_params=0,
                keyword_only_params=0,
                has_varargs=False,
                has_varkw=False,
                is_async=False,
                is_generator=False,
                return_count=0,
                yield_count=0,
                raise_count=0,
                cyclomatic_complexity=1,
                max_nesting_depth=1,
                stmt_count=1,
                decorator_count=0,
                has_docstring=True,
                complexity_bucket="low",
                created_at=datetime.now(tz=UTC),
            )
        ],
    )
    insert_modules(
        gateway,
        [
            ModuleRow(
                module="pkg.foo",
                path="pkg/foo.py",
                repo=repo,
                commit=commit,
            )
        ],
    )

    builder = ConfigBuilder.from_snapshot(repo=repo, commit=commit, repo_root=repo_root)
    cfg = builder.function_history()
    compute_function_history(gateway, cfg, runner=git_ctx.runner)

    rows = con.execute("SELECT * FROM analytics.function_history").fetchall()
    expect_equal(len(rows), 1, "Expected a single function history row.")
    columns = TABLE_SCHEMAS["analytics.function_history"].column_names()
    result = dict(zip(columns, rows[0], strict=True))
    expected_commit_count = len(git_ctx.commits)
    expect_equal(result["commit_count"], expected_commit_count)
    expect_equal(result["author_count"], 1)
    expect_true(
        result["lines_added"] >= MIN_EXPECTED_LINES_ADDED,
        "Expected lines_added to reflect git history",
    )
    expect_in(result["stability_bucket"], EXPECTED_STABILITY_BUCKETS)
    expect_equal(result["function_goid_h128"], 1)
    expect_equal(result["rel_path"], "pkg/foo.py")
    expect_equal(result["module"], "pkg.foo")


def test_function_history_respects_min_threshold(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Respect the minimum line threshold when populating function history."""
    git_ctx = init_git_repo_with_history(tmp_path)
    repo_root = git_ctx.repo_root
    repo = "demo/repo"
    commit = git_ctx.commits[0]
    gateway = fresh_gateway
    con = gateway.con

    insert_function_metrics(
        gateway,
        [
            FunctionMetricsRow(
                function_goid_h128=2,
                urn="urn:fn2",
                repo=repo,
                commit=commit,
                rel_path="pkg/foo.py",
                language="python",
                kind="function",
                qualname="pkg.foo",
                start_line=1,
                end_line=3,
                loc=3,
                logical_loc=3,
                param_count=0,
                positional_params=0,
                keyword_only_params=0,
                has_varargs=False,
                has_varkw=False,
                is_async=False,
                is_generator=False,
                return_count=0,
                yield_count=0,
                raise_count=0,
                cyclomatic_complexity=1,
                max_nesting_depth=1,
                stmt_count=1,
                decorator_count=0,
                has_docstring=True,
                complexity_bucket="low",
                created_at=datetime.now(tz=UTC),
            )
        ],
    )
    builder = ConfigBuilder.from_snapshot(repo=repo, commit=commit, repo_root=repo_root)
    cfg = builder.function_history(min_lines_threshold=10)
    compute_function_history(gateway, cfg, runner=git_ctx.runner)
    rows = con.execute(
        "SELECT commit_count, lines_added FROM analytics.function_history"
    ).fetchall()
    expect_equal(len(rows), 1)
    commit_count, lines_added = rows[0]
    expect_equal(commit_count, 0)
    expect_equal(lines_added, 0)
