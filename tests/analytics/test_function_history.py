"""Tests for per-function git history aggregation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from codeintel.analytics.function_history import compute_function_history
from codeintel.config.models import FunctionHistoryConfig
from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import expect_equal, expect_in
from tests._helpers.builders import (
    FunctionMetricsRow,
    ModuleRow,
    insert_function_metrics,
    insert_modules,
)
from tests._helpers.fakes import FakeToolRunner

EXPECTED_STABILITY_BUCKETS = {"new_hot", "stable", "churning", "legacy_hot"}
EXPECTED_LINES_ADDED = 3


def _git_log_payload() -> str:
    return "\n".join(
        [
            "@@@c123\tauthor@example.com\t2024-01-01T00:00:00+00:00",
            f"@@ -1,0 +1,{EXPECTED_LINES_ADDED} @@",
        ]
    )


def test_function_history_populates_rows(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Compute_function_history should persist metrics for touched functions."""
    repo_root = tmp_path
    repo = "demo/repo"
    commit = "abc123"
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
                rel_path="foo.py",
                language="python",
                kind="function",
                qualname="foo",
                start_line=1,
                end_line=5,
                loc=5,
                logical_loc=5,
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
                path="foo.py",
                repo=repo,
                commit=commit,
            )
        ],
    )

    runner = FakeToolRunner(
        cache_dir=repo_root / ".tool_cache",
        payloads={"git": _git_log_payload()},
    )
    cfg = FunctionHistoryConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    compute_function_history(con, cfg, runner=runner)

    rows = con.execute("SELECT * FROM analytics.function_history").fetchall()
    expect_equal(len(rows), 1, "Expected a single function history row.")
    columns = TABLE_SCHEMAS["analytics.function_history"].column_names()
    result = dict(zip(columns, rows[0], strict=True))
    expect_equal(result["commit_count"], 1)
    expect_equal(result["author_count"], 1)
    expect_equal(result["lines_added"], EXPECTED_LINES_ADDED)
    expect_in(result["stability_bucket"], EXPECTED_STABILITY_BUCKETS)
    expect_equal(result["function_goid_h128"], 1)
    expect_equal(result["rel_path"], "foo.py")
    expect_equal(result["module"], "pkg.foo")


def test_function_history_respects_min_threshold(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Respect the minimum line threshold when populating function history."""
    repo_root = tmp_path
    repo = "demo/repo"
    commit = "abc123"
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
                rel_path="bar.py",
                language="python",
                kind="function",
                qualname="bar",
                start_line=1,
                end_line=2,
                loc=2,
                logical_loc=2,
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
    runner = FakeToolRunner(cache_dir=repo_root / ".tool_cache", payloads={"git": ""})
    cfg = FunctionHistoryConfig.from_paths(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        overrides=FunctionHistoryConfig.Overrides(min_lines_threshold=10),
    )
    compute_function_history(con, cfg, runner=runner)
    rows = con.execute("SELECT commit_count, lines_added FROM analytics.function_history").fetchall()
    expect_equal(len(rows), 1)
    commit_count, lines_added = rows[0]
    expect_equal(commit_count, 0)
    expect_equal(lines_added, 0)
