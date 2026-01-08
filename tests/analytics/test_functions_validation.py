"""Tests for function analytics validation flows."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.build.analytics.functions.metrics import (
    FunctionAnalyticsResult,
    compute_function_analytics_result,
)
from codeintel.config.primitives import SnapshotRef
from tests._helpers import TestScenario
from tests._helpers.fixtures.rows import GoidRow, insert_rows
from tests._helpers.sql import run_query

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from tests._helpers.context import TestContext


def _insert_goid(
    ctx: TestContext,
    *,
    rel_path: str,
    qualname: str,
    start_line: int = 1,
    end_line: int = 2,
) -> None:
    now = datetime.now(UTC)
    insert_rows(
        ctx.gateway,
        [
            GoidRow(
                goid_h128=1,
                urn=f"urn:{ctx.repo}:{ctx.commit}:{rel_path}#{qualname}",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=rel_path,
                kind="function",
                qualname=qualname,
                start_line=start_line,
                end_line=end_line,
                created_at=now,
            )
        ],
    )


def _get_snapshot(ctx: TestContext) -> SnapshotRef:
    return SnapshotRef(repo=ctx.repo, commit=ctx.commit, repo_root=ctx.repo_root)


def _write_function_results(ctx: TestContext, result: FunctionAnalyticsResult) -> None:
    backend = ctx.gateway.policy
    snapshot = _get_snapshot(ctx)
    if result.types_rows:
        backend.delete_for_snapshot(
            "analytics.function_types",
            repo=snapshot.repo,
            commit=snapshot.commit,
        )
        backend.bulk_insert_mappings("analytics.function_types", result.types_rows)

    validation_rows = result.reporter.to_rows()
    if validation_rows:
        backend.delete_for_snapshot(
            "analytics.function_validation",
            repo=snapshot.repo,
            commit=snapshot.commit,
        )
        backend.bulk_insert("analytics.function_validation", validation_rows)


@pytest.fixture
def ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Create a test context for function validation scenarios.

    Yields
    ------
    TestContext
        Context configured for function validation tests.
    """
    context = TestScenario().build(tmp_path)
    try:
        yield context
    finally:
        context.close()


def test_records_validation_when_parse_fails(ctx: TestContext) -> None:
    """Parse errors are persisted to analytics.function_validation."""
    rel_path = "mod.py"
    file_path = ctx.repo_root / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("def broken(:\n    return 1\n", encoding="utf-8")
    _insert_goid(ctx, rel_path=rel_path, qualname="pkg.mod.broken")

    snapshot = _get_snapshot(ctx)
    goids_input = ctx.gateway.con.execute("SELECT * FROM core.goids").arrow().read_all()
    result = compute_function_analytics_result(goids_input, snapshot)
    _write_function_results(ctx, result)

    types_rows = run_query(ctx.gateway, "SELECT * FROM analytics.function_types")
    validation_rows = run_query(
        ctx.gateway,
        """
        SELECT function_goid_h128, issue
        FROM analytics.function_validation
        WHERE repo = ? AND commit = ?
        """,
        [ctx.repo, ctx.commit],
    )

    if types_rows:
        pytest.fail(f"Expected no types rows, found {types_rows}")
    if validation_rows != [(1, "parse_failed")]:
        pytest.fail(f"Unexpected validation rows: {validation_rows}")
    if result.reporter.parse_failed != 1:
        pytest.fail(f"Unexpected parse_failed count: {result.reporter.parse_failed}")


def test_span_not_found_is_recorded(ctx: TestContext) -> None:
    """Missing spans produce span_not_found validation rows."""
    rel_path = "mod.py"
    file_path = ctx.repo_root / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("def foo():\n    return 1\n", encoding="utf-8")
    _insert_goid(ctx, rel_path=rel_path, qualname="pkg.mod.foo", start_line=50, end_line=55)

    snapshot = _get_snapshot(ctx)
    goids_input = ctx.gateway.con.execute("SELECT * FROM core.goids").arrow().read_all()
    result = compute_function_analytics_result(goids_input, snapshot)
    _write_function_results(ctx, result)

    validation_rows = run_query(
        ctx.gateway,
        """
        SELECT function_goid_h128, issue
        FROM analytics.function_validation
        WHERE repo = ? AND commit = ?
        """,
        [ctx.repo, ctx.commit],
    )

    if validation_rows != [(1, "span_not_found")]:
        pytest.fail(f"Unexpected validation rows: {validation_rows}")
    if result.reporter.span_not_found != 1:
        pytest.fail(f"Unexpected span_not_found count: {result.reporter.span_not_found}")
