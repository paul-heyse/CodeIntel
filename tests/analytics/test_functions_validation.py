"""Tests for function analytics validation flows."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeintel.analytics.functions import compute_function_metrics_and_types
from codeintel.config import SnapshotInit
from codeintel.config.steps_analytics import FunctionAnalyticsStepConfig
from tests._helpers.builders import GoidRow, insert_rows
from tests._helpers.config_factory import function_analytics_cfg
from tests._helpers.context import TestContext, create_test_context
from tests._helpers.env_options import EnvOptions


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


def _function_analytics_cfg(
    ctx: TestContext, *, fail_on_missing_spans: bool = False
) -> FunctionAnalyticsStepConfig:
    snapshot = SnapshotInit(repo=ctx.repo, commit=ctx.commit, repo_root=ctx.repo_root)
    return function_analytics_cfg(
        snapshot,
        fail_on_missing_spans=fail_on_missing_spans,
    )


@pytest.fixture
def ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Create a test context for function validation scenarios.

    Yields
    ------
    TestContext
        Context configured for function validation tests.
    """
    options = EnvOptions(repo="demo/repo", commit="deadbeef")
    context = create_test_context(tmp_path, options=options)
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

    cfg = _function_analytics_cfg(ctx, fail_on_missing_spans=False)
    summary = compute_function_metrics_and_types(ctx.gateway, cfg)

    metrics_rows = ctx.gateway.con.execute("SELECT * FROM analytics.function_metrics").fetchall()
    validation_rows = ctx.gateway.con.execute(
        """
        SELECT function_goid_h128, issue
        FROM analytics.function_validation
        WHERE repo = ? AND commit = ?
        """,
        [ctx.repo, ctx.commit],
    ).fetchall()

    if metrics_rows:
        pytest.fail(f"Expected no metrics rows, found {metrics_rows}")
    if validation_rows != [(1, "parse_failed")]:
        pytest.fail(f"Unexpected validation rows: {validation_rows}")
    if summary["validation_parse_failed"] != 1:
        pytest.fail(f"Unexpected parse_failed count: {summary['validation_parse_failed']}")


def test_span_not_found_is_recorded(ctx: TestContext) -> None:
    """Missing spans produce span_not_found validation rows."""
    rel_path = "mod.py"
    file_path = ctx.repo_root / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("def foo():\n    return 1\n", encoding="utf-8")
    _insert_goid(ctx, rel_path=rel_path, qualname="pkg.mod.foo", start_line=50, end_line=55)

    cfg = _function_analytics_cfg(ctx, fail_on_missing_spans=False)
    summary = compute_function_metrics_and_types(ctx.gateway, cfg)

    validation_rows = ctx.gateway.con.execute(
        """
        SELECT function_goid_h128, issue
        FROM analytics.function_validation
        WHERE repo = ? AND commit = ?
        """,
        [ctx.repo, ctx.commit],
    ).fetchall()

    if validation_rows != [(1, "span_not_found")]:
        pytest.fail(f"Unexpected validation rows: {validation_rows}")
    if summary["validation_span_not_found"] != 1:
        pytest.fail(
            f"Unexpected span_not_found count: {summary['validation_span_not_found']}",
        )
