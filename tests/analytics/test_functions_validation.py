"""Tests for function analytics validation flows."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeintel.analytics.functions.metrics import compute_function_metrics_and_types
from codeintel.config.models import FunctionAnalyticsConfig, FunctionAnalyticsOverrides
from codeintel.storage.gateway import StorageGateway
from tests._helpers.builders import GoidRow, insert_goids


def _insert_goid(
    gateway: StorageGateway,
    *,
    rel_path: str,
    qualname: str,
    start_line: int = 1,
    end_line: int = 2,
) -> None:
    now = datetime.now(UTC)
    insert_goids(
        gateway,
        [
            GoidRow(
                goid_h128=1,
                urn=f"urn:{qualname}",
                repo="demo/repo",
                commit="deadbeef",
                rel_path=rel_path,
                kind="function",
                qualname=qualname,
                start_line=start_line,
                end_line=end_line,
                created_at=now,
            )
        ],
    )


def test_records_validation_when_parse_fails(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """Parse errors are persisted to analytics.function_validation."""
    gateway = fresh_gateway
    con = gateway.con

    rel_path = "mod.py"
    file_path = tmp_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("def broken(:\n    return 1\n", encoding="utf-8")
    _insert_goid(gateway, rel_path=rel_path, qualname="pkg.mod.broken")

    cfg = FunctionAnalyticsConfig.from_paths(
        repo="demo/repo",
        commit="deadbeef",
        repo_root=tmp_path,
        overrides=FunctionAnalyticsOverrides(fail_on_missing_spans=False),
    )
    summary = compute_function_metrics_and_types(gateway, cfg)

    metrics_rows = con.execute("SELECT * FROM analytics.function_metrics").fetchall()
    validation_rows = con.execute(
        """
        SELECT function_goid_h128, issue
        FROM analytics.function_validation
        WHERE repo = ? AND commit = ?
        """,
        ["demo/repo", "deadbeef"],
    ).fetchall()

    if metrics_rows:
        pytest.fail(f"Expected no metrics rows, found {metrics_rows}")
    if validation_rows != [(1, "parse_failed")]:
        pytest.fail(f"Unexpected validation rows: {validation_rows}")
    if summary["validation_parse_failed"] != 1:
        pytest.fail(f"Unexpected parse_failed count: {summary['validation_parse_failed']}")


def test_span_not_found_is_recorded(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """Missing spans produce span_not_found validation rows."""
    gateway = fresh_gateway
    con = gateway.con

    rel_path = "mod.py"
    file_path = tmp_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("def foo():\n    return 1\n", encoding="utf-8")
    _insert_goid(gateway, rel_path=rel_path, qualname="pkg.mod.foo", start_line=50, end_line=55)

    cfg = FunctionAnalyticsConfig.from_paths(
        repo="demo/repo",
        commit="deadbeef",
        repo_root=tmp_path,
        overrides=FunctionAnalyticsOverrides(fail_on_missing_spans=False),
    )
    summary = compute_function_metrics_and_types(gateway, cfg)

    validation_rows = con.execute(
        """
        SELECT function_goid_h128, issue
        FROM analytics.function_validation
        WHERE repo = ? AND commit = ?
        """,
        ["demo/repo", "deadbeef"],
    ).fetchall()

    if validation_rows != [(1, "span_not_found")]:
        pytest.fail(f"Unexpected validation rows: {validation_rows}")
    if summary["validation_span_not_found"] != 1:
        pytest.fail(
            f"Unexpected span_not_found count: {summary['validation_span_not_found']}",
        )
