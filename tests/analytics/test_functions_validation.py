"""Tests for function analytics validation and parser hooks."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from codeintel.analytics.functions import ParsedFile, compute_function_metrics_and_types
from codeintel.config.models import FunctionAnalyticsConfig, FunctionAnalyticsOverrides
from codeintel.ingestion.ast_utils import AstSpanIndex
from codeintel.storage.schemas import apply_all_schemas


def _insert_goid(con: duckdb.DuckDBPyConnection, *, rel_path: str, qualname: str) -> None:
    now = datetime.now(UTC)
    con.execute(
        """
        INSERT INTO core.goids (
            goid_h128,
            urn,
            repo,
            commit,
            rel_path,
            language,
            kind,
            qualname,
            start_line,
            end_line,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            1,
            f"urn:{qualname}",
            "demo/repo",
            "deadbeef",
            rel_path,
            "python",
            "function",
            qualname,
            1,
            2,
            now,
        ],
    )


def test_records_validation_when_parser_returns_none(tmp_path: Path) -> None:
    """Parser hook returning None records validation rows instead of failing silently."""
    con = duckdb.connect(":memory:")
    apply_all_schemas(con)

    rel_path = "mod.py"
    file_path = tmp_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("def missing_span():\n    return 1\n", encoding="utf-8")
    _insert_goid(con, rel_path=rel_path, qualname="pkg.mod.missing_span")

    called = {"count": 0}

    def parser(_path: Path) -> ParsedFile | None:
        called["count"] += 1
        return None

    cfg = FunctionAnalyticsConfig.from_paths(
        repo="demo/repo",
        commit="deadbeef",
        repo_root=tmp_path,
        overrides=FunctionAnalyticsOverrides(fail_on_missing_spans=False),
    )
    compute_function_metrics_and_types(con, cfg, parser=parser)

    metrics_rows = con.execute("SELECT * FROM analytics.function_metrics").fetchall()
    validation_rows = con.execute(
        "SELECT issue FROM analytics.function_validation WHERE repo = ? AND commit = ?",
        ["demo/repo", "deadbeef"],
    ).fetchall()

    if called["count"] != 1:
        pytest.fail("parser hook should be invoked once")
    if metrics_rows:
        pytest.fail("metrics should be empty when parsing fails")
    if validation_rows != [("parse_failed",)]:
        pytest.fail(f"unexpected validation rows: {validation_rows}")


def test_custom_parser_hook_is_used(tmp_path: Path) -> None:
    """Custom parser hook is invoked and metrics are emitted when it returns a ParsedFile."""
    con = duckdb.connect(":memory:")
    apply_all_schemas(con)

    rel_path = "mod.py"
    qualname = "pkg.mod.foo"
    file_path = tmp_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("def foo():\n    return 1\n", encoding="utf-8")
    _insert_goid(con, rel_path=rel_path, qualname=qualname)

    source = "def foo():\n    return 1\n"
    parsed = ParsedFile(
        lines=list(source.splitlines()),
        index=AstSpanIndex.from_tree(ast.parse(source), (ast.FunctionDef, ast.AsyncFunctionDef)),
    )
    called = {"count": 0}

    def parser(_path: Path) -> ParsedFile | None:
        called["count"] += 1
        return parsed

    cfg = FunctionAnalyticsConfig.from_paths(
        repo="demo/repo",
        commit="deadbeef",
        repo_root=tmp_path,
    )
    summary = compute_function_metrics_and_types(con, cfg, parser=parser)

    metrics_rows = con.execute(
        "SELECT qualname FROM analytics.function_metrics WHERE repo = ? AND commit = ?",
        ["demo/repo", "deadbeef"],
    ).fetchall()
    validation_rows = con.execute(
        "SELECT issue FROM analytics.function_validation WHERE repo = ? AND commit = ?",
        ["demo/repo", "deadbeef"],
    ).fetchall()

    if called["count"] != 1:
        pytest.fail("parser hook should be invoked once")
    if metrics_rows != [(qualname,)]:
        pytest.fail(f"unexpected metrics rows: {metrics_rows}")
    if validation_rows:
        pytest.fail(f"validation rows should be empty: {validation_rows}")
    if summary != {
        "metrics_rows": 1,
        "types_rows": 1,
        "validation_total": 0,
        "validation_parse_failed": 0,
        "validation_span_not_found": 0,
    }:
        pytest.fail(f"unexpected summary: {summary}")
