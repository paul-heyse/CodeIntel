"""Tests for function analytics validation and parser hooks."""

from __future__ import annotations

import ast
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from codeintel.analytics.function_parsing import (
    FunctionParserKind,
    FunctionParserRegistry,
    ParsedFile,
)
from codeintel.analytics.functions import (
    FunctionAnalyticsOptions,
    ProcessContext,
    ProcessState,
    ValidationReporter,
    build_function_analytics,
    compute_function_metrics_and_types,
    persist_function_analytics,
)
from codeintel.analytics.functions import (
    GoidRow as AnalyticsGoidRow,
)
from codeintel.config.models import FunctionAnalyticsConfig, FunctionAnalyticsOverrides
from codeintel.ingestion.ast_utils import AstSpanIndex
from codeintel.storage.gateway import StorageGateway
from tests._helpers.builders import GoidRow, insert_goids


def _insert_goid(gateway: StorageGateway, *, rel_path: str, qualname: str) -> None:
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
                start_line=1,
                end_line=2,
                created_at=now,
            )
        ],
    )


def test_records_validation_when_parser_returns_none(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Parser hook returning None records validation rows instead of failing silently."""
    gateway = fresh_gateway
    con = gateway.con

    rel_path = "mod.py"
    file_path = tmp_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("def missing_span():\n    return 1\n", encoding="utf-8")
    _insert_goid(gateway, rel_path=rel_path, qualname="pkg.mod.missing_span")

    called = {"count": 0}

    def parser(_path: Path) -> ParsedFile | None:
        called["count"] += 1
        return None

    registry = FunctionParserRegistry()
    registry.register(FunctionParserKind.PYTHON, parser)
    cfg = FunctionAnalyticsConfig.from_paths(
        repo="demo/repo",
        commit="deadbeef",
        repo_root=tmp_path,
        overrides=FunctionAnalyticsOverrides(fail_on_missing_spans=False),
    )
    compute_function_metrics_and_types(
        gateway,
        cfg,
        options=FunctionAnalyticsOptions(parser_registry=registry),
    )

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


def test_custom_parser_hook_is_used(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """Custom parser hook is invoked and metrics are emitted when it returns a ParsedFile."""
    gateway = fresh_gateway
    con = gateway.con

    rel_path = "mod.py"
    qualname = "pkg.mod.foo"
    file_path = tmp_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("def foo():\n    return 1\n", encoding="utf-8")
    _insert_goid(gateway, rel_path=rel_path, qualname=qualname)

    source = "def foo():\n    return 1\n"
    parsed_index = AstSpanIndex.from_tree(
        ast.parse(source),
        (ast.FunctionDef, ast.AsyncFunctionDef),
    )
    parsed = ParsedFile(
        lines=list(source.splitlines()),
        index=parsed_index,
    )
    called = {"count": 0}

    def parser(_path: Path) -> ParsedFile | None:
        called["count"] += 1
        return parsed

    registry = FunctionParserRegistry()
    registry.register(FunctionParserKind.PYTHON, parser)
    cfg = FunctionAnalyticsConfig.from_paths(
        repo="demo/repo",
        commit="deadbeef",
        repo_root=tmp_path,
    )
    summary = compute_function_metrics_and_types(
        gateway,
        cfg,
        options=FunctionAnalyticsOptions(parser_registry=registry),
    )

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


def test_unknown_parser_kind_raises() -> None:
    """Parser registry should reject unknown parser values."""
    registry = FunctionParserRegistry()
    with pytest.raises(ValueError, match=r"^Unsupported function parser: unknown$"):
        registry.get(cast("FunctionParserKind", "unknown"))  # type: ignore[arg-type]


def test_config_invalid_parser_string_raises(tmp_path: Path) -> None:
    """FunctionAnalyticsConfig should reject invalid parser strings."""
    with pytest.raises(ValueError, match="is not a valid FunctionParserKind"):
        FunctionAnalyticsConfig.from_paths(
            repo="demo/repo",
            commit="deadbeef",
            repo_root=tmp_path,
            overrides=FunctionAnalyticsOverrides(parser=cast("FunctionParserKind", "bogus")),
        )


def test_build_and_persist_paths_are_separate(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """build_function_analytics remains pure until persisted."""
    gateway = fresh_gateway
    con = gateway.con

    file_path = tmp_path / "mod.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("def foo():\n    return 1\n", encoding="utf-8")
    _insert_goid(gateway, rel_path="mod.py", qualname="pkg.mod.foo")

    cfg = FunctionAnalyticsConfig.from_paths(
        repo="demo/repo",
        commit="deadbeef",
        repo_root=tmp_path,
    )
    rel_path = "mod.py"
    goid_row: AnalyticsGoidRow = {
        "goid_h128": 1,
        "urn": "urn:pkg.mod.foo",
        "repo": cfg.repo,
        "commit": cfg.commit,
        "rel_path": rel_path,
        "language": "python",
        "kind": "function",
        "qualname": "pkg.mod.foo",
        "start_line": 1,
        "end_line": 2,
    }
    goids_by_file: dict[str, list[AnalyticsGoidRow]] = {rel_path: [goid_row]}

    parsed = ParsedFile(
        lines=["def foo():", "    return 1"],
        index=AstSpanIndex.from_tree(
            ast.parse("def foo():\n    return 1\n"),
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ),
    )
    ctx = ProcessContext(cfg=cfg, now=datetime.now(UTC))
    state = ProcessState(
        cfg=cfg,
        cache={},
        parser=lambda _path: parsed,
        ctx=ctx,
    )

    result = build_function_analytics(goids_by_file=goids_by_file, state=state)
    metrics_before_row = con.execute("SELECT COUNT(*) FROM analytics.function_metrics").fetchone()
    types_before_row = con.execute("SELECT COUNT(*) FROM analytics.function_types").fetchone()
    if metrics_before_row is None or types_before_row is None:
        pytest.fail("unexpected NULL count rows")
    if metrics_before_row[0] != 0 or types_before_row[0] != 0:
        pytest.fail("builder should not persist results")

    summary = persist_function_analytics(gateway, cfg, result, created_at=ctx.now)
    counts_after = con.execute(
        """
        SELECT COUNT(*) FROM analytics.function_metrics WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetchone()
    if counts_after is None:
        pytest.fail("missing metrics count after persist")
    if counts_after[0] != 1:
        pytest.fail(f"unexpected persisted metrics count: {counts_after[0]}")
    if summary["metrics_rows"] != 1 or summary["types_rows"] != 1:
        pytest.fail(f"unexpected summary after persist: {summary}")


def test_validation_reporter_emits_counts(
    fresh_gateway: StorageGateway, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Validation reporter produces structured counters."""
    gateway = fresh_gateway

    rel_path = "mod.py"
    file_path = tmp_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("def foo():\n    return 1\n", encoding="utf-8")
    _insert_goid(gateway, rel_path=rel_path, qualname="pkg.mod.missing_span")

    cfg = FunctionAnalyticsConfig.from_paths(
        repo="demo/repo",
        commit="deadbeef",
        repo_root=tmp_path,
        overrides=FunctionAnalyticsOverrides(fail_on_missing_spans=False),
    )
    registry = FunctionParserRegistry()
    registry.register(FunctionParserKind.PYTHON, lambda _path: None)
    caplog.set_level(logging.INFO)
    reporter_calls: list[tuple[str, int]] = []
    reporter = ValidationReporter(
        emit_counter=lambda name, value: reporter_calls.append((name, value)),
        logger=logging.getLogger("validation-test"),
    )

    compute_function_metrics_and_types(
        gateway,
        cfg,
        options=FunctionAnalyticsOptions(
            parser_registry=registry,
            validation_reporter=reporter,
        ),
    )

    log_levels = {(rec.name, rec.levelno) for rec in caplog.records}
    if ("validation-test", logging.INFO) not in log_levels:
        pytest.fail("validation reporter did not log at INFO")
    expected_calls = {
        ("function_validation.total", 1),
        ("function_validation.parse_failed", 1),
        ("function_validation.span_not_found", 0),
    }
    if set(reporter_calls) != expected_calls:
        pytest.fail(f"unexpected reporter counters: {reporter_calls}")
