"""Scaffolding tests for v1 analytics Hamilton nodes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from codeintel.build.analytics.functions.metrics import FunctionAnalyticsResult
from codeintel.build.hamilton.env import BuildEnv
from codeintel.core.validation.reporters import FunctionValidationReporter
from tests._helpers.schemas import ensure_schema_service

try:
    from codeintel.build.hamilton.native.analytics.function_ast_features import (
        function_ast_features__base,
    )
    from codeintel.build.hamilton.native.analytics.function_types import function_types__base
except RuntimeError as exc:
    if "SchemaService has not been configured" in str(exc):
        pytest.skip(
            "SchemaService is required for v1 analytics scaffold nodes.",
            allow_module_level=True,
        )
    raise

try:
    import polars as pl
except ModuleNotFoundError:
    pytest.skip("polars is required for analytics scaffold tests", allow_module_level=True)

pytestmark = pytest.mark.no_runtime_env


@dataclass(frozen=True)
class _FakeSnapshot:
    repo: str
    commit: str
    repo_root: Path


@dataclass(frozen=True)
class _FakeEnv:
    snapshot: _FakeSnapshot
    repo: str
    commit: str


def _fake_env(repo_root: Path) -> BuildEnv:
    snapshot = _FakeSnapshot(repo="repo", commit="commit", repo_root=repo_root)
    env = _FakeEnv(snapshot=snapshot, repo=snapshot.repo, commit=snapshot.commit)
    return cast("BuildEnv", env)


def _sample_goids_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "goid_h128": [1],
            "urn": ["urn:goid:1"],
            "repo": ["repo"],
            "commit": ["commit"],
            "rel_path": ["src/app.py"],
            "language": ["python"],
            "kind": ["function"],
            "qualname": ["app.main"],
            "start_line": [1],
            "end_line": [2],
            "created_at": [datetime(2024, 1, 1, tzinfo=UTC)],
        }
    )


def _sample_modules_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "repo": ["repo"],
            "commit": ["commit"],
            "module": ["app"],
            "path": ["src/app.py"],
            "language": ["python"],
        }
    )


def _sample_function_types_row() -> dict[str, object]:
    return {
        "function_goid_h128": 1,
        "urn": "urn:goid:1",
        "repo": "repo",
        "commit": "commit",
        "rel_path": "src/app.py",
        "language": "python",
        "kind": "function",
        "qualname": "app.main",
        "start_line": 1,
        "end_line": 2,
        "total_params": 0,
        "return_type": None,
        "type_comment": None,
        "param_types": None,
        "created_at": datetime(2024, 1, 1, tzinfo=UTC),
    }


def _columns(frame: object) -> list[str]:
    column_names = getattr(frame, "column_names", None)
    if isinstance(column_names, Sequence):
        return list(column_names)
    columns = getattr(frame, "columns", None)
    if isinstance(columns, Sequence):
        return list(columns)
    collect = getattr(frame, "collect", None)
    if callable(collect):
        collected = collect()
        collected_columns = getattr(collected, "columns", None)
        if isinstance(collected_columns, Sequence):
            return list(collected_columns)
    msg = f"Unsupported metrics frame type: {type(frame).__name__}"
    raise TypeError(msg)


def test_function_types_base_columns() -> None:
    """Ensure function types base nodes expose the expected columns.

    Raises
    ------
    RuntimeError
        If required schema services are unavailable for the test runtime.
    """
    try:
        ensure_schema_service()
    except RuntimeError as exc:
        if "ContractService has not been configured" in str(exc):
            pytest.skip("ContractService is required for analytics scaffold nodes.")
        raise
    reporter = FunctionValidationReporter(repo="repo", commit="commit")
    analytics_result = FunctionAnalyticsResult(
        types_rows=[_sample_function_types_row()],
        reporter=reporter,
    )
    frame = function_types__base(analytics_result)
    assert _columns(frame) == [
        "function_goid_h128",
        "urn",
        "repo",
        "commit",
        "rel_path",
        "language",
        "kind",
        "qualname",
        "start_line",
        "end_line",
        "total_params",
        "return_type",
        "type_comment",
        "param_types",
        "created_at",
    ]


def test_function_ast_features_base_columns(tmp_path: Path) -> None:
    """Ensure function AST feature base nodes expose the expected columns.

    Raises
    ------
    RuntimeError
        If required schema services are unavailable for the test runtime.
    """
    try:
        ensure_schema_service()
    except RuntimeError as exc:
        if "ContractService has not been configured" in str(exc):
            pytest.skip("ContractService is required for analytics scaffold nodes.")
        raise
    env = _fake_env(tmp_path)
    frame = function_ast_features__base(
        env,
        _sample_goids_frame(),
        _sample_modules_frame(),
    )
    assert _columns(frame) == [
        "repo",
        "commit",
        "function_goid_h128",
        "rel_path",
        "qualname",
        "is_async",
        "uses_network",
        "uses_db",
        "uses_filesystem",
        "uses_subprocess",
        "uses_concurrency_lib",
        "uses_threading",
        "uses_asyncio_lib",
        "http_client_libs",
        "http_server_libs",
        "db_libs",
        "message_libs",
        "config_read_count",
        "feature_flag_count",
        "decorators",
        "libraries_used",
        "created_at",
    ]
