"""Unit tests for analytics.tests_profiles module.

This module consolidates unit tests for test profiles helpers, wrappers,
registry guards, and snapshot validation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path as PathLib
from typing import TYPE_CHECKING

import pytest

from codeintel.build.analytics.testing.behavioral import importance
from codeintel.build.analytics.testing.behavioral.importance import (
    compute_flakiness_score,
    compute_importance_score,
)
from codeintel.build.analytics.testing.behavioral.tags import infer_behavior_tags
from codeintel.build.analytics.testing.profiles import rows
from codeintel.build.analytics.testing.profiles.types import (
    FunctionCoverageEntry,
    ImportanceInputs,
    IoFlags,
    SubsystemCoverageEntry,
    TestAstInfo,
    TestGraphMetrics,
    TestProfileContext,
    TestProfileOptions,
    TestRecord,
)
from codeintel.build.schemas import configure_schema_service
from codeintel.config.datasets.columns import load_columns_by_table, serialize_row
from codeintel.config.primitives import SnapshotRef
from codeintel.runtime.runtime_bundle import RuntimeBundle
from tests._helpers.factories import blank_test_profile_row

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


@pytest.fixture(autouse=True)
def _configure_schema_provider(hamilton_runtime: RuntimeBundle) -> None:
    configure_schema_service(runtime=hamilton_runtime)


def _columns_by_table() -> dict[str, tuple[str, ...]]:
    columns = load_columns_by_table()
    return {key: tuple(value) for key, value in columns.items()}


def serialize_test_profile_row(row: Mapping[str, object]) -> tuple[object, ...]:
    """Serialize a test profile mapping using schema-derived column order.

    Returns
    -------
    tuple[object, ...]
        Tuple of values in storage column order.
    """
    columns = _columns_by_table()["analytics.test_profile"]
    return serialize_row(row, columns)


def _make_snapshot(repo_root: Path | None = None) -> SnapshotRef:
    """Create a standard test snapshot reference.

    Parameters
    ----------
    repo_root
        Optional repo root path; defaults to Path.cwd() if not provided.

    Returns
    -------
    SnapshotRef
        Configured snapshot reference with test defaults.
    """
    return SnapshotRef(
        repo="demo/repo",
        commit="deadbeef",
        repo_root=repo_root if repo_root is not None else PathLib.cwd(),
    )


def _snapshot_cfg() -> tuple[SnapshotRef, TestProfileOptions]:
    """Create snapshot and options for tests without file I/O.

    Returns
    -------
    tuple[SnapshotRef, TestProfileOptions]
        Snapshot reference and test profile options.
    """
    return _make_snapshot(), TestProfileOptions()


def test_importance_guardrails_and_monotonicity() -> None:
    """Importance/flakiness scoring should remain bounded and monotonic."""
    io_none = IoFlags()
    io_network = IoFlags(uses_network=True)
    slow_threshold = 1000.0
    fast_score = importance.compute_flakiness_score(
        status="passed",
        markers=["fast"],
        duration_ms=100.0,
        io_flags=io_none,
        slow_test_threshold_ms=slow_threshold,
    )
    slow_score = importance.compute_flakiness_score(
        status="failed",
        markers=["slow"],
        duration_ms=5000.0,
        io_flags=io_network,
        slow_test_threshold_ms=slow_threshold,
    )
    if not 0.0 <= fast_score <= 1.0 or not 0.0 <= slow_score <= 1.0:
        msg = "Flakiness scores escaped [0, 1] bounds."
        pytest.fail(msg)
    if slow_score <= fast_score:
        msg = "Flakiness scoring is not monotonic with worse signals."
        pytest.fail(msg)

    baseline = importance.compute_importance_score(
        ImportanceInputs(
            functions_covered_count=1,
            weighted_degree=0.5,
            max_function_count=5,
            max_weighted_degree=5.0,
            subsystem_risk=0.1,
            max_subsystem_risk=1.0,
        )
    )
    improved = importance.compute_importance_score(
        ImportanceInputs(
            functions_covered_count=3,
            weighted_degree=4.0,
            max_function_count=5,
            max_weighted_degree=5.0,
            subsystem_risk=0.5,
            max_subsystem_risk=1.0,
        )
    )
    if baseline is None or improved is None:
        msg = "Importance scoring returned None unexpectedly."
        pytest.fail(msg)
    if improved <= baseline:
        msg = "Importance score did not increase with stronger signals."
        pytest.fail(msg)


def test_importance_and_flakiness_scoring() -> None:
    """Validate flakiness and importance scoring produce bounded values.

    Raises
    ------
    AssertionError
        If scores fall outside expected ranges.
    """
    io_flags = IoFlags(
        uses_network=True, uses_db=False, uses_filesystem=False, uses_subprocess=False
    )
    flakiness = compute_flakiness_score(
        status="xfail",
        markers=["slow", "network"],
        duration_ms=3000.0,
        io_flags=io_flags,
        slow_test_threshold_ms=2000.0,
    )
    min_expected_flakiness = 0.4
    if not (min_expected_flakiness <= flakiness <= 1.0):
        message = "Flakiness score out of expected range."
        raise AssertionError(message)

    inputs = ImportanceInputs(
        functions_covered_count=2,
        weighted_degree=1.0,
        max_function_count=4,
        max_weighted_degree=2.0,
        subsystem_risk=0.5,
        max_subsystem_risk=1.0,
    )
    imp_score = compute_importance_score(inputs)
    if imp_score is None or not (0.0 <= imp_score <= 1.0):
        message = "Importance score out of expected range."
        raise AssertionError(message)


def test_build_test_profile_rows_round_trip() -> None:
    """Tuple-to-model mapping should align with schema constants for new helpers."""
    created_at = datetime(2024, 1, 1, tzinfo=UTC)
    snapshot, test_options = _snapshot_cfg()
    test_record = TestRecord(
        test_id="test-id",
        test_goid_h128=1,
        urn="urn",
        rel_path="rel.py",
        module="mod",
        qualname="qual",
        language="python",
        kind="function",
        status="passed",
        duration_ms=12.5,
        markers=["fast"],
        flaky=False,
        start_line=1,
        end_line=2,
    )
    functions_covered = {
        "test-id": FunctionCoverageEntry(
            functions=[{"function_goid_h128": 1}],
            count=1,
            primary=[1],
        )
    }
    subsystems_covered = {
        "test-id": SubsystemCoverageEntry(
            subsystems=[{"subsystem_id": "sub"}],
            count=1,
            primary_subsystem_id="sub",
            max_risk_score=0.2,
        )
    }
    tg_metrics = {
        "test-id": TestGraphMetrics(
            degree=1,
            weighted_degree=2.0,
            proj_degree=1,
            proj_weight=2.0,
            proj_clustering=0.1,
            proj_betweenness=0.2,
        )
    }
    ast_info = {"test-id": TestAstInfo(assert_count=1, raise_count=0)}
    inputs = rows.TestProfileInputs(
        functions_covered=functions_covered,
        subsystems_covered=subsystems_covered,
        tg_metrics=tg_metrics,
        ast_info=ast_info,
    )
    ctx = rows.build_test_profile_context(
        snapshot=snapshot,
        inputs=inputs,
        options=test_options,
    )

    frozen_ctx = TestProfileContext(
        snapshot=ctx.snapshot,
        options=ctx.options,
        now=created_at,
        max_function_count=ctx.max_function_count,
        max_weighted_degree=ctx.max_weighted_degree,
        max_subsystem_risk=ctx.max_subsystem_risk,
        functions_covered=ctx.functions_covered,
        subsystems_covered=ctx.subsystems_covered,
        tg_metrics=ctx.tg_metrics,
        ast_info=ctx.ast_info,
    )
    models = rows.build_test_profile_rows([test_record], frozen_ctx)
    if len(models) != 1:
        msg = "Expected exactly one model."
        pytest.fail(msg)
    serialized = serialize_test_profile_row(models[0])
    if len(serialized) != len(_columns_by_table()["analytics.test_profile"]):
        msg = "Serialized tuple length mismatch for test_profile."
        pytest.fail(msg)
    if models[0]["created_at"] != created_at:
        msg = "Created_at did not preserve frozen context value."
        pytest.fail(msg)


def test_infer_behavior_tags_basic() -> None:
    """Ensure behavior tag inference captures core markers.

    Raises
    ------
    AssertionError
        If expected tags are missing.
    """
    ast_info = TestAstInfo(
        assert_count=0,
        raise_count=0,
        uses_pytest_raises=True,
        uses_concurrency_lib=False,
        has_boundary_asserts=False,
        uses_fixtures=False,
        io_flags=IoFlags(),
    )
    tags = infer_behavior_tags(
        name="test_network_error_path",
        markers=["network", "slow"],
        io_flags=IoFlags(
            uses_network=True, uses_db=False, uses_filesystem=False, uses_subprocess=False
        ),
        ast_info=ast_info,
    )
    if "network_interaction" not in tags or "error_paths" not in tags:
        message = f"Unexpected tags: {tags}"
        raise AssertionError(message)


def test_test_profile_model_snapshot() -> None:
    """Deterministic snapshot of test_profile row model to catch drift."""
    snapshot, test_options = _snapshot_cfg()
    created_at = datetime(2024, 1, 1, tzinfo=UTC)
    test_record = TestRecord(
        test_id="t1",
        test_goid_h128=101,
        urn="urn:t1",
        rel_path="rel.py",
        module="mod.a",
        qualname="A::test",
        language="python",
        kind="function",
        status="passed",
        duration_ms=10.5,
        markers=["fast"],
        flaky=False,
        start_line=1,
        end_line=5,
    )
    functions = {
        "t1": FunctionCoverageEntry(
            functions=[{"function_goid_h128": 1}],
            count=1,
            primary=[1],
        )
    }
    subsystems = {
        "t1": SubsystemCoverageEntry(
            subsystems=[{"subsystem_id": "s1"}],
            count=1,
            primary_subsystem_id="s1",
            max_risk_score=0.4,
        )
    }
    tg_metrics = {
        "t1": TestGraphMetrics(
            degree=2,
            weighted_degree=3.0,
            proj_degree=1,
            proj_weight=1.5,
            proj_clustering=0.2,
            proj_betweenness=0.1,
        )
    }
    ast_info = {"t1": TestAstInfo(io_flags=IoFlags(uses_network=True))}
    inputs = rows.TestProfileInputs(
        functions_covered=functions,
        subsystems_covered=subsystems,
        tg_metrics=tg_metrics,
        ast_info=ast_info,
    )
    ctx = rows.build_test_profile_context(
        snapshot=snapshot,
        inputs=inputs,
        options=test_options,
    )
    frozen_ctx = TestProfileContext(
        snapshot=ctx.snapshot,
        options=ctx.options,
        now=created_at,
        max_function_count=ctx.max_function_count,
        max_weighted_degree=ctx.max_weighted_degree,
        max_subsystem_risk=ctx.max_subsystem_risk,
        functions_covered=ctx.functions_covered,
        subsystems_covered=ctx.subsystems_covered,
        tg_metrics=ctx.tg_metrics,
        ast_info=ctx.ast_info,
    )
    model = rows.build_test_profile_rows([test_record], frozen_ctx)[0]

    expected = {
        "repo": "demo/repo",
        "commit": "deadbeef",
        "test_id": "t1",
        "test_goid_h128": 101,
        "urn": "urn:t1",
        "rel_path": "rel.py",
        "module": "mod.a",
        "qualname": "A::test",
        "language": "python",
        "kind": "function",
        "status": "passed",
        "duration_ms": 10.5,
        "markers": ["fast"],
        "flaky": False,
        "last_run_at": created_at,
        "functions_covered": [{"function_goid_h128": 1}],
        "functions_covered_count": 1,
        "primary_function_goids": [1],
        "subsystems_covered": [{"subsystem_id": "s1"}],
        "subsystems_covered_count": 1,
        "primary_subsystem_id": "s1",
        "assert_count": 0,
        "raise_count": 0,
        "uses_parametrize": False,
        "uses_fixtures": False,
        "io_bound": True,
        "uses_network": True,
        "uses_db": False,
        "uses_filesystem": False,
        "uses_subprocess": False,
        "flakiness_score": pytest.approx(0.15),
        "importance_score": pytest.approx(1.0),
        "notes": None,
        "tg_degree": 2,
        "tg_weighted_degree": 3.0,
        "tg_proj_degree": 1,
        "tg_proj_weight": 1.5,
        "tg_proj_clustering": 0.2,
        "tg_proj_betweenness": 0.1,
        "created_at": created_at,
    }
    if model != expected:
        pytest.fail(f"Snapshot mismatch for test_profile model: {model}")
    serialized = serialize_test_profile_row(model)
    if len(serialized) != len(_columns_by_table()["analytics.test_profile"]):
        pytest.fail("Serialized tuple length mismatch for test_profile.")


def test_test_profile_serialization() -> None:
    """Ensure test profile row serialization produces correct tuple length."""
    sample_row = blank_test_profile_row()
    sample_row["repo"] = "r"
    sample_row["commit"] = "c"
    sample_row["test_id"] = "t"
    sample_row["rel_path"] = "p"
    sample_row["markers"] = []
    sample_row["functions_covered"] = []
    sample_row["primary_function_goids"] = []
    sample_row["subsystems_covered"] = []
    sample_row["created_at"] = datetime.now(tz=UTC)

    serialized = serialize_test_profile_row(sample_row)
    if len(serialized) != len(_columns_by_table()["analytics.test_profile"]):
        pytest.fail("Serialized tuple length mismatch for test_profile.")
