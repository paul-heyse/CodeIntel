"""Multi-table pipeline template tests.

This module validates that the multi-table pipeline utilities in
``codeintel.build.hamilton.templates.multi_table_pipeline`` correctly combine
multiple materialization results and that the row extractor factory works.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from codeintel.build.contracts import OutputContract
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.templates.multi_table_pipeline import (
    create_row_extractor,
    multi_table_record,
)
from codeintel.build.targets import OutputTarget, TargetGraph
from tests._helpers.assertions import assert_record_row_counts, assert_target_ok
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_none,
    expect_true,
)
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness


def _make_env(harness: HamiltonBuildHarness) -> BuildEnv:
    """Create a BuildEnv for testing.

    Returns
    -------
    BuildEnv
        Build environment configured for testing.
    """
    return replace(harness.build_env(), force_targets=frozenset({"function_metrics"}))


def _make_graph() -> TargetGraph:
    """Create a minimal TargetGraph with function_metrics target.

    Returns
    -------
    TargetGraph
        Target graph with function_metrics target registered.
    """
    graph = TargetGraph()
    graph.register(
        OutputTarget(
            name="function_metrics",
            module="analytics",
            contract=OutputContract.simple(
                table_keys=(
                    "analytics.function_metrics",
                    "analytics.function_types",
                    "analytics.function_validation",
                )
            ),
        )
    )
    return graph


@dataclass
class MaterializationConfig:
    """Configuration for creating materialization metadata dicts."""

    status: str
    table_key: str
    row_count: int | None = None
    input_hash: str = "hash123"
    duration_ms: float = 100.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to materialization metadata dict.

        Returns
        -------
        dict[str, Any]
            Materialization metadata dict for testing.
        """
        result: dict[str, Any] = {
            "status": self.status,
            "table_key": self.table_key,
            "input_hash": self.input_hash,
            "duration_ms": self.duration_ms,
        }
        if self.row_count is not None:
            result["row_count"] = self.row_count
        if self.error is not None:
            result["error"] = self.error
        return result


def _materialization(
    status: str,
    table_key: str,
    row_count: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Create a materialization metadata dict for testing.

    Returns
    -------
    dict[str, Any]
        Materialization metadata dict.
    """
    return MaterializationConfig(
        status=status,
        table_key=table_key,
        row_count=row_count,
        error=error,
    ).to_dict()


def test_multi_table_record_all_succeeded(build_harness: HamiltonBuildHarness) -> None:
    """Verify multi_table_record produces succeeded when all tables succeed."""
    env = _make_env(build_harness)
    graph = _make_graph()

    materializations = {
        "analytics.function_metrics": _materialization(
            status="succeeded",
            table_key="analytics.function_metrics",
            row_count=100,
        ),
        "analytics.function_types": _materialization(
            status="succeeded",
            table_key="analytics.function_types",
            row_count=100,
        ),
        "analytics.function_validation": _materialization(
            status="succeeded",
            table_key="analytics.function_validation",
            row_count=5,
        ),
    }

    record = multi_table_record(env, graph, "function_metrics", materializations)

    assert_target_ok(record)
    expect_equal(record.target, expected="function_metrics", label="record.target")
    assert_record_row_counts(
        record,
        {
            "analytics.function_metrics": 100,
            "analytics.function_types": 100,
            "analytics.function_validation": 5,
        },
    )


def test_multi_table_record_partial_failure(build_harness: HamiltonBuildHarness) -> None:
    """Verify multi_table_record produces failed when one table fails."""
    env = _make_env(build_harness)
    graph = _make_graph()

    materializations = {
        "analytics.function_metrics": _materialization(
            status="succeeded",
            table_key="analytics.function_metrics",
            row_count=100,
        ),
        "analytics.function_types": _materialization(
            status="failed",
            table_key="analytics.function_types",
            error="Write failed: disk full",
        ),
        "analytics.function_validation": _materialization(
            status="succeeded",
            table_key="analytics.function_validation",
            row_count=5,
        ),
    }

    record = multi_table_record(env, graph, "function_metrics", materializations)

    assert_target_ok(record, expected_status="failed")
    expect_true(
        record.error is not None and "Write failed" in record.error,
        message=f"Expected error message containing 'Write failed', got: {record.error}",
    )


def test_multi_table_record_all_skipped(build_harness: HamiltonBuildHarness) -> None:
    """Verify multi_table_record produces skipped when all tables skipped."""
    env = _make_env(build_harness)
    graph = _make_graph()

    materializations = {
        "analytics.function_metrics": _materialization(
            status="skipped",
            table_key="analytics.function_metrics",
        ),
        "analytics.function_types": _materialization(
            status="skipped",
            table_key="analytics.function_types",
        ),
        "analytics.function_validation": _materialization(
            status="skipped",
            table_key="analytics.function_validation",
        ),
    }

    record = multi_table_record(env, graph, "function_metrics", materializations)

    assert_target_ok(record, expected_status="skipped")


def test_multi_table_record_mixed_success_skip(
    build_harness: HamiltonBuildHarness,
) -> None:
    """Verify multi_table_record handles mix of succeeded and skipped."""
    env = _make_env(build_harness)
    graph = _make_graph()

    materializations = {
        "analytics.function_metrics": _materialization(
            status="succeeded",
            table_key="analytics.function_metrics",
            row_count=100,
        ),
        "analytics.function_types": _materialization(
            status="skipped",
            table_key="analytics.function_types",
        ),
        "analytics.function_validation": _materialization(
            status="succeeded",
            table_key="analytics.function_validation",
            row_count=5,
        ),
    }

    record = multi_table_record(env, graph, "function_metrics", materializations)

    # Mixed succeeded/skipped should still be succeeded overall
    assert_target_ok(record)


# ============================================================================
# Row Extractor Factory Tests
# ============================================================================


@dataclass
class MockResult:
    """Mock compute result for testing row extraction."""

    metrics_rows: list[dict[str, object]] = field(default_factory=list)
    types_rows: list[dict[str, object]] = field(default_factory=list)
    items: list[object] = field(default_factory=list)


@dataclass
class MockItem:
    """Mock item for custom converter testing."""

    id: int
    name: str


def test_create_row_extractor_with_columns() -> None:
    """Verify create_row_extractor converts dict rows to tuples using columns."""
    columns = ("id", "name", "value")
    extractor = create_row_extractor("metrics_rows", columns=columns)

    result = MockResult(
        metrics_rows=[
            {"id": 1, "name": "a", "value": 10},
            {"id": 2, "name": "b", "value": 20},
        ]
    )

    rows = extractor(result)

    expect_true(rows is not None, message="Expected non-None rows")
    if rows is None:
        return  # Type narrowing; expect_true already failed if we reach this
    expect_equal(len(rows), expected=2, label="len(rows)")
    expect_equal(rows[0], expected=(1, "a", 10), label="rows[0]")
    expect_equal(rows[1], expected=(2, "b", 20), label="rows[1]")


def test_create_row_extractor_with_converter() -> None:
    """Verify create_row_extractor uses custom row converter."""
    extractor = create_row_extractor(
        "items",
        row_converter=lambda item: (item.id, item.name),
    )

    result = MockResult(
        items=[
            MockItem(id=1, name="first"),
            MockItem(id=2, name="second"),
        ]
    )

    rows = extractor(result)

    expect_true(rows is not None, message="Expected non-None rows")
    if rows is None:
        return  # Type narrowing; expect_true already failed if we reach this
    expect_equal(len(rows), expected=2, label="len(rows)")
    expect_equal(rows[0], expected=(1, "first"), label="rows[0]")
    expect_equal(rows[1], expected=(2, "second"), label="rows[1]")


def test_create_row_extractor_none_input() -> None:
    """Verify create_row_extractor returns None for None input."""
    columns = ("id", "name")
    extractor = create_row_extractor("metrics_rows", columns=columns)

    rows = extractor(None)

    expect_is_none(rows, label="rows for None input")


def test_create_row_extractor_empty_rows() -> None:
    """Verify create_row_extractor returns None for empty rows."""
    columns = ("id", "name")
    extractor = create_row_extractor("metrics_rows", columns=columns)

    result = MockResult(metrics_rows=[])

    rows = extractor(result)

    expect_is_none(rows, label="rows for empty input")


def test_create_row_extractor_missing_column() -> None:
    """Verify create_row_extractor handles missing columns gracefully."""
    columns = ("id", "name", "missing_col")
    extractor = create_row_extractor("metrics_rows", columns=columns)

    result = MockResult(
        metrics_rows=[
            {"id": 1, "name": "a"},  # missing_col not present
        ]
    )

    rows = extractor(result)

    expect_true(rows is not None, message="Expected non-None rows")
    if rows is None:
        return  # Type narrowing; expect_true already failed if we reach this
    expect_equal(len(rows), expected=1, label="len(rows)")
    # Missing column should produce None
    expect_equal(rows[0], expected=(1, "a", None), label="rows[0]")


def test_create_row_extractor_missing_attribute() -> None:
    """Verify create_row_extractor returns None for missing attribute."""
    columns = ("id", "name")
    extractor = create_row_extractor("nonexistent_attr", columns=columns)

    result = MockResult()

    rows = extractor(result)

    expect_is_none(rows, label="rows for missing attribute")
