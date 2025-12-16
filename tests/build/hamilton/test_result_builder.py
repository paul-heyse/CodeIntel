"""Tests for Hamilton result builders."""

from __future__ import annotations

import time
from typing import cast

import pytest

from codeintel.build.hamilton.result_builder import (
    BuildExecutionResult,
    BuildResultBuilder,
    DictResultBuilder,
    NodeResult,
    ResultStatus,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_instance,
    expect_true,
)


class TestResultStatus:
    """Test suite for ResultStatus enum."""

    @staticmethod
    def test_values() -> None:
        """Test enum values."""
        expect_equal(ResultStatus.SUCCESS.value, "success")
        expect_equal(ResultStatus.PARTIAL.value, "partial")
        expect_equal(ResultStatus.FAILED.value, "failed")
        expect_equal(ResultStatus.SKIPPED.value, "skipped")


class TestNodeResult:
    """Test suite for NodeResult dataclass."""

    @staticmethod
    def test_successful_result() -> None:
        """Test creating a successful node result."""
        result = NodeResult(
            node_name="test_node",
            value=42,
        )
        expect_equal(result.node_name, "test_node")
        expect_equal(result.value, 42)
        expect_equal(result.status, ResultStatus.SUCCESS)
        expect_true(result.is_success)
        expect_in("error_message", result.__dict__)
        expect_true(result.error_message is None)

    @staticmethod
    def test_failed_result() -> None:
        """Test creating a failed node result."""
        result = NodeResult(
            node_name="test_node",
            value=None,
            status=ResultStatus.FAILED,
            error_message="Something went wrong",
        )
        expect_false(result.is_success)
        expect_equal(result.error_message, "Something went wrong")

    @staticmethod
    def test_with_duration() -> None:
        """Test creating result with duration."""
        result = NodeResult(
            node_name="test_node",
            value="data",
            duration_seconds=1.5,
        )
        expect_equal(result.duration_seconds, 1.5)


class TestBuildExecutionResult:
    """Test suite for BuildExecutionResult dataclass."""

    @staticmethod
    def test_empty_result() -> None:
        """Test creating an empty result."""
        result = BuildExecutionResult(status=ResultStatus.SKIPPED)
        expect_equal(result.success_count, 0)
        expect_equal(result.failure_count, 0)
        expect_equal(result.node_results, {})

    @staticmethod
    def test_success_count() -> None:
        """Test success count calculation."""
        result = BuildExecutionResult(
            status=ResultStatus.SUCCESS,
            node_results={
                "a": NodeResult("a", 1),
                "b": NodeResult("b", 2),
                "c": NodeResult("c", None, status=ResultStatus.FAILED),
            },
        )
        expect_equal(result.success_count, 2)
        expect_equal(result.failure_count, 1)

    @staticmethod
    def test_get_output() -> None:
        """Test getting specific output."""
        result = BuildExecutionResult(
            status=ResultStatus.SUCCESS,
            node_results={
                "output1": NodeResult("output1", "value1"),
                "output2": NodeResult("output2", "value2"),
            },
        )
        expect_equal(result.get_output("output1"), "value1")
        expect_equal(result.get_output("output2"), "value2")

    @staticmethod
    def test_get_output_not_found() -> None:
        """Test getting non-existent output raises KeyError."""
        result = BuildExecutionResult(status=ResultStatus.SUCCESS)
        with pytest.raises(KeyError, match="not found"):
            result.get_output("nonexistent")

    @staticmethod
    def test_get_outputs() -> None:
        """Test getting all outputs."""
        result = BuildExecutionResult(
            status=ResultStatus.SUCCESS,
            node_results={
                "a": NodeResult("a", 1),
                "b": NodeResult("b", 2),
            },
        )
        outputs = result.get_outputs()
        expect_equal(outputs, {"a": 1, "b": 2})

    @staticmethod
    def test_summary() -> None:
        """Test generating summary."""
        result = BuildExecutionResult(
            status=ResultStatus.SUCCESS,
            node_results={
                "output1": NodeResult("output1", "value"),
            },
            total_duration_seconds=1.5,
            requested_outputs=["output1"],
        )
        summary = result.summary()
        expect_in("SUCCESS", summary)
        expect_in("1 succeeded", summary)
        expect_in("0 failed", summary)
        expect_in("1.50s", summary)
        expect_in("output1", summary)

    @staticmethod
    def test_summary_with_failures() -> None:
        """Test summary includes failure details."""
        result = BuildExecutionResult(
            status=ResultStatus.PARTIAL,
            node_results={
                "good": NodeResult("good", "value"),
                "bad": NodeResult(
                    "bad",
                    None,
                    status=ResultStatus.FAILED,
                    error_message="Error occurred",
                ),
            },
        )
        summary = result.summary()
        expect_in("Failed nodes:", summary)
        expect_in("bad", summary)
        expect_in("Error occurred", summary)

    @staticmethod
    def test_to_dict() -> None:
        """Test converting to dictionary."""
        result = BuildExecutionResult(
            status=ResultStatus.SUCCESS,
            node_results={
                "output": NodeResult("output", "value"),
            },
            total_duration_seconds=1.0,
            metadata={"key": "value"},
        )
        d = result.to_dict()
        expect_equal(d["status"], "success")
        expect_equal(d["success_count"], 1)
        expect_equal(d["failure_count"], 0)
        expect_equal(d["total_duration_seconds"], 1.0)
        expect_equal(d["metadata"], {"key": "value"})
        node_results = d["node_results"]
        expect_true(isinstance(node_results, dict))
        expect_in("output", cast("dict[str, object]", node_results))


class TestBuildResultBuilder:
    """Test suite for BuildResultBuilder."""

    @staticmethod
    def test_build_success() -> None:
        """Test building successful result."""
        builder = BuildResultBuilder()
        result = builder.build_result(output1="value1", output2="value2")

        expect_is_instance(result, BuildExecutionResult)
        expect_equal(result.status, ResultStatus.SUCCESS)
        expect_equal(result.success_count, 2)
        expect_equal(result.get_output("output1"), "value1")

    @staticmethod
    def test_build_with_failure() -> None:
        """Test building result with failure."""
        builder = BuildResultBuilder()
        error = ValueError("Test error")
        result = builder.build_result(good="value", bad=error)

        expect_equal(result.status, ResultStatus.PARTIAL)
        expect_equal(result.failure_count, 1)
        expect_equal(result.node_results["bad"].error_message, "Test error")

    @staticmethod
    def test_build_empty() -> None:
        """Test building empty result."""
        builder = BuildResultBuilder()
        result = builder.build_result()

        expect_equal(result.status, ResultStatus.SKIPPED)
        expect_equal(result.success_count, 0)

    @staticmethod
    def test_include_values_false() -> None:
        """Test excluding values from results."""
        builder = BuildResultBuilder(include_values=False)
        result = builder.build_result(output="large_data")

        expect_true(result.node_results["output"].value is None)
        expect_equal(result.status, ResultStatus.SUCCESS)

    @staticmethod
    def test_metadata() -> None:
        """Test including metadata."""
        metadata: dict[str, object] = {"run_id": "123", "version": "1.0"}
        builder = BuildResultBuilder(metadata=metadata)
        result = builder.build_result(output="value")

        expect_equal(result.metadata, metadata)

    @staticmethod
    def test_timing() -> None:
        """Test timing functionality."""
        builder = BuildResultBuilder()
        builder.start_timing()
        time.sleep(0.01)  # Small delay
        result = builder.build_result(output="value")

        expect_true(result.total_duration_seconds > 0)
        expect_true(result.start_time > 0)
        expect_true(result.end_time > result.start_time)

    @staticmethod
    def test_input_types() -> None:
        """Test input_types method."""
        builder = BuildResultBuilder()
        types = builder.input_types()
        expect_in(object, types)

    @staticmethod
    def test_output_type() -> None:
        """Test output_type method."""
        builder = BuildResultBuilder()
        expect_equal(builder.output_type(), BuildExecutionResult)


class TestDictResultBuilder:
    """Test suite for DictResultBuilder."""

    @staticmethod
    def test_build_result() -> None:
        """Test building dictionary result."""
        builder = DictResultBuilder()
        result = builder.build_result(a=1, b=2, c=3)

        expect_is_instance(result, dict)
        expect_equal(result, {"a": 1, "b": 2, "c": 3})

    @staticmethod
    def test_output_type() -> None:
        """Test output_type method."""
        builder = DictResultBuilder()
        expect_equal(builder.output_type(), dict)


@pytest.mark.parametrize(
    ("outputs", "expected_status"),
    [
        pytest.param({}, ResultStatus.SKIPPED, id="empty"),
        pytest.param({"a": 1}, ResultStatus.SUCCESS, id="single_success"),
        pytest.param(
            {"a": ValueError("error")},
            ResultStatus.FAILED,
            id="single_failure",
        ),
        pytest.param(
            {"a": 1, "b": ValueError("error")},
            ResultStatus.PARTIAL,
            id="mixed",
        ),
    ],
)
def test_result_status_determination(
    outputs: dict[str, object],
    expected_status: ResultStatus,
) -> None:
    """Parametrized test for status determination."""
    builder = BuildResultBuilder()
    result = builder.build_result(**outputs)
    expect_equal(result.status, expected_status)
