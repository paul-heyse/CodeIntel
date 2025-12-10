"""Contract tests for CliResult.

These tests ensure CliResult maintains its contract across changes:
- Serialization format is stable
- Error structure matches RFC 9457
- Success/failure semantics are consistent
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from codeintel.cli.errors import ProblemDetail
from codeintel.cli.core import CliResult
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_none,
    expect_is_not_none,
    expect_true,
)

# Test constants
TEST_COUNT = 42
METADATA_VALUE = 123
STATUS_SUCCESS = 400
STATUS_ERROR = 500
WARNING_COUNT = 2


def test_success_result_has_data() -> None:
    """Success results must have data attribute."""
    result = CliResult.ok({"key": "value"})

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_none(result.error)


def test_error_result_has_problem_detail() -> None:
    """Error results must have RFC 9457 Problem Detail."""
    error = ProblemDetail(
        type="urn:test:error",
        title="Test Error",
        detail="Details",
        status=STATUS_SUCCESS,
    )
    result: CliResult[dict[str, str]] = CliResult.fail(error)

    expect_false(result.success)
    expect_is_none(result.data)
    expect_is_not_none(result.error)
    if result.error is not None:
        expect_equal(result.error.type, "urn:test:error")


def test_json_serialization_success() -> None:
    """Success result JSON has required fields."""
    result = CliResult.ok({"count": TEST_COUNT})
    json_str = result.to_json()
    parsed = json.loads(json_str)

    expect_in("success", parsed)
    expect_true(parsed["success"])
    expect_in("data", parsed)
    expect_equal(parsed["data"]["count"], TEST_COUNT)


def test_json_serialization_error() -> None:
    """Error result JSON follows RFC 9457 structure."""
    error = ProblemDetail(
        type="urn:codeintel:cli:validation-error",
        title="Validation Failed",
        detail="Field 'name' is required",
        status=STATUS_SUCCESS,
        instance="/operations/test",
        extensions={"field": "name"},
    )
    result: CliResult[dict[str, str]] = CliResult.fail(error)
    json_str = result.to_json()
    parsed = json.loads(json_str)

    expect_false(parsed["success"])
    expect_in("error", parsed)
    err = parsed["error"]

    # RFC 9457 required fields
    expect_in("type", err)
    expect_in("title", err)

    # RFC 9457 optional fields
    expect_in("detail", err)
    expect_in("status", err)
    expect_in("instance", err)


def test_warnings_preserved() -> None:
    """Warnings are preserved in result."""
    result = CliResult(
        success=True,
        data={"data": 1},
        warnings=["Warning 1", "Warning 2"],
    )

    expect_equal(len(result.warnings), WARNING_COUNT)
    expect_in("Warning 1", result.warnings)


def test_metadata_preserved() -> None:
    """Metadata is preserved in result."""
    result = CliResult.ok(
        {"data": 1},
        metadata={"duration_ms": METADATA_VALUE},
    )

    expect_equal(result.metadata.get("duration_ms"), METADATA_VALUE)


def test_to_dict_structure() -> None:
    """to_dict returns expected structure."""
    result = CliResult.ok({"key": "value"})
    data = result.to_dict()

    expect_in("success", data)
    expect_in("data", data)
    expect_true(data["success"])


def test_error_to_dict_structure() -> None:
    """Error to_dict includes error field."""
    error = ProblemDetail(
        type="urn:test:error",
        title="Error",
        status=STATUS_ERROR,
    )
    result: CliResult[None] = CliResult.fail(error)
    data = result.to_dict()

    expect_in("success", data)
    expect_in("error", data)
    expect_false(data["success"])


def test_ok_classmethod_creates_success() -> None:
    """ok() classmethod creates successful result."""
    result = CliResult.ok("data")

    expect_true(result.success)
    expect_equal(result.data, "data")
    expect_is_none(result.error)
    expect_equal(result.warnings, [])


def test_fail_classmethod_creates_failure() -> None:
    """fail() classmethod creates failed result."""
    error = ProblemDetail(
        type="urn:test:error",
        title="Error",
        status=STATUS_ERROR,
    )
    result: CliResult[str] = CliResult.fail(error)

    expect_false(result.success)
    expect_is_none(result.data)
    expect_equal(result.error, error)


def test_fail_with_warnings() -> None:
    """fail() can include warnings."""
    error = ProblemDetail(
        type="urn:test:error",
        title="Error",
        status=STATUS_ERROR,
    )
    result: CliResult[str] = CliResult.fail(
        error,
        warnings=["Warning 1"],
    )

    expect_equal(len(result.warnings), 1)
    expect_equal(result.warnings[0], "Warning 1")


def test_json_serialization_roundtrip() -> None:
    """JSON serialization is stable."""
    result = CliResult.ok({"nested": {"key": "value"}, "list": [1, 2, 3]})
    json_str = result.to_json()
    parsed = json.loads(json_str)

    expect_true(parsed["success"])
    expect_equal(parsed["data"]["nested"]["key"], "value")
    expect_equal(parsed["data"]["list"], [1, 2, 3])


def test_dataclass_data_serialization() -> None:
    """Dataclass data is serialized correctly."""

    @dataclass
    class TestData:
        """Test data structure."""

        name: str
        value: int

        def to_dict(self) -> dict[str, object]:
            """Convert to dictionary for serialization.

            Returns
            -------
            dict[str, object]
                Dictionary representation.
            """
            return {"name": self.name, "value": self.value}

    test_value = 42
    result = CliResult.ok(TestData(name="test", value=test_value))
    data = result.to_dict()

    # Data should be serialized via to_dict
    expect_equal(data["data"], {"name": "test", "value": test_value})


def test_problem_detail_required_fields() -> None:
    """ProblemDetail requires type, title, status."""
    error = ProblemDetail(
        type="urn:test:error",
        title="Test Error",
        status=STATUS_SUCCESS,
    )

    expect_equal(error.type, "urn:test:error")
    expect_equal(error.title, "Test Error")
    expect_equal(error.status, STATUS_SUCCESS)


def test_problem_detail_optional_fields() -> None:
    """ProblemDetail optional fields have sensible defaults."""
    error = ProblemDetail(
        type="urn:test:error",
        title="Test Error",
        status=STATUS_SUCCESS,
    )

    expect_is_none(error.detail)
    expect_is_none(error.instance)
    # extensions should have a default
    expect_true(isinstance(error.extensions, dict))


def test_full_problem_detail() -> None:
    """Full ProblemDetail with all fields."""
    error = ProblemDetail(
        type="urn:test:error",
        title="Test Error",
        detail="Detailed explanation",
        status=STATUS_SUCCESS,
        instance="/path/to/resource",
        extensions={"extra": "data"},
    )

    expect_equal(error.type, "urn:test:error")
    expect_equal(error.title, "Test Error")
    expect_equal(error.detail, "Detailed explanation")
    expect_equal(error.status, STATUS_SUCCESS)
    expect_equal(error.instance, "/path/to/resource")
    expect_equal(error.extensions, {"extra": "data"})


def test_to_dict_rfc9457_compliance() -> None:
    """to_dict follows RFC 9457 format."""
    error = ProblemDetail(
        type="urn:test:error",
        title="Test Error",
        detail="Details",
        status=STATUS_SUCCESS,
        instance="/test",
        extensions={"field": "value"},
    )
    data = error.to_dict()

    # RFC 9457 field names
    expect_in("type", data)
    expect_in("title", data)
    expect_in("detail", data)
    expect_in("status", data)
    expect_in("instance", data)

    # Extensions are flattened into the dict per RFC 9457
    # or may be nested under "extensions"
    has_field = "field" in data or ("extensions" in data and "field" in data["extensions"])
    expect_true(has_field)
