"""Tests for storage handlers following the unified handler pattern."""

from __future__ import annotations

from pathlib import Path

from codeintel.cli.handlers.storage import (
    GenerateMacrosResult,
    MacroRequirement,
    ProfileStorageResult,
    ValidateMacrosResult,
    generate_macros_handler,
    profile_storage_handler,
    validate_macros_handler,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
)
from tests.cli.handlers.conftest import StorageHandlerHarness

HTTP_BAD_REQUEST = 400


def test_validate_macros_handler_returns_ok_when_valid(
    storage_macro_harness_fixture: StorageHandlerHarness,
) -> None:
    """Handler returns success when macros are valid."""
    with storage_macro_harness_fixture.command_context(
        {"macro_requirement": MacroRequirement.REQUIRE}
    ) as ctx:
        result = validate_macros_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, ValidateMacrosResult)
    if result.data is not None:
        expect_equal(result.data.status, "valid")


def test_generate_macros_handler_fails_when_no_tables(
    storage_macro_harness_fixture: StorageHandlerHarness,
) -> None:
    """Handler returns error when no tables provided."""
    with storage_macro_harness_fixture.command_context({}) as ctx:
        result = generate_macros_handler(ctx)

    expect_true(not result.success)
    expect_is_not_none(result.error)
    if result.error is not None:
        expect_equal(result.error.status, HTTP_BAD_REQUEST)


def test_generate_macros_handler_returns_ok_with_tables(
    storage_macro_harness_fixture: StorageHandlerHarness,
) -> None:
    """Handler returns success when tables are provided."""
    with storage_macro_harness_fixture.command_context({"tables": ["test_table"]}) as ctx:
        result = generate_macros_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, GenerateMacrosResult)
    if result.data is not None:
        expect_equal(result.data.count, 1)


def test_profile_storage_handler_fails_when_no_output_dir(
    storage_macro_harness_fixture: StorageHandlerHarness,
) -> None:
    """Handler returns error when output_dir not provided."""
    with storage_macro_harness_fixture.command_context({}) as ctx:
        result = profile_storage_handler(ctx)

    expect_true(not result.success)
    expect_is_not_none(result.error)
    if result.error is not None:
        expect_equal(result.error.status, HTTP_BAD_REQUEST)


def test_profile_storage_handler_returns_ok(
    tmp_path: Path, storage_macro_harness_fixture: StorageHandlerHarness
) -> None:
    """Handler returns success when output_dir is provided."""
    output_dir = tmp_path / "profile"
    include_views = True
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "output_dir": str(output_dir),
        "include_views": include_views,
        "db_path": str(storage_macro_harness_fixture.db_path),
    }
    with storage_macro_harness_fixture.command_context(params) as ctx:
        result = profile_storage_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, ProfileStorageResult)
    if result.data is not None:
        expect_equal(result.data.include_views, include_views)


def test_validate_macros_result_to_dict() -> None:
    """ValidateMacrosResult.to_dict returns expected structure."""
    result = ValidateMacrosResult(
        status="valid",
        missing_ingest=[],
        present_ingest=["test"],
        dataset_rows_only=[],
    )

    data = result.to_dict()

    expect_equal(data["status"], "valid")
    expect_equal(data["missing_ingest"], [])
    expect_equal(data["present_ingest"], ["test"])


def test_generate_macros_result_to_dict() -> None:
    """GenerateMacrosResult.to_dict returns expected structure."""
    result = GenerateMacrosResult(
        macros=[{"macro_name": "test", "ddl": "CREATE"}],
        count=1,
    )

    data = result.to_dict()

    expect_equal(data["count"], 1)
    expect_equal(len(result.macros), 1)


def test_profile_storage_result_to_dict() -> None:
    """ProfileStorageResult.to_dict returns expected structure."""
    result = ProfileStorageResult(
        db_path="/path/to/db",
        output_dir="/path/to/output",
        include_views=True,
    )

    data = result.to_dict()

    expect_equal(data["db_path"], "/path/to/db")
    expect_equal(data["output_dir"], "/path/to/output")
    expect_true(data["include_views"])
