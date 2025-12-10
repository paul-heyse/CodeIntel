"""Tests for storage handlers following the unified handler pattern."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from codeintel.cli.config.model import CliConfig
from codeintel.cli.handlers.protocol import EnhancedHandlerContext
from codeintel.cli.handlers.storage import (
    GenerateMacrosResult,
    MacroRequirement,
    ProfileStorageResult,
    ValidateMacrosResult,
    generate_macros_handler,
    profile_storage_handler,
    validate_macros_handler,
)
from codeintel.cli.resolution.types import ResolvedRuntime
from codeintel.config.serving_models import ServingConfig
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
)

HTTP_BAD_REQUEST = 400


def test_validate_macros_handler_returns_ok_when_valid() -> None:
    """Handler returns success when macros are valid."""
    with _mock_storage_gateway() as mock_gateway:
        mock_gateway.con = MagicMock()
        ctx = _build_test_context(params={"macro_requirement": MacroRequirement.REQUIRE})

        with (
            patch("codeintel.cli.handlers.storage.open_gateway", return_value=mock_gateway),
            patch("codeintel.cli.handlers.storage._assert_macro_coverage"),
            patch("codeintel.cli.handlers.storage.validate_macro_registry"),
            patch("codeintel.cli.handlers.storage.validate_dataset_schema_registry"),
            patch("codeintel.cli.handlers.storage.validate_normalized_macro_schemas"),
            patch(
                "codeintel.cli.handlers.storage.ingest_macro_coverage", return_value=([], ["test"])
            ),
            patch("codeintel.cli.handlers.storage.dataset_rows_only_entries", return_value=[]),
        ):
            result = validate_macros_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, ValidateMacrosResult)
    if result.data is not None:
        expect_equal(result.data.status, "valid")


def test_generate_macros_handler_fails_when_no_tables() -> None:
    """Handler returns error when no tables provided."""
    ctx = _build_test_context(params={})

    result = generate_macros_handler(ctx)

    expect_true(not result.success)
    expect_is_not_none(result.error)
    if result.error is not None:
        expect_equal(result.error.status, HTTP_BAD_REQUEST)


def test_generate_macros_handler_returns_ok_with_tables() -> None:
    """Handler returns success when tables are provided."""
    mock_macro = MagicMock()
    mock_macro.macro_name = "test_macro"
    mock_macro.ddl = "CREATE TABLE test()"

    ctx = _build_test_context(params={"tables": ["test_table"]})

    with patch("codeintel.cli.handlers.storage.render_macro", return_value=mock_macro):
        result = generate_macros_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, GenerateMacrosResult)
    if result.data is not None:
        expect_equal(result.data.count, 1)


def test_profile_storage_handler_fails_when_no_output_dir() -> None:
    """Handler returns error when output_dir not provided."""
    ctx = _build_test_context(params={})

    result = profile_storage_handler(ctx)

    expect_true(not result.success)
    expect_is_not_none(result.error)
    if result.error is not None:
        expect_equal(result.error.status, HTTP_BAD_REQUEST)


def test_profile_storage_handler_returns_ok(tmp_path: Path) -> None:
    """Handler returns success when output_dir is provided."""
    output_dir = tmp_path / "profile"
    include_views = True
    ctx = _build_test_context(
        params={
            "output_dir": str(output_dir),
            "include_views": include_views,
        }
    )

    with patch("codeintel.cli.handlers.storage.run_profile"):
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


@contextmanager
def _mock_storage_gateway() -> Iterator[MagicMock]:
    """Create mock storage gateway.

    Yields
    ------
    MagicMock
        Mock gateway.
    """
    mock = MagicMock(spec=StorageGateway)
    mock.con = MagicMock()
    mock.close = MagicMock()
    yield mock


def _build_test_context(
    params: dict[str, object],
) -> EnhancedHandlerContext:
    """Build a test context with mocked dependencies.

    Parameters
    ----------
    params
        Handler parameters.

    Returns
    -------
    EnhancedHandlerContext
        Test context.
    """
    mock_serving = MagicMock(spec=ServingConfig)
    mock_runtime = MagicMock(spec=ResolvedRuntime)
    mock_runtime.serving = mock_serving
    mock_runtime.paths = MagicMock()
    mock_runtime.paths.db_path = Path("build/test.duckdb")
    mock_config = MagicMock(spec=CliConfig)
    mock_gateway = MagicMock(spec=StorageGateway)
    mock_graph_runtime = MagicMock()

    return EnhancedHandlerContext(
        config=mock_config,
        runtime=mock_runtime,
        params=params,
        verbosity=0,
        _gateway=mock_gateway,
        _graph_runtime=mock_graph_runtime,
        _operation_name="storage.test",
    )
