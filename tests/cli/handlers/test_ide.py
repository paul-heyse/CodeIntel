"""Tests for IDE handlers following the unified handler pattern."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from codeintel.cli.config.model import CliConfig
from codeintel.cli.handlers.ide import IdeHintsResult, ide_hints_handler
from codeintel.cli.handlers.protocol import EnhancedHandlerContext
from codeintel.cli.resolution.types import ResolvedRuntime
from codeintel.config.serving_models import ServingConfig
from codeintel.serving.mcp.models import FileHintsResponse, ResponseMeta, ViewRow
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
)

HTTP_NOT_FOUND = 404


def test_ide_hints_handler_returns_ok_when_hints_found() -> None:
    """Handler returns success result when hints are found."""
    hint_row = ViewRow.model_validate(
        {
            "rel_path": "pkg/mod.py",
            "module": "pkg.mod",
            "subsystem_id": "core",
            "subsystem_name": "Core",
            "role": "model",
        }
    )
    mock_response = FileHintsResponse(
        found=True,
        hints=[hint_row],
        meta=ResponseMeta(),
    )

    with _mock_backend_returning(mock_response):
        ctx = _build_test_context(params={"rel_path": "pkg/mod.py"})
        result = ide_hints_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, IdeHintsResult)
    if result.data is not None:
        expect_equal(result.data.rel_path, "pkg/mod.py")
        expect_equal(len(result.data.hints), 1)


def test_ide_hints_handler_returns_fail_when_no_hints() -> None:
    """Handler returns failure result when no hints are found."""
    mock_response = FileHintsResponse(
        found=False,
        hints=[],
        meta=ResponseMeta(),
    )

    with _mock_backend_returning(mock_response):
        ctx = _build_test_context(params={"rel_path": "missing.py"})
        result = ide_hints_handler(ctx)

    expect_true(not result.success)
    expect_is_not_none(result.error)
    if result.error is not None:
        expect_equal(result.error.status, HTTP_NOT_FOUND)
        expect_is_not_none(result.error.detail)
        if result.error.detail is not None:
            expect_true("missing.py" in result.error.detail)


def test_ide_hints_handler_raises_when_rel_path_missing() -> None:
    """Handler raises ValueError when rel_path is missing."""
    mock_response = FileHintsResponse(found=True, hints=[], meta=ResponseMeta())

    with _mock_backend_returning(mock_response):
        ctx = _build_test_context(params={})

        with pytest.raises(ValueError, match="rel_path parameter is required"):
            ide_hints_handler(ctx)


def test_ide_hints_handler_raises_when_rel_path_empty() -> None:
    """Handler raises ValueError when rel_path is empty."""
    mock_response = FileHintsResponse(found=True, hints=[], meta=ResponseMeta())

    with _mock_backend_returning(mock_response):
        ctx = _build_test_context(params={"rel_path": "  "})

        with pytest.raises(ValueError, match="rel_path cannot be empty"):
            ide_hints_handler(ctx)


def test_ide_hints_result_to_dict() -> None:
    """Result to_dict returns expected structure."""
    result = IdeHintsResult(
        rel_path="pkg/mod.py",
        hints=[{"module": "pkg.mod", "subsystem_id": "core"}],
        meta={"total_count": 1},
    )

    data = result.to_dict()

    expect_equal(data["rel_path"], "pkg/mod.py")
    expect_equal(data["hints"], [{"module": "pkg.mod", "subsystem_id": "core"}])
    expect_equal(data["meta"], {"total_count": 1})


@contextmanager
def _mock_backend_returning(response: FileHintsResponse) -> Iterator[None]:
    """Create patch for build_backend_resource.

    Parameters
    ----------
    response
        Response to return from get_file_hints.

    Yields
    ------
    None
        Context manager for patch.
    """
    mock_backend = MagicMock()
    mock_backend.get_file_hints.return_value = response
    mock_resource = MagicMock()
    mock_resource.backend = mock_backend

    with patch(
        "codeintel.cli.handlers.ide.build_backend_resource",
        return_value=mock_resource,
    ):
        yield


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
        _operation_name="ide.hints",
    )
