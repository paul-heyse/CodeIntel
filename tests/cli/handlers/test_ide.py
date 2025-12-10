"""Tests for IDE handlers following the unified handler pattern."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from codeintel.cli.handlers.context import ParameterError
from codeintel.cli.handlers.ide import IdeHintsResult, ide_hints_handler
from codeintel.serving.mcp.models import FileHintsResponse, ResponseMeta, ViewRow
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.serving_contexts import ProvisionedServiceContext
from tests.cli.handlers.conftest import HandlerContextBuilder

HTTP_NOT_FOUND = 404


def test_ide_hints_handler_returns_ok_when_hints_found(
    handler_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
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

    with patch.object(
        handler_service_context.backend,
        "get_file_hints",
        return_value=mock_response,
    ):
        ctx = handler_context_builder(
            handler_service_context, "ide.hints", {"rel_path": "pkg/mod.py"}
        )
        result = ide_hints_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, IdeHintsResult)
    if result.data is not None:
            expect_equal(result.data.rel_path, "pkg/mod.py")
            expect_equal(len(result.data.hints), 1)


def test_ide_hints_handler_returns_fail_when_no_hints(
    handler_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
    """Handler returns failure result when no hints are found."""
    mock_response = FileHintsResponse(
        found=False,
        hints=[],
        meta=ResponseMeta(),
    )

    with patch.object(
        handler_service_context.backend,
        "get_file_hints",
        return_value=mock_response,
    ):
        ctx = handler_context_builder(
            handler_service_context,
            "ide.hints",
            {"rel_path": "missing.py"},
        )
        result = ide_hints_handler(ctx)

    expect_true(not result.success)
    expect_is_not_none(result.error)
    if result.error is not None:
        expect_equal(result.error.status, HTTP_NOT_FOUND)
        expect_is_not_none(result.error.detail)
        if result.error.detail is not None:
            expect_true("missing.py" in result.error.detail)


def test_ide_hints_handler_raises_when_rel_path_missing(
    handler_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
    """Handler raises ParameterError when rel_path is missing."""
    mock_response = FileHintsResponse(found=True, hints=[], meta=ResponseMeta())

    with patch.object(
        handler_service_context.backend,
        "get_file_hints",
        return_value=mock_response,
    ):
        ctx = handler_context_builder(handler_service_context, "ide.hints", {})

        with pytest.raises(ParameterError, match="Required parameter 'rel_path' not provided"):
            ide_hints_handler(ctx)


def test_ide_hints_handler_raises_when_rel_path_empty(
    handler_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
    """Handler raises ValueError when rel_path is empty after strip."""
    mock_response = FileHintsResponse(found=True, hints=[], meta=ResponseMeta())

    with patch.object(
        handler_service_context.backend,
        "get_file_hints",
        return_value=mock_response,
    ):
        ctx = handler_context_builder(
            handler_service_context,
            "ide.hints",
            {"rel_path": "  "},
        )

        # The handler gets "  " as a string, which is non-empty but whitespace.
        # After stripping, it becomes empty and should raise ValueError.
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
