"""Tests for IDE handlers following the unified handler pattern."""

from __future__ import annotations

import pytest

from codeintel.cli.handlers.ide import IdeHintsResult, ide_hints_handler
from codeintel.cli.services.params import ParamError
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.serving_contexts import ProvisionedServiceContext
from tests.cli.handlers.conftest import CommandContextBuilder_

HTTP_NOT_FOUND = 404
KNOWN_REL_PATH = "pkg/mod.py"


def test_ide_hints_handler_returns_ok_when_hints_found(
    handler_service_context: ProvisionedServiceContext,
    handler_context_builder: CommandContextBuilder_,
) -> None:
    """Handler returns success result when hints are found."""
    ctx = handler_context_builder(
        handler_service_context, "ide.hints", {"rel_path": KNOWN_REL_PATH}
    )
    result = ide_hints_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, IdeHintsResult)
    if result.data is not None:
        expect_equal(result.data.rel_path, KNOWN_REL_PATH)
        expect_true(len(result.data.hints) >= 0)


def test_ide_hints_handler_returns_fail_when_no_hints(
    handler_service_context: ProvisionedServiceContext,
    handler_context_builder: CommandContextBuilder_,
) -> None:
    """Handler returns failure result when no hints are found."""
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


def test_ide_hints_handler_raises_when_rel_path_missing(
    handler_service_context: ProvisionedServiceContext,
    handler_context_builder: CommandContextBuilder_,
) -> None:
    """Handler raises ParamError when rel_path is missing."""
    ctx = handler_context_builder(handler_service_context, "ide.hints", {})

    with pytest.raises(ParamError, match="Required parameter 'rel_path' not provided"):
        ide_hints_handler(ctx)


def test_ide_hints_handler_raises_when_rel_path_empty(
    handler_service_context: ProvisionedServiceContext,
    handler_context_builder: CommandContextBuilder_,
) -> None:
    """Handler raises ValueError when rel_path is empty after strip."""
    ctx = handler_context_builder(
        handler_service_context,
        "ide.hints",
        {"rel_path": "  "},
    )

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
