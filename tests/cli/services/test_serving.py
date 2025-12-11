"""Tests for ServingService."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from codeintel.cli.services.serving import ServingError, ServingService
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_true,
)

# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def test_init() -> None:
    """Create service with runtime and storage."""
    runtime = MagicMock()
    storage = MagicMock()
    service = ServingService(runtime, storage)
    expect_true(service.runtime_service is runtime)
    expect_true(service.storage_service is storage)


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------


def test_invoke_unknown_operation() -> None:
    """Raise ServingError for unknown operation."""
    runtime = MagicMock()
    storage = MagicMock()
    service = ServingService(runtime, storage)

    with patch("codeintel.cli.services.serving.get_operation") as mock_get:
        mock_get.return_value = None

        with pytest.raises(ServingError) as exc_info:
            service.invoke("unknown.operation", {})

        expect_equal(exc_info.value.operation_id, "unknown.operation")
        expect_in("Unknown", str(exc_info.value))


def test_invoke_runs_prereqs_by_default() -> None:
    """Run prerequisites unless skipped."""
    runtime = MagicMock()
    runtime.runtime.snapshot = MagicMock()
    runtime.runtime.paths = MagicMock()
    runtime.runtime.tools = MagicMock()
    runtime.runtime.serving = MagicMock()

    storage = MagicMock()

    with (
        patch("codeintel.cli.services.serving.get_operation") as mock_get_op,
        patch("codeintel.cli.services.serving.run_operation_prereqs") as mock_prereqs,
        patch("codeintel.cli.services.serving.build_service_stack") as mock_stack,
    ):
        mock_op = MagicMock()
        mock_op.backend_method = "test_method"
        mock_get_op.return_value = mock_op

        mock_service = MagicMock()
        mock_service.test_method.return_value = {"result": "ok"}
        mock_stack.return_value.service = mock_service

        service = ServingService(runtime, storage)
        service.invoke("test.op", {"param": "value"})

        mock_prereqs.assert_called_once()


def test_invoke_skips_prereqs_when_requested() -> None:
    """Skip prerequisites when skip_prereqs=True."""
    runtime = MagicMock()
    runtime.runtime.serving = MagicMock()

    storage = MagicMock()

    with (
        patch("codeintel.cli.services.serving.get_operation") as mock_get_op,
        patch("codeintel.cli.services.serving.run_operation_prereqs") as mock_prereqs,
        patch("codeintel.cli.services.serving.build_service_stack") as mock_stack,
    ):
        mock_op = MagicMock()
        mock_op.backend_method = "test_method"
        mock_get_op.return_value = mock_op

        mock_service = MagicMock()
        mock_service.test_method.return_value = {"result": "ok"}
        mock_stack.return_value.service = mock_service

        service = ServingService(runtime, storage)
        service.invoke("test.op", {"param": "value"}, skip_prereqs=True)

        mock_prereqs.assert_not_called()


def test_invoke_backend_method_not_found() -> None:
    """Raise ServingError when backend method missing."""
    runtime = MagicMock()
    runtime.runtime.serving = MagicMock()

    storage = MagicMock()

    with (
        patch("codeintel.cli.services.serving.get_operation") as mock_get_op,
        patch("codeintel.cli.services.serving.run_operation_prereqs"),
        patch("codeintel.cli.services.serving.build_service_stack") as mock_stack,
    ):
        mock_op = MagicMock()
        mock_op.backend_method = "nonexistent_method"
        mock_get_op.return_value = mock_op

        mock_service = MagicMock(spec=[])
        mock_stack.return_value.service = mock_service

        service = ServingService(runtime, storage)

        with pytest.raises(ServingError) as exc_info:
            service.invoke("test.op", {})

        expect_in("Backend method not found", str(exc_info.value))


# ---------------------------------------------------------------------------
# Batch invocation
# ---------------------------------------------------------------------------


def test_invoke_batch_success() -> None:
    """Batch invoke returns success results."""
    runtime = MagicMock()
    runtime.runtime.serving = MagicMock()
    storage = MagicMock()

    with (
        patch("codeintel.cli.services.serving.get_operation") as mock_get_op,
        patch("codeintel.cli.services.serving.run_operation_prereqs"),
        patch("codeintel.cli.services.serving.build_service_stack") as mock_stack,
    ):
        mock_op = MagicMock()
        mock_op.backend_method = "test_method"
        mock_get_op.return_value = mock_op

        mock_service = MagicMock()
        mock_service.test_method.return_value = {"result": "ok"}
        mock_stack.return_value.service = mock_service

        service = ServingService(runtime, storage)
        results = service.invoke_batch(
            "test.op",
            [{"id": 1}, {"id": 2}],
        )

        expect_equal(len(results), 2)
        expect_true(all(r["success"] for r in results))


def test_invoke_batch_partial_failure() -> None:
    """Batch invoke captures individual failures."""
    runtime = MagicMock()
    storage = MagicMock()

    with patch("codeintel.cli.services.serving.get_operation") as mock_get_op:
        mock_get_op.return_value = None

        service = ServingService(runtime, storage)
        results = service.invoke_batch(
            "unknown.op",
            [{"id": 1}, {"id": 2}],
        )

        expect_equal(len(results), 2)
        expect_true(not any(r["success"] for r in results))
        expect_true(all("error" in r for r in results))


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_serialize_pydantic_model() -> None:
    """Serialize Pydantic model to dict."""
    mock_model = MagicMock()
    mock_model.model_dump.return_value = {"field": "value"}

    result = ServingService.serialize_result(mock_model)
    expect_equal(result, {"field": "value"})


def test_serialize_dataclass() -> None:
    """Serialize object with __dict__."""

    class DataLike:
        def __init__(self) -> None:
            self.field = "value"

    obj = DataLike()
    result = ServingService.serialize_result(obj)
    expect_equal(result, {"field": "value"})


def test_serialize_primitive() -> None:
    """Wrap primitive in value dict."""
    result = ServingService.serialize_result("simple")
    expect_equal(result, {"value": "simple"})
