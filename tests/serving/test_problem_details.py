"""Problem detail conversions across serving transports."""

from __future__ import annotations

from typing import cast

import pytest

from codeintel.serving.mcp import errors as mcp_errors
from codeintel.serving.mcp import tool_utils
from codeintel.serving.mcp.models import ProblemDetail as ProblemDetailModel
from codeintel.serving.services.errors import DatasetNotFoundError


def _require(*, condition: bool, message: str) -> None:
    """Fail the test when a condition is false."""
    if not condition:
        pytest.fail(message)


def test_dataset_not_found_round_trip_preserves_code_and_context() -> None:
    """Domain → transport → domain preserves codes and extras."""
    detail = DatasetNotFoundError.for_name("unknown-dataset").detail

    model = ProblemDetailModel.from_domain(detail)
    _require(condition=model.code == "dataset-not-found", message="Code not preserved")
    _require(condition=model.extras == {"dataset": "unknown-dataset"}, message="Extras lost")

    domain_roundtrip = model.to_domain()
    _require(condition=domain_roundtrip.code == detail.code, message="Code changed on round trip")
    _require(
        condition=domain_roundtrip.extras == detail.extras, message="Extras changed on round trip"
    )


def test_wrap_serializes_mcp_error_to_problem_detail_payload() -> None:
    """MCP wrapper emits a serialized problem payload for errors."""
    dataset_error = DatasetNotFoundError.for_name("missing-dataset")

    @tool_utils.wrap_tool
    def _tool() -> object:
        raise mcp_errors.McpError(dataset_error.detail)

    payload = _tool()
    _require(condition=isinstance(payload, dict), message="Wrapper must return a mapping")
    payload_dict = cast("dict[str, object]", payload)
    error = payload_dict.get("error")
    _require(condition=isinstance(error, dict), message="Error payload missing")
    error_dict = cast("dict[str, object]", error)
    _require(condition=error_dict["code"] == "dataset-not-found", message="Code mismatch")
    _require(
        condition=cast("str", error_dict["type"]).endswith("/dataset-not-found"),
        message="Problem type not propagated",
    )
    _require(
        condition=cast("dict[str, object]", error_dict["extras"])["dataset"] == "missing-dataset",
        message="Dataset context missing",
    )
