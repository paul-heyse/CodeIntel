"""Tests for MCP backend surface helpers and service forwarding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.serving.mcp import backend as mcp_backend
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.backend import DatasetBackendMixin
from codeintel.serving.mcp.models import DatasetDescriptor
from codeintel.serving.services.errors import ProblemDetail
from tests._helpers import FakeQueryService, RecordingAsyncClient
from tests._helpers.assertions import expect_equal, expect_in, expect_true

if TYPE_CHECKING:
    from codeintel.serving.services.query_service import QueryService
    from tests._helpers import ModelLike


@pytest.mark.anyio
async def test_async_get_and_close_use_client_methods() -> None:
    """Async helpers should delegate to client.get/aclose and be reusable."""
    client = RecordingAsyncClient()
    get_async = mcp_backend.__dict__["_get_async"]
    response = await get_async(client, "/path", {"q": "1"})
    expect_true(response.json()["ok"] is True)
    expect_equal(client.get_calls, [("/path", {"q": "1"})])

    aclose_client = mcp_backend.__dict__["_aclose_client"]
    await aclose_client(client)
    await aclose_client(client)
    expect_equal(client.closed_count, 2)


class _DatasetService:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.include_bad = False

    def list_datasets(self) -> list[object]:
        self.calls.append("list_datasets")
        datasets: list[object] = [
            DatasetDescriptor(
                name="docs.functions",
                table="docs.v_functions",
                description="fn docs",
                family="docs",
            ),
            _DatasetDataclass(
                name="analytics.risks",
                table="analytics.goid_risk_factors",
                description="risk",
                family="analytics",
            ),
            _DatasetModel(value="model_dump"),
        ]
        if self.include_bad:
            datasets.append(_BadDataset())
        return datasets

    def enable_bad_dataset(self) -> None:
        self.include_bad = True


@dataclass
class _DatasetDataclass:
    name: str
    table: str
    description: str
    family: str | None = None


class _DatasetModel:
    def __init__(self, value: str) -> None:
        self.value = value

    def model_dump(self) -> dict[str, object]:
        return {
            "name": f"docs.{self.value}",
            "table": f"docs.v_{self.value}",
            "description": self.value,
            "family": "docs",
        }


class _BadDataset:
    def model_dump(self) -> dict[str, object]:
        _ = self
        message = "fail"
        raise errors.backend_failure(message)


class _DatasetBackend(DatasetBackendMixin):
    def __init__(self, service: object) -> None:
        self.service = cast("QueryService", service)


def test_list_datasets_serializes_mixed_inputs() -> None:
    """DatasetBackendMixin should normalize dataclasses, model_dump, and raw models."""
    service = _DatasetService()
    backend = _DatasetBackend(service)
    datasets = backend.list_datasets()
    expect_equal(
        {d.name for d in datasets},
        {
            "docs.functions",
            "analytics.risks",
            "docs.model_dump",
        },
    )

    def _raise_validation(*_: object, **__: object) -> DatasetDescriptor:
        message = "fail"
        raise errors.backend_failure(message)

    service.enable_bad_dataset()
    with pytest.raises(errors.McpError):
        _ = backend.list_datasets()


class _ForwardingBackend(DatasetBackendMixin):
    def __init__(self, service: FakeQueryService) -> None:
        self.service = cast("QueryService", service)

    def get_function_profile(self, *, goid_h128: int) -> dict[str, object]:
        result = cast("ModelLike", self.service.get_function_profile(goid_h128=goid_h128))
        return result.model_dump()

    def get_file_profile(self, *, rel_path: str) -> dict[str, object]:
        result = cast("ModelLike", self.service.get_file_profile(rel_path=rel_path))
        return result.model_dump()

    def get_module_profile(self, *, module: str) -> dict[str, object]:
        result = cast("ModelLike", self.service.get_module_profile(module=module))
        return result.model_dump()


def test_backend_forwards_service_calls_and_serializes_profiles() -> None:
    """Profile-facing backend methods should forward args and return serialized payloads."""
    service = FakeQueryService()
    backend = _ForwardingBackend(service)

    fn_profile = backend.get_function_profile(goid_h128=1)
    file_profile = backend.get_file_profile(rel_path="pkg/file.py")
    mod_profile = backend.get_module_profile(module="pkg.mod")

    expect_equal(fn_profile["value"], "pkg.func.1")
    expect_equal(file_profile["value"], "pkg/file.py")
    expect_equal(mod_profile["value"], "pkg.mod")
    expect_in(("get_function_profile", {"goid_h128": 1}), service.calls)
    expect_in(("get_file_profile", {"rel_path": "pkg/file.py"}), service.calls)
    expect_in(("get_module_profile", {"module": "pkg.mod"}), service.calls)


def test_dataset_rows_and_schema_forwarding() -> None:
    """Dataset helpers should serialize domain rows/schema and propagate errors."""
    service = FakeQueryService()
    backend = _ForwardingBackend(service)

    rows = backend.read_dataset_rows(dataset_name="docs.functions", limit=2, offset=1)
    expect_equal(rows.dataset, "docs.functions")
    expect_equal(rows.dataset_name, "docs.functions")
    expect_equal(rows.limit, 2)
    expect_equal(rows.offset, 1)
    expect_equal(rows.rows[0].model_dump()["goid"], 1)

    schema = backend.dataset_schema(dataset_name="docs.functions", sample_limit=1)
    expect_equal(schema.dataset, "docs.functions")
    expect_equal(schema.table_key, "docs.v_functions")
    expect_equal(schema.sample_rows[0].model_dump()["goid"], 1)

    service.enable_rows_failure()
    with pytest.raises(errors.McpError):
        backend.read_dataset_rows(dataset_name="docs.functions", limit=1, offset=0)


def test_backend_raises_problem_detail_on_service_errors() -> None:
    """Backend should propagate service ProblemErrors as McpError."""

    class _BadService(FakeQueryService):
        def get_function_profile(self, *, goid_h128: int) -> ModelLike:
            _ = self
            _ = goid_h128
            detail = ProblemDetail(type="t", title="bad", detail="fail", status=500, code="boom")
            raise errors.McpError(detail)

    backend = _ForwardingBackend(_BadService())
    with pytest.raises(errors.McpError):
        _ = backend.get_function_profile(goid_h128=2)
