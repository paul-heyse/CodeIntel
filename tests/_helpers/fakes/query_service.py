"""Fakes for QueryService-like behavior in MCP backend tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from codeintel.serving.backend.pagination import BackendLimits
from codeintel.serving.backend.query_api import DuckDBQueryApi
from codeintel.serving.domain_models import DatasetRows, DatasetSchema, ResponseMeta
from codeintel.serving.services.errors import DatasetNotFoundError, ProblemDetail
from codeintel.storage.gateway import StorageGateway
from tests._helpers.gateway import GatewayFactory
from tests._helpers.serving_stubs import (
    HookedDatasetQueries,
    HookedFunctionQueries,
    HookedProfileQueries,
    HookedSubsystemQueries,
)


@dataclass
class ModelLike:
    """Pydantic-like stub with validation hooks."""

    value: str

    @classmethod
    def from_domain(cls, payload: object) -> ModelLike:
        return cls(value=str(payload))

    @classmethod
    def model_validate(cls, payload: Mapping[str, Any]) -> ModelLike:
        return cls(value=str(payload.get("value", payload)))

    def model_dump(self) -> dict[str, object]:
        return {"value": self.value}


class FakeQueryService:
    """Lightweight QueryService fake with deterministic payloads."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.limits = BackendLimits(default_limit=25, max_rows_per_call=100)
        self._rows_fail: bool = False

    def list_datasets(self) -> list[object]:
        self.calls.append(("list_datasets", {}))
        return [
            {"name": "docs.functions", "table": "docs.v_functions", "description": "fn docs"},
            ModelLike(value="model_dump"),
        ]

    def dataset_specs(self) -> list[object]:
        self.calls.append(("dataset_specs", {}))
        return [
            {
                "name": "docs.functions",
                "table_key": "docs.v_functions",
                "family": "docs",
                "is_view": True,
                "schema_columns": ["goid", "name"],
            }
        ]

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRows:
        self.calls.append(
            ("read_dataset_rows", {"dataset_name": dataset_name, "limit": limit, "offset": offset})
        )
        if self._rows_fail:
            detail = ProblemDetail(
                type="error",
                title="fail",
                detail="bad rows",
                status=400,
                code="bad",
            )
            raise DatasetNotFoundError(detail)
        meta = ResponseMeta(
            requested_limit=limit,
            applied_limit=limit or self.limits.default_limit,
            requested_offset=offset,
            applied_offset=offset,
            truncated=False,
        )
        return DatasetRows(
            dataset_name=dataset_name,
            limit=limit or self.limits.default_limit,
            offset=offset,
            rows=[{"goid": 1, "name": "fn"}, {"goid": 2, "name": "fn2"}],
            meta=meta,
        )

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> DatasetSchema:
        self.calls.append(
            ("dataset_schema", {"dataset_name": dataset_name, "sample_limit": sample_limit})
        )
        meta = ResponseMeta(
            requested_limit=sample_limit,
            applied_limit=sample_limit,
            requested_offset=0,
            applied_offset=0,
        )
        return DatasetSchema(
            dataset_name=dataset_name,
            table_key="docs.v_functions",
            duckdb_schema=[{"name": "goid", "type": "BIGINT", "nullable": False}],
            json_schema={"type": "object"},
            sample_rows=[{"goid": 1, "name": "fn"}],
            capabilities={"validation": True},
            owner="analytics",
            freshness_sla=None,
            retention_policy=None,
            schema_version="1",
            stable_id="stable-1",
            validation_profile="strict",
            meta=meta,
        )

    def enable_rows_failure(self) -> None:
        """Toggle failures for read_dataset_rows to simulate ProblemDetail errors."""
        self._rows_fail = True

    def get_function_profile(self, *, goid_h128: int) -> ModelLike:
        self.calls.append(("get_function_profile", {"goid_h128": goid_h128}))
        return ModelLike.from_domain(f"pkg.func.{goid_h128}")

    def get_file_profile(self, *, rel_path: str) -> ModelLike:
        self.calls.append(("get_file_profile", {"rel_path": rel_path}))
        return ModelLike.from_domain(rel_path)

    def get_module_profile(self, *, module: str) -> ModelLike:
        self.calls.append(("get_module_profile", {"module": module}))
        return ModelLike.from_domain(module)


class DummyDuckDBQueryApi(DuckDBQueryApi):
    """Minimal DuckDBQueryApi implementation for service-layer tests."""

    def __init__(self) -> None:
        self._gateway = GatewayFactory().with_macros().open()
        self._limits = BackendLimits()
        self._functions = HookedFunctionQueries()
        self._modules = HookedProfileQueries()
        self._subsystems = HookedSubsystemQueries()
        self._datasets = HookedDatasetQueries()

    @property
    def gateway(self) -> StorageGateway:
        return self._gateway

    @property
    def limits(self) -> BackendLimits:
        return self._limits

    @property
    def functions(self) -> HookedFunctionQueries:
        return self._functions

    @property
    def modules(self) -> HookedProfileQueries:
        return self._modules

    @property
    def subsystems(self) -> HookedSubsystemQueries:
        return self._subsystems

    @property
    def datasets(self) -> HookedDatasetQueries:
        return self._datasets

    def __getattr__(self, name: str) -> object:
        raise AttributeError(name)


__all__ = ["DummyDuckDBQueryApi", "FakeQueryService", "ModelLike"]
