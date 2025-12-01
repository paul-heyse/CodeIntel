"""Dataset delegates for local and HTTP query services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

from codeintel.config.dataset_contract import DatasetContract
from codeintel.serving import domain_models as dm
from codeintel.serving.backend import BackendLimits, clamp_limit_value, clamp_offset_value
from codeintel.serving.backend.query_api import DuckDBQueryApi
from codeintel.serving.mcp import errors as mcp_errors
from codeintel.serving.mcp.models import (
    DatasetDescriptor,
    DatasetRowsResponse,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    ResponseMeta,
    ViewRow,
)
from codeintel.serving.services.errors import DatasetNotFoundError
from codeintel.serving.services.http_transport import _HttpTransportMixin
from codeintel.storage.datasets import load_dataset_registry


def _normalize_validation_profile(
    value: str | None,
) -> Literal["strict", "lenient"] | None:
    """
    Normalize validation profile strings to allowed literal values.

    Returns
    -------
    Literal["strict", "lenient"] | None
        Normalized validation profile when valid.
    """
    if value == "strict":
        return "strict"
    if value == "lenient":
        return "lenient"
    return None


class _LocalDatasetMixin:
    """
    Dataset listing and retrieval helpers for LocalQueryService.

    Expects ``self`` to provide:
      - query: DuckDBQueryService
      - dataset_tables: dict[str, str] | None
      - describe_dataset_fn: Callable[[str, str], str]
      - limits: BackendLimits
      - _call(name, func, *, dataset, schema_version, retries)
    """

    query: DuckDBQueryApi
    dataset_tables: dict[str, str] | None
    describe_dataset_fn: Callable[[str, str], str]
    limits: BackendLimits
    _call: Callable[..., Any]

    def list_datasets(self) -> list[DatasetDescriptor]:
        """
        List datasets available through the dataset registry.

        Returns
        -------
        list[DatasetDescriptor]
            Dataset descriptors with names, tables, and descriptions.
        """

        def _list() -> list[DatasetDescriptor]:
            mapping: dict[str, str] = self.dataset_tables or {}
            registry = None
            if not mapping:
                query_gateway = getattr(self.query, "gateway", None)
                if query_gateway is not None:
                    mapping = query_gateway.datasets.mapping
                    registry = load_dataset_registry(query_gateway.con)
            if registry is None:
                registry = load_dataset_registry(self.query.gateway.con)
            results: list[DatasetDescriptor] = []
            for name, table in sorted(mapping.items()):
                ds: DatasetContract | None = (
                    registry.by_name.get(name) if registry is not None else None
                )
                description = (
                    ds.description
                    if ds is not None and ds.description is not None
                    else self.describe_dataset_fn(name, table)
                )
                results.append(
                    DatasetDescriptor(
                        name=name,
                        table=table,
                        family=ds.family if ds is not None else None,
                        description=description,
                        owner=ds.owner if ds is not None else None,
                        freshness_sla=ds.freshness_sla if ds is not None else None,
                        retention_policy=ds.retention_policy if ds is not None else None,
                        schema_version=ds.schema_version if ds is not None else None,
                        stable_id=ds.stable_id if ds is not None else None,
                        validation_profile=_normalize_validation_profile(
                            ds.validation_profile if ds is not None else None
                        ),
                        capabilities=ds.capabilities() if ds is not None else {},
                    )
                )
            return results

        return self._call("list_datasets", _list)

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        """
        Return canonical dataset specs with filenames and schema metadata.

        Returns
        -------
        list[DatasetSpecDescriptor]
            Dataset specs sorted by name.
        """

        def _list_specs() -> list[DatasetSpecDescriptor]:
            return self.query.datasets.dataset_specs()

        return self._call("dataset_specs", _list_specs)

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> dm.DatasetSchema:
        """
        Return DuckDB + JSON Schema details and sample rows for a dataset.

        Returns
        -------
        dm.DatasetSchema
            Dataset schema details with sample rows.

        Raises
        ------
        McpError
            When the requested dataset is missing from the registry.
        """
        registry = load_dataset_registry(self.query.gateway.con)
        schema_version = None
        if dataset_name in registry.by_name:
            schema_version = registry.by_name[dataset_name].schema_version
        else:
            problem_detail = DatasetNotFoundError.for_name(dataset_name).detail
            raise mcp_errors.McpError(problem_detail)

        def _schema() -> DatasetSchemaResponse:
            return self.query.datasets.dataset_schema(
                dataset_name=dataset_name, sample_limit=sample_limit
            )

        pydantic_resp: DatasetSchemaResponse = self._call(
            "dataset_schema",
            _schema,
            dataset=dataset_name,
            schema_version=schema_version,
        )
        return pydantic_resp.to_domain()

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> dm.DatasetRows:
        """
        Read dataset rows with clamping and messaging.

        Returns
        -------
        dm.DatasetRows
            Dataset slice and metadata for truncation/messaging.

        Raises
        ------
        McpError
            When the dataset is not registered.
        """
        applied_limit = self.query.limits.default_limit if limit is None else limit
        registry = load_dataset_registry(self.query.gateway.con)
        schema_version = None
        if dataset_name in registry.by_name:
            schema_version = registry.by_name[dataset_name].schema_version
        else:
            problem_detail = DatasetNotFoundError.for_name(dataset_name).detail
            raise mcp_errors.McpError(problem_detail)
        clamped_offset = clamp_offset_value(offset)
        clamped_limit = clamp_limit_value(
            applied_limit,
            default=self.query.limits.default_limit,
            max_limit=self.query.limits.max_rows_per_call,
        )
        messages = [*clamped_offset.messages, *clamped_limit.messages]
        if clamped_limit.has_error or clamped_offset.has_error:
            response = DatasetRowsResponse(
                dataset_name=dataset_name,
                limit=clamped_limit.applied or 0,
                offset=clamped_offset.applied,
                rows=[],
                meta=ResponseMeta(
                    requested_limit=limit,
                    applied_limit=clamped_limit.applied,
                    requested_offset=offset,
                    applied_offset=clamped_offset.applied,
                    truncated=False,
                    messages=messages,
                ),
            )
            return response.to_domain()

        def _run() -> DatasetRowsResponse:
            raw_rows = self.query.datasets.read_dataset_rows(
                dataset_name=dataset_name,
                limit=clamped_limit.applied,
                offset=clamped_offset.applied,
            )
            return DatasetRowsResponse(
                dataset_name=dataset_name,
                limit=clamped_limit.applied or 0,
                offset=clamped_offset.applied,
                rows=[ViewRow.model_validate(row) for row in raw_rows],
                meta=ResponseMeta(
                    requested_limit=limit,
                    applied_limit=clamped_limit.applied,
                    requested_offset=offset,
                    applied_offset=clamped_offset.applied,
                    truncated=False,
                    messages=messages,
                ),
            )

        pydantic_resp: DatasetRowsResponse = self._call(
            "read_dataset_rows",
            _run,
            dataset=dataset_name,
            schema_version=schema_version,
        )
        return pydantic_resp.to_domain()


class _HttpDatasetQueryMixin(_HttpTransportMixin):
    """HTTP-based dataset query APIs used by HttpQueryService."""

    limits: BackendLimits

    def list_datasets(self) -> list[DatasetDescriptor]:
        def _run() -> list[DatasetDescriptor]:
            data = cast("list[dict[str, object]]", self.request_json("/datasets", {}))
            return [DatasetDescriptor.model_validate(item) for item in data]

        return self._http_call("list_datasets", _run)

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        def _run() -> list[DatasetSpecDescriptor]:
            payload = cast(
                "list[dict[str, object]]",
                self.request_json("/datasets/specs", {}),
            )
            return [DatasetSpecDescriptor.model_validate(entry) for entry in payload]

        return self._http_call("dataset_specs", _run)

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> dm.DatasetRows:
        def _run() -> DatasetRowsResponse:
            clamp = clamp_limit_value(
                limit,
                default=self.limits.default_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            offset_clamp = clamp_offset_value(offset)
            messages = [*clamp.messages, *offset_clamp.messages]
            if clamp.has_error or offset_clamp.has_error:
                meta = ResponseMeta(
                    requested_limit=limit,
                    applied_limit=clamp.applied,
                    requested_offset=offset,
                    applied_offset=offset_clamp.applied,
                    messages=messages,
                    truncated=False,
                )
                return DatasetRowsResponse(
                    dataset_name=dataset_name,
                    limit=clamp.applied,
                    offset=offset_clamp.applied,
                    rows=[],
                    meta=meta,
                )
            data = self.request_json(
                f"/datasets/{dataset_name}",
                {"limit": clamp.applied, "offset": offset_clamp.applied},
            )
            if isinstance(data, dict) and "dataset" in data and "dataset_name" not in data:
                data = {**data, "dataset_name": data["dataset"]}
            response = DatasetRowsResponse.model_validate(data)
            existing_meta = response.meta if response.meta is not None else ResponseMeta()
            merged_meta = ResponseMeta(
                requested_limit=limit,
                applied_limit=clamp.applied,
                requested_offset=offset,
                applied_offset=offset_clamp.applied,
                truncated=existing_meta.truncated,
                messages=[*messages, *(existing_meta.messages or [])],
            )
            return response.model_copy(update={"meta": merged_meta})

        pydantic_resp: DatasetRowsResponse = self._http_call(
            "read_dataset_rows", _run, dataset=dataset_name
        )
        return pydantic_resp.to_domain()

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> dm.DatasetSchema:
        def _run() -> DatasetSchemaResponse:
            data = self.request_json(
                f"/datasets/{dataset_name}/schema",
                {"limit": sample_limit},
            )
            return DatasetSchemaResponse.model_validate(data)

        pydantic_resp: DatasetSchemaResponse = self._http_call(
            "dataset_schema", _run, dataset=dataset_name
        )
        return pydantic_resp.to_domain()


__all__ = ["_HttpDatasetQueryMixin", "_LocalDatasetMixin"]
