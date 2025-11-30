"""Dataset delegates for local and HTTP query services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

from codeintel.serving.backend import (
    BackendLimits,
    DuckDBQueryService,
    clamp_limit_value,
    clamp_offset_value,
)
from codeintel.serving.mcp.models import (
    DatasetDescriptor,
    DatasetRowsResponse,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    ResponseMeta,
)
from codeintel.serving.services.http_transport import _HttpTransportMixin
from codeintel.storage.datasets import Dataset, load_dataset_registry


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

    query: DuckDBQueryService
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
                ds: Dataset | None = registry.by_name.get(name) if registry is not None else None
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
            return self.query.dataset_specs()

        return self._call("dataset_specs", _list_specs)

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> DatasetSchemaResponse:
        """
        Return DuckDB + JSON Schema details and sample rows for a dataset.

        Returns
        -------
        DatasetSchemaResponse
            Composite schema and sample payload.
        """

        def _schema() -> DatasetSchemaResponse:
            return self.query.dataset_schema(dataset_name=dataset_name, sample_limit=sample_limit)

        registry = load_dataset_registry(self.query.gateway.con)
        schema_version = None
        if dataset_name in registry.by_name:
            schema_version = registry.by_name[dataset_name].schema_version
        return self._call(
            "dataset_schema",
            _schema,
            dataset=dataset_name,
            schema_version=schema_version,
        )

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        """
        Read dataset rows with clamping and messaging.

        Returns
        -------
        DatasetRowsResponse
            Dataset slice and metadata for truncation/messaging.
        """
        applied_limit = self.query.limits.default_limit if limit is None else limit
        registry = load_dataset_registry(self.query.gateway.con)
        schema_version = None
        if dataset_name in registry.by_name:
            schema_version = registry.by_name[dataset_name].schema_version
        return self._call(
            "read_dataset_rows",
            lambda: self.query.read_dataset_rows(
                dataset_name=dataset_name,
                limit=applied_limit,
                offset=offset,
            ),
            dataset=dataset_name,
            schema_version=schema_version,
        )


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
    ) -> DatasetRowsResponse:
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
                    dataset=dataset_name,
                    limit=clamp.applied,
                    offset=offset_clamp.applied,
                    rows=[],
                    meta=meta,
                )
            data = self.request_json(
                f"/datasets/{dataset_name}",
                {"limit": clamp.applied, "offset": offset_clamp.applied},
            )
            response = DatasetRowsResponse.model_validate(data)
            existing_meta = response.meta if response.meta is not None else ResponseMeta()
            merged_meta = ResponseMeta(
                requested_limit=limit,
                applied_limit=clamp.applied,
                requested_offset=offset,
                applied_offset=offset_clamp.applied,
                truncated=existing_meta.truncated,
                messages=[*messages, *existing_meta.messages],
            )
            return response.model_copy(update={"meta": merged_meta})

        return self._http_call("read_dataset_rows", _run, dataset=dataset_name)

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> DatasetSchemaResponse:
        def _run() -> DatasetSchemaResponse:
            data = self.request_json(
                f"/datasets/{dataset_name}/schema",
                {"limit": sample_limit},
            )
            return DatasetSchemaResponse.model_validate(data)

        return self._http_call("dataset_schema", _run, dataset=dataset_name)


__all__ = ["_HttpDatasetQueryMixin", "_LocalDatasetMixin"]
