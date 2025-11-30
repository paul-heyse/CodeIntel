"""Dataset HTTP routes."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Literal

from fastapi import APIRouter, Request, status
from starlette.responses import Response

from codeintel.serving.http.dependencies import ServiceDep
from codeintel.serving.mcp.models import (
    DatasetDescriptor,
    DatasetRowsResponse,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
)
from codeintel.serving.registry import OperationSpec, get_operation_spec

LOG = logging.getLogger("codeintel.serving.http.routes.datasets")


def _compute_etag(payload: object) -> str:
    """
    Compute a weak ETag for a JSON-serializable payload.

    Returns
    -------
    str
        Weak ETag header value.
    """
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return f'W/"{hashlib.sha256(encoded).hexdigest()}"'


def _filter_datasets(
    datasets: list[DatasetDescriptor],
    *,
    docs_view: Literal["include", "exclude", "only"],
    read_only: Literal["include", "exclude", "only"],
) -> list[DatasetDescriptor]:
    filtered: list[DatasetDescriptor] = []
    for ds in datasets:
        caps = ds.capabilities or {}
        is_docs = bool(caps.get("docs_view"))
        is_read_only = bool(caps.get("read_only"))
        docs_ok = (docs_view != "only" or is_docs) and (docs_view != "exclude" or not is_docs)
        read_only_ok = (read_only != "only" or is_read_only) and (
            read_only != "exclude" or not is_read_only
        )
        if docs_ok and read_only_ok:
            filtered.append(ds)
    return filtered


def _require_spec(op_id: str) -> OperationSpec:
    spec = get_operation_spec(op_id)
    if spec is None:
        message = f"OperationSpec {op_id} is not registered"
        raise ValueError(message)
    return spec


def build_datasets_router() -> APIRouter:
    """
    Construct the router for dataset browsing endpoints.

    Raises
    ------
    ValueError
        If OperationSpec entries are missing or lack http_path values.

    Returns
    -------
    APIRouter
        Router exposing dataset discovery and access endpoints.
    """
    router = APIRouter()
    spec_list = _require_spec("datasets.list")
    spec_specs = _require_spec("datasets.specs")
    spec_rows = _require_spec("datasets.rows")
    spec_schema = _require_spec("datasets.schema")
    if spec_list.http_path is None or spec_specs.http_path is None:
        message = "Dataset OperationSpec entries must define http_path"
        raise ValueError(message)
    if spec_rows.http_path is None or spec_schema.http_path is None:
        message = "Dataset OperationSpec entries must define http_path"
        raise ValueError(message)
    list_path = spec_list.http_path
    specs_path = spec_specs.http_path
    rows_path = spec_rows.http_path
    schema_path = spec_schema.http_path

    @router.get(
        list_path,
        response_model=list[DatasetDescriptor],
        summary=spec_list.summary,
        tags=[spec_list.category],
    )
    def list_datasets(
        *,
        service: ServiceDep,
        request: Request,
        response: Response,
        docs_view: Literal["include", "exclude", "only"] = "include",
        read_only: Literal["include", "exclude", "only"] = "include",
    ) -> Response | list[DatasetDescriptor]:
        """
        Return dataset descriptors available through the backend.

        Returns
        -------
        list[DatasetDescriptor]
            Dataset descriptors sorted by name.
        """
        datasets = service.list_datasets()
        filtered = _filter_datasets(
            datasets,
            docs_view=docs_view,
            read_only=read_only,
        )
        payload = [ds.model_dump() for ds in filtered]
        etag = _compute_etag(payload)
        response.headers["Cache-Control"] = "public, max-age=60"
        response.headers["ETag"] = etag
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=response.headers)
        LOG.info(
            "Listed %d datasets (docs_view=%s read_only=%s)", len(filtered), docs_view, read_only
        )
        return filtered

    @router.get(
        specs_path,
        response_model=list[DatasetSpecDescriptor],
        summary=spec_specs.summary,
        tags=[spec_specs.category],
    )
    def list_dataset_specs(
        *,
        service: ServiceDep,
        request: Request,
        response: Response,
    ) -> Response | list[DatasetSpecDescriptor]:
        """
        Return canonical dataset specs including filenames and schema IDs.

        Returns
        -------
        list[DatasetSpecDescriptor]
            Dataset specs sorted by name.
        """
        specs = service.dataset_specs()
        payload = [spec.model_dump() for spec in specs]
        etag = _compute_etag(payload)
        response.headers["Cache-Control"] = "public, max-age=60"
        response.headers["ETag"] = etag
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=response.headers)
        LOG.info("Listed %d dataset specs", len(specs))
        return specs

    @router.get(
        rows_path,
        response_model=DatasetRowsResponse,
        summary=spec_rows.summary,
        tags=[spec_rows.category],
    )
    def read_dataset_rows(
        *,
        service: ServiceDep,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        """
        Read a window of rows from a configured dataset.

        Returns
        -------
        DatasetRowsResponse
            Dataset slice with pagination metadata.
        """
        resp = service.read_dataset_rows(dataset_name=dataset_name, limit=limit, offset=offset)
        LOG.info(
            "Read dataset=%s limit=%s offset=%s returned_rows=%d",
            dataset_name,
            limit,
            offset,
            len(resp.rows),
        )
        return resp

    @router.get(
        schema_path,
        response_model=DatasetSchemaResponse,
        summary=spec_schema.summary,
        tags=[spec_schema.category],
    )
    def dataset_schema(
        *,
        service: ServiceDep,
        dataset_name: str,
        limit: int = 5,
    ) -> DatasetSchemaResponse:
        """
        Return schema metadata, JSON Schema (when present), and sample rows for a dataset.

        Returns
        -------
        DatasetSchemaResponse
            Schema detail payload.
        """
        detail = service.dataset_schema(dataset_name=dataset_name, sample_limit=limit)
        LOG.info("Returned schema detail for dataset=%s", dataset_name)
        return detail

    return router


__all__ = ["build_datasets_router"]
