"""Dataset HTTP routes."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from fastapi import APIRouter, Depends, Request, status
from starlette.responses import Response

from codeintel.serving.http import dependencies as deps
from codeintel.serving.mcp.models import (
    DatasetDescriptor,
    DatasetRowsResponse,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
)
from codeintel.serving.operations import get_operation

if TYPE_CHECKING:
    from codeintel.serving import domain_models as dm
    from codeintel.serving.http.routes.functions import RouterOptions
    from codeintel.serving.operations import Operation

LOG = logging.getLogger("codeintel.serving.http.routes.datasets")
_FASTAPI_REQUEST = Request


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
    datasets: list[dm.DatasetDescriptorDomain],
    *,
    docs_view: Literal["include", "exclude", "only"],
    read_only: Literal["include", "exclude", "only"],
) -> list[dm.DatasetDescriptorDomain]:
    filtered: list[dm.DatasetDescriptorDomain] = []
    for ds in datasets:
        is_docs = bool(getattr(ds, "is_docs_view", False))
        is_read_only = bool(getattr(ds, "is_read_only", False))
        docs_ok = (docs_view != "only" or is_docs) and (docs_view != "exclude" or not is_docs)
        read_only_ok = (read_only != "only" or is_read_only) and (
            read_only != "exclude" or not is_read_only
        )
        if docs_ok and read_only_ok:
            filtered.append(ds)
    return filtered


def _require_spec(op_id: str) -> Operation:
    spec = get_operation(op_id)
    if spec is None:
        message = f"Operation {op_id} is not registered"
        raise ValueError(message)
    return spec


def _load_dataset_specs() -> tuple[dict[str, Operation], dict[str, str]]:
    """Load dataset operation specifications.

    Returns
    -------
    tuple[dict[str, Operation], dict[str, str]]
        Tuple of (specs, paths) dicts keyed by operation ID.

    Raises
    ------
    ValueError
        If required specs are missing or lack http_path.
    """
    ids = ["datasets.list", "datasets.specs", "datasets.rows", "datasets.schema"]
    specs: dict[str, Operation] = {}
    paths: dict[str, str] = {}
    for op_id in ids:
        spec = _require_spec(op_id)
        if spec.http_path is None:
            msg = f"Operation {op_id} is missing http_path"
            raise ValueError(msg)
        specs[op_id] = spec
        paths[op_id] = spec.http_path
    return specs, paths


def _build_dataset_deps(options: RouterOptions | None) -> dict[str, list[Any]]:
    """Build dependencies for each dataset operation.

    Parameters
    ----------
    options
        Router options with auto_pipeline flag.

    Returns
    -------
    dict[str, list[Any]]
        Mapping of operation ID to dependency list.
    """
    if options is None or not options.auto_pipeline:
        return {}
    ids = ["datasets.list", "datasets.specs", "datasets.rows", "datasets.schema"]
    return {op_id: [Depends(deps.make_op_prereq_dependency(op_id))] for op_id in ids}


def build_datasets_router(options: RouterOptions | None = None) -> APIRouter:
    """Construct the router for dataset browsing endpoints.

    Parameters
    ----------
    options
        Router configuration options. When auto_pipeline is enabled,
        dependencies are attached that automatically run prerequisites.

    Returns
    -------
    APIRouter
        Router exposing dataset discovery and access endpoints.
    """
    router = APIRouter()
    specs, paths = _load_dataset_specs()
    deps_by_operation = _build_dataset_deps(options)

    spec_list = specs["datasets.list"]
    spec_specs = specs["datasets.specs"]
    spec_rows = specs["datasets.rows"]
    spec_schema = specs["datasets.schema"]

    @router.get(
        paths["datasets.list"],
        response_model=list[DatasetDescriptor],
        summary=spec_list.summary,
        tags=[spec_list.category],
        dependencies=list(deps_by_operation.get("datasets.list", [])),
    )
    def list_datasets(
        *,
        service: deps.ServiceDep,
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
        response_models = [
            DatasetDescriptor.model_validate(
                asdict(ds) if is_dataclass(ds) else cast("dict[str, object]", ds)
            )
            for ds in filtered
        ]
        payload = [descriptor.model_dump() for descriptor in response_models]
        etag = _compute_etag(payload)
        response.headers["Cache-Control"] = "public, max-age=60"
        response.headers["ETag"] = etag
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=response.headers)
        LOG.info(
            "Listed %d datasets (docs_view=%s read_only=%s)", len(filtered), docs_view, read_only
        )
        return response_models

    @router.get(
        paths["datasets.specs"],
        response_model=list[DatasetSpecDescriptor],
        summary=spec_specs.summary,
        tags=[spec_specs.category],
        dependencies=list(deps_by_operation.get("datasets.specs", [])),
    )
    def list_dataset_specs(
        *,
        service: deps.ServiceDep,
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
        paths["datasets.rows"],
        response_model=DatasetRowsResponse,
        summary=spec_rows.summary,
        tags=[spec_rows.category],
        dependencies=list(deps_by_operation.get("datasets.rows", [])),
    )
    def read_dataset_rows(
        *,
        service: deps.ServiceDep,
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
        domain_rows: dm.DatasetRows = service.read_dataset_rows(
            dataset_name=dataset_name, limit=limit, offset=offset
        )
        resp = DatasetRowsResponse.from_domain(domain_rows)
        LOG.info(
            "Read dataset=%s limit=%s offset=%s returned_rows=%d",
            dataset_name,
            limit,
            offset,
            len(resp.rows),
        )
        return resp

    @router.get(
        paths["datasets.schema"],
        response_model=DatasetSchemaResponse,
        summary=spec_schema.summary,
        tags=[spec_schema.category],
        dependencies=list(deps_by_operation.get("datasets.schema", [])),
    )
    def dataset_schema(
        *,
        service: deps.ServiceDep,
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
        domain_schema: dm.DatasetSchema = service.dataset_schema(
            dataset_name=dataset_name, sample_limit=limit
        )
        detail = DatasetSchemaResponse.from_domain(domain_schema)
        LOG.info("Returned schema detail for dataset=%s", dataset_name)
        return detail

    return router


__all__ = ["build_datasets_router"]
