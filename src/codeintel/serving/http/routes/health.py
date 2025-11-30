"""Health and diagnostics HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter

from codeintel.serving.http.dependencies import BackendDep, ConfigDep
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.registry import OperationSpec, get_operation_spec
from codeintel.storage.gateway import DuckDBError


def _require_spec(op_id: str) -> OperationSpec:
    spec = get_operation_spec(op_id)
    if spec is None:
        message = f"OperationSpec {op_id} is not registered"
        raise ValueError(message)
    return spec


def build_health_router() -> APIRouter:
    """
    Construct the router for health and diagnostics endpoints.

    Raises
    ------
    ValueError
        If the OperationSpec for health is missing or incomplete.

    Returns
    -------
    APIRouter
        Router exposing health status endpoints.
    """
    router = APIRouter()
    spec = _require_spec("health.status")
    if spec.http_path is None:
        message = "OperationSpec health.status is missing http_path"
        raise ValueError(message)
    path = spec.http_path

    @router.get(
        path,
        summary=spec.summary,
        tags=[spec.category],
    )
    def health(
        *,
        backend: BackendDep,
        config: ConfigDep,
    ) -> dict[str, object]:
        """
        Report server health and connectivity.

        Returns
        -------
        dict[str, object]
            Health payload including repo/commit and read-only state.

        Raises
        ------
        errors.backend_failure
            If the backend connection is unavailable.
        """
        limits: dict[str, int] | None = None
        service = getattr(backend, "service", None)
        service_limits = getattr(service, "limits", None)
        if service_limits is None and hasattr(service, "query"):
            query_obj = getattr(service, "query", None)
            service_limits = getattr(query_obj, "limits", None)
        if service_limits is not None:
            limits = {
                "default_limit": service_limits.default_limit,
                "max_rows_per_call": service_limits.max_rows_per_call,
            }

        if isinstance(backend, DuckDBBackend):
            con = backend.gateway.con
            try:
                con.execute("SELECT 1;")
            except DuckDBError as exc:
                message = "Backend connection failed health probe."
                raise errors.backend_failure(message) from exc
        payload: dict[str, object] = {
            "status": "ok",
            "repo": config.repo,
            "commit": config.commit,
            "read_only": config.read_only,
        }
        if limits is not None:
            payload["limits"] = limits
        return payload

    return router


__all__ = ["build_health_router"]
