"""Serving surfaces exposing CodeIntel data via HTTP (FastAPI) and MCP protocol."""

from codeintel.serving.mcp.backend import DuckDBBackend, HttpBackend, QueryBackend
from codeintel.serving.protocols import HasModelDump
from codeintel.serving.services.factory import (
    BackendResource,
    build_backend_resource,
    build_service_from_config,
)
from codeintel.serving.services.query_service import (
    HttpQueryService,
    LocalQueryService,
    QueryService,
)

__all__ = [
    "BackendResource",
    "DuckDBBackend",
    "HasModelDump",
    "HttpBackend",
    "HttpQueryService",
    "LocalQueryService",
    "QueryBackend",
    "QueryService",
    "build_backend_resource",
    "build_service_from_config",
]
