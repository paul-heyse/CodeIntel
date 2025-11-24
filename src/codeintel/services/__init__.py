"""Shared application services for CodeIntel surfaces."""

from __future__ import annotations

from codeintel.services.query_service import HttpQueryService, LocalQueryService
from codeintel.services.wiring import BackendResource, build_backend_resource

__all__ = ["BackendResource", "HttpQueryService", "LocalQueryService", "build_backend_resource"]
