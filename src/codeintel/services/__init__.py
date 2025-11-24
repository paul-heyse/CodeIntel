"""Shared application services for CodeIntel surfaces."""

from __future__ import annotations

from codeintel.services.query_service import HttpQueryService, LocalQueryService

__all__ = ["HttpQueryService", "LocalQueryService"]
