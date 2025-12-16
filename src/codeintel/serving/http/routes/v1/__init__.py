"""Versioned v1 router aggregation for serving HTTP endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from codeintel.serving.http.routes.v1.export import router as export_router
from codeintel.serving.http.routes.v1.search import router as search_router
from codeintel.serving.http.routes.v1.semantic import router as semantic_router

router = APIRouter()
router.include_router(semantic_router)
router.include_router(search_router)
router.include_router(export_router)

__all__ = ["router"]
