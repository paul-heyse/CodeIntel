"""Versioned v1 router aggregation for serving HTTP endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from codeintel.serving.features import ServingFeatureSet
from codeintel.serving.http.routes.v1.export import router as export_router
from codeintel.serving.http.routes.v1.search import router as search_router
from codeintel.serving.http.routes.v1.semantic import router as semantic_router


def build_v1_router(features: ServingFeatureSet) -> APIRouter:
    """Build the versioned v1 router with feature gating.

    Returns
    -------
    APIRouter
        Router with v1 endpoints for enabled features.
    """
    router = APIRouter()
    router.include_router(semantic_router)
    router.include_router(search_router)
    if features.enable_http_export:
        router.include_router(export_router)
    return router


router = build_v1_router(ServingFeatureSet.all_enabled())

__all__ = ["build_v1_router", "router"]
