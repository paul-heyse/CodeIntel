"""Route aggregation with API versioning."""

from __future__ import annotations

from fastapi import APIRouter

from codeintel.serving.features import ServingFeatureSet
from codeintel.serving.http.routes.v1 import build_v1_router


def build_http_router(features: ServingFeatureSet) -> APIRouter:
    """Build the root router with all versioned routes.

    Returns
    -------
    APIRouter
        Router with all enabled versioned routes included.
    """
    router = APIRouter()
    router.include_router(build_v1_router(features), prefix="/v1", tags=["v1"])
    return router


router = build_http_router(ServingFeatureSet.all_enabled())

__all__ = ["build_http_router", "router"]
