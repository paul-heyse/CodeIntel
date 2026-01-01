"""FastMCP tool: serving_meta."""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from codeintel.serving.features import ServingFeatureSet
from codeintel.serving.mcp.models import DEFAULT_RESOURCE_TEMPLATES, ServingMetaResponse
from codeintel.serving.mcp.runtime import QueryLimiter
from codeintel.serving.mcp.tools.shared import (
    READ_ONLY_LOCAL_ANNOTATIONS,
    TAG_META,
    TAG_READ,
    mcp_correlation_id,
)
from codeintel.serving.meta.service import ServingMetaExtras, build_serving_meta_payload
from codeintel.serving.metrics import QueryMetrics, log_query_metrics
from codeintel.serving.operations.ops import ServingOperations

if TYPE_CHECKING:
    from codeintel.serving.settings import ServingSettings

_CURRENT_CONTEXT = CurrentContext()


async def _catalog_view_count(ops: ServingOperations, limiter: QueryLimiter) -> int:
    catalog = await limiter.run(ops.catalog)
    return len(catalog.views)


def register_meta_tool(
    mcp: FastMCP,
    ops: ServingOperations,
    limiter: QueryLimiter,
    *,
    settings: ServingSettings,
    started_at: datetime,
) -> None:
    """Register serving_meta tool."""

    @mcp.tool(
        name="serving_meta",
        description="Get serving layer metadata including snapshot info",
        annotations=READ_ONLY_LOCAL_ANNOTATIONS,
        tags={TAG_META, TAG_READ},
    )
    async def serving_meta(*, ctx: Context = _CURRENT_CONTEXT) -> ServingMetaResponse:
        start = time.perf_counter()
        await ctx.info("Retrieving serving metadata")

        view_count = await _catalog_view_count(ops, limiter)
        feature_set = ServingFeatureSet.from_settings(settings)
        features = {
            "supports_explain": feature_set.enable_mcp_explain,
            "supports_export": feature_set.enable_mcp_export,
            "supports_export_tasks": feature_set.enable_mcp_export_tasks,
            "supports_search": feature_set.enable_mcp_search,
            "supports_resources": True,
            "supports_sampling": feature_set.enable_mcp_sampling,
        }
        extras = ServingMetaExtras(
            features=features,
            inventories={"views": view_count},
            resource_templates=DEFAULT_RESOURCE_TEMPLATES,
        )
        payload = build_serving_meta_payload(
            ops,
            settings=settings,
            started_at=started_at,
            extras=extras,
        )

        duration_ms = (time.perf_counter() - start) * 1000
        log_query_metrics(
            QueryMetrics(
                endpoint="mcp:serving_meta",
                view_id=None,
                query=None,
                row_count=0,
                truncated=False,
                duration_ms=duration_ms,
                correlation_id=mcp_correlation_id(ctx),
            )
        )

        return payload


__all__ = ["register_meta_tool"]
