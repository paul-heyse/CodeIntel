"""FastMCP resources: semantic/meta discovery and snapshot metadata."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.serving.meta.models import DEFAULT_RESOURCE_TEMPLATES
from codeintel.serving.meta.service import (
    build_environment_meta_payload,
    build_resource_templates_payload,
)
from codeintel.serving.uris import (
    META_ENVIRONMENT_URI,
    META_RESOURCES_URI,
    META_SERVING_URI,
    SEMANTIC_VIEW_URI_TEMPLATE,
    SEMANTIC_VIEWS_URI,
)

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from codeintel.serving.operations.ops import ServingOperations
    from codeintel.serving.settings import ServingSettings


def _build_resource_templates_response(ops: ServingOperations) -> dict[str, object]:
    payload = build_resource_templates_payload(
        ops,
        templates=DEFAULT_RESOURCE_TEMPLATES,
        generated_at=datetime.now(UTC),
    )
    return payload.model_dump(mode="json")


def register_meta_resources(
    mcp: FastMCP,
    ops: ServingOperations,
    *,
    settings: ServingSettings,
) -> None:
    """Register meta and semantic discovery resources."""

    @mcp.resource(SEMANTIC_VIEWS_URI)
    def semantic_views() -> dict[str, object]:
        return ops.catalog().model_dump(mode="json")

    @mcp.resource(SEMANTIC_VIEW_URI_TEMPLATE)
    def view_description(view_id: str) -> dict[str, object]:
        return ops.describe(view_id).model_dump(mode="json")

    @mcp.resource(META_SERVING_URI)
    def serving_meta_resource() -> dict[str, object]:
        return ops.meta().model_dump(mode="json")

    @mcp.resource(META_RESOURCES_URI)
    def resource_templates() -> dict[str, object]:
        return _build_resource_templates_response(ops)

    @mcp.resource(META_ENVIRONMENT_URI, mime_type="application/json", tags={"meta"})
    def environment() -> dict[str, object]:
        return build_environment_meta_payload(ops, settings=settings)


__all__ = ["register_meta_resources"]
