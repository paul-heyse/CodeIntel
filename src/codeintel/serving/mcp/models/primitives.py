"""Primitive Pydantic models and Annotated types for FastMCP surfaces."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from codeintel.serving.models.primitives import ResourceTemplate, SnapshotRef
from codeintel.serving.uris import (
    EXPORT_META_URI_TEMPLATE,
    META_SERVING_URI,
    SEMANTIC_VIEW_URI_TEMPLATE,
)

CodeIntelURI = Annotated[
    str,
    Field(
        pattern=r"^codeintel://.+",
        description="CodeIntel resource URI (codeintel://...).",
        examples=[
            META_SERVING_URI,
            SEMANTIC_VIEW_URI_TEMPLATE.format(view_id="call_graph_enriched"),
        ],
    ),
]

RFC6570TemplateURI = Annotated[
    str,
    Field(
        description="RFC 6570 URI template (may include {placeholders}).",
        examples=[
            SEMANTIC_VIEW_URI_TEMPLATE,
            EXPORT_META_URI_TEMPLATE,
        ],
    ),
]

ViewId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_.-]+$",
        description="Semantic view identifier (stable).",
        examples=["call_graph_enriched", "data_models", "symbol_graph_metrics"],
    ),
]

ExportId = Annotated[
    str,
    Field(
        min_length=8,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Export identifier (opaque token).",
        examples=["01HZY9E1K8ZQ6N9J3W2K9M3A8B"],
    ),
]

Sha256Hex = Annotated[
    str,
    Field(
        pattern=r"^[a-f0-9]{64}$",
        description="SHA-256 hex digest (64 lowercase hex characters).",
        examples=["e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"],
    ),
]


__all__ = [
    "CodeIntelURI",
    "ExportId",
    "RFC6570TemplateURI",
    "ResourceTemplate",
    "Sha256Hex",
    "SnapshotRef",
    "ViewId",
]
