"""Build-side helpers for producing immutable serving snapshots."""

from __future__ import annotations

from codeintel.build.serving.manifest import ServingSnapshotManifest
from codeintel.build.serving.publisher import publish_serving_snapshot
from codeintel.build.serving.semantic_compile import (
    CompiledSemanticRegistry,
    compile_semantic_registry_from_views,
    write_semantic_registry,
)
from codeintel.build.serving.semantic_tags import SEMANTIC_VIEW_TAG_ATTR, semantic_view

__all__ = [
    "SEMANTIC_VIEW_TAG_ATTR",
    "CompiledSemanticRegistry",
    "ServingSnapshotManifest",
    "compile_semantic_registry_from_views",
    "publish_serving_snapshot",
    "semantic_view",
    "write_semantic_registry",
]
