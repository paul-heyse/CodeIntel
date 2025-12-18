"""Build-side helpers for producing immutable serving snapshots.

This package uses lazy exports to avoid import cycles with the storage layer.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.build.serving.manifest import ServingSnapshotManifest
    from codeintel.build.serving.publisher import publish_serving_snapshot
    from codeintel.build.serving.semantic_compile import (
        CompiledSemanticRegistry,
        compile_semantic_registry_from_views,
        write_semantic_registry,
    )
    from codeintel.core.hamilton.semantic_tags import SEMANTIC_VIEW_TAG_ATTR, semantic_view

_EXPORTS: dict[str, tuple[str, str]] = {
    "CompiledSemanticRegistry": (
        "codeintel.build.serving.semantic_compile",
        "CompiledSemanticRegistry",
    ),
    "SEMANTIC_VIEW_TAG_ATTR": ("codeintel.core.hamilton.semantic_tags", "SEMANTIC_VIEW_TAG_ATTR"),
    "ServingSnapshotManifest": ("codeintel.build.serving.manifest", "ServingSnapshotManifest"),
    "compile_semantic_registry_from_views": (
        "codeintel.build.serving.semantic_compile",
        "compile_semantic_registry_from_views",
    ),
    "publish_serving_snapshot": ("codeintel.build.serving.publisher", "publish_serving_snapshot"),
    "semantic_view": ("codeintel.core.hamilton.semantic_tags", "semantic_view"),
    "write_semantic_registry": (
        "codeintel.build.serving.semantic_compile",
        "write_semantic_registry",
    ),
}


def __getattr__(name: str) -> object:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = [
    "SEMANTIC_VIEW_TAG_ATTR",
    "CompiledSemanticRegistry",
    "ServingSnapshotManifest",
    "compile_semantic_registry_from_views",
    "publish_serving_snapshot",
    "semantic_view",
    "write_semantic_registry",
]
