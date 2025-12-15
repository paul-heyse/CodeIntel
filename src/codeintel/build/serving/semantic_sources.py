"""Build-side adapters for discovering semantic views.

This module bridges from view-builder registries (e.g., Ibis view builders) to
the semantic registry compiler by extracting semantic tag metadata.
"""

from __future__ import annotations

import codeintel.storage.views.ibis_views as _ibis_views
from codeintel.build.serving.semantic_tags import get_semantic_view_tags
from codeintel.storage.views.ibis_registry import VIEW_BUILDERS


def collect_semantic_view_tags() -> dict[str, dict[str, str]]:
    """Collect semantic tag metadata for all registered views.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping of registered view table_key to semantic tag mapping.
    """
    _ = _ibis_views
    tags_by_view: dict[str, dict[str, str]] = {}
    for view_name, builder in VIEW_BUILDERS.items():
        tags = get_semantic_view_tags(builder)
        if tags is None:
            continue
        tags_by_view[view_name] = tags
    return tags_by_view


__all__ = ["collect_semantic_view_tags"]
