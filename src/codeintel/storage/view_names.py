"""View name constants aggregated from all view modules.

This module provides a single import point for view name tuples without
importing any gateway dependencies. This breaks the circular import chain
between config.datasets and storage.views.

Note
----
Do NOT import from gateway, ibis_views, or MinimalStorageGateway here.
This module must remain dependency-free to avoid circular imports.
"""

from __future__ import annotations

from codeintel.storage.views.data_model_views import DATA_MODEL_VIEW_NAMES
from codeintel.storage.views.function_views import FUNCTION_VIEW_NAMES
from codeintel.storage.views.graph_views import GRAPH_VIEW_NAMES
from codeintel.storage.views.ide_views import IDE_VIEW_NAMES
from codeintel.storage.views.module_views import MODULE_VIEW_NAMES
from codeintel.storage.views.subsystem_views import SUBSYSTEM_VIEW_NAMES
from codeintel.storage.views.test_views import TEST_VIEW_NAMES

__all__ = [
    "ALIAS_DOCS_VIEWS",
    "DATA_MODEL_VIEW_NAMES",
    "DERIVED_DOCS_VIEWS",
    "DOCS_VIEWS",
    "FUNCTION_VIEW_NAMES",
    "GRAPH_VIEW_NAMES",
    "IDE_VIEW_NAMES",
    "MODULE_VIEW_NAMES",
    "SUBSYSTEM_VIEW_NAMES",
    "TEST_VIEW_NAMES",
]

ALIAS_DOCS_VIEWS: dict[str, str] = {
    "docs.v_function_profile": "analytics.function_profile",
    "docs.v_file_profile": "analytics.file_profile",
    "docs.v_module_profile": "analytics.module_profile",
    "docs.v_config_graph_metrics_keys": "analytics.config_graph_metrics_keys",
    "docs.v_config_graph_metrics_modules": "analytics.config_graph_metrics_modules",
    "docs.v_config_projection_key_edges": "analytics.config_projection_key_edges",
    "docs.v_config_projection_module_edges": "analytics.config_projection_module_edges",
}

DOCS_VIEWS: tuple[str, ...] = (
    *FUNCTION_VIEW_NAMES,
    *MODULE_VIEW_NAMES,
    *TEST_VIEW_NAMES,
    *SUBSYSTEM_VIEW_NAMES,
    *GRAPH_VIEW_NAMES,
    *IDE_VIEW_NAMES,
    *DATA_MODEL_VIEW_NAMES,
)

DERIVED_DOCS_VIEWS: tuple[str, ...] = tuple(
    view for view in DOCS_VIEWS if view not in ALIAS_DOCS_VIEWS
)
