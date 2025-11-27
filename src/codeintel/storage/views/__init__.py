"""Docs view registry and creation helpers."""

from __future__ import annotations

from collections.abc import Callable

from duckdb import DuckDBPyConnection

from codeintel.storage.views.data_model_views import (
    DATA_MODEL_VIEW_NAMES,
    create_data_model_views,
)
from codeintel.storage.views.function_views import FUNCTION_VIEW_NAMES, create_function_views
from codeintel.storage.views.graph_views import GRAPH_VIEW_NAMES, create_graph_views
from codeintel.storage.views.ide_views import IDE_VIEW_NAMES, create_ide_views
from codeintel.storage.views.module_views import MODULE_VIEW_NAMES, create_module_views
from codeintel.storage.views.subsystem_views import (
    SUBSYSTEM_VIEW_NAMES,
    create_subsystem_views,
)
from codeintel.storage.views.test_views import TEST_VIEW_NAMES, create_test_views

DOCS_VIEWS: tuple[str, ...] = (
    *FUNCTION_VIEW_NAMES,
    *MODULE_VIEW_NAMES,
    *TEST_VIEW_NAMES,
    *SUBSYSTEM_VIEW_NAMES,
    *GRAPH_VIEW_NAMES,
    *IDE_VIEW_NAMES,
    *DATA_MODEL_VIEW_NAMES,
)

_VIEW_CREATORS: tuple[Callable[[DuckDBPyConnection], None], ...] = (
    create_function_views,
    create_module_views,
    create_test_views,
    create_subsystem_views,
    create_graph_views,
    create_ide_views,
    create_data_model_views,
)


def create_all_views(con: DuckDBPyConnection) -> None:
    """Create or replace all docs.* views."""
    for create in _VIEW_CREATORS:
        create(con)


__all__ = ["DOCS_VIEWS", "create_all_views"]
