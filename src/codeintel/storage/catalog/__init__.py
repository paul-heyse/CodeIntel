"""Storage-backed function catalog services."""

from __future__ import annotations

from codeintel.storage.catalog.service import (
    CatalogService,
    FunctionCatalog,
    FunctionCatalogProvider,
    FunctionSpan,
    FunctionSpanIndex,
    build_function_catalog_from_rows,
    load_function_catalog,
    load_function_index,
    load_function_spans,
)

__all__ = [
    "CatalogService",
    "FunctionCatalog",
    "FunctionCatalogProvider",
    "FunctionSpan",
    "FunctionSpanIndex",
    "build_function_catalog_from_rows",
    "load_function_catalog",
    "load_function_index",
    "load_function_spans",
]
