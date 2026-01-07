"""Tabular helpers for building function catalog providers."""

from __future__ import annotations

from collections.abc import Mapping

import pyarrow as pa

from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.storage.catalog import CatalogService, build_function_catalog_from_rows

_FUNCTION_KINDS = {"function", "method"}


def module_map_from_frame(modules_frame: pa.Table) -> dict[str, str]:
    """Build module mapping from core.modules frame.

    Returns
    -------
    dict[str, str]
        Mapping of file path to module name.
    """
    module_map: dict[str, str] = {}
    for row in iter_rows(modules_frame):
        path = row.get("path")
        module = row.get("module")
        if isinstance(path, str) and isinstance(module, str):
            module_map[path] = module
    return module_map


def catalog_provider_from_frames(
    *,
    goids_frame: pa.Table,
    modules_frame: pa.Table,
    module_map_override: Mapping[str, str] | None = None,
) -> CatalogService:
    """Build a CatalogService from goids and modules frames.

    Returns
    -------
    CatalogService
        Catalog provider backed by the input frames.
    """
    module_map = dict(module_map_override or module_map_from_frame(modules_frame))
    rows: list[dict[str, object]] = []
    for row in iter_rows(goids_frame):
        kind = row.get("kind")
        if kind is not None and str(kind) not in _FUNCTION_KINDS:
            continue
        rows.append(
            {
                "goid_h128": row.get("goid_h128"),
                "rel_path": row.get("rel_path"),
                "qualname": row.get("qualname"),
                "start_line": row.get("start_line"),
                "end_line": row.get("end_line"),
                "urn": row.get("urn"),
            }
        )
    catalog = build_function_catalog_from_rows(rows, module_by_path=module_map)
    return CatalogService(catalog)


__all__ = [
    "catalog_provider_from_frames",
    "module_map_from_frame",
]
