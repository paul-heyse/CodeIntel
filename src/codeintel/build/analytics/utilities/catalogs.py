"""Tabular helpers for building function catalog providers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pyarrow as pa

from codeintel.build.analytics.utilities.snapshot import snapshot_plan, snapshot_table
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.plan_ops import materialize_plan
from codeintel.storage.catalog import CatalogService, build_function_catalog_from_rows

_FUNCTION_KINDS = {"function", "method"}
type RowSource = pa.Table | pa.RecordBatchReader


def module_map_from_frame(
    modules_frame: RowSource,
    *,
    repo: str | None = None,
    commit: str | None = None,
) -> dict[str, str]:
    """Build module mapping from core.modules frame.

    Returns
    -------
    dict[str, str]
        Mapping of file path to module name.
    """
    module_map: dict[str, str] = {}
    source = _snapshot_source(
        modules_frame,
        repo=repo,
        commit=commit,
        columns=("path", "module"),
    )
    for row in _iter_rows_from_source(source):
        path = row.get("path")
        module = row.get("module")
        if isinstance(path, str) and isinstance(module, str):
            module_map[path] = module
    return module_map


def catalog_provider_from_frames(
    *,
    goids_frame: RowSource,
    modules_frame: RowSource,
    module_map_override: Mapping[str, str] | None = None,
    repo: str | None = None,
    commit: str | None = None,
) -> CatalogService:
    """Build a CatalogService from goids and modules frames.

    Returns
    -------
    CatalogService
        Catalog provider backed by the input frames.
    """
    module_map = dict(
        module_map_override or module_map_from_frame(modules_frame, repo=repo, commit=commit)
    )
    rows: list[dict[str, object]] = []
    source = _goids_source(goids_frame, repo=repo, commit=commit)
    for row in _iter_rows_from_source(source):
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


def _iter_rows_from_source(source: RowSource) -> Iterable[dict[str, object]]:
    if isinstance(source, pa.Table):
        yield from iter_rows(source)
        return
    for batch in source:
        yield from iter_rows(batch)


def _snapshot_source(
    source: RowSource,
    *,
    repo: str | None,
    commit: str | None,
    columns: Sequence[str],
) -> RowSource:
    if not isinstance(source, pa.Table):
        return source
    if not set(columns).issubset(source.column_names):
        return source
    return snapshot_table(
        source,
        repo=repo,
        commit=commit,
        columns=columns,
    )


def _goids_source(
    source: RowSource,
    *,
    repo: str | None,
    commit: str | None,
) -> RowSource:
    if not isinstance(source, pa.Table):
        return source
    required = (
        "goid_h128",
        "rel_path",
        "qualname",
        "start_line",
        "end_line",
        "urn",
        "kind",
    )
    if not set(required).issubset(source.column_names):
        return source
    plan = snapshot_plan(source, repo=repo, commit=commit, columns=required)
    plan = plan.filter(E.in_("kind", sorted(_FUNCTION_KINDS)))
    return materialize_plan(plan, use_threads=True)


__all__ = [
    "catalog_provider_from_frames",
    "module_map_from_frame",
]
