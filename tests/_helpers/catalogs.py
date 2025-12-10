"""Helpers for working with function catalogs in tests."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from codeintel.graphs.catalog import FunctionCatalog, FunctionCatalogService, FunctionMeta
from tests._helpers.context import TestContext
from tests._helpers.fakes.function_catalogs import MockFunctionCatalog

CatalogLike = FunctionCatalog | FunctionCatalogService | MockFunctionCatalog


def _normalize_catalog(catalog: CatalogLike) -> FunctionCatalog | MockFunctionCatalog:
    if isinstance(catalog, (FunctionCatalog, MockFunctionCatalog)):
        return catalog
    if hasattr(catalog, "catalog"):
        maybe = catalog.catalog
        if callable(maybe):
            return maybe()  # type: ignore[no-any-return]
        return maybe  # type: ignore[return-value]
    if hasattr(catalog, "get"):
        return catalog.get()  # type: ignore[no-any-return]
    raise TypeError(f"Unsupported catalog type: {type(catalog)}")


def _iter_functions(catalog: FunctionCatalog | MockFunctionCatalog) -> Iterable[FunctionMeta]:
    """Yield FunctionMeta entries from a catalog using public accessors when possible."""
    funcs_by_path = getattr(catalog, "functions_by_path", {})
    if funcs_by_path:
        for funcs in funcs_by_path.values():
            yield from funcs
        return

    direct = getattr(catalog, "_functions", None)
    if direct:
        yield from direct
        return

    funcs_attr = getattr(catalog, "functions", None)
    if funcs_attr:
        yield from funcs_attr


def seed_goids_from_catalog(ctx: TestContext, catalog: CatalogLike) -> None:
    """Insert core.goids rows for every function in a FunctionCatalog."""
    catalog_obj = _normalize_catalog(catalog)
    now = datetime.now(tz=UTC)
    rows: list[tuple[object, ...]] = []
    for func in _iter_functions(catalog_obj):
        rows.append(
            (
                func.urn,
                ctx.repo,
                ctx.commit,
                func.rel_path or "",
                "python",
                "function",
                func.qualname,
                func.goid,
                func.start_line or 1,
                func.end_line or func.start_line or 1,
                now,
            )
        )
    if not rows:
        return
    ctx.gateway.con.executemany(
        """
        INSERT INTO core.goids (
            urn,
            repo,
            commit,
            rel_path,
            language,
            kind,
            qualname,
            goid_h128,
            start_line,
            end_line,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def ensure_catalog_with_goids(ctx: TestContext, catalog: CatalogLike) -> CatalogLike:
    """Normalize a catalog/provider and ensure GOIDs are seeded for its functions."""
    catalog_obj = _normalize_catalog(catalog)
    seed_goids_from_catalog(ctx, catalog_obj)
    return catalog_obj


__all__ = [
    "ensure_catalog_with_goids",
    "seed_goids_from_catalog",
]
