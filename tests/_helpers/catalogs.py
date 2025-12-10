"""Helpers for working with function catalogs in tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.catalog import FunctionCatalog, FunctionCatalogService, FunctionMeta
from tests._helpers.fakes.function_catalogs import MockFunctionCatalog

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.context import TestContext


class CatalogCtx(Protocol):
    """Minimal context required to seed GOIDs."""

    gateway: StorageGateway
    repo: str
    commit: str


CatalogLike = FunctionCatalog | FunctionCatalogService | MockFunctionCatalog
CatalogInput = CatalogLike | object
type CatalogCtxLike = CatalogCtx | "TestContext"


def _normalize_catalog(catalog: CatalogInput) -> FunctionCatalog | MockFunctionCatalog:
    """Return a concrete FunctionCatalog from supported providers.

    Parameters
    ----------
    catalog
        Catalog instance or provider wrapper.

    Returns
    -------
    FunctionCatalog | MockFunctionCatalog
        Underlying catalog object for iteration.

    Raises
    ------
    TypeError
        If the catalog type is not supported.
    """
    if isinstance(catalog, (FunctionCatalog, MockFunctionCatalog)):
        return catalog
    if isinstance(catalog, FunctionCatalogService):
        return catalog.catalog()
    maybe = getattr(catalog, "catalog", None)
    if maybe is not None:
        candidate = maybe() if callable(maybe) else maybe
        if isinstance(candidate, (FunctionCatalog, MockFunctionCatalog)):
            return candidate
    getter: Callable[[], Any] | None = getattr(catalog, "get", None)
    if getter is not None:
        candidate = getter()
        if isinstance(candidate, (FunctionCatalog, MockFunctionCatalog)):
            return candidate
    message = f"Unsupported catalog type: {type(catalog)}"
    raise TypeError(message)


def _iter_functions(catalog: FunctionCatalog | MockFunctionCatalog) -> Iterable[FunctionMeta]:
    """Yield FunctionMeta entries from a catalog using public accessors when possible.

    Yields
    ------
    FunctionMeta
        Function metadata entries from the provided catalog.
    """
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


def _build_goid_rows(
    catalog: FunctionCatalog | MockFunctionCatalog,
    *,
    repo: str,
    commit: str,
    kinds: Mapping[int, str] | None = None,
) -> list[tuple[object, ...]]:
    """Build goid rows from a catalog with optional per-goid kind overrides.

    Returns
    -------
    list[tuple[object, ...]]
        Row payloads ready for insertion into `core.goids`.
    """
    kind_map = kinds or {}
    now = datetime.now(tz=UTC)
    return [
        (
            func.urn,
            repo,
            commit,
            func.rel_path or "",
            "python",
            kind_map.get(func.goid) or getattr(func, "kind", None) or "function",
            func.qualname,
            func.goid,
            func.start_line or 1,
            func.end_line or func.start_line or 1,
            now,
        )
        for func in _iter_functions(catalog)
    ]


def seed_goids_from_catalog(ctx: CatalogCtxLike, catalog: CatalogInput) -> None:
    """Insert core.goids rows for every function in a FunctionCatalog."""
    catalog_obj = _normalize_catalog(catalog)
    rows = _build_goid_rows(
        catalog_obj,
        repo=ctx.repo,
        commit=ctx.commit,
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


def ensure_catalog_with_goids(ctx: CatalogCtxLike, catalog: CatalogInput) -> CatalogLike:
    """Normalize a catalog/provider and ensure GOIDs are seeded for its functions.

    Returns
    -------
    CatalogLike
        Catalog object with GOIDs ensured in the database.
    """
    catalog_obj = _normalize_catalog(catalog)
    seed_goids_from_catalog(ctx, catalog_obj)
    return catalog_obj


def seed_goids_for_snapshot(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    catalog: CatalogLike,
    *,
    kinds: Mapping[int, str] | None = None,
) -> CatalogLike:
    """Seed core.goids rows for a snapshot without constructing a full TestContext.

    Returns
    -------
    CatalogLike
        Catalog object that had GOIDs seeded for the given snapshot.
    """
    catalog_obj = _normalize_catalog(catalog)
    rows = _build_goid_rows(
        catalog_obj,
        repo=snapshot.repo,
        commit=snapshot.commit,
        kinds=kinds,
    )
    if rows:
        gateway.con.executemany(
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
    return catalog_obj


__all__ = [
    "ensure_catalog_with_goids",
    "seed_goids_for_snapshot",
    "seed_goids_from_catalog",
]
