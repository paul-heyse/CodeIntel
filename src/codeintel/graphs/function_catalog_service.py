"""Service wrapper for function span/catalog access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from codeintel.graphs.function_catalog import (
    FunctionCatalog,
    FunctionMeta,
    FunctionSpan,
    load_function_catalog,
)
from codeintel.storage.gateway import StorageGateway


class FunctionCatalogProvider(Protocol):
    """Protocol for objects that supply a function catalog."""

    def catalog(self) -> FunctionCatalog:
        """
        Return a function catalog instance.

        Returns
        -------
        FunctionCatalog
            Catalog containing spans and module mapping.
        """
        raise NotImplementedError

    def urn_for_goid(self, goid: int) -> str | None:
        """
        Return a URN for a GOID when available.

        Returns
        -------
        str | None
            URN string if present in the catalog.
        """
        raise NotImplementedError

    def lookup_goid(
        self, rel_path: str, start_line: int, end_line: int | None, qualname: str | None
    ) -> int | None:
        """
        Resolve a GOID for a span and optional qualname.

        Returns
        -------
        int | None
            GOID when found, otherwise None.
        """
        raise NotImplementedError


@dataclass
class FunctionCatalogService(FunctionCatalogProvider):
    """Typed wrapper around FunctionCatalog construction and access."""

    _catalog: FunctionCatalog

    @classmethod
    def from_db(cls, gateway: StorageGateway, *, repo: str, commit: str) -> FunctionCatalogService:
        """
        Load catalog state for a repo snapshot from a storage gateway.

        Returns
        -------
        FunctionCatalogService
            Service wrapping the loaded catalog.
        """
        return cls(load_function_catalog(gateway, repo=repo, commit=commit))

    def catalog(self) -> FunctionCatalog:
        """
        Return the underlying catalog instance.

        Returns
        -------
        FunctionCatalog
            Backing catalog with spans and module mapping.
        """
        return self._catalog

    def local_name_map(self, rel_path: str) -> dict[str, int]:
        """
        Return local name map for a given relative path.

        Returns
        -------
        dict[str, int]
            Mapping of local names to GOIDs for the path.
        """
        return self._catalog.function_index.local_name_map(rel_path)

    def functions_by_path(self) -> dict[str, list[FunctionMeta]]:
        """
        Return functions keyed by path.

        Returns
        -------
        dict[str, list[FunctionMeta]]
            Mapping of normalized path to function metadata.
        """
        return self._catalog.functions_by_path

    def urn_for_goid(self, goid: int) -> str | None:
        """
        Return URN for GOID when present.

        Returns
        -------
        str | None
            URN string when available.
        """
        return self._catalog.urn_for_goid(goid)

    def lookup_goid(
        self, rel_path: str, start_line: int, end_line: int | None, qualname: str | None
    ) -> int | None:
        """
        Resolve a GOID using span and optional qualname.

        Returns
        -------
        int | None
            GOID when found, otherwise None.
        """
        return self._catalog.function_index.lookup(rel_path, start_line, end_line, qualname)

    def module_for_path(self, rel_path: str) -> str | None:
        """
        Return module name for a given relative path.

        Returns
        -------
        str | None
            Module name if known.
        """
        return self._catalog.module_by_path.get(rel_path)

    @property
    def index(self) -> object:
        """Expose the underlying FunctionSpanIndex for advanced consumers."""
        return self._catalog.function_index

    @property
    def spans(self) -> list[FunctionSpan]:
        """
        Return all function spans in the catalog.

        Returns
        -------
        list[FunctionSpan]
            Spans for functions/methods in the repo snapshot.
        """
        return self._catalog.function_spans
