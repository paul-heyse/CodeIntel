"""Catalog protocol definitions.

This module provides protocols for catalog access and lazy loading,
enabling dependency injection and testability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.core.catalog.function_span import FunctionSpan
    from codeintel.core.catalog.span_index import SpanIndex


@runtime_checkable
class CatalogProtocol(Protocol):
    """Protocol for function catalog access.

    Implementations provide access to function spans, URNs, and module
    mappings for a repository snapshot.

    Examples
    --------
    >>> def analyze(catalog: CatalogProtocol) -> int:
    ...     return len(catalog.function_spans)
    """

    @property
    def function_spans(self) -> Sequence[FunctionSpan]:
        """Return all function spans in the catalog.

        Returns
        -------
        Sequence[FunctionSpan]
            All indexed function spans.
        """
        ...

    @property
    def function_index(self) -> SpanIndex:
        """Return the span lookup index.

        Returns
        -------
        SpanIndex
            Index supporting path/span lookups.
        """
        ...

    @property
    def module_by_path(self) -> Mapping[str, str]:
        """Return module name mapping keyed by normalized path.

        Returns
        -------
        Mapping[str, str]
            Mapping of relative path to module name.
        """
        ...

    def lookup_goid(
        self,
        rel_path: str,
        start_line: int,
        end_line: int | None = None,
        qualname: str | None = None,
    ) -> int | None:
        """Resolve GOID from span and optional qualname.

        Parameters
        ----------
        rel_path
            Relative file path.
        start_line
            Starting line number.
        end_line
            Optional ending line number.
        qualname
            Optional qualified name for disambiguation.

        Returns
        -------
        int | None
            GOID value when resolved, otherwise None.
        """
        ...

    def urn_for_goid(self, goid: int) -> str | None:
        """Return URN for GOID when present.

        Parameters
        ----------
        goid
            Global object identifier.

        Returns
        -------
        str | None
            URN string when available.
        """
        ...

    def module_for_path(self, rel_path: str) -> str | None:
        """Return module name for a given relative path.

        Parameters
        ----------
        rel_path
            Relative file path.

        Returns
        -------
        str | None
            Module name if known.
        """
        ...


@runtime_checkable
class CatalogProviderProtocol(Protocol):
    """Protocol for lazy catalog loading.

    Implementations provide lazy loading of catalogs with caching
    and invalidation support.

    Examples
    --------
    >>> class MyCatalogProvider:
    ...     RESOURCE_NAME: ClassVar[str] = "catalog"
    ...
    ...     def get(self) -> CatalogProtocol:
    ...         return self._load_catalog()
    ...
    ...     def invalidate(self) -> None:
    ...         self._cached = None
    """

    RESOURCE_NAME: ClassVar[str]

    def get(self) -> CatalogProtocol:
        """Return the catalog instance.

        Returns
        -------
        CatalogProtocol
            The loaded or cached catalog.
        """
        ...

    def invalidate(self) -> None:
        """Invalidate cached catalog data.

        After calling this, the next get() call will reload the catalog.
        """
        ...


__all__ = [
    "CatalogProtocol",
    "CatalogProviderProtocol",
]
