"""Catalog resource provider.

This module provides a resource provider for function catalog access,
bridging the hexagonal architecture (ports/adapters/resources) with
the production FunctionCatalog implementation in `graphs.catalog`.

Integration Architecture
------------------------
- FunctionCatalog (graphs.catalog): Production implementation with all logic
- CatalogResource (this module): Resource provider wrapper for DI/context
- CatalogPort (ports/catalog.py): Protocol interface for abstraction

The CatalogResource wraps FunctionCatalog and exposes it via the CatalogPort
protocol, enabling:
- Dependency injection in GraphPluginExecutionContext
- Protocol-based testing with mock implementations
- Clean separation of production logic from resource lifecycle
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from codeintel.graphs.ports.catalog import FunctionSpanData

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.graphs.catalog import FunctionCatalog
    from codeintel.graphs.ports.catalog import CatalogPort


@dataclass
class CatalogResource:
    """Resource provider for function catalog access.

    Implements both ResourceProvider and CatalogPort protocols,
    providing unified access to function catalog operations.

    Attributes
    ----------
    catalog
        Underlying function catalog.
    """

    RESOURCE_NAME: ClassVar[str] = "catalog"

    catalog: FunctionCatalog
    _cached_spans: tuple[FunctionSpanData, ...] | None = None

    @property
    def resource_name(self) -> str:
        """Resource identifier.

        Returns
        -------
        str
            Resource name.
        """
        return self.RESOURCE_NAME

    def get(self) -> CatalogPort:
        """Get the catalog as a CatalogPort.

        Returns
        -------
        CatalogPort
            Catalog port interface.
        """
        return self

    def invalidate(self) -> None:
        """Invalidate cached data."""
        self._cached_spans = None

    @property
    def cached_spans(self) -> tuple[FunctionSpanData, ...] | None:
        """Return cached spans if already materialized."""
        return self._cached_spans

    @property
    def function_spans(self) -> Sequence[FunctionSpanData]:
        """All function spans in the catalog.

        Returns
        -------
        Sequence[FunctionSpanData]
            All indexed function spans.
        """
        if self._cached_spans is None:
            spans = self.catalog.function_spans
            self._cached_spans = tuple(
                FunctionSpanData(
                    goid=span.goid,
                    rel_path=span.rel_path,
                    qualname=span.qualname,
                    start_line=span.start_line,
                    end_line=span.end_line,
                    urn=self.catalog.urn_for_goid(span.goid),
                )
                for span in spans
            )
        return self._cached_spans

    @property
    def paths(self) -> Sequence[str]:
        """All file paths with indexed functions.

        Returns
        -------
        Sequence[str]
            Unique file paths containing functions.
        """
        return tuple(self.catalog.function_index.paths())

    @property
    def module_by_path(self) -> Mapping[str, str]:
        """Mapping of file paths to module names.

        Returns
        -------
        Mapping[str, str]
            File path to module name mapping.
        """
        return self.catalog.module_by_path

    def spans_for_path(self, rel_path: str) -> Sequence[FunctionSpanData]:
        """Get function spans for a specific file.

        Parameters
        ----------
        rel_path
            Relative file path.

        Returns
        -------
        Sequence[FunctionSpanData]
            Function spans in the file.
        """
        spans = self.catalog.function_index.spans_for_path(rel_path)
        return tuple(
            FunctionSpanData(
                goid=span.goid,
                rel_path=span.rel_path,
                qualname=span.qualname,
                start_line=span.start_line,
                end_line=span.end_line,
                urn=self.catalog.urn_for_goid(span.goid),
            )
            for span in spans
        )

    def local_name_map(self, rel_path: str) -> Mapping[str, int]:
        """Get local name to GOID mapping for a file.

        Parameters
        ----------
        rel_path
            Relative file path.

        Returns
        -------
        Mapping[str, int]
            Local function name to GOID mapping.
        """
        return self.catalog.function_index.local_name_map(rel_path)

    def lookup_goid(
        self,
        rel_path: str,
        start_line: int,
        end_line: int | None = None,
        qualname: str | None = None,
    ) -> int | None:
        """Look up a GOID by file position and optional qualname.

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
            GOID if found, None otherwise.
        """
        return self.catalog.lookup_goid(rel_path, start_line, end_line, qualname)

    def urn_for_goid(self, goid: int) -> str | None:
        """Get URN for a GOID.

        Parameters
        ----------
        goid
            Global object identifier.

        Returns
        -------
        str | None
            URN if found, None otherwise.
        """
        return self.catalog.urn_for_goid(goid)


__all__ = ["CatalogResource"]
