"""Catalog adapter implementing CatalogPort.

This module provides a concrete implementation of CatalogPort that
wraps the existing FunctionCatalog infrastructure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.graphs.ports.catalog import FunctionSpanData

if TYPE_CHECKING:
    from codeintel.graphs.catalog import FunctionCatalog


@dataclass
class CatalogAdapter:
    """CatalogPort implementation wrapping FunctionCatalog.

    Attributes
    ----------
    catalog
        Underlying FunctionCatalog instance.
    """

    catalog: FunctionCatalog

    @property
    def function_spans(self) -> Sequence[FunctionSpanData]:
        """All function spans in the catalog.

        Returns
        -------
        Sequence[FunctionSpanData]
            All indexed function spans.
        """
        spans = self.catalog.function_spans
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


__all__ = ["CatalogAdapter"]
