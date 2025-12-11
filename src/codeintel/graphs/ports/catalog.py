"""Catalog port interface for function catalog access.

This module defines the CatalogPort protocol that abstracts function
catalog operations, providing span lookups and module mappings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class FunctionSpanData:
    """Function span data for catalog operations.

    Attributes
    ----------
    goid
        Global object identifier.
    rel_path
        Relative file path.
    qualname
        Fully qualified name.
    start_line
        Starting line number.
    end_line
        Ending line number.
    urn
        Optional URN identifier.
    """

    goid: int
    rel_path: str
    qualname: str
    start_line: int
    end_line: int
    urn: str | None = None

    @property
    def local_name(self) -> str:
        """Extract the local (unqualified) function name.

        Returns
        -------
        str
            Local function name without module/class prefix.
        """
        return self.qualname.rsplit(".", maxsplit=1)[-1]


@runtime_checkable
class CatalogPort(Protocol):
    """Protocol for function catalog operations.

    Implementations provide access to function spans, GOID lookups,
    and module mappings without exposing storage details.
    """

    @property
    def function_spans(self) -> Sequence[FunctionSpanData]:
        """All function spans in the catalog.

        Returns
        -------
        Sequence[FunctionSpanData]
            All indexed function spans.
        """
        ...

    @property
    def paths(self) -> Sequence[str]:
        """All file paths with indexed functions.

        Returns
        -------
        Sequence[str]
            Unique file paths containing functions.
        """
        ...

    @property
    def module_by_path(self) -> Mapping[str, str]:
        """Mapping of file paths to module names.

        Returns
        -------
        Mapping[str, str]
            File path to module name mapping.
        """
        ...

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
        ...

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
        ...

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
        ...

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
        ...


__all__ = [
    "CatalogPort",
    "FunctionSpanData",
]
