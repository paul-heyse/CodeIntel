"""Catalog port interface for function catalog access.

This module provides backward-compatible exports for catalog operations.
New code should use the unified types from ``codeintel.graphs.catalog``.

Data Classes
------------
- FunctionSpanData: Deprecated alias for FunctionSpan

Deprecated
----------
- CatalogPort: Use CatalogService from ``codeintel.graphs.catalog`` directly
- FunctionSpanData: Use FunctionSpan from ``codeintel.graphs.catalog``

.. deprecated:: 5.0.0
    The CatalogPort protocol and FunctionSpanData are deprecated.
    Use CatalogService and FunctionSpan from ``codeintel.graphs.catalog``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from codeintel.graphs.catalog import FunctionSpan

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _create_function_span_data(**kwargs: object) -> FunctionSpan:
    """Create a FunctionSpan from legacy FunctionSpanData arguments.

    .. deprecated:: 5.0.0
        Use FunctionSpan from ``codeintel.graphs.catalog`` directly.

    Parameters
    ----------
    **kwargs
        Arguments matching FunctionSpanData fields.

    Returns
    -------
    FunctionSpan
        Unified function span.
    """
    warnings.warn(
        "FunctionSpanData is deprecated. Use FunctionSpan from codeintel.graphs.catalog.",
        DeprecationWarning,
        stacklevel=2,
    )
    return FunctionSpan(
        goid=int(kwargs["goid"]),  # type: ignore[arg-type]
        rel_path=str(kwargs["rel_path"]),
        qualname=str(kwargs["qualname"]),
        start_line=int(kwargs["start_line"]),  # type: ignore[arg-type]
        end_line=int(kwargs["end_line"]),  # type: ignore[arg-type]
        urn=kwargs.get("urn"),  # type: ignore[arg-type]
    )


class _FunctionSpanDataCompat:
    """Compatibility class providing FunctionSpanData-like construction.

    .. deprecated:: 5.0.0
        Use FunctionSpan from ``codeintel.graphs.catalog`` directly.
    """

    def __new__(  # noqa: PLR0913, PLR0917 - matches legacy dataclass signature
        cls,
        goid: int,
        rel_path: str,
        qualname: str,
        start_line: int,
        end_line: int,
        urn: str | None = None,
    ) -> FunctionSpan:
        """Create a FunctionSpan from legacy FunctionSpanData arguments."""
        return _create_function_span_data(
            goid=goid,
            rel_path=rel_path,
            qualname=qualname,
            start_line=start_line,
            end_line=end_line,
            urn=urn,
        )


# Deprecated alias - use FunctionSpan directly
FunctionSpanData = _FunctionSpanDataCompat


# Type alias for backward compatibility in type annotations
FunctionSpanDataType = FunctionSpan


@runtime_checkable
class CatalogPort(Protocol):
    """Protocol for function catalog operations.

    .. deprecated:: 5.0.0
        Use CatalogService from ``codeintel.graphs.catalog`` directly.
        The protocol is retained for backward compatibility but new code should
        use CatalogService directly.

    Implementations provide access to function spans, GOID lookups,
    and module mappings without exposing storage details.
    """

    @property
    def function_spans(self) -> Sequence[FunctionSpan]:
        """All function spans in the catalog.

        Returns
        -------
        Sequence[FunctionSpan]
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

    def spans_for_path(self, rel_path: str) -> Sequence[FunctionSpan]:
        """Get function spans for a specific file.

        Parameters
        ----------
        rel_path
            Relative file path.

        Returns
        -------
        Sequence[FunctionSpan]
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
    "FunctionSpan",  # Re-export for convenience
    "FunctionSpanData",  # Deprecated alias
    "FunctionSpanDataType",  # Type alias for annotations
]
