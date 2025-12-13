"""Unified function catalog for graph builders.

This module consolidates function span indexing and the catalog service
into a single cohesive module.

Key Components
--------------
- FunctionSpan: Unified function metadata (GOID, path, qualname, lines, optional URN)
- FunctionSpanIndex: Lookup structure for resolving GOIDs from file spans
- FunctionCatalog: Centralized access to spans, URNs, and module mappings
- FunctionCatalogProvider: Protocol for catalog access (DI)
- CatalogService: Unified service for graphs and analytics (replaces FunctionCatalogService)

Hexagonal Architecture Integration
----------------------------------
This module is the production implementation of function catalog functionality.
CatalogService implements both the FunctionCatalogProvider protocol and the
resource provider pattern, enabling dependency injection and testability.

For plugin execution, prefer using CatalogService via ctx.require() rather
than direct imports.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

from codeintel.ingestion.infrastructure.paths import normalize_rel_path
from codeintel.storage.helpers.module_index import load_module_map
from codeintel.storage.ibis_types import filter_by, ibis_bool

if TYPE_CHECKING:
    from collections.abc import Callable as TypingCallable
    from collections.abc import Iterable, Mapping, Sequence

    import pandas as pd

    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class FunctionSpan:
    """Unified function span representation with optional URN.

    Attributes
    ----------
    goid
        Global object identifier (128-bit hash).
    rel_path
        Relative file path within the repository.
    qualname
        Fully qualified name of the function.
    start_line
        Starting line number (1-indexed).
    end_line
        Ending line number (1-indexed).
    urn
        Optional URN identifier. Populated when loaded via catalog.
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


class FunctionSpanIndex:
    """Lookup structure for resolving GOIDs from file spans."""

    def __init__(self, spans: Iterable[FunctionSpan]) -> None:
        """
        Initialize the index from an iterable of function spans.

        Parameters
        ----------
        spans : Iterable[FunctionSpan]
            Function spans to index.
        """
        self._by_path: dict[str, list[FunctionSpan]] = {}
        for span in spans:
            path = normalize_rel_path(span.rel_path)
            self._by_path.setdefault(path, []).append(span)

        for path_spans in self._by_path.values():
            path_spans.sort(key=lambda s: (s.start_line, s.end_line))

    def paths(self) -> list[str]:
        """
        Return paths with at least one function span.

        Returns
        -------
        list[str]
            Paths present in the index.
        """
        return list(self._by_path.keys())

    def spans_for_path(self, rel_path: str) -> list[FunctionSpan]:
        """
        Return spans for a given relative path.

        Returns
        -------
        list[FunctionSpan]
            Spans for the requested path (empty when missing).
        """
        return list(self._by_path.get(normalize_rel_path(rel_path), []))

    def local_name_map(self, rel_path: str) -> dict[str, int]:
        """
        Map local names and qualnames to GOIDs for a single file.

        Returns
        -------
        dict[str, int]
            Mapping from short/qualified names to GOIDs.
        """
        mapping: dict[str, int] = {}
        for span in self.spans_for_path(rel_path):
            local_name = span.qualname.rsplit(".", maxsplit=1)[-1]
            mapping.setdefault(local_name, span.goid)
            mapping.setdefault(span.qualname, span.goid)
        return mapping

    def lookup(
        self,
        rel_path: str,
        start_line: int,
        end_line: int | None = None,
        qualname: str | None = None,
    ) -> int | None:
        """
        Resolve a GOID for the given path and span.

        Resolution order favors exact span matches, then qualname matches
        overlapping the span, then any enclosing span, and finally a fallback
        to functions starting on the same line.

        Returns
        -------
        int | None
            GOID when found; otherwise None.
        """
        spans_list = self._by_path.get(normalize_rel_path(rel_path))
        if spans_list is None:
            return None
        spans: list[FunctionSpan] = spans_list
        if not spans:
            return None

        start = int(start_line)
        end = int(end_line) if end_line is not None else start

        def _first_match(predicate: TypingCallable[[FunctionSpan], bool]) -> int | None:
            for span in spans:
                if predicate(span):
                    return span.goid
            return None

        predicates: list[TypingCallable[[FunctionSpan], bool]] = []
        if qualname:
            predicates.append(
                lambda span: span.start_line == start
                and span.end_line == end
                and _qualname_matches(span.qualname, qualname)
            )
        predicates.append(lambda span: span.start_line == start and span.end_line == end)
        if qualname:
            predicates.append(
                lambda span: _qualname_matches(span.qualname, qualname)
                and span.start_line <= start <= span.end_line
            )
        predicates.append(lambda span: span.start_line <= start <= span.end_line)
        predicates.append(lambda span: span.start_line == start)

        for predicate in predicates:
            match = _first_match(predicate)
            if match is not None:
                return match
        return None


def _qualname_matches(full: str, candidate: str) -> bool:
    """
    Check if a qualname matches a candidate.

    Returns
    -------
    bool
        True if the candidate matches the full qualname.
    """
    if full == candidate:
        return True
    suffix = candidate.rsplit(".", maxsplit=1)[-1]
    return full.endswith(f".{suffix}")


def load_function_spans(gateway: StorageGateway, *, repo: str, commit: str) -> list[FunctionSpan]:
    """
    Load function spans from `core.goids` for a repo snapshot.

    Returns
    -------
    list[FunctionSpan]
        Normalized function spans keyed by GOID.
    """
    goids = cast("Any", gateway.ibis.table("core.goids"))
    filtered = filter_by(
        goids,
        ibis_bool(goids.repo == repo),
        ibis_bool(goids.commit == commit),
        ibis_bool(goids.kind.isin(["function", "method"])),
    )
    df = cast(
        "pd.DataFrame",
        filtered.select("goid_h128", "rel_path", "qualname", "start_line", "end_line").execute(),
    )
    rows = df.to_dict(orient="records")
    spans: list[FunctionSpan] = []
    for row in rows:
        start_line = row["start_line"]
        end_line = row["end_line"]
        if start_line is None:
            continue
        spans.append(
            FunctionSpan(
                goid=int(row["goid_h128"]),
                rel_path=normalize_rel_path(row["rel_path"]),
                qualname=str(row["qualname"]),
                start_line=int(start_line),
                end_line=int(end_line) if end_line is not None else int(start_line),
            )
        )
    return spans


def load_function_index(gateway: StorageGateway, *, repo: str, commit: str) -> FunctionSpanIndex:
    """
    Create a `FunctionSpanIndex` from DuckDB state.

    Returns
    -------
    FunctionSpanIndex
        Index seeded from `core.goids` for the repo/commit snapshot.
    """
    return FunctionSpanIndex(load_function_spans(gateway, repo=repo, commit=commit))


def _create_function_meta(**kwargs: object) -> FunctionSpan:
    """Create a FunctionSpan from legacy FunctionMeta arguments.

    .. deprecated:: 5.0.0
        Use FunctionSpan directly with urn parameter.

    Parameters
    ----------
    **kwargs
        Arguments matching FunctionMeta fields (goid, urn, rel_path, qualname,
        start_line, end_line).

    Returns
    -------
    FunctionSpan
        Unified function span.
    """
    warnings.warn(
        "FunctionMeta is deprecated. Use FunctionSpan with urn parameter instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return FunctionSpan(
        goid=int(kwargs["goid"]),  # type: ignore[arg-type]
        rel_path=str(kwargs["rel_path"]),
        qualname=str(kwargs["qualname"]),
        start_line=int(kwargs["start_line"]),  # type: ignore[arg-type]
        end_line=int(kwargs["end_line"]),  # type: ignore[arg-type]
        urn=str(kwargs["urn"]),
    )


class _FunctionMetaCompat:
    """Compatibility class providing FunctionMeta-like construction.

    .. deprecated:: 5.0.0
        Use FunctionSpan directly with urn parameter.
    """

    def __new__(  # noqa: PLR0913, PLR0917 - matches legacy dataclass signature
        cls,
        goid: int,
        urn: str,
        rel_path: str,
        qualname: str,
        start_line: int,
        end_line: int,
    ) -> FunctionSpan:
        """Create a FunctionSpan from legacy FunctionMeta arguments."""
        return _create_function_meta(
            goid=goid,
            urn=urn,
            rel_path=rel_path,
            qualname=qualname,
            start_line=start_line,
            end_line=end_line,
        )


# Deprecated alias - use FunctionSpan directly
FunctionMeta = _FunctionMetaCompat


class FunctionCatalog:
    """Centralized access to function spans, URNs, and module mappings."""

    def __init__(
        self,
        *,
        functions: Iterable[FunctionSpan],
        module_by_path: dict[str, str],
    ) -> None:
        """
        Initialize the catalog from function spans and module mapping.

        Parameters
        ----------
        functions : Iterable[FunctionSpan]
            Function spans to catalog (with optional URN populated).
        module_by_path : dict[str, str]
            Mapping of file paths to module names.
        """
        self._functions: list[FunctionSpan] = list(functions)
        self._index = FunctionSpanIndex(self._functions)
        self._urn_by_goid = {
            fn.goid: fn.urn for fn in self._functions if fn.urn is not None
        }
        self._module_by_path = {
            normalize_rel_path(path): mod for path, mod in module_by_path.items()
        }
        self._funcs_by_path: dict[str, list[FunctionSpan]] = {}
        for fn in self._functions:
            self._funcs_by_path.setdefault(fn.rel_path, []).append(fn)

    @property
    def function_spans(self) -> list[FunctionSpan]:
        """Return all function spans in the catalog."""
        spans: list[FunctionSpan] = []
        for path in self._index.paths():
            spans.extend(self._index.spans_for_path(path))
        return spans

    @property
    def function_index(self) -> FunctionSpanIndex:
        """
        Return the span lookup index.

        Returns
        -------
        FunctionSpanIndex
            Index supporting path/span lookups.
        """
        return self._index

    @property
    def functions_by_path(self) -> dict[str, list[FunctionSpan]]:
        """
        Return functions keyed by normalized path.

        Returns
        -------
        dict[str, list[FunctionSpan]]
            Mapping of relative path to function spans.
        """
        return self._funcs_by_path

    @property
    def module_by_path(self) -> dict[str, str]:
        """
        Return module name mapping keyed by normalized path.

        Returns
        -------
        dict[str, str]
            Mapping of relative path to module name.
        """
        return self._module_by_path

    def urn_for_goid(self, goid: int) -> str | None:
        """
        Return URN for a GOID if known.

        Returns
        -------
        str | None
            URN string when present, otherwise None.
        """
        return self._urn_by_goid.get(goid)

    def lookup_goid(
        self, rel_path: str, start_line: int, end_line: int | None, qualname: str | None
    ) -> int | None:
        """
        Resolve GOID from span and optional qualname.

        Returns
        -------
        int | None
            GOID value when resolved, otherwise None.
        """
        return self._index.lookup(rel_path, start_line, end_line, qualname)


def load_function_catalog(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
) -> FunctionCatalog:
    """
    Load function spans and module map for a repo snapshot via a gateway.

    Returns
    -------
    FunctionCatalog
        Catalog containing spans, URNs, and module mapping.
    """
    goids = cast("Any", gateway.ibis.table("core.goids"))
    goids_filtered = filter_by(
        goids,
        ibis_bool(goids.repo == repo),
        ibis_bool(goids.commit == commit),
        ibis_bool(goids.kind.isin(["function", "method"])),
    )
    df = cast(
        "pd.DataFrame",
        goids_filtered.select(
            "goid_h128", "urn", "rel_path", "qualname", "start_line", "end_line"
        ).execute(),
    )
    rows = df.to_dict(orient="records")

    functions: list[FunctionSpan] = []
    for row in rows:
        start_line = row["start_line"]
        end_line = row["end_line"]
        if start_line is None:
            continue
        end_val = int(end_line) if end_line is not None else int(start_line)
        functions.append(
            FunctionSpan(
                goid=int(row["goid_h128"]),
                rel_path=normalize_rel_path(row["rel_path"]),
                qualname=str(row["qualname"]),
                start_line=int(start_line),
                end_line=end_val,
                urn=str(row["urn"]),
            )
        )

    module_by_path = load_module_map(gateway, repo, commit)
    return FunctionCatalog(functions=functions, module_by_path=module_by_path)


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

    def module_for_path(self, rel_path: str) -> str | None:
        """
        Return module name for a given relative path.

        Returns
        -------
        str | None
            Module name if known.
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
class CatalogService(FunctionCatalogProvider):
    """Unified catalog access for graphs and analytics.

    This class consolidates the functionality of the former FunctionCatalogService
    and CatalogResource into a single service that implements:

    - FunctionCatalogProvider protocol for analytics
    - ResourceProvider pattern for dependency injection
    - CatalogPort-equivalent methods for graph plugins

    Attributes
    ----------
    RESOURCE_NAME
        Resource identifier for DI registration.
    _catalog
        Underlying function catalog.
    """

    RESOURCE_NAME: ClassVar[str] = "catalog"

    _catalog: FunctionCatalog
    _cached_spans: tuple[FunctionSpan, ...] | None = None

    @classmethod
    def from_db(cls, gateway: StorageGateway, *, repo: str, commit: str) -> CatalogService:
        """
        Load catalog state for a repo snapshot from a storage gateway.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        repo
            Repository identifier.
        commit
            Commit hash.

        Returns
        -------
        CatalogService
            Service wrapping the loaded catalog.
        """
        return cls(load_function_catalog(gateway, repo=repo, commit=commit))

    def get(self) -> CatalogService:
        """Return self to satisfy ResourceProvider interface.

        Returns
        -------
        CatalogService
            Self reference.
        """
        return self

    def invalidate(self) -> None:
        """Invalidate cached data."""
        self._cached_spans = None

    @property
    def resource_name(self) -> str:
        """Resource identifier.

        Returns
        -------
        str
            Resource name.
        """
        return self.RESOURCE_NAME

    def catalog(self) -> FunctionCatalog:
        """Return the underlying catalog instance.

        Returns
        -------
        FunctionCatalog
            Backing catalog with spans and module mapping.
        """
        return self._catalog

    @property
    def function_spans(self) -> Sequence[FunctionSpan]:
        """All function spans in the catalog.

        Returns
        -------
        Sequence[FunctionSpan]
            All indexed function spans.
        """
        if self._cached_spans is None:
            self._cached_spans = tuple(self._catalog.function_spans)
        return self._cached_spans

    @property
    def paths(self) -> Sequence[str]:
        """All file paths with indexed functions.

        Returns
        -------
        Sequence[str]
            Unique file paths containing functions.
        """
        return tuple(self._catalog.function_index.paths())

    @property
    def module_by_path(self) -> Mapping[str, str]:
        """Mapping of file paths to module names.

        Returns
        -------
        Mapping[str, str]
            File path to module name mapping.
        """
        return self._catalog.module_by_path

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
        return tuple(self._catalog.function_index.spans_for_path(rel_path))

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
        return self._catalog.function_index.local_name_map(rel_path)

    def get_functions_by_path(self) -> dict[str, list[FunctionSpan]]:
        """
        Return functions keyed by path.

        Returns
        -------
        dict[str, list[FunctionSpan]]
            Mapping of normalized path to function spans.
        """
        return self._catalog.functions_by_path

    def urn_for_goid(self, goid: int) -> str | None:
        """
        Return URN for GOID when present.

        Parameters
        ----------
        goid
            Global object identifier.

        Returns
        -------
        str | None
            URN string when available.
        """
        return self._catalog.urn_for_goid(goid)

    def lookup_goid(
        self,
        rel_path: str,
        start_line: int,
        end_line: int | None = None,
        qualname: str | None = None,
    ) -> int | None:
        """
        Resolve a GOID using span and optional qualname.

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
            GOID when found, otherwise None.
        """
        return self._catalog.function_index.lookup(rel_path, start_line, end_line, qualname)

    def module_for_path(self, rel_path: str) -> str | None:
        """
        Return module name for a given relative path.

        Parameters
        ----------
        rel_path
            Relative file path.

        Returns
        -------
        str | None
            Module name if known.
        """
        return self._catalog.module_by_path.get(rel_path)

    @property
    def index(self) -> FunctionSpanIndex:
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


def FunctionCatalogService(catalog: FunctionCatalog) -> CatalogService:  # noqa: N802
    """Create a CatalogService from a FunctionCatalog.

    .. deprecated:: 5.0.0
        Use CatalogService directly.

    Parameters
    ----------
    catalog
        Function catalog to wrap.

    Returns
    -------
    CatalogService
        Unified catalog service.
    """
    warnings.warn(
        "FunctionCatalogService is deprecated. Use CatalogService directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return CatalogService(catalog)


# Keep a type alias for backward compatibility in type annotations
FunctionCatalogServiceType = CatalogService


__all__ = [
    "CatalogService",
    "FunctionCatalog",
    "FunctionCatalogProvider",
    "FunctionCatalogService",  # Deprecated alias for CatalogService
    "FunctionMeta",  # Deprecated alias for FunctionSpan
    "FunctionSpan",
    "FunctionSpanIndex",
    "load_function_catalog",
    "load_function_index",
    "load_function_spans",
]
