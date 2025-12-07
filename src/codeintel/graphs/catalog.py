"""Unified function catalog for graph builders.

This module consolidates function span indexing, function metadata catalog,
and the catalog service into a single cohesive module.

Key Components
--------------
- FunctionSpan: Normalized function metadata (GOID, path, qualname, lines)
- FunctionSpanIndex: Lookup structure for resolving GOIDs from file spans
- FunctionMeta: Extended function metadata including URN
- FunctionCatalog: Centralized access to spans, URNs, and module mappings
- FunctionCatalogProvider: Protocol for catalog access (DI)
- FunctionCatalogService: Service wrapper for catalog operations

Hexagonal Architecture Integration
----------------------------------
This module is the production implementation of function catalog functionality.
It integrates with the hexagonal architecture via:

- resources/catalog.py: CatalogResource wraps FunctionCatalog for DI
- ports/catalog.py: CatalogPort defines the abstraction protocol
- adapters/catalog_adapter.py: CatalogAdapter provides port implementation

For plugin execution, prefer using CatalogResource via ctx.require() rather
than direct imports, enabling dependency injection and testability.
"""

from __future__ import annotations

from collections.abc import Callable as TypingCallable
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from codeintel.ingestion.infrastructure.paths import normalize_rel_path
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.helpers.module_index import load_module_map

# =============================================================================
# Function Span Types (from function_index.py)
# =============================================================================


@dataclass(frozen=True)
class FunctionSpan:
    """Normalized function metadata used by graph builders."""

    goid: int
    rel_path: str
    qualname: str
    start_line: int
    end_line: int


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
    con = gateway.con
    rows = con.execute(
        """
        SELECT goid_h128, rel_path, qualname, start_line, end_line
        FROM core.goids
        WHERE repo = ? AND commit = ?
          AND kind IN ('function', 'method')
        """,
        [repo, commit],
    ).fetchall()

    spans: list[FunctionSpan] = []
    for goid_h128, rel_path, qualname, start_line, end_line in rows:
        if start_line is None:
            continue
        spans.append(
            FunctionSpan(
                goid=int(goid_h128),
                rel_path=normalize_rel_path(rel_path),
                qualname=str(qualname),
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


# =============================================================================
# Function Metadata and Catalog (from function_catalog.py)
# =============================================================================


@dataclass(frozen=True)
class FunctionMeta:
    """Function metadata used across graph builders."""

    goid: int
    urn: str
    rel_path: str
    qualname: str
    start_line: int
    end_line: int


class FunctionCatalog:
    """Centralized access to function spans, URNs, and module mappings."""

    def __init__(
        self,
        *,
        functions: Iterable[FunctionMeta],
        module_by_path: dict[str, str],
    ) -> None:
        """
        Initialize the catalog from function metadata and module mapping.

        Parameters
        ----------
        functions : Iterable[FunctionMeta]
            Function metadata to catalog.
        module_by_path : dict[str, str]
            Mapping of file paths to module names.
        """
        self._functions: list[FunctionMeta] = list(functions)
        self._index = FunctionSpanIndex(
            [
                FunctionSpan(
                    goid=fn.goid,
                    rel_path=fn.rel_path,
                    qualname=fn.qualname,
                    start_line=fn.start_line,
                    end_line=fn.end_line,
                )
                for fn in self._functions
            ]
        )
        self._urn_by_goid = {fn.goid: fn.urn for fn in self._functions}
        self._module_by_path = {
            normalize_rel_path(path): mod for path, mod in module_by_path.items()
        }
        self._funcs_by_path: dict[str, list[FunctionMeta]] = {}
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
    def functions_by_path(self) -> dict[str, list[FunctionMeta]]:
        """
        Return functions keyed by normalized path.

        Returns
        -------
        dict[str, list[FunctionMeta]]
            Mapping of relative path to function metadata.
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
    Load function metadata and module map for a repo snapshot via a gateway.

    Returns
    -------
    FunctionCatalog
        Catalog containing spans, URNs, and module mapping.
    """
    con = gateway.con
    rows = con.execute(
        """
        SELECT goid_h128, urn, rel_path, qualname, start_line, end_line
        FROM core.goids
        WHERE repo = ? AND commit = ? AND kind IN ('function', 'method')
        """,
        [repo, commit],
    ).fetchall()

    functions: list[FunctionMeta] = []
    for goid_h128, urn, rel_path, qualname, start_line, end_line in rows:
        if start_line is None:
            continue
        end_val = int(end_line) if end_line is not None else int(start_line)
        functions.append(
            FunctionMeta(
                goid=int(goid_h128),
                urn=str(urn),
                rel_path=normalize_rel_path(rel_path),
                qualname=str(qualname),
                start_line=int(start_line),
                end_line=end_val,
            )
        )

    module_by_path = load_module_map(gateway, repo, commit)
    return FunctionCatalog(functions=functions, module_by_path=module_by_path)


# =============================================================================
# Catalog Provider and Service (from function_catalog_service.py)
# =============================================================================


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

    def get(self) -> FunctionCatalogService:
        """Return self to satisfy LazyResource-like interface.

        This allows FunctionCatalogService to be used in contexts that expect
        a .get() method (e.g., analytics CatalogProvider wrapper).

        Returns
        -------
        FunctionCatalogService
            Self reference.
        """
        return self

    def catalog(self) -> FunctionCatalog:
        """Return the underlying catalog instance.

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


__all__ = [
    "FunctionCatalog",
    "FunctionCatalogProvider",
    "FunctionCatalogService",
    "FunctionMeta",
    "FunctionSpan",
    "FunctionSpanIndex",
    "load_function_catalog",
    "load_function_index",
    "load_function_spans",
]
