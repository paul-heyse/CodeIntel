"""Function catalog services and persistence helpers.

This module provides the production implementation of function catalog loading
and indexing from storage, alongside a small service wrapper used by both the
graphs and analytics domains.

Key components
--------------
- FunctionSpanIndex: SpanIndex specialization with path normalization
- FunctionCatalog: Spans + URNs + module mapping
- CatalogService: Unified, injectable access to the catalog
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

from codeintel.core.catalog.function_span import FunctionSpan
from codeintel.core.catalog.span_index import SpanIndex
from codeintel.core.paths import normalize_path
from codeintel.storage.helpers.module_index import load_module_map
from codeintel.storage.helpers.sql_params import render_sql
from codeintel.storage.query_results import records_from_relation

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from codeintel.storage.gateway import StorageGateway


class FunctionSpanIndex(SpanIndex):
    """Lookup structure for resolving GOIDs from file spans.

    This class extends the core SpanIndex with the repo-aware path normalization
    used across the storage-backed catalog implementation.
    """

    def __init__(self, spans: Iterable[FunctionSpan]) -> None:
        """Initialize the index from an iterable of function spans.

        Parameters
        ----------
        spans
            Function spans to index.
        """
        super().__init__(spans, path_normalizer=normalize_path)


def _load_function_rows(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    include_urn: bool = False,
) -> list[dict[str, Any]]:
    """Load raw function rows from core.goids.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    repo
        Repository identifier.
    commit
        Commit identifier.
    include_urn
        Whether to include URN column in results.

    Returns
    -------
    list[dict[str, Any]]
        Raw row dictionaries from the query.
    """
    columns = ["goid_h128", "rel_path", "qualname", "start_line", "end_line"]
    if include_urn:
        columns.insert(1, "urn")
    select_cols = ", ".join(columns)
    relation = gateway.con.sql(
        render_sql(
            f"""
            SELECT {select_cols}
            FROM core.goids
            WHERE repo = $repo
              AND commit = $commit
              AND kind IN ('function', 'method')
            """,
            {"repo": repo, "commit": commit},
        )
    )
    return cast("list[dict[str, Any]]", records_from_relation(relation))


def load_function_spans(gateway: StorageGateway, *, repo: str, commit: str) -> list[FunctionSpan]:
    """Load function spans from core.goids for a repo snapshot.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    repo
        Repository identifier.
    commit
        Commit identifier.

    Returns
    -------
    list[FunctionSpan]
        Normalized function spans keyed by GOID.
    """
    rows = _load_function_rows(gateway, repo=repo, commit=commit, include_urn=False)
    spans: list[FunctionSpan] = []
    for row in rows:
        start_line = row["start_line"]
        if start_line is None:
            continue
        end_line = row["end_line"]
        spans.append(
            FunctionSpan(
                goid=int(row["goid_h128"]),
                rel_path=normalize_path(row["rel_path"]),
                qualname=str(row["qualname"]),
                start_line=int(start_line),
                end_line=int(end_line) if end_line is not None else int(start_line),
            )
        )
    return spans


def load_function_index(gateway: StorageGateway, *, repo: str, commit: str) -> FunctionSpanIndex:
    """Create a `FunctionSpanIndex` from DuckDB state.

    Returns
    -------
    FunctionSpanIndex
        Index seeded from `core.goids` for the repo/commit snapshot.
    """
    return FunctionSpanIndex(load_function_spans(gateway, repo=repo, commit=commit))


class FunctionCatalog:
    """Centralized access to function spans, URNs, and module mappings."""

    def __init__(
        self,
        *,
        functions: Iterable[FunctionSpan],
        module_by_path: dict[str, str],
    ) -> None:
        """Initialize the catalog from function spans and module mapping.

        Parameters
        ----------
        functions
            Function spans to catalog (with optional URN populated).
        module_by_path
            Mapping of file paths to module names.
        """
        self._functions: list[FunctionSpan] = list(functions)
        self._index = FunctionSpanIndex(self._functions)
        self._urn_by_goid = {fn.goid: fn.urn for fn in self._functions if fn.urn is not None}
        self._module_by_path = {normalize_path(path): mod for path, mod in module_by_path.items()}
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
        """Return the span lookup index.

        Returns
        -------
        FunctionSpanIndex
            Index supporting path/span lookups.
        """
        return self._index

    @property
    def functions_by_path(self) -> dict[str, list[FunctionSpan]]:
        """Return functions keyed by normalized path."""
        return self._funcs_by_path

    @property
    def module_by_path(self) -> dict[str, str]:
        """Return module name mapping keyed by normalized path."""
        return self._module_by_path

    def urn_for_goid(self, goid: int) -> str | None:
        """Return URN for a GOID if known.

        Parameters
        ----------
        goid
            GOID to lookup.

        Returns
        -------
        str | None
            URN when available, otherwise None.
        """
        return self._urn_by_goid.get(goid)

    def lookup_goid(
        self, rel_path: str, start_line: int, end_line: int | None, qualname: str | None
    ) -> int | None:
        """Resolve GOID from span and optional qualname.

        Parameters
        ----------
        rel_path
            Normalized repository-relative path.
        start_line
            Start line of the span (1-based).
        end_line
            Optional end line of the span (1-based).
        qualname
            Optional qualified name to disambiguate overlapping spans.

        Returns
        -------
        int | None
            Matching GOID when found, otherwise None.
        """
        return self._index.lookup(rel_path, start_line, end_line, qualname)


def load_function_catalog(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
) -> FunctionCatalog:
    """Load function spans and module map for a repo snapshot.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    repo
        Repository identifier.
    commit
        Commit identifier.

    Returns
    -------
    FunctionCatalog
        Catalog containing spans, lookup index, URN mappings, and module mappings.
    """
    rows = _load_function_rows(gateway, repo=repo, commit=commit, include_urn=True)
    functions: list[FunctionSpan] = []
    for row in rows:
        start_line = row["start_line"]
        if start_line is None:
            continue
        end_line = row["end_line"]
        functions.append(
            FunctionSpan(
                goid=int(row["goid_h128"]),
                rel_path=normalize_path(row["rel_path"]),
                qualname=str(row["qualname"]),
                start_line=int(start_line),
                end_line=int(end_line) if end_line is not None else int(start_line),
                urn=str(row["urn"]),
            )
        )
    module_by_path = load_module_map(gateway, repo, commit)
    return FunctionCatalog(functions=functions, module_by_path=module_by_path)


class FunctionCatalogProvider(Protocol):
    """Protocol for objects that supply a function catalog."""

    def catalog(self) -> FunctionCatalog:
        """Return a function catalog instance."""
        raise NotImplementedError

    def urn_for_goid(self, goid: int) -> str | None:
        """Return a URN for a GOID when available."""
        raise NotImplementedError

    def module_for_path(self, rel_path: str) -> str | None:
        """Return module name for a given relative path."""
        raise NotImplementedError

    def lookup_goid(
        self, rel_path: str, start_line: int, end_line: int | None, qualname: str | None
    ) -> int | None:
        """Resolve a GOID for a span and optional qualname."""
        raise NotImplementedError


@dataclass
class CatalogService(FunctionCatalogProvider):
    """Unified catalog access for graphs and analytics.

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
        """Load catalog state for a repo snapshot from a storage gateway.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        repo
            Repository identifier.
        commit
            Commit identifier.

        Returns
        -------
        CatalogService
            Service instance populated from storage.
        """
        return cls(load_function_catalog(gateway, repo=repo, commit=commit))

    def get(self) -> CatalogService:
        """Return self to satisfy ResourceProvider interface.

        Returns
        -------
        CatalogService
            The current service instance.
        """
        return self

    def invalidate(self) -> None:
        """Invalidate cached data."""
        self._cached_spans = None

    @property
    def resource_name(self) -> str:
        """Resource identifier."""
        return self.RESOURCE_NAME

    def catalog(self) -> FunctionCatalog:
        """Return the underlying catalog instance.

        Returns
        -------
        FunctionCatalog
            Underlying catalog implementation.
        """
        return self._catalog

    @property
    def function_spans(self) -> Sequence[FunctionSpan]:
        """All function spans in the catalog."""
        if self._cached_spans is None:
            self._cached_spans = tuple(self._catalog.function_spans)
        return self._cached_spans

    @property
    def paths(self) -> Sequence[str]:
        """All file paths with indexed functions."""
        return tuple(self._catalog.function_index.paths())

    @property
    def module_by_path(self) -> Mapping[str, str]:
        """Mapping of file paths to module names."""
        return self._catalog.module_by_path

    def spans_for_path(self, rel_path: str) -> Sequence[FunctionSpan]:
        """Get function spans for a specific file.

        Parameters
        ----------
        rel_path
            Repository-relative path to lookup.

        Returns
        -------
        Sequence[FunctionSpan]
            Function spans defined within the given file.
        """
        return tuple(self._catalog.function_index.spans_for_path(rel_path))

    def local_name_map(self, rel_path: str) -> Mapping[str, int]:
        """Get local name to GOID mapping for a file.

        Parameters
        ----------
        rel_path
            Repository-relative path to lookup.

        Returns
        -------
        Mapping[str, int]
            Mapping from local name to GOID.
        """
        return self._catalog.function_index.local_name_map(rel_path)

    def get_functions_by_path(self) -> dict[str, list[FunctionSpan]]:
        """Return functions keyed by path.

        Returns
        -------
        dict[str, list[FunctionSpan]]
            Mapping from normalized path to spans in that file.
        """
        return self._catalog.functions_by_path

    def urn_for_goid(self, goid: int) -> str | None:
        """Return URN for GOID when present.

        Parameters
        ----------
        goid
            GOID to lookup.

        Returns
        -------
        str | None
            URN when known, otherwise None.
        """
        return self._catalog.urn_for_goid(goid)

    def lookup_goid(
        self,
        rel_path: str,
        start_line: int,
        end_line: int | None = None,
        qualname: str | None = None,
    ) -> int | None:
        """Resolve a GOID using span and optional qualname.

        Parameters
        ----------
        rel_path
            Repository-relative file path.
        start_line
            Start line of the span (1-based).
        end_line
            Optional end line of the span (1-based).
        qualname
            Optional qualified name to disambiguate.

        Returns
        -------
        int | None
            Matching GOID when found, otherwise None.
        """
        return self._catalog.function_index.lookup(rel_path, start_line, end_line, qualname)

    def module_for_path(self, rel_path: str) -> str | None:
        """Return module name for a given relative path.

        Parameters
        ----------
        rel_path
            Repository-relative file path.

        Returns
        -------
        str | None
            Module name when present in the module map, otherwise None.
        """
        return self._catalog.module_by_path.get(rel_path)

    @property
    def index(self) -> FunctionSpanIndex:
        """Expose the underlying FunctionSpanIndex for advanced consumers."""
        return self._catalog.function_index

    @property
    def spans(self) -> list[FunctionSpan]:
        """Return all function spans in the catalog."""
        return self._catalog.function_spans


__all__ = [
    "CatalogService",
    "FunctionCatalog",
    "FunctionCatalogProvider",
    "FunctionSpan",
    "FunctionSpanIndex",
    "load_function_catalog",
    "load_function_index",
    "load_function_spans",
]
