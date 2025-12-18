"""Transport-agnostic serving operations.

This layer centralizes serving business logic so HTTP and FastMCP transports can
remain thin adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.serving.operations.protocols import ServingDBManagerProtocol, ServingKernelProtocol

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.serving.search.models import SearchQueryRequest, SearchQueryResponse
    from codeintel.serving.semantic.models import (
        SemanticExplainResponse,
        SemanticExportRequest,
        SemanticQueryRequest,
        SemanticQueryResponse,
    )


@dataclass(frozen=True, slots=True)
class ServingOperations:
    """Facade over the serving kernel for transport adapters."""

    kernel: ServingKernelProtocol

    @property
    def db(self) -> ServingDBManagerProtocol:
        """Expose the DB manager for adapters that require pointer metadata.

        Returns
        -------
        ServingDBManagerProtocol
            Serving DB manager used for pointer access and lifecycle.
        """
        return self.kernel.db

    def catalog(self) -> dict[str, object]:
        """Return the semantic catalog.

        Returns
        -------
        dict[str, object]
            Catalog payload.
        """
        return self.kernel.catalog()

    def describe(self, view_id: str) -> dict[str, object]:
        """Describe a semantic view.

        Parameters
        ----------
        view_id
            Semantic view identifier.

        Returns
        -------
        dict[str, object]
            View description payload.
        """
        return self.kernel.describe(view_id)

    def query(self, request: SemanticQueryRequest) -> SemanticQueryResponse:
        """Execute a semantic query.

        Parameters
        ----------
        request
            Semantic query request.

        Returns
        -------
        SemanticQueryResponse
            Query response.
        """
        return self.kernel.query(request)

    def explain(self, request: SemanticQueryRequest) -> SemanticExplainResponse:
        """Explain a semantic query.

        Parameters
        ----------
        request
            Semantic query request.

        Returns
        -------
        SemanticExplainResponse
            Explain response with SQL and plan.
        """
        return self.kernel.explain(request)

    def compile_query_sql(self, request: SemanticQueryRequest) -> str:
        """Compile semantic query SQL.

        Parameters
        ----------
        request
            Semantic query request.

        Returns
        -------
        str
            Compiled SQL string.
        """
        return self.kernel.compile_query_sql(request)

    def search(self, request: SearchQueryRequest) -> SearchQueryResponse:
        """Execute a search request.

        Parameters
        ----------
        request
            Search query request.

        Returns
        -------
        SearchQueryResponse
            Search response.
        """
        return self.kernel.search(request)

    def meta(self) -> dict[str, object]:
        """Return serving metadata.

        Returns
        -------
        dict[str, object]
            Metadata payload.
        """
        return self.kernel.meta()

    def export_rows(self, request: SemanticExportRequest) -> Iterator[dict[str, object]]:
        """Return export rows iterator.

        Parameters
        ----------
        request
            Export request.

        Returns
        -------
        Iterator[dict[str, object]]
            Iterator of row dictionaries.
        """
        return self.kernel.export_rows(request)

    def export_sql(self, request: SemanticExportRequest) -> str:
        """Return export SQL.

        Parameters
        ----------
        request
            Export request.

        Returns
        -------
        str
            Compiled SQL string.
        """
        return self.kernel.export_sql(request)

    def export_fingerprint(self, request: SemanticExportRequest) -> tuple[str, str | None]:
        """Return export fingerprints.

        Parameters
        ----------
        request
            Export request.

        Returns
        -------
        tuple[str, str | None]
            Query hash and optional schema hash.
        """
        return self.kernel.export_fingerprint(request)

    def export_to_parquet(self, request: SemanticExportRequest, *, output_path: Path) -> None:
        """Write Parquet export to disk.

        Parameters
        ----------
        request
            Export request.
        output_path
            Output path for the Parquet file.
        """
        return self.kernel.export_to_parquet(request, output_path=output_path)

    def export_to_arrow_ipc(self, request: SemanticExportRequest, *, output_path: Path) -> int:
        """Write Arrow IPC export to disk and return row count.

        Parameters
        ----------
        request
            Export request.
        output_path
            Output path for the Arrow IPC file.

        Returns
        -------
        int
            Number of rows written.
        """
        return self.kernel.export_to_arrow_ipc(request, output_path=output_path)


__all__ = ["ServingOperations"]
