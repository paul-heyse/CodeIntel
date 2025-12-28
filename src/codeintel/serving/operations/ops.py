"""Transport-agnostic serving operations.

This layer centralizes serving business logic so HTTP and FastMCP transports can
remain thin adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.serving.errors import (
    CodeIntelDomainError,
    ExportTooLargeError,
    SemanticViewNotFoundError,
)
from codeintel.serving.operations.protocols import ServingDBManagerProtocol, ServingKernelProtocol
from codeintel.serving.settings import ServingSettings

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from codeintel.serving.meta.models import ServingKernelMetaResponse
    from codeintel.serving.operations.cancellation import CancelCheck
    from codeintel.serving.search.models import SearchQueryRequest, SearchQueryResponse
from codeintel.serving.semantic.models import (
    SemanticCatalogResponse,
    SemanticExplainResponse,
    SemanticExportRequest,
    SemanticQueryRequest,
    SemanticQueryResponse,
    SemanticViewDescriptionResponse,
)


@dataclass(frozen=True, slots=True)
class ServingOperations:
    """Facade over the serving kernel for transport adapters."""

    kernel: ServingKernelProtocol
    settings: ServingSettings

    def _export_limit_exceeded(self, *, limit: int) -> bool:
        return limit > self.settings.export_max_rows

    @property
    def db(self) -> ServingDBManagerProtocol:
        """Expose the DB manager for adapters that require pointer metadata.

        Returns
        -------
        ServingDBManagerProtocol
            Serving DB manager used for pointer access and lifecycle.
        """
        return self.kernel.db

    def catalog(self) -> SemanticCatalogResponse:
        """Return the semantic catalog.

        Returns
        -------
        SemanticCatalogResponse
            Catalog payload.
        """
        return self.kernel.catalog()

    def describe(self, view_id: str) -> SemanticViewDescriptionResponse:
        """Describe a semantic view.

        Parameters
        ----------
        view_id
            Semantic view identifier.

        Returns
        -------
        SemanticViewDescriptionResponse
            View description payload.

        Raises
        ------
        SemanticViewNotFoundError
            When the requested view cannot be resolved.
        """
        try:
            return self.kernel.describe(view_id)
        except KeyError as exc:
            raise SemanticViewNotFoundError(view_id) from exc

    def query(
        self, request: SemanticQueryRequest, *, cancel_check: CancelCheck | None = None
    ) -> SemanticQueryResponse:
        """Execute a semantic query.

        Parameters
        ----------
        request
            Semantic query request.
        cancel_check
            Optional cancellation hook invoked during query execution.

        Returns
        -------
        SemanticQueryResponse
            Query response.

        Raises
        ------
        SemanticViewNotFoundError
            When the requested view cannot be resolved.
        CodeIntelDomainError
            When the request is invalid.
        """
        try:
            return self.kernel.query(request, cancel_check=cancel_check)
        except KeyError as exc:
            raise SemanticViewNotFoundError(request.view_id) from exc
        except ValueError as exc:
            raise CodeIntelDomainError(
                code="CODEINTEL_SEMANTIC_INVALID_QUERY",
                details={"reason": str(exc)},
            ) from exc

    def query_ipc_stream(
        self, request: SemanticQueryRequest, *, cancel_check: CancelCheck | None = None
    ) -> Iterable[bytes]:
        """Execute a semantic query and return Arrow IPC stream bytes.

        Parameters
        ----------
        request
            Semantic query request.
        cancel_check
            Optional cancellation hook invoked during query execution.

        Returns
        -------
        Iterable[bytes]
            Iterable of Arrow IPC stream bytes.

        Raises
        ------
        SemanticViewNotFoundError
            When the requested view cannot be resolved.
        CodeIntelDomainError
            When the request is invalid.
        """
        try:
            return self.kernel.query_ipc_stream(request, cancel_check=cancel_check)
        except KeyError as exc:
            raise SemanticViewNotFoundError(request.view_id) from exc
        except ValueError as exc:
            raise CodeIntelDomainError(
                code="CODEINTEL_SEMANTIC_INVALID_QUERY",
                details={"reason": str(exc)},
            ) from exc

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

        Raises
        ------
        SemanticViewNotFoundError
            When the requested view cannot be resolved.
        CodeIntelDomainError
            When query compilation fails.
        """
        try:
            return self.kernel.explain(request)
        except KeyError as exc:
            raise SemanticViewNotFoundError(request.view_id) from exc
        except ValueError as exc:
            raise CodeIntelDomainError(
                code="CODEINTEL_SEMANTIC_INVALID_QUERY",
                details={"reason": str(exc)},
            ) from exc

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

        Raises
        ------
        SemanticViewNotFoundError
            When the requested view cannot be resolved.
        CodeIntelDomainError
            When query compilation fails.
        """
        try:
            return self.kernel.compile_query_sql(request)
        except KeyError as exc:
            raise SemanticViewNotFoundError(request.view_id) from exc
        except ValueError as exc:
            raise CodeIntelDomainError(
                code="CODEINTEL_SEMANTIC_INVALID_QUERY",
                details={"reason": str(exc)},
            ) from exc

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

        Raises
        ------
        CodeIntelDomainError
            When the request is invalid.
        """
        try:
            return self.kernel.search(request)
        except ValueError as exc:
            raise CodeIntelDomainError(
                code="CODEINTEL_SEMANTIC_INVALID_QUERY",
                details={"reason": str(exc)},
            ) from exc

    def meta(self) -> ServingKernelMetaResponse:
        """Return serving metadata.

        Returns
        -------
        ServingKernelMetaResponse
            Metadata payload.
        """
        return self.kernel.meta()

    def export_rows(
        self, request: SemanticExportRequest, *, cancel_check: CancelCheck | None = None
    ) -> Iterator[dict[str, object]]:
        """Yield export rows.

        Parameters
        ----------
        request
            Export request.
        cancel_check
            Optional cancellation hook invoked during export.

        Yields
        ------
        dict[str, object]
            Row dictionary for each exported record.

        Raises
        ------
        ExportTooLargeError
            When the requested export exceeds the configured maximum rows.
        SemanticViewNotFoundError
            When the requested view cannot be resolved.
        CodeIntelDomainError
            When the request is invalid.
        """
        if self._export_limit_exceeded(limit=request.limit):
            raise ExportTooLargeError(row_count=request.limit)
        try:
            yield from self.kernel.export_rows(request, cancel_check=cancel_check)
        except KeyError as exc:
            raise SemanticViewNotFoundError(request.view_id) from exc
        except ValueError as exc:
            raise CodeIntelDomainError(
                code="CODEINTEL_EXPORT_INVALID_REQUEST",
                details={"reason": str(exc)},
            ) from exc

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

        Raises
        ------
        ExportTooLargeError
            When the requested export exceeds the configured maximum rows.
        SemanticViewNotFoundError
            When the requested view cannot be resolved.
        CodeIntelDomainError
            When the request is invalid.
        """
        if self._export_limit_exceeded(limit=request.limit):
            raise ExportTooLargeError(row_count=request.limit)
        try:
            return self.kernel.export_sql(request)
        except KeyError as exc:
            raise SemanticViewNotFoundError(request.view_id) from exc
        except ValueError as exc:
            raise CodeIntelDomainError(
                code="CODEINTEL_EXPORT_INVALID_REQUEST",
                details={"reason": str(exc)},
            ) from exc

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

        Raises
        ------
        ExportTooLargeError
            When the requested export exceeds the configured maximum rows.
        SemanticViewNotFoundError
            When the requested view cannot be resolved.
        CodeIntelDomainError
            When the request is invalid.
        """
        if self._export_limit_exceeded(limit=request.limit):
            raise ExportTooLargeError(row_count=request.limit)
        try:
            return self.kernel.export_fingerprint(request)
        except KeyError as exc:
            raise SemanticViewNotFoundError(request.view_id) from exc
        except ValueError as exc:
            raise CodeIntelDomainError(
                code="CODEINTEL_EXPORT_INVALID_REQUEST",
                details={"reason": str(exc)},
            ) from exc

    def export_to_parquet(
        self,
        request: SemanticExportRequest,
        *,
        output_path: Path,
        cancel_check: CancelCheck | None = None,
    ) -> int:
        """Write Parquet export to disk and return row count.

        Parameters
        ----------
        request
            Export request.
        output_path
            Output path for the Parquet file.
        cancel_check
            Optional cancellation hook invoked during export.

        Returns
        -------
        int
            Number of rows written.

        Raises
        ------
        ExportTooLargeError
            When the requested export exceeds the configured maximum rows.
        SemanticViewNotFoundError
            When the requested view cannot be resolved.
        CodeIntelDomainError
            When the request is invalid.
        """
        if self._export_limit_exceeded(limit=request.limit):
            raise ExportTooLargeError(row_count=request.limit)
        try:
            return self.kernel.export_to_parquet(
                request, output_path=output_path, cancel_check=cancel_check
            )
        except KeyError as exc:
            raise SemanticViewNotFoundError(request.view_id) from exc
        except ValueError as exc:
            raise CodeIntelDomainError(
                code="CODEINTEL_EXPORT_INVALID_REQUEST",
                details={"reason": str(exc)},
            ) from exc

    def export_to_arrow_ipc(
        self,
        request: SemanticExportRequest,
        *,
        output_path: Path,
        cancel_check: CancelCheck | None = None,
    ) -> int:
        """Write Arrow IPC export to disk and return row count.

        Parameters
        ----------
        request
            Export request.
        output_path
            Output path for the Arrow IPC file.
        cancel_check
            Optional cancellation hook invoked during export.

        Returns
        -------
        int
            Number of rows written.

        Raises
        ------
        ExportTooLargeError
            When the requested export exceeds the configured maximum rows.
        SemanticViewNotFoundError
            When the requested view cannot be resolved.
        CodeIntelDomainError
            When the request is invalid.
        """
        if self._export_limit_exceeded(limit=request.limit):
            raise ExportTooLargeError(row_count=request.limit)
        try:
            return self.kernel.export_to_arrow_ipc(
                request, output_path=output_path, cancel_check=cancel_check
            )
        except KeyError as exc:
            raise SemanticViewNotFoundError(request.view_id) from exc
        except ValueError as exc:
            raise CodeIntelDomainError(
                code="CODEINTEL_EXPORT_INVALID_REQUEST",
                details={"reason": str(exc)},
            ) from exc


__all__ = ["ServingOperations"]
