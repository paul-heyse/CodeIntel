"""Protocol definitions for transport-agnostic serving operations."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator
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


class ServingSnapshotPointerProtocol(Protocol):
    """Protocol for serving snapshot pointer metadata."""

    @property
    def repo(self) -> str:
        """Repository identifier."""
        ...

    @property
    def commit(self) -> str:
        """Git commit SHA."""
        ...

    @property
    def run_id(self) -> str:
        """Build run identifier."""
        ...

    @property
    def published_at(self) -> datetime:
        """When the snapshot was published."""
        ...

    @property
    def snapshot_root(self) -> Path:
        """Root directory for the snapshot."""
        ...

    @property
    def snapshot_manifest_path(self) -> Path:
        """Path to snapshot_manifest.json for the snapshot."""
        ...

    @property
    def schema_manifest_path(self) -> Path:
        """Path to the schema manifest for the snapshot."""
        ...

    @property
    def semantic_layer_version(self) -> str:
        """Semantic layer version/hash for the snapshot."""
        ...


class ServingDBManagerProtocol(Protocol):
    """Protocol for the serving DB manager access used by serving operations."""

    def current_pointer(self) -> ServingSnapshotPointerProtocol:
        """Return the current serving snapshot pointer."""
        ...

    def current_summary(self) -> dict[str, object]:
        """Return cached snapshot summary metadata."""
        ...

    async def wait_ready(self, *, timeout_s: float | None = None) -> bool:
        """Wait for a snapshot pointer to become available."""
        ...


class ServingKernelProtocol(Protocol):
    """Protocol for the serving kernel interface consumed by operations."""

    @property
    def db(self) -> ServingDBManagerProtocol:
        """Return the serving DB manager."""
        ...

    def catalog(self) -> SemanticCatalogResponse:
        """Return the semantic view catalog."""
        ...

    def describe(self, view_id: str) -> SemanticViewDescriptionResponse:
        """Describe a semantic view."""
        ...

    def query(
        self, request: SemanticQueryRequest, *, cancel_check: CancelCheck | None = None
    ) -> SemanticQueryResponse:
        """Execute a semantic query and return typed results."""
        ...

    def query_ipc_stream(
        self, request: SemanticQueryRequest, *, cancel_check: CancelCheck | None = None
    ) -> Generator[bytes]:
        """Execute a semantic query and return Arrow IPC stream bytes."""
        ...

    def explain(self, request: SemanticQueryRequest) -> SemanticExplainResponse:
        """Compile a semantic query and return SQL and plan text."""
        ...

    def compile_query_sql(self, request: SemanticQueryRequest) -> str:
        """Compile a semantic query into SQL."""
        ...

    def search(self, request: SearchQueryRequest) -> SearchQueryResponse:
        """Execute a search query."""
        ...

    def meta(self) -> ServingKernelMetaResponse:
        """Return serving metadata."""
        ...

    def export_rows(
        self, request: SemanticExportRequest, *, cancel_check: CancelCheck | None = None
    ) -> Iterator[dict[str, object]]:
        """Return an iterator over export rows for a view."""
        ...

    def export_sql(self, request: SemanticExportRequest) -> str:
        """Return compiled export SQL."""
        ...

    def export_fingerprint(self, request: SemanticExportRequest) -> tuple[str, str | None]:
        """Return query hash and optional schema hash for the export request."""
        ...

    def export_to_parquet(
        self,
        request: SemanticExportRequest,
        *,
        output_path: Path,
        cancel_check: CancelCheck | None = None,
    ) -> int:
        """Write an export payload as Parquet and return rows written."""
        ...

    def export_to_arrow_ipc(
        self,
        request: SemanticExportRequest,
        *,
        output_path: Path,
        cancel_check: CancelCheck | None = None,
    ) -> int:
        """Write an Arrow IPC file to the provided path and return rows written."""
        ...


__all__ = ["ServingDBManagerProtocol", "ServingKernelProtocol", "ServingSnapshotPointerProtocol"]
