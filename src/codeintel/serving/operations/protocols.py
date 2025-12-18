"""Protocol definitions for transport-agnostic serving operations."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

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


class ServingKernelProtocol(Protocol):
    """Protocol for the serving kernel interface consumed by operations."""

    @property
    def db(self) -> ServingDBManagerProtocol:
        """Return the serving DB manager."""
        ...

    def catalog(self) -> dict[str, object]:
        """Return the semantic view catalog."""
        ...

    def describe(self, view_id: str) -> dict[str, object]:
        """Describe a semantic view."""
        ...

    def query(self, request: SemanticQueryRequest) -> SemanticQueryResponse:
        """Execute a semantic query and return typed results."""
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

    def meta(self) -> dict[str, object]:
        """Return serving metadata dictionary."""
        ...

    def export_rows(self, request: SemanticExportRequest) -> Iterator[dict[str, object]]:
        """Return an iterator over export rows for a view."""
        ...

    def export_sql(self, request: SemanticExportRequest) -> str:
        """Return compiled export SQL."""
        ...

    def export_fingerprint(self, request: SemanticExportRequest) -> tuple[str, str | None]:
        """Return query hash and optional schema hash for the export request."""
        ...

    def export_to_parquet(self, request: SemanticExportRequest, *, output_path: Path) -> None:
        """Write an export payload as Parquet to the provided path."""
        ...

    def export_to_arrow_ipc(self, request: SemanticExportRequest, *, output_path: Path) -> int:
        """Write an Arrow IPC file to the provided path and return rows written."""
        ...


__all__ = ["ServingDBManagerProtocol", "ServingKernelProtocol", "ServingSnapshotPointerProtocol"]
