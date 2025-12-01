"""Shared helpers for isolated gateway/DuckDB test setup."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from codeintel.serving.backend.duckdb_service import DuckDBQueryService
from codeintel.serving.backend.limits import BackendLimits
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.models import FunctionSummaryResponse
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway
from codeintel.storage.gateway import open_memory_gateway as _open_memory_gateway


def open_fresh_duckdb(db_path: Path) -> StorageGateway:
    """
    Return a fresh DuckDB connection for tests.

    Returns
    -------
    StorageGateway
        Open gateway (caller must close).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=True,
        ensure_views=True,
        validate_schema=True,
    )
    return open_gateway(cfg)


def seed_tables(gateway: StorageGateway, ddl: list[str]) -> None:
    """Apply defensive DDL statements (DROP/CREATE) to avoid cross-test conflicts."""
    for stmt in ddl:
        gateway.con.execute(stmt)


def open_ingestion_gateway(
    *,
    apply_schema: bool = True,
    ensure_views: bool = False,
    validate_schema: bool = True,
    strict_schema: bool = True,
) -> StorageGateway:
    """
    Return an in-memory gateway prepped for ingestion runners.

    Parameters mirror `open_memory_gateway`; schema application is enabled by default so
    ingestion steps can write tables without extra setup.

    Returns
    -------
    StorageGateway
        Gateway configured for ingestion tests.
    """
    effective_ensure_views = ensure_views or strict_schema
    effective_validate_schema = validate_schema or strict_schema
    return _open_memory_gateway(
        apply_schema=apply_schema,
        ensure_views=effective_ensure_views,
        validate_schema=effective_validate_schema,
    )


def build_duckdb_backend(
    gateway: StorageGateway,
    *,
    repo: str | None = None,
    commit: str | None = None,
    service_override: LocalQueryService | None = None,
) -> DuckDBBackend:
    """
    Construct a DuckDBBackend with gateway config fallbacks.

    Parameters
    ----------
    gateway
        Active storage gateway for the backend to use.
    repo
        Optional repo override; falls back to ``gateway.config.repo``.
    commit
        Optional commit override; falls back to ``gateway.config.commit``.
    service_override
        Optional LocalQueryService to bypass default wiring.

    Returns
    -------
    DuckDBBackend
        Backend ready for adapter/service tests.
    """
    repo_value = repo or gateway.config.repo
    commit_value = commit or gateway.config.commit
    return DuckDBBackend(
        gateway=gateway,
        repo=repo_value,
        commit=commit_value,
        service_override=service_override,
    )


class ScopeCapturingQuery(DuckDBQueryService):
    """Minimal DuckDBQueryService that delegates to a provided callable and records scopes."""

    def __init__(self, delegate: Callable[..., object]) -> None:
        self._delegate = delegate
        self.gateway = cast(
            "StorageGateway", SimpleNamespace(datasets=SimpleNamespace(mapping={}), config={})
        )
        self.repo = "demo/repo"
        self.commit = "deadbeef"
        self.limits = BackendLimits()
        self.graph_engine = None

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: object | None = None,
    ) -> FunctionSummaryResponse:
        result = self._delegate(
            urn=urn,
            goid_h128=goid_h128,
            rel_path=rel_path,
            qualname=qualname,
            scope=scope,
        )
        if isinstance(result, FunctionSummaryResponse):
            return result
        return FunctionSummaryResponse.model_validate(result)


def build_scope_parsing_service(delegate: Callable[..., object]) -> LocalQueryService:
    """
    Build a LocalQueryService that forwards to the provided delegate while preserving scope.

    Parameters
    ----------
    delegate
        Callable invoked by the stub query service.

    Returns
    -------
    LocalQueryService
        Service that can be used to validate scope parsing behavior.
    """
    query = ScopeCapturingQuery(delegate=delegate)
    return LocalQueryService(query=query)
