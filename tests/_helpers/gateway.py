"""Shared helpers for isolated gateway/DuckDB test setup."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from codeintel.graphs.engine import GraphEngine
from codeintel.serving.backend import (
    BackendContext,
    DuckDBQueryService,
    DuckDBRepositories,
    GraphEngineProvider,
)
from codeintel.serving.backend.pagination import BackendLimits
from codeintel.serving.backend.query_api import DuckDBQueryApi
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.models import FunctionSummaryResponse
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway
from codeintel.storage.gateway import open_memory_gateway as _open_memory_gateway
from codeintel.storage.ingest_macros import ensure_ingest_macros


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
    gateway = _open_memory_gateway(
        apply_schema=apply_schema,
        ensure_views=effective_ensure_views,
        validate_schema=effective_validate_schema,
    )
    ensure_ingest_macros(gateway.con)
    return gateway


def open_ingestion_gateway_with_macros(
    *,
    apply_schema: bool = True,
    ensure_views: bool = True,
    validate_schema: bool = True,
    strict_schema: bool = True,
) -> StorageGateway:
    """
    Return an in-memory gateway with schemas/views/macros ensured for ingestion/graph tests.

    This helper always registers ingest macros and ensures views to avoid missing
    metadata.* table-function errors in graph and analytics tests.

    Returns
    -------
    StorageGateway
        Gateway configured for ingestion/graph tests with macros registered.
    """
    return open_ingestion_gateway(
        apply_schema=apply_schema,
        ensure_views=ensure_views,
        validate_schema=validate_schema,
        strict_schema=strict_schema,
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
    repo_value = repo or gateway.config.repo or "demo/repo"
    commit_value = commit or gateway.config.commit or "deadbeef"
    return DuckDBBackend(
        gateway=gateway,
        repo=repo_value,
        commit=commit_value,
        service_override=service_override,
    )


class ScopeCapturingQuery:
    """Minimal query stub that delegates to a provided callable and records scopes."""

    def __init__(self, delegate: Callable[..., object]) -> None:
        self._delegate = delegate
        self.gateway = cast(
            "StorageGateway", SimpleNamespace(datasets=SimpleNamespace(mapping={}), config={})
        )
        self.repo = "demo/repo"
        self.commit = "deadbeef"
        self.limits = BackendLimits()
        self.graph_engine = None
        self.functions = self
        self.modules = self
        self.subsystems = self
        self.datasets = self

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
    query = cast("DuckDBQueryApi", ScopeCapturingQuery(delegate=delegate))
    return LocalQueryService(query=query)


def build_duckdb_query_service(
    gateway: StorageGateway,
    *,
    repo: str | None = None,
    commit: str | None = None,
    limits: BackendLimits | None = None,
    graph_engine: GraphEngine | None = None,
) -> DuckDBQueryService:
    """
    Construct a DuckDBQueryService using the new context/repository wiring.

    Parameters
    ----------
    gateway
        Active storage gateway.
    repo
        Repository identifier.
    commit
        Commit hash.
    limits
        Optional BackendLimits; defaults to BackendLimits().
    graph_engine
        Optional graph engine instance.

    Returns
    -------
    DuckDBQueryService
        Constructed query service bound to the provided gateway/snapshot.
    """
    effective_limits = limits or BackendLimits()
    repo_value = repo or gateway.config.repo or "demo/repo"
    commit_value = commit or gateway.config.commit or "deadbeef"
    context = BackendContext(
        gateway=gateway,
        repo=repo_value,
        commit=commit_value,
        limits=effective_limits,
        graph_engine=graph_engine,
    )
    repositories = DuckDBRepositories(gateway, repo_value, commit_value)
    engine_provider = GraphEngineProvider(context=context, graph_engine=graph_engine)
    return DuckDBQueryService(
        context=context, repositories=repositories, engine_provider=engine_provider
    )
