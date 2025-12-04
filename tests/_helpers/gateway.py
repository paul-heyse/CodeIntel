"""Shared helpers for isolated gateway/DuckDB test setup.

This module provides gateway and DuckDB connection helpers for tests,
including functions for creating fresh gateways, ensuring macros are
registered, and building backend services.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import duckdb

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
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway
from codeintel.storage.gateway import open_memory_gateway as _open_memory_gateway
from codeintel.storage.macros import ensure_ingest_macros, list_ingest_macros
from codeintel.storage.metadata import INGEST_MACROS
from tests._helpers.fakes.serving import ScopeRecordingQuery

# Type alias for DuckDB connections (originally from duckdb.py)
DuckDBConnection = duckdb.DuckDBPyConnection

# Expected macros that must be registered for tests (originally from duckdb.py)
MACROS_EXPECTED = {m.lower() for m in INGEST_MACROS.values()}


# =============================================================================
# Gateway Factory
# =============================================================================


class GatewayFactory:
    """Unified gateway creation with composable options.

    Provide a fluent builder interface for creating test gateways with
    consistent configuration. This consolidates the various gateway creation
    functions into a single, composable interface.

    Example
    -------
    >>> gateway = GatewayFactory().with_macros().open()
    >>> gateway = GatewayFactory().file_backed(db_path).with_schema().open()
    """

    def __init__(self) -> None:
        """Initialize factory with defaults."""
        self._apply_schema: bool = True
        self._ensure_views: bool = True
        self._ensure_macros: bool = True
        self._validate_schema: bool = True
        self._strict_schema: bool = True
        self._file_backed: bool = False
        self._db_path: Path | None = None
        self._repo: str | None = None
        self._commit: str | None = None

    def with_schema(self) -> GatewayFactory:
        """Enable schema application (default).

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._apply_schema = True
        return self

    def without_schema(self) -> GatewayFactory:
        """Disable schema application.

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._apply_schema = False
        return self

    def with_views(self) -> GatewayFactory:
        """Enable view creation (default).

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._ensure_views = True
        return self

    def without_views(self) -> GatewayFactory:
        """Disable view creation.

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._ensure_views = False
        return self

    def with_macros(self) -> GatewayFactory:
        """Enable macro registration (default).

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._ensure_macros = True
        return self

    def without_macros(self) -> GatewayFactory:
        """Disable macro registration.

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._ensure_macros = False
        return self

    def with_validation(self) -> GatewayFactory:
        """Enable schema validation (default).

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._validate_schema = True
        return self

    def without_validation(self) -> GatewayFactory:
        """Disable schema validation.

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._validate_schema = False
        return self

    def strict(self) -> GatewayFactory:
        """Enable strict schema mode (default).

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._strict_schema = True
        return self

    def relaxed(self) -> GatewayFactory:
        """Disable strict schema mode.

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._strict_schema = False
        return self

    def file_backed(self, db_path: Path) -> GatewayFactory:
        """Use a file-backed database instead of in-memory.

        Parameters
        ----------
        db_path
            Path to the database file.

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._file_backed = True
        self._db_path = db_path
        return self

    def in_memory(self) -> GatewayFactory:
        """Use an in-memory database (default).

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._file_backed = False
        self._db_path = None
        return self

    def with_snapshot(self, repo: str, commit: str) -> GatewayFactory:
        """Set the repository snapshot.

        Parameters
        ----------
        repo
            Repository identifier.
        commit
            Commit hash.

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._repo = repo
        self._commit = commit
        return self

    def open(self) -> StorageGateway:
        """Create and return the configured gateway.

        Returns
        -------
        StorageGateway
            Configured gateway ready for use.

        Raises
        ------
        ValueError
            If db_path is not set for file-backed gateway.
        RuntimeError
            If macros cannot be registered.
        """
        if self._file_backed:
            if self._db_path is None:
                msg = "db_path must be set for file-backed gateway"
                raise ValueError(msg)
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            cfg = StorageConfig(
                db_path=self._db_path,
                read_only=False,
                apply_schema=self._apply_schema,
                ensure_views=self._ensure_views,
                validate_schema=self._validate_schema,
            )
            gateway = open_gateway(cfg)
        else:
            effective_ensure_views = self._ensure_views or self._strict_schema
            effective_validate_schema = self._validate_schema or self._strict_schema
            gateway = _open_memory_gateway(
                apply_schema=self._apply_schema,
                ensure_views=effective_ensure_views,
                validate_schema=effective_validate_schema,
                repo=self._repo,
                commit=self._commit,
            )

        if self._ensure_macros:
            ensure_ingest_macros(gateway.con)
            registered = list_ingest_macros(gateway.con)
            missing = MACROS_EXPECTED - registered
            if missing:
                ensure_ingest_macros(gateway.con)
                registered = list_ingest_macros(gateway.con)
                missing = MACROS_EXPECTED - registered
            if missing:
                gateway.close()
                message = f"Missing ingest macros on gateway: {sorted(missing)}"
                raise RuntimeError(message)

        return gateway


def memory_con_with_macros() -> DuckDBConnection:
    """
    Create an in-memory DuckDB connection with ingest macros registered.

    Returns
    -------
    DuckDBConnection
        Connection to an in-memory DuckDB instance with macros ensured.
    """
    con = duckdb.connect(database=":memory:")
    ensure_ingest_macros(con)
    return con


def gateway_with_macros(
    *,
    apply_schema: bool = True,
    ensure_views: bool = True,
    validate_schema: bool = True,
    repo: str | None = None,
    commit: str | None = None,
) -> StorageGateway:
    """
    Create an in-memory StorageGateway with schemas/views/macros ensured.

    Prefer using ``GatewayFactory`` for new code. Delegate to ``GatewayFactory.open()``
    which may raise ``RuntimeError`` if ingest macros cannot be registered.

    Parameters
    ----------
    apply_schema
        Whether to apply database schema on creation.
    ensure_views
        Whether to ensure views are created.
    validate_schema
        Whether to validate the schema after creation.
    repo
        Optional repository identifier.
    commit
        Optional commit hash.

    Returns
    -------
    StorageGateway
        Gateway backed by an in-memory DuckDB connection with ingest macros present.
    """
    factory = GatewayFactory()
    if not apply_schema:
        factory = factory.without_schema()
    if not ensure_views:
        factory = factory.without_views()
    if not validate_schema:
        factory = factory.without_validation()
    if repo is not None and commit is not None:
        factory = factory.with_snapshot(repo, commit)
    return factory.open()


def open_fresh_duckdb(db_path: Path) -> StorageGateway:
    """
    Return a fresh DuckDB connection for tests.

    Prefer using ``GatewayFactory().file_backed(db_path).open()`` for new code.

    Parameters
    ----------
    db_path
        Path to the database file.

    Returns
    -------
    StorageGateway
        Open gateway (caller must close).
    """
    return GatewayFactory().file_backed(db_path).open()


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

    Prefer using ``GatewayFactory`` for new code.

    Parameters
    ----------
    apply_schema
        Whether to apply database schema.
    ensure_views
        Whether to ensure views are created.
    validate_schema
        Whether to validate schema.
    strict_schema
        Whether to enforce strict schema mode.

    Returns
    -------
    StorageGateway
        Gateway configured for ingestion tests.
    """
    factory = GatewayFactory()
    if not apply_schema:
        factory = factory.without_schema()
    factory = factory.with_views() if ensure_views else factory.without_views()
    if not validate_schema:
        factory = factory.without_validation()
    factory = factory.strict() if strict_schema else factory.relaxed()
    return factory.open()


# Backward compatibility alias
open_ingestion_gateway_with_macros = open_ingestion_gateway


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
    query = cast("DuckDBQueryApi", ScopeRecordingQuery(delegate=delegate))
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


__all__ = [
    "MACROS_EXPECTED",
    "DuckDBConnection",
    "GatewayFactory",
    "ScopeRecordingQuery",  # Re-exported from fakes.serving
    "build_duckdb_backend",
    "build_duckdb_query_service",
    "build_scope_parsing_service",
    "gateway_with_macros",
    "memory_con_with_macros",
    "open_fresh_duckdb",
    "open_ingestion_gateway",
    "open_ingestion_gateway_with_macros",  # Alias for backward compatibility
    "seed_tables",
]
