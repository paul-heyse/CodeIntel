"""Shared helpers for isolated gateway/DuckDB test setup (ibis-only).

This module now builds gateways without ingest macro registration. Tests should
exercise ibis/sqlglot paths and avoid DuckDB macros or legacy storage adapters.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast
from warnings import warn

import duckdb

from codeintel.serving.backend import (
    BackendContext,
    DuckDBQueryService,
    DuckDBRepositories,
    GraphEngineProvider,
)
from codeintel.serving.backend.pagination import BackendLimits
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.storage.gateway import DuckDBConnection, StorageConfig, open_gateway
from codeintel.storage.gateway import open_memory_gateway as _open_memory_gateway
from tests._helpers.fakes.serving import ScopeRecordingQuery

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from codeintel.graphs.engine import GraphEngine
    from codeintel.serving.backend.query_api import DuckDBQueryApi
    from codeintel.serving.services.observability import ServiceObservability
    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.env_options import GatewayOptions

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
    >>> gateway = GatewayFactory().file_backed(db_path).with_schema().open()
    >>> gateway = GatewayFactory.from_options(opts).open()
    """

    def __init__(self) -> None:
        """Initialize factory with defaults."""
        self._apply_schema: bool = True
        self._ensure_views: bool = True
        self._validate_schema: bool = True
        self._strict_schema: bool = True
        self._file_backed: bool = False
        self._db_path: Path | None = None
        self._repo: str | None = None
        self._commit: str | None = None

    @classmethod
    def from_options(cls, options: GatewayOptions) -> GatewayFactory:
        """Create a factory configured from a GatewayOptions dataclass.

        Parameters
        ----------
        options
            Gateway configuration options.

        Returns
        -------
        GatewayFactory
            Factory configured with the provided options.
        """
        factory = cls()
        factory._apply_schema = options.apply_schema
        factory._ensure_views = options.ensure_views
        factory._validate_schema = options.validate_schema
        factory._strict_schema = options.strict_schema
        factory._file_backed = options.file_backed
        factory._db_path = options.db_path
        factory._repo = options.repo
        factory._commit = options.commit
        return factory

    def with_macros(self) -> GatewayFactory:
        """Deprecate legacy macro toggle (no-op).

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        warn(
            "GatewayFactory.with_macros is deprecated and a no-op; macros are no longer used.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self

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

        return gateway


@contextmanager
def analytics_gateway(options: GatewayOptions | None = None) -> Iterator[StorageGateway]:
    """Context-managed gateway creation for analytics tests.

    Parameters
    ----------
    options
        Optional GatewayOptions to configure the gateway.

    Yields
    ------
    StorageGateway
        Gateway with schema/views/macros applied.
    """
    factory = GatewayFactory.from_options(options) if options else GatewayFactory()
    gateway = factory.open()
    try:
        yield gateway
    finally:
        gateway.close()


def memory_con_with_macros() -> DuckDBConnection:
    """Return an in-memory DuckDB connection (deprecated name).

    Returns
    -------
    DuckDBConnection
        Connection to an in-memory DuckDB instance.
    """
    warn(
        "memory_con_with_macros is deprecated; macros are not registered in tests.",
        DeprecationWarning,
        stacklevel=2,
    )
    return duckdb.connect(database=":memory:")


def gateway_with_macros() -> StorageGateway:
    """Return an in-memory gateway with schema/views (deprecated name).

    Returns
    -------
    StorageGateway
        Gateway configured with schema/views.
    """
    warn(
        "gateway_with_macros is deprecated; use GatewayFactory().open() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return GatewayFactory().open()


def seed_tables(gateway: StorageGateway, ddl: list[str]) -> None:
    """Apply defensive DDL statements (DROP/CREATE) to avoid cross-test conflicts."""
    for stmt in ddl:
        gateway.con.execute(stmt)


def seed_repo_identity(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    modules: dict[str, str] | None = None,
) -> None:
    """
    Insert a repo identity row for serving-layer verification.

    Parameters
    ----------
    gateway
        Target gateway with an applied schema.
    repo
        Repository slug to record.
    commit
        Commit hash to record.
    modules
        Optional module->path mappings to persist alongside identity.
    """
    modules_payload = modules or {}
    gateway.con.execute(
        "DELETE FROM core.repo_map WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    gateway.con.execute(
        """
        INSERT INTO core.repo_map (repo, commit, modules, overlays, generated_at)
        VALUES (?, ?, ?, '{}', CURRENT_TIMESTAMP)
        """,
        [repo, commit, json.dumps(modules_payload)],
    )
    if modules_payload:
        gateway.con.execute(
            "DELETE FROM core.modules WHERE repo = ? AND commit = ?",
            [repo, commit],
        )
        gateway.con.executemany(
            """
            INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
            VALUES (?, ?, ?, ?, 'python', '[]', '[]')
            """,
            [(module, path, repo, commit) for module, path in modules_payload.items()],
        )


@dataclass(frozen=True)
class BackendOptions:
    """Options for building a DuckDBBackend."""

    limits: BackendLimits = field(default_factory=BackendLimits)
    observability: ServiceObservability | None = None
    query_engine: GraphEngine | None = None


def build_duckdb_backend(
    gateway: StorageGateway,
    *,
    repo: str | None = None,
    commit: str | None = None,
    service: LocalQueryService | None = None,
    options: BackendOptions | None = None,
) -> DuckDBBackend:
    """Construct a DuckDBBackend with gateway config fallbacks.

    When ``service`` is not provided, a default LocalQueryService is constructed
    using the gateway and repo/commit configuration.

    Parameters
    ----------
    gateway
        Active storage gateway for the backend to use.
    repo
        Optional repo override; falls back to ``gateway.config.repo``.
    commit
        Optional commit override; falls back to ``gateway.config.commit``.
    service
        Optional LocalQueryService. When not provided, one is built internally.
    options
        Optional backend options (limits, observability, query_engine).

    Returns
    -------
    DuckDBBackend
        Backend ready for adapter/service tests.
    """
    opts = options or BackendOptions()
    repo_value = repo or gateway.config.repo or "demo/repo"
    commit_value = commit or gateway.config.commit or "deadbeef"

    if service is None:
        context = BackendContext(
            gateway=gateway,
            repo=repo_value,
            commit=commit_value,
            limits=opts.limits,
            graph_engine=opts.query_engine,
        )
        repositories = DuckDBRepositories(gateway, repo_value, commit_value)
        provider = GraphEngineProvider(context=context, graph_engine=opts.query_engine)
        query = DuckDBQueryService(
            context=context,
            repositories=repositories,
            engine_provider=provider,
        )
        service = LocalQueryService(query=query, observability=opts.observability)

    return DuckDBBackend(
        service=service,
        gateway=gateway,
        repo=repo_value,
        commit=commit_value,
        limits=opts.limits,
        observability=opts.observability,
        query_engine=opts.query_engine,
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
    "BackendOptions",
    "DuckDBConnection",
    "GatewayFactory",
    "ScopeRecordingQuery",  # Re-exported from fakes.serving
    "build_duckdb_backend",
    "build_duckdb_query_service",
    "build_scope_parsing_service",
    "gateway_with_macros",
    "memory_con_with_macros",
    "seed_repo_identity",
    "seed_tables",
]
