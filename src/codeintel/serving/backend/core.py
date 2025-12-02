"""Core context and repository helpers for DuckDB-backed backends."""

from __future__ import annotations

from dataclasses import dataclass, field

from codeintel.graphs.engine import GraphEngine
from codeintel.serving.backend.pagination import BackendLimits
from codeintel.serving.mcp import errors
from codeintel.storage.gateway import DuckDBConnection, StorageGateway
from codeintel.storage.repositories import (
    DatasetReadRepository,
    FunctionRepository,
    GraphRepository,
    ModuleRepository,
    SubsystemRepository,
    TestRepository,
)


@dataclass(frozen=True)
class BackendContext:
    """Shared context for DuckDB query services."""

    gateway: StorageGateway
    repo: str
    commit: str
    limits: BackendLimits
    graph_engine: GraphEngine | None = None


@dataclass
class GraphEngineProvider:
    """Resolve and cache a graph engine for DuckDB query services."""

    context: BackendContext
    graph_engine: GraphEngine | None = None
    _engine: GraphEngine | None = field(default=None, init=False, repr=False)

    def require(self) -> GraphEngine:
        """
        Return a graph engine or raise when unavailable.

        Returns
        -------
        GraphEngine
            Resolved graph engine for the configured repo/commit.

        Raises
        ------
        backend_failure
            When no graph engine is available.
        """
        if self._engine is not None:
            return self._engine
        if self.graph_engine is not None:
            self._engine = self.graph_engine
            return self._engine
        if self.context.graph_engine is None:
            message = "Graph engine must be provided to DuckDBQueryService."
            raise errors.backend_failure(message)
        self._engine = self.context.graph_engine
        return self._engine


@dataclass
class DuckDBRepositories:
    """Lazily constructed repositories for DuckDB-backed services."""

    gateway: StorageGateway
    repo: str
    commit: str
    _functions: FunctionRepository | None = field(default=None, init=False, repr=False)
    _modules: ModuleRepository | None = field(default=None, init=False, repr=False)
    _subsystems: SubsystemRepository | None = field(default=None, init=False, repr=False)
    _tests: TestRepository | None = field(default=None, init=False, repr=False)
    _datasets: DatasetReadRepository | None = field(default=None, init=False, repr=False)
    _graphs: GraphRepository | None = field(default=None, init=False, repr=False)

    @property
    def functions(self) -> FunctionRepository:
        """Return a lazily constructed function repository."""
        if self._functions is None:
            self._functions = FunctionRepository(self.gateway, self.repo, self.commit)
        return self._functions

    @property
    def modules(self) -> ModuleRepository:
        """Return a lazily constructed module repository."""
        if self._modules is None:
            self._modules = ModuleRepository(self.gateway, self.repo, self.commit)
        return self._modules

    @property
    def subsystems(self) -> SubsystemRepository:
        """Return a lazily constructed subsystem repository."""
        if self._subsystems is None:
            self._subsystems = SubsystemRepository(self.gateway, self.repo, self.commit)
        return self._subsystems

    @property
    def tests(self) -> TestRepository:
        """Return a lazily constructed test repository."""
        if self._tests is None:
            self._tests = TestRepository(self.gateway, self.repo, self.commit)
        return self._tests

    @property
    def datasets(self) -> DatasetReadRepository:
        """Return a lazily constructed dataset repository."""
        if self._datasets is None:
            self._datasets = DatasetReadRepository(self.gateway, self.repo, self.commit)
        return self._datasets

    @property
    def graphs(self) -> GraphRepository:
        """Return a lazily constructed graph repository."""
        if self._graphs is None:
            self._graphs = GraphRepository(self.gateway, self.repo, self.commit)
        return self._graphs


__all__ = [
    "BackendContext",
    "DuckDBConnection",
    "DuckDBRepositories",
    "GraphEngineProvider",
    "StorageGateway",
]
