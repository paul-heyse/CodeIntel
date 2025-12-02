"""DuckDB-backed query service shared by all serving surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field

from codeintel.graphs.engine import GraphEngine
from codeintel.serving.backend.core import (
    BackendContext,
    DuckDBConnection,
    DuckDBRepositories,
    GraphEngineProvider,
    StorageGateway,
)
from codeintel.serving.backend.dataset_backend import DatasetBackend
from codeintel.serving.backend.function_backend import FunctionBackend
from codeintel.serving.backend.pagination import BackendLimits
from codeintel.serving.backend.profile_backend import ProfileBackend
from codeintel.serving.backend.query_api import (
    DatasetQueriesApi,
    DuckDBQueryApi,
    FunctionQueriesApi,
    ProfileQueriesApi,
    SubsystemQueriesApi,
)
from codeintel.serving.backend.subsystem_backend import SubsystemBackend


@dataclass
class DuckDBQueryService(DuckDBQueryApi):
    """Shared query runner facade delegating to backend services."""

    context: BackendContext
    repositories: DuckDBRepositories
    engine_provider: GraphEngineProvider

    _functions: FunctionBackend = field(init=False, repr=False)
    _modules: ProfileBackend = field(init=False, repr=False)
    _subsystems: SubsystemBackend = field(init=False, repr=False)
    _datasets: DatasetBackend = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Construct backend delegates backed by shared context/repos."""
        self._functions = FunctionBackend(
            context=self.context,
            repositories=self.repositories,
            engine_provider=self.engine_provider,
        )
        self._modules = ProfileBackend(
            context=self.context,
            repositories=self.repositories,
        )
        self._subsystems = SubsystemBackend(
            context=self.context,
            repositories=self.repositories,
        )
        self._datasets = DatasetBackend(
            context=self.context,
            repositories=self.repositories,
        )

    def __getattr__(self, name: str) -> object:
        """
        Delegate attribute lookups to the backend services.

        This preserves the prior behavior where attributes defined on the
        helper classes are visible on DuckDBQueryService directly.

        Returns
        -------
        object
            Attribute value resolved from a backend delegate.

        Raises
        ------
        AttributeError
            If the attribute is not provided by any delegate.
        """
        for helper in (self._functions, self._modules, self._subsystems, self._datasets):
            if hasattr(helper, name):
                return getattr(helper, name)
        raise AttributeError(name)

    def __dir__(self) -> list[str]:
        """
        Include delegate attributes in dir() for easier introspection.

        Returns
        -------
        list[str]
            Sorted attribute names across the service and its delegates.
        """
        names = set(super().__dir__())
        names.update(
            {
                "context",
                "repositories",
                "engine_provider",
                "con",
                "gateway",
                "limits",
                "graph_engine",
                "functions",
                "modules",
                "subsystems",
                "datasets",
            }
        )
        for helper in (self._functions, self._modules, self._subsystems, self._datasets):
            names.update(dir(helper))
        return sorted(names)

    @property
    def con(self) -> DuckDBConnection:
        """Return the underlying DuckDB connection."""
        return self.context.gateway.con

    @property
    def gateway(self) -> StorageGateway:
        """Expose the storage gateway for callers needing direct access."""
        return self.context.gateway

    @property
    def limits(self) -> BackendLimits:
        """Expose backend limits for services consuming the query facade."""
        return self.context.limits

    @property
    def graph_engine(self) -> GraphEngine | None:
        """Optional graph engine provided via context or engine provider."""
        return self.engine_provider.graph_engine or self.context.graph_engine

    @property
    def functions(self) -> FunctionQueriesApi:
        """Helper for function queries."""
        return self._functions

    @property
    def modules(self) -> ProfileQueriesApi:
        """Helper for module/file queries."""
        return self._modules

    @property
    def subsystems(self) -> SubsystemQueriesApi:
        """Helper for subsystem queries."""
        return self._subsystems

    @property
    def datasets(self) -> DatasetQueriesApi:
        """Helper for dataset queries."""
        return self._datasets


__all__ = ["DuckDBQueryService"]
