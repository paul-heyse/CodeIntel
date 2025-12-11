"""Abstract base classes for service query delegates.

Template Method Pattern
-----------------------
These base classes implement the Template Method pattern where:

- The base class defines the public query method signatures
- Subclasses provide transport-specific execution via ``_execute()``

This design eliminates code duplication between Local and HTTP implementations
while maintaining the ability to customize transport behavior.

Architecture
------------
::

    ┌─────────────────────────────────────────────┐
    │           BaseFunctionQueries               │
    │  - get_function_summary()                   │
    │  - list_high_risk_functions()               │
    │  - ... (defines interface)                  │
    └─────────────────────────────────────────────┘
                       ▲
           ┌──────────┴──────────┐
           │                      │
    ┌──────┴──────┐        ┌─────┴──────┐
    │   Local     │        │    HTTP    │
    │ _execute()  │        │ _execute() │
    │ uses        │        │ uses       │
    │ DuckDBQuery │        │ HTTP req   │
    └─────────────┘        └────────────┘

Usage
-----
Subclasses must implement the ``_execute()`` method to provide transport:

::

    class LocalFunctionQueries(BaseFunctionQueries):
        query: DuckDBQueryApi
        observability: ServiceObservability | None

        def _execute[T](self, operation: str, executor: Callable[[], T], **kwargs) -> T:
            return _observe_call(self.observability, "local", operation, executor)


    class HttpFunctionQueries(BaseFunctionQueries):
        request_json: Callable[..., object]
        limits: BackendLimits

        def _execute[T](self, operation: str, executor: Callable[[], T], **kwargs) -> T:
            return _observe_call(self.observability, "http", operation, executor)

Current Implementation
----------------------
The existing delegate classes (``_FunctionQueryDelegates``, ``_HttpFunctionQueryMixin``)
do not yet inherit from these base classes. They will be migrated in a future refactor.

For now, this module serves as documentation for the intended pattern.

See Also
--------
- ``codeintel.serving.services.functions`` : Function query implementations
- ``codeintel.serving.services.profiles`` : Profile query implementations
- ``codeintel.serving.services.subsystems`` : Subsystem query implementations
- ``codeintel.serving.services.datasets`` : Dataset query implementations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.serving import domain_models as dm
    from codeintel.serving.mcp.models import GraphScopePayload

T = TypeVar("T")


class BaseFunctionQueries(ABC):
    """Abstract base class for function query operations.

    This class defines the interface for function-related queries.
    Subclasses must implement ``_execute()`` to provide transport-specific
    execution (local DuckDB or HTTP forwarding).

    All methods return domain models (``dm.*``).
    """

    @abstractmethod
    def _execute(
        self,
        operation: str,
        executor: Callable[[], T],
        *,
        dataset: str | None = None,
    ) -> T:
        """Execute a query operation through the transport.

        Parameters
        ----------
        operation
            Name of the operation for observability.
        executor
            Callable that performs the actual query.
        dataset
            Optional dataset name for context.

        Returns
        -------
        T
            Result from the executor.
        """
        ...

    @abstractmethod
    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.FunctionSummaryResult:
        """Return a function summary for the given identifiers."""
        ...

    @abstractmethod
    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphScopePayload | None = None,
    ) -> dm.HighRiskFunctionsResult:
        """List high-risk functions with optional filters."""
        ...

    @abstractmethod
    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.CallGraphNeighbors:
        """Return call graph neighbors for a function."""
        ...

    @abstractmethod
    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.TestsForFunctionResult:
        """Return tests that exercise a function."""
        ...

    @abstractmethod
    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> dm.GraphNeighborhood:
        """Return a bounded ego neighborhood in the call graph."""
        ...

    @abstractmethod
    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> dm.ImportBoundary:
        """Return import edges crossing a subsystem boundary."""
        ...

    @abstractmethod
    def get_file_summary(
        self,
        *,
        rel_path: str,
        scope: GraphScopePayload | None = None,
    ) -> dm.FileSummaryResult:
        """Return a file summary with nested function rows."""
        ...


class BaseProfileQueries(ABC):
    """Abstract base class for profile and architecture queries.

    This class defines the interface for profile-related queries.
    Subclasses must implement ``_execute()`` to provide transport-specific
    execution.

    All methods return domain models (``dm.*``).
    """

    @abstractmethod
    def _execute(
        self,
        operation: str,
        executor: Callable[[], T],
        *,
        dataset: str | None = None,
    ) -> T:
        """Execute a query operation through the transport."""
        ...

    @abstractmethod
    def get_function_profile(self, *, goid_h128: int) -> dm.FunctionProfileResult:
        """Return a denormalized function profile."""
        ...

    @abstractmethod
    def get_file_profile(self, *, rel_path: str) -> dm.FileProfileResult:
        """Return a denormalized file profile."""
        ...

    @abstractmethod
    def get_module_profile(self, *, module: str) -> dm.ModuleProfileResult:
        """Return a profile for a module."""
        ...

    @abstractmethod
    def get_function_architecture(self, *, goid_h128: int) -> dm.FunctionArchitectureResult:
        """Return architecture metrics for a function."""
        ...

    @abstractmethod
    def get_module_architecture(self, *, module: str) -> dm.ModuleArchitectureResult:
        """Return architecture metrics for a module."""
        ...

    @abstractmethod
    def get_file_hints(self, *, rel_path: str) -> dm.FileHintsResult:
        """Return IDE hints for a file."""
        ...


class BaseSubsystemQueries(ABC):
    """Abstract base class for subsystem queries.

    This class defines the interface for subsystem-related queries.
    Subclasses must implement ``_execute()`` to provide transport-specific
    execution.

    All methods return domain models (``dm.*``).
    """

    @abstractmethod
    def _execute(
        self,
        operation: str,
        executor: Callable[[], T],
        *,
        dataset: str | None = None,
    ) -> T:
        """Execute a query operation through the transport."""
        ...

    @abstractmethod
    def list_subsystems(
        self,
        *,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> dm.SubsystemSummaryResult:
        """List inferred subsystems with optional filters."""
        ...

    @abstractmethod
    def get_module_subsystems(self, *, module: str) -> dm.ModuleSubsystemResult:
        """Return subsystem memberships for a module."""
        ...

    @abstractmethod
    def get_subsystem_modules(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> dm.SubsystemModulesResult:
        """Return subsystem detail and member modules."""
        ...

    @abstractmethod
    def search_subsystems(
        self,
        *,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> dm.SubsystemSearchResult:
        """Search subsystems by role or label."""
        ...

    @abstractmethod
    def summarize_subsystem(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> dm.SubsystemModulesResult:
        """Summarize a subsystem with optional module truncation."""
        ...

    @abstractmethod
    def list_subsystem_profiles(
        self,
        *,
        limit: int | None = None,
    ) -> dm.SubsystemProfileResult:
        """List subsystem profiles from docs views."""
        ...

    @abstractmethod
    def list_subsystem_coverage(
        self,
        *,
        limit: int | None = None,
    ) -> dm.SubsystemCoverageResult:
        """List subsystem coverage rollups from docs views."""
        ...


class BaseDatasetQueries(ABC):
    """Abstract base class for dataset queries.

    This class defines the interface for dataset-related queries.
    Subclasses must implement ``_execute()`` to provide transport-specific
    execution.

    All methods return domain models (``dm.*``).
    """

    @abstractmethod
    def _execute(
        self,
        operation: str,
        executor: Callable[[], T],
        *,
        dataset: str | None = None,
    ) -> T:
        """Execute a query operation through the transport."""
        ...

    @abstractmethod
    def list_datasets(self) -> list[dm.DatasetDescriptorDomain]:
        """List available datasets."""
        ...

    @abstractmethod
    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> dm.DatasetRows:
        """Read rows from a dataset."""
        ...

    @abstractmethod
    def dataset_schema(
        self,
        *,
        dataset_name: str,
        sample_limit: int = 5,
    ) -> dm.DatasetSchema:
        """Return schema and samples for a dataset."""
        ...


__all__ = [
    "BaseDatasetQueries",
    "BaseFunctionQueries",
    "BaseProfileQueries",
    "BaseSubsystemQueries",
]
