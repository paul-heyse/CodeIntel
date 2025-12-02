Here’s a concrete, step‑by‑step migration plan for **Epic 2 – Split `DuckDBQueryService` into composable backend services**, tailored to your actual layout (`duckdb_service.py`, `query_api.py`, `serving/services/*`, and tests).

I’ll:

* Keep **public behavior and types** stable (`DuckDBQueryApi`, `DuckDBQueryService`, `BackendContext`, etc.).
* Move logic out of `duckdb_service.py` into focused backend modules.
* Keep `BackendContext` / `DuckDBRepositories` as shared infra.
* Show *exact* file‑level edits and where to add tests.

---

## 0. Guard rails / invariants

These are the things you **keep unchanged** across the refactor:

* `codeintel.serving.backend.DuckDBQueryService` still implements `DuckDBQueryApi`.
* `BackendContext`, `DuckDBRepositories`, `BackendLimits`, `GraphEngineProvider` remain importable from `codeintel.serving.backend`.
* `LocalQueryService` and HTTP/MCP layers still talk to `DuckDBQueryApi` only.
* Existing tests using `tests._helpers.gateway.build_duckdb_query_service` keep compiling.

The refactor is purely **internal architecture**; everything above `serving/backend` shouldn’t notice.

---

## 1. Introduce a backend “core” module for shared infra

Right now `duckdb_service.py` defines:

* `BackendContext`
* `GraphEngineProvider`
* `DuckDBRepositories`
* `_FunctionQueries`
* `_ModuleQueries`
* `_SubsystemQueries`
* `_DatasetQueries`
* `DuckDBQueryService`

That makes splitting problematic because new backend modules would need these infra types and `duckdb_service` would need the backends → circular imports.

### 1.1 Create `serving/backend/core.py`

New file: `codeintel/serving/backend/core.py`:

```python
# codeintel/serving/backend/core.py
from __future__ import annotations

from dataclasses import dataclass, field

from codeintel.graphs.engine import GraphEngine
from codeintel.serving.backend.pagination import BackendLimits
from codeintel.serving.services import errors
from codeintel.storage.gateway import StorageGateway, DuckDBConnection
from codeintel.storage.repositories import (
    DatasetReadRepository,
    FunctionRepository,
    GraphRepository,
    ModuleRepository,
    SubsystemRepository,
    TestRepository,
)


@dataclass
class BackendContext:
    """Shared context for DuckDB-backed backend services."""

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

        Raises
        ------
        ProblemError
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
        if self._functions is None:
            self._functions = FunctionRepository(self.gateway, self.repo, self.commit)
        return self._functions

    @property
    def modules(self) -> ModuleRepository:
        if self._modules is None:
            self._modules = ModuleRepository(self.gateway, self.repo, self.commit)
        return self._modules

    @property
    def subsystems(self) -> SubsystemRepository:
        if self._subsystems is None:
            self._subsystems = SubsystemRepository(self.gateway, self.repo, self.commit)
        return self._subsystems

    @property
    def tests(self) -> TestRepository:
        if self._tests is None:
            self._tests = TestRepository(self.gateway, self.repo, self.commit)
        return self._tests

    @property
    def datasets(self) -> DatasetReadRepository:
        if self._datasets is None:
            self._datasets = DatasetReadRepository(self.gateway, self.repo, self.commit)
        return self._datasets

    @property
    def graphs(self) -> GraphRepository:
        if self._graphs is None:
            self._graphs = GraphRepository(self.gateway, self.repo, self.commit)
        return self._graphs


# Re-export common storage types so backends can import from one place.
__all__ = [
    "BackendContext",
    "DuckDBRepositories",
    "GraphEngineProvider",
    "DuckDBConnection",
    "StorageGateway",
]
```

**Implementation detail:** This is a straight move of `BackendContext`, `GraphEngineProvider`, and `DuckDBRepositories` from `duckdb_service.py` into `core.py` (copy/paste bodies, no logic changes).

### 1.2 Update imports

* In `serving/backend/duckdb_service.py`, replace direct definitions with imports:

```python
# old (near top of duckdb_service.py)
# class BackendContext: ...
# class GraphEngineProvider: ...
# class DuckDBRepositories: ...

# new
from codeintel.serving.backend.core import (
    BackendContext,
    DuckDBRepositories,
    GraphEngineProvider,
    DuckDBConnection,
    StorageGateway,
)
```

* In `serving/backend/__init__.py`, change re‑exports to point to `core` for infra:

```python
# old
from codeintel.serving.backend.duckdb_service import (
    BackendContext,
    DuckDBQueryService,
    DuckDBRepositories,
    GraphEngineProvider,
)

# new
from codeintel.serving.backend.core import (
    BackendContext,
    DuckDBRepositories,
    GraphEngineProvider,
)
from codeintel.serving.backend.duckdb_service import DuckDBQueryService
```

Everything importing `BackendContext` / `DuckDBRepositories` / `GraphEngineProvider` from `codeintel.serving.backend` continues to work.

---

## 2. Extract `FunctionBackend` from `_FunctionQueries`

Goal: `_FunctionQueries` currently lives inside `duckdb_service.py` and implements the `FunctionQueriesApi` methods. We’ll:

* Move it to `function_backend.py`.
* Rename to `FunctionBackend`.
* Declare it as implementing `FunctionQueriesApi`.
* Keep method bodies identical.

### 2.1 Create `serving/backend/function_backend.py`

New file:

```python
# codeintel/serving/backend/function_backend.py
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Literal, cast

import networkx as nx

from codeintel.config.steps_graphs import GraphRunScope
from codeintel.serving.backend.core import (
    BackendContext,
    DuckDBRepositories,
    GraphEngineProvider,
    DuckDBConnection,
)
from codeintel.serving.backend.pagination import (
    BackendLimits,
    clamp_limit_value,
)
from codeintel.serving.backend.query_api import FunctionQueriesApi
from codeintel.serving.backend.response_builders import (
    build_callgraph_neighbors_response,
    build_function_architecture_response,
    build_function_profile_response,
    build_function_summary_response,
    build_high_risk_functions_response,
    build_import_boundary_response,
    build_tests_for_function_response,
)
from codeintel.serving.domain_models import Message, ResponseMeta
from codeintel.serving.services import errors
from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    FunctionArchitectureResponse,
    FunctionProfileResponse,
    FunctionSummaryResponse,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    TestsForFunctionResponse,
)
from codeintel.storage.repositories import FunctionRepository, GraphRepository


@dataclass
class FunctionBackend(FunctionQueriesApi):
    """
    DuckDB-backed implementation of FunctionQueriesApi.

    Thin wrapper around DuckDB repositories and graph engine, responsible for:
    - limit clamping and ResponseMeta construction
    - resolving function identifiers (URN / goid_h128 / path+qualname)
    - delegating to repositories and response builders
    """

    context: BackendContext
    repositories: DuckDBRepositories
    engine_provider: GraphEngineProvider

    # --- convenience accessors (copied from _FunctionQueries) ---

    @property
    def con(self) -> DuckDBConnection:
        return self.context.gateway.con

    @property
    def functions(self) -> FunctionRepository:
        return self.repositories.functions

    @property
    def graphs(self) -> GraphRepository:
        return self.repositories.graphs

    # --- the following methods should be moved verbatim
    #     from _FunctionQueries, changing only "self" type ---

    def _require_graph_engine(self) -> nx.DiGraph:
        """
        Move the body of _FunctionQueries._require_graph_engine here unchanged.
        """
        # copy existing logic calling self.engine_provider.require()
        ...

    def _resolve_function_goid(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> int | None:
        """
        Copy existing _FunctionQueries._resolve_function_goid implementation.
        """
        ...

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphRunScope | None = None,
    ) -> FunctionSummaryResponse:
        # body = current _FunctionQueries.get_function_summary
        # uses ResponseMeta, Message, clamp_limit_value, etc.
        ...

    def list_high_risk_functions(
        self,
        *,
        limit: int | None = None,
        risk: Literal["any", "high"] = "any",
        scope: GraphRunScope | None = None,
    ) -> HighRiskFunctionsResponse:
        # move body from _FunctionQueries.list_high_risk_functions
        ...

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphRunScope | None = None,
    ) -> CallGraphNeighborsResponse:
        # move body from _FunctionQueries.get_callgraph_neighbors
        ...

    def get_tests_for_function(
        self,
        *,
        goid_h128: int,
        limit: int | None = None,
        scope: GraphRunScope | None = None,
    ) -> TestsForFunctionResponse:
        # move body from _FunctionQueries.get_tests_for_function
        ...

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        scope: GraphRunScope | None = None,
    ) -> CallGraphNeighborsResponse:
        # move body from _FunctionQueries.get_callgraph_neighborhood
        ...

    def get_import_boundary(
        self,
        *,
        goid_h128: int,
        scope: GraphRunScope | None = None,
    ) -> ImportBoundaryResponse:
        # move body from _FunctionQueries.get_import_boundary
        ...

    def get_function_profile(
        self,
        *,
        goid_h128: int,
    ) -> FunctionProfileResponse:
        # move body from _FunctionQueries.get_function_profile
        ...

    def get_function_architecture(
        self,
        *,
        goid_h128: int,
    ) -> FunctionArchitectureResponse:
        # move body from _FunctionQueries.get_function_architecture
        ...
```

> **Mechanics:** Literally cut each method (and the two private helpers) out of `_FunctionQueries` and paste into `FunctionBackend`, adjusting imports to use `BackendContext`, `DuckDBRepositories`, etc. No semantic changes.

### 2.2 Delete `_FunctionQueries` from `duckdb_service.py`

Remove the entire `class _FunctionQueries:` definition from `duckdb_service.py` once you’ve migrated its content.

---

## 3. Extract `ProfileBackend` from `_ModuleQueries`

This handles file and module profiles/summaries.

### 3.1 Create `serving/backend/profile_backend.py`

```python
# codeintel/serving/backend/profile_backend.py
from __future__ import annotations

from dataclasses import dataclass

from codeintel.config.steps_graphs import GraphRunScope
from codeintel.serving.backend.core import (
    BackendContext,
    DuckDBRepositories,
    DuckDBConnection,
)
from codeintel.serving.backend.query_api import ProfileQueriesApi
from codeintel.serving.backend.response_builders import (
    build_file_profile_response,
    build_file_summary_response,
    build_module_architecture_response,
    build_module_profile_response,
)
from codeintel.serving.domain_models import ResponseMeta
from codeintel.serving.services import errors
from codeintel.serving.mcp.models import (
    FileHintsResponse,
    FileProfileResponse,
    FileSummaryResponse,
    ModuleArchitectureResponse,
    ModuleProfileResponse,
)
from codeintel.storage.repositories import ModuleRepository, SubsystemRepository


@dataclass
class ProfileBackend(ProfileQueriesApi):
    """DuckDB-backed implementation of ProfileQueriesApi for files/modules."""

    context: BackendContext
    repositories: DuckDBRepositories

    @property
    def con(self) -> DuckDBConnection:
        return self.context.gateway.con

    @property
    def modules(self) -> ModuleRepository:
        return self.repositories.modules

    @property
    def subsystems(self) -> SubsystemRepository:
        return self.repositories.subsystems

    # Methods: direct copies from _ModuleQueries

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        # copy from _ModuleQueries.get_file_profile
        ...

    def get_file_summary(
        self,
        *,
        rel_path: str,
        scope: GraphRunScope | None = None,
    ) -> FileSummaryResponse:
        # copy from _ModuleQueries.get_file_summary
        ...

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        # copy from _ModuleQueries.get_module_profile
        ...

    def get_module_architecture(
        self,
        *,
        module: str,
    ) -> ModuleArchitectureResponse:
        # copy from _ModuleQueries.get_module_architecture
        ...

    def get_file_hints(self, *, rel_path: str) -> FileHintsResponse:
        # copy from _ModuleQueries.get_file_hints
        ...
```

### 3.2 Delete `_ModuleQueries` from `duckdb_service.py`

Remove `class _ModuleQueries` once migrated.

---

## 4. Extract `SubsystemBackend` from `_SubsystemQueries`

### 4.1 Create `serving/backend/subsystem_backend.py`

```python
# codeintel/serving/backend/subsystem_backend.py
from __future__ import annotations

from dataclasses import dataclass

from codeintel.serving.backend.core import BackendContext, DuckDBRepositories
from codeintel.serving.backend.pagination import clamp_limit_value
from codeintel.serving.backend.query_api import SubsystemQueriesApi
from codeintel.serving.backend.response_builders import (
    build_subsystem_coverage_response,
    build_subsystem_modules_response,
    build_subsystem_profile_response,
    build_subsystem_summary_response,
    build_paginated_subsystems_response,
)
from codeintel.serving.domain_models import ResponseMeta
from codeintel.serving.services import errors
from codeintel.serving.mcp.models import (
    SubsystemCoverageResponse,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemSummaryResponse,
)
from codeintel.storage.repositories import SubsystemRepository


@dataclass
class SubsystemBackend(SubsystemQueriesApi):
    """DuckDB-backed implementation of SubsystemQueriesApi."""

    context: BackendContext
    repositories: DuckDBRepositories

    @property
    def subsystems(self) -> SubsystemRepository:
        return self.repositories.subsystems

    def list_subsystems(
        self,
        *,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> SubsystemSummaryResponse:
        # copy from _SubsystemQueries.list_subsystems
        ...

    def get_module_subsystems(self, *, module: str) -> SubsystemModulesResponse:
        # copy from _SubsystemQueries.get_module_subsystems
        ...

    def get_subsystem_modules(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> SubsystemModulesResponse:
        # copy from _SubsystemQueries.get_subsystem_modules
        ...

    def search_subsystems(
        self,
        *,
        q: str,
        limit: int | None = None,
    ) -> SubsystemSummaryResponse:
        # copy from _SubsystemQueries.search_subsystems
        ...

    def summarize_subsystem(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> SubsystemModulesResponse:
        # copy from _SubsystemQueries.summarize_subsystem
        ...

    def list_subsystem_profiles(
        self,
        *,
        limit: int | None = None,
    ) -> SubsystemProfileResponse:
        # copy from _SubsystemQueries.list_subsystem_profiles
        ...

    def list_subsystem_coverage(
        self,
        *,
        limit: int | None = None,
    ) -> SubsystemCoverageResponse:
        # copy from _SubsystemQueries.list_subsystem_coverage
        ...
```

### 4.2 Delete `_SubsystemQueries` from `duckdb_service.py`

Once moved, delete `class _SubsystemQueries`.

---

## 5. Extract `DatasetBackend` from `_DatasetQueries`

### 5.1 Create `serving/backend/dataset_backend.py`

```python
# codeintel/serving/backend/dataset_backend.py
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import cast

from codeintel.config.dataset_contract import DatasetContract
from codeintel.serving.backend.core import BackendContext, DuckDBRepositories, StorageGateway
from codeintel.serving.backend.pagination import clamp_limit_value, clamp_offset_value
from codeintel.serving.backend.query_api import DatasetQueriesApi
from codeintel.serving.backend.response_builders import (
    build_dataset_rows_response,
    build_dataset_schema_response,
)
from codeintel.serving.domain_models import Message, ResponseMeta
from codeintel.serving.services import errors
from codeintel.serving.mcp.models import (
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    FileHintsResponse,
)
from codeintel.storage.datasets import (
    dataset_for_name,
    list_dataset_specs,
    load_dataset_registry,
)
from codeintel.storage.gateway import DuckDBConnection, StorageGateway
from codeintel.storage.views.datasets import read_dataset_rows


@dataclass
class DatasetBackend(DatasetQueriesApi):
    """DuckDB-backed implementation of DatasetQueriesApi."""

    context: BackendContext
    repositories: DuckDBRepositories

    @property
    def datasets(self):
        return self.repositories.datasets

    @property
    def gateway(self) -> StorageGateway:
        return self.context.gateway

    @property
    def con(self) -> DuckDBConnection:
        return self.context.gateway.con

    def list_datasets(self) -> list[DatasetSpecDescriptor]:
        # copy from _DatasetQueries.list_datasets
        ...

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        # copy from _DatasetQueries.dataset_specs
        ...

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[Mapping[str, object]]:
        # move the low-level logic to call dataset reader view
        # (existing body from _DatasetQueries.read_dataset_rows)
        ...

    def dataset_schema(
        self,
        *,
        dataset_name: str,
        sample_limit: int = 5,
    ) -> DatasetSchemaResponse:
        # copy from _DatasetQueries.dataset_schema
        ...
```

### 5.2 Delete `_DatasetQueries` from `duckdb_service.py`

Remove `class _DatasetQueries`.

---

## 6. Slim `DuckDBQueryService` into a composition root

Now that function/profile/subsystem/dataset logic lives in dedicated backends, `DuckDBQueryService` just needs to:

* Own a `BackendContext`, `DuckDBRepositories`, and `GraphEngineProvider`.
* Create backend instances in `__post_init__`.
* Implement `DuckDBQueryApi` by exposing them via `functions`, `modules`, `subsystems`, `datasets`.
* Keep `con`, `gateway`, `limits`, `graph_engine`, `__getattr__`, `__dir__` as before.

### 6.1 Rewrite `serving/backend/duckdb_service.py`

Shrink it to something like:

```python
# codeintel/serving/backend/duckdb_service.py
from __future__ import annotations

from dataclasses import dataclass, field

from codeintel.serving.backend.core import (
    BackendContext,
    DuckDBRepositories,
    GraphEngineProvider,
    DuckDBConnection,
    StorageGateway,
)
from codeintel.serving.backend.function_backend import FunctionBackend
from codeintel.serving.backend.profile_backend import ProfileBackend
from codeintel.serving.backend.subsystem_backend import SubsystemBackend
from codeintel.serving.backend.dataset_backend import DatasetBackend
from codeintel.serving.backend.pagination import BackendLimits
from codeintel.serving.backend.query_api import (
    DuckDBQueryApi,
    FunctionQueriesApi,
    ProfileQueriesApi,
    SubsystemQueriesApi,
    DatasetQueriesApi,
)


@dataclass
class DuckDBQueryService(DuckDBQueryApi):
    """Shared query runner facade delegating to backend services.

    This class is now a thin composition root over:
    - FunctionBackend
    - ProfileBackend
    - SubsystemBackend
    - DatasetBackend
    """

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

    # --- DuckDBQueryApi core properties ---

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
    def graph_engine(self):
        """Optional graph engine provided via context or engine provider."""
        return self.engine_provider.graph_engine or self.context.graph_engine

    # --- typed backend accessors for QueryService ---

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

    # --- dynamic delegation for legacy call-sites (if any) ---

    def __getattr__(self, name: str) -> object:
        """
        Delegate attribute lookups to the backend services.

        This preserves the prior behavior where attributes defined on the
        helper classes are visible on DuckDBQueryService directly.
        """
        for helper in (self._functions, self._modules, self._subsystems, self._datasets):
            if hasattr(helper, name):
                return getattr(helper, name)
        raise AttributeError(name)

    def __dir__(self) -> list[str]:
        """Include delegate attributes in dir() for easier introspection."""
        names = set(super().__dir__())
        for helper in (self._functions, self._modules, self._subsystems, self._datasets):
            names.update(dir(helper))
        return sorted(names)
```

**Key points:**

* Type hints align with `DuckDBQueryApi`: `functions`, `modules`, `subsystems`, `datasets`.
* `__getattr__` still offers dynamic delegation, so any legacy code calling `DuckDBQueryService.get_function_summary(...)` directly still works.
* `BackendContext` / `DuckDBRepositories` / `GraphEngineProvider` remain as they were, just imported from `core.py`.

---

## 7. Wiring and bootstrap: keep construction the same

Anywhere you currently do:

```python
context = BackendContext(...)
repositories = DuckDBRepositories(gateway, repo, commit)
engine_provider = GraphEngineProvider(context=context, graph_engine=graph_engine)
backend = DuckDBQueryService(
    context=context,
    repositories=repositories,
    engine_provider=engine_provider,
)
```

still works unchanged (e.g. in:

* `serving/bootstrap.build_service_stack`
* `serving/mcp/backend.QueryBackend`
* `tests/_helpers/gateway.build_duckdb_query_service`)

No changes required beyond import paths already fixed via `backend/__init__.py`.

---

## 8. New backend‑focused tests

You already have strong end‑to‑end coverage via:

* `tests/serving/test_serving_runtime_analytics_e2e.py`
* `tests/serving/test_dataset_specs.py`
* plus many HTTP/MCP tests.

To test the new backends in isolation, add:

### 8.1 `tests/serving/backend/test_function_backend.py`

Focus: limit clamping, GraphRunScope behavior, and graph engine usage.

Examples:

```python
# tests/serving/backend/test_function_backend.py
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from codeintel.serving.backend import BackendContext, BackendLimits, DuckDBRepositories
from codeintel.serving.backend.function_backend import FunctionBackend
from codeintel.serving.backend.core import GraphEngineProvider
from codeintel.serving.services import errors
from codeintel.serving.mcp.models import FunctionSummaryResponse
from tests._helpers.gateway import build_test_gateway  # or similar helper


def make_fake_context_and_repos(gateway):
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    context = BackendContext(
        gateway=gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
        graph_engine=None,
    )
    repos = DuckDBRepositories(gateway, context.repo, context.commit)
    engine_provider = GraphEngineProvider(context=context, graph_engine=None)
    return context, repos, engine_provider


def test_get_function_summary_requires_identifier(test_gateway):
    context, repos, engine_provider = make_fake_context_and_repos(test_gateway)
    backend = FunctionBackend(context=context, repositories=repos, engine_provider=engine_provider)

    # Missing urn/goid/path+qualname should raise invalid_argument
    with pytest.raises(errors.ProblemError) as excinfo:
        backend.get_function_summary()
    assert excinfo.value.problem.code == "invalid_argument"


def test_get_function_summary_returns_response_meta(test_gateway, sample_function):
    context, repos, engine_provider = make_fake_context_and_repos(test_gateway)
    backend = FunctionBackend(context=context, repositories=repos, engine_provider=engine_provider)

    resp: FunctionSummaryResponse = backend.get_function_summary(goid_h128=sample_function.goid_h128)
    assert resp.meta is not None
    # optional: assert messages / pagination as expected
```

You can also add a `FakeGraphEngine` to verify that `get_callgraph_neighbors` calls into it with the clamped limit.

### 8.2 `tests/serving/backend/test_profile_backend.py`

Check file/module profile paths:

```python
def test_get_file_profile_not_found_raises(test_gateway):
    context, repos, _ = make_fake_context_and_repos(test_gateway)
    backend = ProfileBackend(context=context, repositories=repos)

    with pytest.raises(errors.ProblemError) as excinfo:
        backend.get_file_profile(rel_path="nonexistent.py")
    assert excinfo.value.problem.code == "not_found"
```

### 8.3 `tests/serving/backend/test_subsystem_backend.py`

Check pagination clamps and not_found errors:

```python
def test_list_subsystems_respects_backend_limits(test_gateway):
    context, repos, _ = make_fake_context_and_repos(test_gateway)
    backend = SubsystemBackend(context=context, repositories=repos)

    result = backend.list_subsystems(limit=10)
    assert len(result.subsystems) <= context.limits.max_rows_per_call
    assert result.meta.limit <= context.limits.max_rows_per_call
```

### 8.4 `tests/serving/backend/test_dataset_backend.py`

Specifically test:

* `list_datasets` uses registry.
* `dataset_schema` populates validation profile and capabilities.

```python
def test_dataset_schema_includes_validation_profile(test_gateway):
    context, repos, _ = make_fake_context_and_repos(test_gateway)
    backend = DatasetBackend(context=context, repositories=repos)

    resp = backend.dataset_schema(dataset_name="docs.functions")
    assert resp.dataset_name == "docs.functions"
    assert resp.schema.columns  # not empty
    assert resp.meta is not None
```

---

## 9. Regression & cleanup

1. **Run all tests** after the refactor:

   * Especially `tests/serving/test_serving_runtime_analytics_e2e.py` and `tests/serving/test_dataset_specs.py`.
2. Delete any now‑unused imports from `duckdb_service.py` (e.g. `FunctionRepository`, `GraphRepository` etc if only referenced in backends).
3. Update docstrings:

   * `duckdb_service.py`: clarify it’s a composition root.
   * New backend modules: document that they implement the query protocols.

---

