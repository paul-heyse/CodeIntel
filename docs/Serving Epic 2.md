
You’re totally right to call that out—my previous reply *was* effectively talking to the pre-backend world. I’ve just gone through the **latest** `serving.zip` you attached (and confirmed `serving/backend/{__init__,duckdb_service,limits,datasets}.py` etc.), plus the current `services/query_service.py`, `http/fastapi.py`, and `mcp/registry.py`.

Below is a **revised, code-accurate** implementation plan for refactor item 2, explicitly **building on the existing `serving.backend`** layout you now have.

I’ll break it into 3 chunks:

1. Split `services.query_service` by vertical slice (functions / profiles / subsystems / datasets).
2. Split `http.fastapi` into route modules per feature.
3. Split `mcp.registry` into tool modules per feature.

For each, I’ll tell you:

* **Which new files to create**
* **Exactly what to cut-and-paste from the existing code**
* **How to update imports / re-exports**

---

## 0. Current baseline (from your updated zip)

What I’m seeing now (relevant bits):

```text
codeintel/serving/
  backend/
    __init__.py          # re-exports DuckDBQueryService, BackendLimits, clamp_*
    duckdb_service.py    # full DuckDBQueryService impl
    limits.py            # BackendLimits, ClampResult, clamp_* helpers
    datasets.py          # dataset registry helpers, build_registry_and_limits, etc.

  http/
    __init__.py
    fastapi.py           # app + dependencies + *all* routers
    datasets.py          # shim: re-exports backend.datasets helpers
    openapi_codeintel.json

  mcp/
    backend.py           # QueryBackend protocol + DuckDBBackend, HttpBackend
    errors.py
    models.py            # all MCP response models + GraphScopePayload + parse_graph_scope
    registry.py          # _wrap + *_register_*_tools + register_tools
    query_service.py     # shim: re-exports BackendLimits/DuckDBQueryService from backend
    server.py            # create_mcp_server() → registry.register_tools(...)

  services/
    factory.py           # uses backend.BackendLimits/DuckDBQueryService + backend.datasets.*
    wiring.py            # uses backend.BackendLimits + backend.datasets.build_registry_and_limits
    errors.py
    query_service.py     # BIG file: QueryService protocol, all delegates, LocalQueryService, HttpQueryService, observability, HTTP mixin, dataset logic, etc.

  protocols.py           # HasModelDump + SCIP / pytest typed protocols
```

So item 1 (backend split) *is* live. Now we’re just carving out vertical slices from:

* `serving/services/query_service.py`
* `serving/http/fastapi.py`
* `serving/mcp/registry.py`

---

## 1. Split `serving.services.query_service` by feature

### 1.1 Goal structure for `services`

We’re going to introduce a handful of helper modules under `serving/services`:

```text
serving/services/
  query_service.py      # LocalQueryService, HttpQueryService, QueryService protocol, plus re-exports
  functions.py          # _FunctionQueryDelegates, _HttpFunctionQueryMixin
  profiles.py           # _ProfileQueryDelegates, _HttpProfileQueryMixin
  subsystems.py         # _SubsystemQueryDelegates, _HttpSubsystemQueryMixin
  datasets.py           # dataset methods for LocalQueryService + HTTP dataset mixin
  http_transport.py     # _HttpTransportMixin
  observability.py      # ServiceCallContext, ServiceCallMetrics, ServiceObservability, _observe_call
```

We **won’t** touch `serving.backend` or `factory.py`/`wiring.py` – they already use the backend correctly.

---

### 1.2 Extract observability into `services/observability.py`

**New file**: `serving/services/observability.py`

From your current `services/query_service.py`, copy:

* `LOG = logging.getLogger("codeintel.serving.services.query")`
* `ServiceCallMetrics` dataclass
* `ServiceObservability` dataclass
* `ServiceCallContext` dataclass
* `_infer_row_count`
* `_extract_message_count`
* `_observe_call`

And the imports they need.

Concretely:

```python
"""Observability primitives for serving query services."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    DatasetRowsResponse,
    FileHintsResponse,
    HighRiskFunctionsResponse,
    ModuleSubsystemResponse,
    SubsystemCoverageResponse,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
    TestsForFunctionResponse,
)

LOG = logging.getLogger("codeintel.serving.services.query")


@dataclass
class ServiceCallMetrics:
    # copy the full field list exactly as in query_service.py
    ...


@dataclass
class ServiceObservability:
    # copy as-is (enabled: bool, logger: logging.Logger, record(...))
    ...


@dataclass
class ServiceCallContext:
    # copy as-is (dataset, schema_version, retries)
    ...


def _infer_row_count(result: object) -> int | None:
    # copy body as-is (DatasetRowsResponse, HighRiskFunctionsResponse, etc.)
    ...


def _extract_message_count(result: object) -> int | None:
    # copy body as-is
    ...


def _observe_call[T](
    observability: ServiceObservability | None,
    *,
    transport: str,
    name: str,
    context: ServiceCallContext | None,
    func: Callable[[], T],
) -> T:
    # copy body as-is (timing, try/except, metrics, etc.)
    ...


__all__ = [
    "LOG",
    "ServiceCallMetrics",
    "ServiceObservability",
    "ServiceCallContext",
    "_observe_call",
]
```

> **In `services/query_service.py`**:
>
> * Remove those definitions.
> * Replace `LOG = logging.getLogger(...)` and the dataclasses/_observe_call with imports:

```python
from codeintel.serving.services.observability import (
    LOG,
    ServiceCallContext,
    ServiceCallMetrics,
    ServiceObservability,
    _observe_call,
)
```

All uses of `LOG`, `ServiceObservability`, `ServiceCallContext`, `ServiceCallMetrics`, `_observe_call` in this file then stay valid.

---

### 1.3 Extract HTTP transport into `services/http_transport.py`

**New file**: `serving/services/http_transport.py`

Copy the entire `_HttpTransportMixin` class from `query_service.py` into this file.

```python
"""HTTP transport mixin for HttpQueryService and HTTP feature mixins."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

from codeintel.serving.backend import BackendLimits
from codeintel.serving.services.observability import (
    ServiceCallContext,
    ServiceCallMetrics,
    ServiceObservability,
    _observe_call,
)

T = TypeVar("T")


class _HttpTransportMixin:
    """Shared HTTP wrapper providing observability & retry metrics."""

    request_json: Callable[[str, dict[str, object]], object]
    limits: BackendLimits
    observability: ServiceObservability | None

    def _http_call[T](
        self,
        name: str,
        func: Callable[[], T],
        *,
        dataset: str | None = None,
        schema_version: str | None = None,
    ) -> T:
        # copy the body from query_service.py exactly:
        # - inspect self.request_json.__self__ for retry attempts
        # - call _observe_call(...)
        # - emit a separate ServiceCallMetrics for retries if present
        ...
```

> **In `query_service.py`**:
>
> * Delete the old `_HttpTransportMixin` definition.
> * Import it instead:

```python
from codeintel.serving.services.http_transport import _HttpTransportMixin
```

---

### 1.4 Extract function-related delegates into `services/functions.py`

**New file**: `serving/services/functions.py`

From `services/query_service.py`, copy:

* Entire `class _FunctionQueryDelegates:`
* Entire `class _HttpFunctionQueryMixin(_HttpTransportMixin):`

You’ll need imports for:

* `Callable`, `Any`
* `DuckDBQueryService` from `codeintel.serving.backend`
* Response models & helpers from `codeintel.serving.mcp.models`:

  * `FunctionSummaryResponse`, `HighRiskFunctionsResponse`, `CallGraphNeighborsResponse`, `GraphNeighborhoodResponse`, `FunctionArchitectureResponse`, `ModuleArchitectureResponse`, `FileSummaryResponse`, `CallGraphEdgeRow`, `ImportBoundaryResponse`, `GraphScopePayload`, `ResponseMeta`, `ViewRow`, and `parse_graph_scope`
* `clamp_limit_value` from `codeintel.serving.backend` (or backend.limits)
* `_HttpTransportMixin` from `codeintel.serving.services.http_transport`
* `ServiceCallContext`, `_observe_call` from `services.observability` if used (they are used indirectly via `_call` in LocalQueryService / `_http_call`)

Skeleton:

```python
"""Function-centric delegates for local and HTTP query services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from codeintel.serving.backend import DuckDBQueryService, clamp_limit_value
from codeintel.serving.mcp.models import (
    CallGraphEdgeRow,
    CallGraphNeighborsResponse,
    FileSummaryResponse,
    FunctionArchitectureResponse,
    FunctionSummaryResponse,
    GraphNeighborhoodResponse,
    GraphScopePayload,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    ModuleArchitectureResponse,
    ResponseMeta,
    ViewRow,
    parse_graph_scope,
)
from codeintel.serving.services.http_transport import _HttpTransportMixin


class _FunctionQueryDelegates:
    """Local delegates that call DuckDBQueryService for function-related APIs."""

    query: DuckDBQueryService
    _call: Callable[..., Any]

    # COPY ALL METHODS FROM _FunctionQueryDelegates IN query_service.py:
    #
    # - get_function_summary
    # - list_high_risk_functions
    # - get_callgraph_neighbors
    # - get_callgraph_neighborhood
    # - get_import_boundary
    # - get_file_summary
    # - get_function_architecture
    # - get_module_architecture
    # etc.
    #
    # Bodies should be unchanged: they call self._call(...) with lambdas that
    # delegate to self.query.get_* and they pass dataset names.


class _HttpFunctionQueryMixin(_HttpTransportMixin):
    """HTTP-based implementation of the function query API."""

    # COPY ALL METHODS FROM _HttpFunctionQueryMixin IN query_service.py:
    #
    # - get_function_summary
    # - list_high_risk_functions
    # - get_callgraph_neighbors
    # - get_callgraph_neighborhood
    # - get_import_boundary
    # - get_file_summary
    #
    # Bodies should stay identical: build params dict, call self._http_call(),
    # wrap results with ResponseMeta, etc.
```

> **In `query_service.py`**:
>
> * Delete `_FunctionQueryDelegates` and `_HttpFunctionQueryMixin` definitions.
> * Import:

```python
from codeintel.serving.services.functions import (
    _FunctionQueryDelegates,
    _HttpFunctionQueryMixin,
)
```

---

### 1.5 Extract profiles into `services/profiles.py`

**New file**: `serving/services/profiles.py`

From `query_service.py`, copy:

* `class _ProfileQueryDelegates:`
* `class _HttpProfileQueryMixin(_HttpTransportMixin):`

Imports needed:

* `Callable`, `Any`
* `DuckDBQueryService`
* `FunctionProfileResponse`, `FileProfileResponse`, `ModuleProfileResponse`
* `_HttpTransportMixin`

Skeleton:

```python
"""Profile and architecture delegates for query services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from codeintel.serving.backend import DuckDBQueryService
from codeintel.serving.mcp.models import (
    FileProfileResponse,
    FunctionProfileResponse,
    ModuleProfileResponse,
)
from codeintel.serving.services.http_transport import _HttpTransportMixin


class _ProfileQueryDelegates:
    """Local profile-query delegates calling DuckDBQueryService."""

    query: DuckDBQueryService
    _call: Callable[..., Any]

    # COPY methods:
    # - get_function_profile
    # - get_file_profile
    # - get_module_profile


class _HttpProfileQueryMixin(_HttpTransportMixin):
    """HTTP-based profile query mixin."""

    # COPY methods from _HttpProfileQueryMixin (same three).
```

> **In `query_service.py`**:
>
> * Remove the original classes.
> * Add:

```python
from codeintel.serving.services.profiles import (
    _ProfileQueryDelegates,
    _HttpProfileQueryMixin,
)
```

---

### 1.6 Extract subsystems into `services/subsystems.py`

**New file**: `serving/services/subsystems.py`

From `query_service.py`, copy:

* `class _SubsystemQueryDelegates:`
* `class _HttpSubsystemQueryMixin(_HttpTransportMixin):`

Imports needed:

* `Callable`, `Any`
* `DuckDBQueryService`
* `FileHintsResponse`, `ModuleSubsystemResponse`, `SubsystemCoverageResponse`,
  `SubsystemModulesResponse`, `SubsystemProfileResponse`, `SubsystemSearchResponse`,
  `SubsystemSummaryResponse`, `ResponseMeta`
* `clamp_limit_value`
* `_HttpTransportMixin`

Skeleton:

```python
"""Subsystem, hints, and coverage delegates for query services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from codeintel.serving.backend import DuckDBQueryService, clamp_limit_value
from codeintel.serving.mcp.models import (
    FileHintsResponse,
    ModuleSubsystemResponse,
    SubsystemCoverageResponse,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
    ResponseMeta,
)
from codeintel.serving.services.http_transport import _HttpTransportMixin


class _SubsystemQueryDelegates:
    """Local delegates for subsystem-related queries."""

    query: DuckDBQueryService
    _call: Callable[..., Any]

    # COPY methods:
    # - list_subsystems
    # - get_module_subsystems
    # - get_file_hints
    # - get_subsystem_modules
    # - search_subsystems
    # - summarize_subsystem
    # - list_subsystem_coverage


class _HttpSubsystemQueryMixin(_HttpTransportMixin):
    """HTTP-based subsystem query APIs."""

    # COPY corresponding HTTP methods from _HttpSubsystemQueryMixin.
```

> **In `query_service.py`**:
>
> * Remove original classes & import from this module.

---

### 1.7 Extract dataset bits into `services/datasets.py`

**New file**: `serving/services/datasets.py`

We’ll put **two things** here:

1. The *local* dataset methods currently defined directly on `LocalQueryService` (list_datasets, dataset_specs, read_dataset_rows, dataset_schema).
2. The `_HttpDatasetQueryMixin` class used inside `HttpQueryService`.

Imports needed:

* `Callable`, `Any`, `cast`
* `DuckDBQueryService`, `BackendLimits`, `clamp_limit_value`, `clamp_offset_value`
* `DatasetDescriptor`, `DatasetRowsResponse`, `DatasetSchemaResponse`, `DatasetSpecDescriptor`, `ResponseMeta`, `ViewRow`
* `Dataset`, `load_dataset_registry` from `codeintel.storage.datasets`
* `describe_dataset` (right now imported in `query_service.py` from `backend.datasets`)
* `_HttpTransportMixin`
* `ServiceCallContext`, `_observe_call` (used by local `_call` on LocalQueryService – we’ll call `_call` from LocalQueryService, not reimplement here)

Skeleton:

```python
"""Dataset delegates for local and HTTP query services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from codeintel.serving.backend import BackendLimits, DuckDBQueryService, clamp_limit_value, clamp_offset_value
from codeintel.serving.backend.datasets import describe_dataset
from codeintel.serving.mcp.models import (
    DatasetDescriptor,
    DatasetRowsResponse,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    ResponseMeta,
    ViewRow,
)
from codeintel.serving.services.http_transport import _HttpTransportMixin
from codeintel.storage.datasets import Dataset, load_dataset_registry


class _LocalDatasetMixin:
    """
    Local dataset listing and retrieval helpers used by LocalQueryService.

    This is extracted purely to shrink query_service.py; behavior is unchanged.
    """

    query: DuckDBQueryService
    dataset_tables: dict[str, str] | None
    describe_dataset_fn: Callable[[str, str], str]
    limits: BackendLimits
    # NOTE: expects LocalQueryService to provide `_call(...)`

    def list_datasets(self) -> list[DatasetDescriptor]:
        # COPY the entire body from LocalQueryService.list_datasets(...)
        ...

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        # COPY body from LocalQueryService.dataset_specs(...)
        ...

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        # COPY body from LocalQueryService.read_dataset_rows(...)
        ...

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> DatasetSchemaResponse:
        # COPY body from LocalQueryService.dataset_schema(...)
        ...


class _HttpDatasetQueryMixin(_HttpTransportMixin):
    """HTTP-based dataset query APIs used by HttpQueryService."""

    # COPY methods from _HttpDatasetQueryMixin in query_service.py:
    #
    # - list_datasets
    # - dataset_specs
    # - read_dataset_rows
    # - dataset_schema
```

> **In `query_service.py`**:
>
> * Remove the dataset methods from `LocalQueryService` and the definition of `_HttpDatasetQueryMixin`.
> * Import:

```python
from codeintel.serving.services.datasets import _LocalDatasetMixin, _HttpDatasetQueryMixin
```

> * Change the `LocalQueryService` base class from:

```python
class LocalQueryService(_FunctionQueryDelegates, _ProfileQueryDelegates, _SubsystemQueryDelegates):
```

to:

```python
class LocalQueryService(
    _FunctionQueryDelegates,
    _ProfileQueryDelegates,
    _SubsystemQueryDelegates,
    _LocalDatasetMixin,
):
```

> * Leave `HttpQueryService` as:

```python
class HttpQueryService(
    _HttpTransportMixin,
    _HttpFunctionQueryMixin,
    _HttpProfileQueryMixin,
    _HttpSubsystemQueryMixin,
    _HttpDatasetQueryMixin,
    QueryService,
):
    ...
```

(Only the import source for `_HttpDatasetQueryMixin` changed.)

---

### 1.8 Clean up & re-exports in `services/query_service.py`

At this point, `services/query_service.py` should roughly contain:

* Imports for `BackendLimits`, `DuckDBQueryService`, `describe_dataset` (though you could also move `describe_dataset` usage into `_LocalDatasetMixin`).
* Imports for all four feature mixins + HTTP mixin + observability.
* The **API Protocols**:

  * `FunctionQueryApi`
  * `ProfileQueryApi`
  * `SubsystemQueryApi`
  * `DatasetQueryApi`
  * `QueryService`
* `LocalQueryService` (now mostly wiring + `_call`, inheriting delegates/mixins).
* `HttpQueryService`.

You can also re-export key types for backwards compatibility:

```python
__all__ = [
    "QueryService",
    "LocalQueryService",
    "HttpQueryService",
    "ServiceCallContext",
    "ServiceCallMetrics",
    "ServiceObservability",
]
```

(using imports from `services.observability`).

---

## 2. Split FastAPI server (`http.fastapi`) into route modules

Your current `serving/http/fastapi.py` has:

* config / backend wiring
* dependency injection (`get_app_config`, `get_backend`, `get_service`, type aliases)
* all routers (`build_functions_router`, `build_profiles_router`, `build_architecture_router`, `build_subsystem_router`, `build_ide_router`, `build_datasets_router`, `build_health_router`)
* app construction (`create_app`, lifespan, etc.)

We’ll carve out:

```text
serving/http/
  dependencies.py      # get_app_config/get_backend/get_service + ConfigDep/BackendDep/ServiceDep
  routes/
    __init__.py
    functions.py       # build_functions_router()
    profiles.py        # build_profiles_router()
    architecture.py    # build_architecture_router()
    subsystems.py      # build_subsystem_router()
    ide.py             # build_ide_router()
    datasets.py        # build_datasets_router()
    health.py          # build_health_router()
  fastapi.py           # now: create_app + includes of routers, imports deps/routes
```

### 2.1 Create `serving/http/dependencies.py`

**New file**: `serving/http/dependencies.py`

Move from `fastapi.py`:

* `get_app_config`
* `get_backend`
* `get_service`
* `ConfigDep`, `BackendDep`, `ServiceDep`

Use the same code, just new home:

```python
"""FastAPI dependency injection helpers for CodeIntel serving."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.backend import QueryBackend
from codeintel.serving.services.query_service import QueryService


def get_app_config(request: Request) -> ServingConfig:
    # COPY body from fastapi.py
    ...


def get_backend(request: Request) -> QueryBackend:
    # COPY body from fastapi.py
    ...


def get_service(request: Request) -> QueryService:
    # COPY body from fastapi.py
    ...


ConfigDep = Annotated[ServingConfig, Depends(get_app_config)]
BackendDep = Annotated[QueryBackend, Depends(get_backend)]
ServiceDep = Annotated[QueryService, Depends(get_service)]

__all__ = ["ConfigDep", "BackendDep", "ServiceDep", "get_app_config", "get_backend", "get_service"]
```

> **In `fastapi.py`**:
>
> * Delete those functions/aliases.
> * Replace their uses with imports:

```python
from codeintel.serving.http.dependencies import ConfigDep, BackendDep, ServiceDep
```

…and update any internal references accordingly (most of them are type hints on endpoints).

---

### 2.2 Split routers into `serving/http/routes/*`

For each `build_*_router` in `fastapi.py`, create a separate module under `serving/http/routes`.

Example for functions:

**New file**: `serving/http/routes/functions.py`

```python
"""Function-centric HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter

from codeintel.serving.http.dependencies import ServiceDep
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    FunctionArchitectureResponse,
    FunctionSummaryResponse,
    GraphNeighborhoodResponse,
    HighRiskFunctionsResponse,
)

def build_functions_router() -> APIRouter:
    """
    Construct the router for function-centric endpoints.
    """
    router = APIRouter()

    # COPY EVERYTHING from fastapi.build_functions_router:
    #
    # - @router.get("/function/summary", ...)
    # - @router.get("/function/callgraph", ...)
    # - @router.get("/function/callgraph/neighborhood", ...)
    # - @router.get("/function/architecture", ...)
    # - @router.get("/module/architecture", ...)
    #
    # Make sure the injected `service` param uses `service: ServiceDep`.

    return router
```

Do the same for:

* `serving/http/routes/profiles.py` (moving `build_profiles_router`).
* `serving/http/routes/architecture.py` (if distinct from functions).
* `serving/http/routes/subsystems.py` (moving `build_subsystem_router`).
* `serving/http/routes/ide.py` (IDE / navigation endpoints).
* `serving/http/routes/datasets.py` (dataset endpoints).
* `serving/http/routes/health.py` (health / ready / live endpoints).

Each module will:

* import `APIRouter`
* import `ServiceDep` (and `ConfigDep`/`BackendDep` if needed)
* import the specific response/request models it uses from `codeintel.serving.mcp.models`
* copy the router body intact from `fastapi.py`.

**New file**: `serving/http/routes/__init__.py`

```python
"""Feature-specific FastAPI routers for CodeIntel serving."""
```

---

### 2.3 Simplify `serving/http/fastapi.py`

Now, in `fastapi.py`:

1. **Imports**: add the route builders and dependencies:

```python
from fastapi import FastAPI

from codeintel.serving.http.dependencies import ConfigDep, BackendDep, ServiceDep
from codeintel.serving.http.routes.architecture import build_architecture_router
from codeintel.serving.http.routes.datasets import build_datasets_router
from codeintel.serving.http.routes.functions import build_functions_router
from codeintel.serving.http.routes.health import build_health_router
from codeintel.serving.http.routes.ide import build_ide_router
from codeintel.serving.http.routes.profiles import build_profiles_router
from codeintel.serving.http.routes.subsystems import build_subsystem_router
```

2. **Remove** the definitions of all `build_*_router` functions and the DI functions (you already moved them).

3. Inside your `create_app` / `build_app` function, where you currently have:

```python
app.include_router(build_functions_router())
app.include_router(build_profiles_router())
app.include_router(build_architecture_router())
app.include_router(build_subsystem_router())
app.include_router(build_ide_router())
app.include_router(build_datasets_router())
app.include_router(build_health_router())
```

leave that intact — it now just calls into the new modules.

The result: `fastapi.py` is mostly:

* app + lifespan
* wiring backend to app.state
* including routers

All endpoint logic lives under `http/routes/*`.

---

## 3. Split MCP registry into feature tool modules

`serving/mcp/registry.py` today has:

* `_wrap` helper
* `_register_function_tools`
* `_register_profile_tools`
* `_register_architecture_tools`
* `_register_dataset_tools`
* `register_tools` that calls these.

We’ll end up with:

```text
serving/mcp/
  tools_base.py        # _wrap, ProblemDetail alias, `register_tools` orchestration OR QueryBackendOrService alias
  function_tools.py    # register_function_tools
  profile_tools.py     # register_profile_tools
  architecture_tools.py# register_architecture_tools (graph plugin)
  dataset_tools.py     # register_dataset_tools
  registry.py          # thin shim that imports register_tools from tools_base (optional)
```

To keep churn small, we can:

* put `_wrap` + `register_tools` in `tools_base.py`
* move the `_register_*` functions into per-feature modules
* have `registry.py` just import `register_tools` from `tools_base` and re-export it.

### 3.1 Create `serving/mcp/tools_base.py`

**New file**: `serving/mcp/tools_base.py`

Copy from `registry.py`:

* `_wrap` function (unchanged)
* `register_tools` function body, **minus** the inline `_register_*` definitions.

You’ll also need the imports:

* `FastMCP`
* `QueryBackend`
* `errors`
* All the feature `register_*_tools` functions (from the new modules you create next)

Example:

```python
"""Common MCP tool registration helpers and error wrapping."""

from __future__ import annotations

from collections.abc import Callable

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp import errors
from codeintel.serving.mcp.backend import QueryBackend
from codeintel.serving.mcp.models import ProblemDetail
from codeintel.serving.mcp.function_tools import register_function_tools
from codeintel.serving.mcp.profile_tools import register_profile_tools
from codeintel.serving.mcp.architecture_tools import register_architecture_tools
from codeintel.serving.mcp.dataset_tools import register_dataset_tools


def _wrap(func: Callable[..., object]) -> Callable[..., object]:
    """
    COPY the existing _wrap implementation from registry.py exactly.
    """
    ...


def register_tools(mcp: FastMCP, backend: QueryBackend) -> None:
    """
    Register all MCP tools against the provided backend.

    COPY the body of the existing register_tools, but instead of defining the
    _register_* helpers inline, just call the imported functions:
    """
    register_function_tools(mcp, backend)
    register_profile_tools(mcp, backend)
    register_architecture_tools(mcp, backend)
    register_dataset_tools(mcp, backend)


__all__ = ["_wrap", "register_tools"]
```

*(We’ll adjust imports once the feature modules exist.)*

---

### 3.2 Create per-feature tool modules

Each of these is just the corresponding `_register_*` function copied over:

#### 3.2.1 `serving/mcp/function_tools.py`

**New file**: `serving/mcp/function_tools.py`

```python
"""Function-oriented MCP tools for CodeIntel."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    FileSummaryResponse,
    FunctionSummaryResponse,
    GraphNeighborhoodResponse,
    HighRiskFunctionsResponse,
    ProblemDetail,
)
from codeintel.serving.mcp.tools_base import _wrap
from codeintel.serving.mcp.backend import QueryBackend


def register_function_tools(mcp: FastMCP, backend: QueryBackend) -> None:
    """Register function-related MCP tools on the given FastMCP instance."""

    # COPY the body of _register_function_tools from registry.py, but:
    # - rename it to register_function_tools
    # - use the passed `backend`
    # - decorate tools with @mcp.tool() and @_wrap as before
    #
    # Example pattern (directly from your existing code):
    #
    # @mcp.tool()
    # @_wrap
    # def get_function_summary(...)->dict[str, object] | dict[str, ProblemDetail]:
    #     resp: FunctionSummaryResponse = backend.get_function_summary(...)
    #     return resp.model_dump()
    #
    # etc.
    ...
```

Similarly:

#### 3.2.2 `serving/mcp/profile_tools.py`

Copy `_register_profile_tools` and rename to `register_profile_tools`.

#### 3.2.3 `serving/mcp/architecture_tools.py`

Copy `_register_architecture_tools` (including graph metric plugin planning) and rename to `register_architecture_tools`.

#### 3.2.4 `serving/mcp/dataset_tools.py`

Copy `_register_dataset_tools` and rename to `register_dataset_tools`.

Each file:

* Imports the specific response models it uses (just mirror the top of `registry.py`).
* Imports `_wrap` from `tools_base` and `QueryBackend` from `mcp.backend`.

---

### 3.3 Slim down `serving/mcp/registry.py`

Finally, make `registry.py` just a thin shim so the rest of the code (e.g. `mcp/server.py`) doesn’t change.

**Edit**: `serving/mcp/registry.py`

Replace its body with:

```python
"""MCP tool registry shim.

Canonical implementations live in feature-specific modules and tools_base.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.backend import QueryBackend
from codeintel.serving.mcp.tools_base import register_tools as _register_tools


def register_tools(mcp: FastMCP, backend: QueryBackend) -> None:
    """Register all MCP tools on the given FastMCP instance."""
    _register_tools(mcp, backend)


__all__ = ["register_tools"]
```

This keeps `from codeintel.serving.mcp.registry import register_tools` working exactly as before.

---

## 4. Sanity pass & suggested order

To implement with minimal pain:

1. **Services split** (step 1):

   * Add `services/observability.py`, `http_transport.py`, `functions.py`, `profiles.py`, `subsystems.py`, `datasets.py`.
   * Adjust `services/query_service.py` imports and remove moved definitions.
   * Run: `pytest tests/services -q`.

2. **HTTP split** (step 2):

   * Add `http/dependencies.py` + `http/routes/*`.
   * Remove routers and DI from `http/fastapi.py` and import instead.
   * Run: `pytest tests/server -q` (or whatever suite hits your HTTP endpoints).

3. **MCP split** (step 3):

   * Add `mcp/tools_base.py` + `mcp/function_tools.py`, `profile_tools.py`, `architecture_tools.py`, `dataset_tools.py`.
   * Slim down `mcp/registry.py`.
   * Run: `pytest tests/mcp tests/services/test_query_service.py -q`.

If you’d like, we can next zoom in on **one specific slice** (for example `serving/services/datasets.py`) and I can give you a “mini diff” where I literally map:

* exact current implementation in `LocalQueryService.read_dataset_rows`
* how to paste it into `_LocalDatasetMixin.read_dataset_rows`
* and how to adjust `LocalQueryService` to inherit/compose it.

# `serving/services/datasets.py` zoom in #

You’re right, this is a good place to get very concrete. Let’s zoom in on **datasets** and treat `read_dataset_rows` as the worked example, with the other three dataset methods following the same pattern.

Below I’ll show:

1. **What you have now** in `LocalQueryService` (real code from your current repo).
2. The **new `_LocalDatasetMixin` in `serving/services/datasets.py`**, with `read_dataset_rows` transplanted.
3. The **small change to `LocalQueryService`** to compose the mixin.
4. A quick note on doing the same for `list_datasets` / `dataset_specs` / `dataset_schema`.

---

## 1. Current dataset methods on `LocalQueryService` (before)

Here’s what your **existing** dataset methods look like right now in `serving/services/query_service.py` (trimmed to the relevant bits):

```python
from codeintel.serving.backend import (
    BackendLimits,
    DuckDBQueryService,
    clamp_limit_value,
    clamp_offset_value,
)
from codeintel.serving.backend.datasets import describe_dataset
from codeintel.serving.mcp.models import (
    DatasetDescriptor,
    DatasetRowsResponse,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    ResponseMeta,
    ViewRow,
)
from codeintel.storage.datasets import Dataset, load_dataset_registry
...
class LocalQueryService(_FunctionQueryDelegates, _ProfileQueryDelegates, _SubsystemQueryDelegates):
    """Application service backed by a local DuckDB query layer."""

    query: DuckDBQueryService
    dataset_tables: dict[str, str] | None = None
    describe_dataset_fn: Callable[[str, str], str] = describe_dataset
    observability: ServiceObservability | None = None
    calls: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Derive dataset registry from the query gateway when not provided."""
        if self.dataset_tables is None:
            gateway = getattr(self.query, "gateway", None)
            self.dataset_tables = dict(gateway.datasets.mapping) if gateway is not None else {}

    def _call[T](
        self,
        name: str,
        func: Callable[[], T],
        *,
        dataset: str | None = None,
        schema_version: str | None = None,
        retries: int | None = None,
    ) -> T:
        ...
        self.calls.append(name)
        return _observe_call(
            self.observability,
            transport="local",
            name=name,
            context=ServiceCallContext(
                dataset=dataset,
                schema_version=schema_version,
                retries=retries,
            ),
            func=func,
        )

    def list_datasets(self) -> list[DatasetDescriptor]:
        """
        List datasets available through the dataset registry.
        """
        def _list() -> list[DatasetDescriptor]:
            mapping: dict[str, str] = self.dataset_tables or {}
            registry = None
            if not mapping:
                query_gateway = getattr(self.query, "gateway", None)
                if query_gateway is not None:
                    mapping = query_gateway.datasets.mapping
                    registry = load_dataset_registry(query_gateway.con)
            if registry is None:
                registry = load_dataset_registry(self.query.gateway.con)
            results: list[DatasetDescriptor] = []
            for name, table in sorted(mapping.items()):
                ds: Dataset | None = registry.by_name.get(name) if registry is not None else None
                description = (
                    ds.description
                    if ds is not None and ds.description is not None
                    else self.describe_dataset_fn(name, table)
                )
                results.append(
                    DatasetDescriptor(
                        name=name,
                        table=table,
                        family=ds.family if ds is not None else None,
                        description=description,
                        owner=ds.owner if ds is not None else None,
                        freshness_sla=ds.freshness_sla if ds is not None else None,
                        retention_policy=ds.retention_policy if ds is not None else None,
                        schema_version=ds.schema_version if ds is not None else None,
                        stable_id=ds.stable_id if ds is not None else None,
                        validation_profile=_normalize_validation_profile(
                            ds.validation_profile if ds is not None else None
                        ),
                        capabilities=ds.capabilities() if ds is not None else {},
                    )
                )
            return results

        return self._call("list_datasets", _list)

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        """
        Return canonical dataset specs with filenames and schema metadata.
        """
        def _list_specs() -> list[DatasetSpecDescriptor]:
            return self.query.dataset_specs()

        return self._call("dataset_specs", _list_specs)

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> DatasetSchemaResponse:
        """
        Return DuckDB + JSON Schema details and sample rows for a dataset.
        """
        def _schema() -> DatasetSchemaResponse:
            return self.query.dataset_schema(dataset_name=dataset_name, sample_limit=sample_limit)

        registry = load_dataset_registry(self.query.gateway.con)
        schema_version = None
        if dataset_name in registry.by_name:
            schema_version = registry.by_name[dataset_name].schema_version
        return self._call(
            "dataset_schema",
            _schema,
            dataset=dataset_name,
            schema_version=schema_version,
        )

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        """
        Read dataset rows with clamping and messaging.
        """
        applied_limit = self.query.limits.default_limit if limit is None else limit
        registry = load_dataset_registry(self.query.gateway.con)
        schema_version = None
        if dataset_name in registry.by_name:
            schema_version = registry.by_name[dataset_name].schema_version
        return self._call(
            "read_dataset_rows",
            lambda: self.query.read_dataset_rows(
                dataset_name=dataset_name,
                limit=applied_limit,
                offset=offset,
            ),
            dataset=dataset_name,
            schema_version=schema_version,
        )
```

And near the top of the file you also have:

```python
def _normalize_validation_profile(
    value: str | None,
) -> Literal["strict", "lenient"] | None:
    """
    Normalize validation profile strings to allowed literal values.
    """
    if value == "strict":
        return "strict"
    if value == "lenient":
        return "lenient"
    return None
```

We’re going to pull **these four methods + the helper** into a mixin module and let `LocalQueryService` just inherit it.

---

## 2. New `_LocalDatasetMixin` (after) in `serving/services/datasets.py`

Create a new file:

> `src/codeintel/serving/services/datasets.py`

with this content:

```python
"""Dataset delegates for local and HTTP query services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

from codeintel.serving.backend import (
    BackendLimits,
    DuckDBQueryService,
    clamp_limit_value,
    clamp_offset_value,
)
from codeintel.serving.backend.datasets import describe_dataset
from codeintel.serving.mcp.models import (
    DatasetDescriptor,
    DatasetRowsResponse,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    ResponseMeta,
    ViewRow,
)
from codeintel.storage.datasets import Dataset, load_dataset_registry
from codeintel.serving.services.http_transport import _HttpTransportMixin


def _normalize_validation_profile(
    value: str | None,
) -> Literal["strict", "lenient"] | None:
    """
    Normalize validation profile strings to allowed literal values.

    Returns
    -------
    Literal["strict", "lenient"] | None
        Normalized validation profile when valid.
    """
    if value == "strict":
        return "strict"
    if value == "lenient":
        return "lenient"
    return None


class _LocalDatasetMixin:
    """
    Dataset listing and retrieval helpers for LocalQueryService.

    Expects `self` to provide:
      - query: DuckDBQueryService
      - dataset_tables: dict[str, str] | None
      - describe_dataset_fn: Callable[[str, str], str]
      - limits: BackendLimits
      - _call(name, func, *, dataset, schema_version, retries)
    """

    # --- these four methods are copied directly from LocalQueryService ---

    def list_datasets(self) -> list[DatasetDescriptor]:
        """
        List datasets available through the dataset registry.

        Returns
        -------
        list[DatasetDescriptor]
            Dataset descriptors with names, tables, and descriptions.
        """

        def _list() -> list[DatasetDescriptor]:
            mapping: dict[str, str] = self.dataset_tables or {}
            registry = None
            if not mapping:
                query_gateway = getattr(self.query, "gateway", None)
                if query_gateway is not None:
                    mapping = query_gateway.datasets.mapping
                    registry = load_dataset_registry(query_gateway.con)
            if registry is None:
                registry = load_dataset_registry(self.query.gateway.con)
            results: list[DatasetDescriptor] = []
            for name, table in sorted(mapping.items()):
                ds: Dataset | None = registry.by_name.get(name) if registry is not None else None
                description = (
                    ds.description
                    if ds is not None and ds.description is not None
                    else self.describe_dataset_fn(name, table)
                )
                results.append(
                    DatasetDescriptor(
                        name=name,
                        table=table,
                        family=ds.family if ds is not None else None,
                        description=description,
                        owner=ds.owner if ds is not None else None,
                        freshness_sla=ds.freshness_sla if ds is not None else None,
                        retention_policy=ds.retention_policy if ds is not None else None,
                        schema_version=ds.schema_version if ds is not None else None,
                        stable_id=ds.stable_id if ds is not None else None,
                        validation_profile=_normalize_validation_profile(
                            ds.validation_profile if ds is not None else None
                        ),
                        capabilities=ds.capabilities() if ds is not None else {},
                    )
                )
            return results

        return self._call("list_datasets", _list)

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        """
        Return canonical dataset specs with filenames and schema metadata.

        Returns
        -------
        list[DatasetSpecDescriptor]
            Dataset specs sorted by name.
        """

        def _list_specs() -> list[DatasetSpecDescriptor]:
            return self.query.dataset_specs()

        return self._call("dataset_specs", _list_specs)

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> DatasetSchemaResponse:
        """
        Return DuckDB + JSON Schema details and sample rows for a dataset.

        Returns
        -------
        DatasetSchemaResponse
            Composite schema and sample payload.
        """

        def _schema() -> DatasetSchemaResponse:
            return self.query.dataset_schema(dataset_name=dataset_name, sample_limit=sample_limit)

        registry = load_dataset_registry(self.query.gateway.con)
        schema_version = None
        if dataset_name in registry.by_name:
            schema_version = registry.by_name[dataset_name].schema_version
        return self._call(
            "dataset_schema",
            _schema,
            dataset=dataset_name,
            schema_version=schema_version,
        )

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        """
        Read dataset rows with clamping and messaging.

        Returns
        -------
        DatasetRowsResponse
            Dataset slice and metadata for truncation/messaging.
        """
        # NOTE: limit/offset clamping happens inside DuckDBQueryService.read_dataset_rows;
        # here we just normalize limit and attach schema_version for observability.
        applied_limit = self.query.limits.default_limit if limit is None else limit
        registry = load_dataset_registry(self.query.gateway.con)
        schema_version = None
        if dataset_name in registry.by_name:
            schema_version = registry.by_name[dataset_name].schema_version
        return self._call(
            "read_dataset_rows",
            lambda: self.query.read_dataset_rows(
                dataset_name=dataset_name,
                limit=applied_limit,
                offset=offset,
            ),
            dataset=dataset_name,
            schema_version=schema_version,
        )


class _HttpDatasetQueryMixin(_HttpTransportMixin):
    """
    HTTP-based dataset query APIs used by HttpQueryService.

    This is a straight move of the existing _HttpDatasetQueryMixin from
    query_service.py into a dedicated module.
    """

    def list_datasets(self) -> list[DatasetDescriptor]:
        def _run() -> list[DatasetDescriptor]:
            data = cast("list[dict[str, object]]", self.request_json("/datasets", {}))
            return [DatasetDescriptor.model_validate(item) for item in data]

        return self._http_call("list_datasets", _run)

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        def _run() -> list[DatasetSpecDescriptor]:
            payload = cast(
                "list[dict[str, object]]",
                self.request_json("/datasets/specs", {}),
            )
            return [DatasetSpecDescriptor.model_validate(entry) for entry in payload]

        return self._http_call("dataset_specs", _run)

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        def _run() -> DatasetRowsResponse:
            clamp = clamp_limit_value(
                limit,
                default=self.limits.default_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            offset_clamp = clamp_offset_value(offset)
            messages = [*clamp.messages, *offset_clamp.messages]
            if clamp.has_error or offset_clamp.has_error:
                meta = ResponseMeta(
                    requested_limit=limit,
                    applied_limit=clamp.applied,
                    requested_offset=offset,
                    applied_offset=offset_clamp.applied,
                    messages=messages,
                    truncated=False,
                )
                return DatasetRowsResponse(
                    dataset_name=dataset_name,
                    limit=clamp.applied,
                    offset=offset_clamp.applied,
                    rows=[],
                    meta=meta,
                )
            data = self.request_json(
                f"/datasets/{dataset_name}",
                {"limit": clamp.applied, "offset": offset_clamp.applied},
            )
            response = DatasetRowsResponse.model_validate(data)
            existing_meta = response.meta if response.meta is not None else ResponseMeta()
            merged_meta = ResponseMeta(
                requested_limit=limit,
                applied_limit=clamp.applied,
                requested_offset=offset,
                applied_offset=offset_clamp.applied,
                truncated=existing_meta.truncated,
                messages=[*messages, *(existing_meta.messages or [])],
            )
            return response.model_copy(update={"meta": merged_meta})

        return self._http_call("read_dataset_rows", _run, dataset=dataset_name)

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> DatasetSchemaResponse:
        def _run() -> DatasetSchemaResponse:
            data = self.request_json(
                f"/datasets/{dataset_name}/schema",
                {"limit": sample_limit},
            )
            return DatasetSchemaResponse.model_validate(data)

        return self._http_call("dataset_schema", _run, dataset=dataset_name)
```

The **key point for `read_dataset_rows`**:
The body is **identical** to what you had on `LocalQueryService`; it just lives on `_LocalDatasetMixin` and still calls `self._call(...)`, which `LocalQueryService` provides.

---

## 3. Adjust `LocalQueryService` to compose the mixin

Now we make `LocalQueryService` inherit from `_LocalDatasetMixin` and drop the inline dataset methods.

### 3.1 Class header – BEFORE

```python
class LocalQueryService(_FunctionQueryDelegates, _ProfileQueryDelegates, _SubsystemQueryDelegates):
    """Application service backed by a local DuckDB query layer."""
```

### 3.2 Class header – AFTER

At the top of `serving/services/query_service.py`, add:

```python
from codeintel.serving.services.datasets import _LocalDatasetMixin, _HttpDatasetQueryMixin
```

Then change the class header:

```python
class LocalQueryService(
    _FunctionQueryDelegates,
    _ProfileQueryDelegates,
    _SubsystemQueryDelegates,
    _LocalDatasetMixin,
):
    """Application service backed by a local DuckDB query layer."""
```

Everything else in the class can stay as-is:

* The fields:

  ```python
  query: DuckDBQueryService
  dataset_tables: dict[str, str] | None = None
  describe_dataset_fn: Callable[[str, str], str] = describe_dataset
  observability: ServiceObservability | None = None
  calls: list[str] = field(default_factory=list)
  ```

* `__post_init__` (still sets `dataset_tables`).

* `_call`.

### 3.3 Remove dataset methods and helper from `LocalQueryService` file

In `serving/services/query_service.py`:

* **Delete** the definitions of:

  * `list_datasets`
  * `dataset_specs`
  * `dataset_schema`
  * `read_dataset_rows`

  from inside `LocalQueryService`.

* **Delete** the `_HttpDatasetQueryMixin` definition (we moved it into `services/datasets.py`).

* **Delete** `_normalize_validation_profile` helper from this file (it now lives in `services/datasets.py` and is only used there).

* **Make `HttpQueryService` use the imported mixin** (which already happens if you imported `_HttpDatasetQueryMixin` from `services.datasets`):

  Before:

  ```python
  class HttpQueryService(
      _HttpDatasetQueryMixin,
      _HttpFunctionQueryMixin,
      _HttpProfileQueryMixin,
      _HttpSubsystemQueryMixin,
  ):
      ...
  ```

  After (same header, just different import source):

  ```python
  from codeintel.serving.services.datasets import _HttpDatasetQueryMixin

  class HttpQueryService(
      _HttpDatasetQueryMixin,
      _HttpFunctionQueryMixin,
      _HttpProfileQueryMixin,
      _HttpSubsystemQueryMixin,
  ):
      ...
  ```

That’s it: `LocalQueryService` now gains dataset behavior purely via `_LocalDatasetMixin`, and `HttpQueryService` uses the HTTP dataset mixin from the same module.

---

## 4. How to apply this pattern to other slices

What we just did for datasets is the pattern you can replicate for **functions / profiles / subsystems**:

* For each slice:

  * Identify the `*_Delegates` and `_Http*Mixin` inside `query_service.py`.
  * Cut those classes into `serving/services/<slice>.py` (e.g. `functions.py`, `profiles.py`, `subsystems.py`).
  * Make `LocalQueryService` and `HttpQueryService` inherit from those mixins.
* Keep `LocalQueryService._call` and the observability machinery in `query_service.py` (or move observability into `services/observability.py` if you want to fully complete the earlier design).

If you’d like, next I can do the **same mini-diff treatment for `_FunctionQueryDelegates` → `services/functions.py`**, focusing on one representative method like `get_function_summary` to show exactly how the cut/paste and class header change would look.
