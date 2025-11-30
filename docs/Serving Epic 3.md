You’re right to be annoyed about the earlier confusion — I went back, **cleared all previously extracted dirs under `/mnt/data`**, and re-unzipped the latest zips you sent. I now see:

* `serving/serving/backend/{__init__,duckdb_service,limits,datasets}.py`
* `serving/serving/http/routes/{functions,profiles,architecture,subsystems,ide,datasets,health}.py`
* `serving/serving/mcp/{backend.py,models.py,tools_base.py,function_tools.py,profile_tools.py,architecture_tools.py,dataset_tools.py,tool_utils.py,registry.py}`
* `serving/serving/services/{query_service.py,functions.py,profiles.py,subsystems.py,datasets.py,...}`

Everything below is based on **this** current layout.

---

## Goal of Refactor 3

You already have:

* **Dataset registry** logic in `serving/backend/datasets.py` and dataset APIs in `QueryService` / HTTP routes.
* **Operations** spread across:

  * HTTP routes in `serving/http/routes/*.py`.
  * MCP tools in `serving/mcp/*_tools.py`.
  * Service layer in `serving/services/query_service.py` & feature mixins.

Refactor 3 introduces a **single registry** that answers:

> “What operations and datasets exist? How do I call them (HTTP/MCP)? What do they touch? What are the safe limits?”

and then exposes that via:

* **HTTP**: `/meta/datasets` and `/meta/operations`
* **MCP**: new meta tools (e.g. `list_datasets_meta`, `list_operations_meta`)

Later, that same registry can be used to **generate routers and MCP tools**, but we don’t *need* to do the full generation in this epic; we can design the registry so that step is easy.

I’ll split the plan into 5 steps:

1. Create `serving/registry.py` with `DatasetMeta` and `OperationSpec`.
2. Add Pydantic meta response models in `serving/mcp/models.py`.
3. Add HTTP `/meta` endpoints using the registry.
4. Add MCP meta tools using the registry.
5. (Optional) Outline how to drive HTTP + MCP from `OperationSpec` for one slice (functions) as a pattern.

---

## 1. New core registry module: `serving/registry.py`

### 1.1. Create the module and core dataclasses

**New file**: `src/codeintel/serving/registry.py`

This houses:

* `DatasetMeta`: environment-specific dataset info (+ limits, docs/read-only flags).
* `OperationSpec`: static, cross-transport description of each operation.

```python
# src/codeintel/serving/registry.py
"""Unified operation and dataset registry for serving.

This module expresses:
- DatasetMeta: enriched dataset metadata (from dataset_specs + limits).
- OperationSpec: cross-transport description of an operation (HTTP + MCP + backend).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.models import DatasetSpecDescriptor
from codeintel.serving.services.query_service import QueryService


@dataclass(frozen=True)
class DatasetMeta:
    """Dataset metadata enriched with serving-specific limits and flags."""

    id: str                 # public id / alias (for now == name)
    name: str               # dataset name, e.g. "docs.v_function_summary"
    table_key: str          # fully qualified table/view name, e.g. "docs.v_function_summary"
    description: str
    schema_version: str | None
    family: str | None
    is_docs_view: bool
    is_read_only: bool
    default_limit: int
    max_limit: int


@dataclass(frozen=True)
class OperationSpec:
    """Cross-transport description of a single serving operation."""

    id: str                        # "function.summary"
    category: str                  # "functions" / "datasets" / "architecture" / ...
    summary: str                   # short human description
    description: str | None        # detailed description for tools/docs

    # HTTP surface (optional for non-HTTP-only ops)
    http_method: Literal["GET", "POST"] | None
    http_path: str | None

    # MCP surface (optional for non-MCP-only ops)
    tool_name: str | None          # MCP tool name, e.g. "get_function_summary"

    # Types / backend binding
    output_model_name: str         # Pydantic response model class name (string)
    backend_method: str            # QueryService method name, e.g. "get_function_summary"

    # Dependencies and safety
    required_datasets: Sequence[str]
    required_graphs: Sequence[str]
    default_limit: int | None      # None = use BackendLimits.default_limit
    max_limit: int | None          # None = use BackendLimits.max_rows_per_call
```

> ❗️Notice: we’re using **string names** (`output_model_name`) instead of actual types to avoid circular imports. The meta endpoints will expose these names as strings so agents know what they’d get back.

---

### 1.2. Build `DatasetMeta` from `QueryService.dataset_specs()`

We want a *runtime* function that, given a `QueryService` instance and `BackendLimits`, produces enriched dataset metadata.

Add this to `serving/registry.py`:

```python
def build_dataset_meta(
    service: QueryService,
    limits: BackendLimits,
) -> list[DatasetMeta]:
    """
    Build dataset metadata entries using dataset_specs and serving limits.

    Parameters
    ----------
    service:
        QueryService instance (LocalQueryService or HttpQueryService).
    limits:
        BackendLimits derived from the current ServingConfig.

    Returns
    -------
    list[DatasetMeta]
        One entry per dataset in the registry.
    """
    specs: list[DatasetSpecDescriptor] = service.dataset_specs()
    metas: list[DatasetMeta] = []

    for spec in specs:
        family = spec.family
        is_docs_view = bool(family == "docs" or spec.table_key.startswith("docs."))
        is_read_only = bool(spec.capabilities.get("read_only", False))
        description = spec.description or f"{spec.name} ({spec.table_key})"

        metas.append(
            DatasetMeta(
                id=spec.name,
                name=spec.name,
                table_key=spec.table_key,
                description=description,
                schema_version=spec.schema_version,
                family=family,
                is_docs_view=is_docs_view,
                is_read_only=is_read_only,
                default_limit=limits.default_limit,
                max_limit=limits.max_rows_per_call,
            )
        )

    return metas
```

This intentionally stays **transport-agnostic**: it only knows about `QueryService` and `BackendLimits`.

(You still keep the lower-level registry + validation logic in `serving/backend/datasets.py`; this is just the *serving-layer* view.)

---

### 1.3. Define `OperationSpec` entries (at least for core operations)

Now we describe the operations in one place.

For this epic, I’d:

* **Fully specify** a few key operations (function summary, high-risk functions, dataset rows) to make the pattern clear.
* Add stubs/placeholders for the rest that you (or an agent) can fill out systematically.

Still inside `serving/registry.py`, add:

```python
# Static registry of operations, keyed by operation id.
# Extend this with additional operations following the same pattern.
_OPERATION_SPECS: dict[str, OperationSpec] = {
    # Functions
    "function.summary": OperationSpec(
        id="function.summary",
        category="functions",
        summary="Summarize a function by GOID, URN, or source location.",
        description=(
            "Look up docs.v_function_summary and related analytics to produce a "
            "single function summary identified by GOID, URN, qualname or file path."
        ),
        http_method="GET",
        http_path="/function/summary",
        tool_name="get_function_summary",            # MCP tool in function_tools.py
        output_model_name="FunctionSummaryResponse", # codeintel.serving.mcp.models
        backend_method="get_function_summary",       # QueryService method
        required_datasets=["docs.v_function_summary"],
        required_graphs=["callgraph"],
        default_limit=1,
        max_limit=1,
    ),
    "functions.high_risk": OperationSpec(
        id="functions.high_risk",
        category="functions",
        summary="List high-risk functions, optionally restricted to tested ones.",
        description=(
            "Backed by analytics.goid_risk_factors and docs.v_function_summary. "
            "Supports min_risk threshold and tested_only filter."
        ),
        http_method="GET",
        http_path="/functions/high-risk",
        tool_name="list_high_risk_functions",
        output_model_name="HighRiskFunctionsResponse",
        backend_method="list_high_risk_functions",
        required_datasets=[
            "analytics.goid_risk_factors",
            "docs.v_function_summary",
        ],
        required_graphs=[],
        default_limit=None,   # use BackendLimits.default_limit
        max_limit=None,       # use BackendLimits.max_rows_per_call
    ),

    # Datasets
    "datasets.list": OperationSpec(
        id="datasets.list",
        category="datasets",
        summary="List datasets available through the backend.",
        description="Uses the dataset registry to expose a filtered list of datasets.",
        http_method="GET",
        http_path="/datasets",
        tool_name=None,  # currently only HTTP
        output_model_name="DatasetDescriptor",
        backend_method="list_datasets",
        required_datasets=[],
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),
    "datasets.specs": OperationSpec(
        id="datasets.specs",
        category="datasets",
        summary="Expose dataset contract and registry metadata.",
        description="Returns DatasetSpecDescriptor entries describing dataset contracts.",
        http_method="GET",
        http_path="/datasets/specs",
        tool_name=None,
        output_model_name="DatasetSpecDescriptor",
        backend_method="dataset_specs",
        required_datasets=[],
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),
    "datasets.rows": OperationSpec(
        id="datasets.rows",
        category="datasets",
        summary="Read rows from a dataset with limits and messaging.",
        description=(
            "Read typed rows from any dataset, applying BackendLimits and dataset-"
            "specific clamping. This is the main entrypoint for ad-hoc dataset reads."
        ),
        http_method="GET",
        http_path="/datasets/{dataset_name}",
        tool_name="read_dataset_rows",  # MCP tool in dataset_tools.py
        output_model_name="DatasetRowsResponse",
        backend_method="read_dataset_rows",
        required_datasets=[],  # dynamic; depends on `dataset_name`
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),
    "datasets.schema": OperationSpec(
        id="datasets.schema",
        category="datasets",
        summary="Describe dataset schema and sample rows.",
        description=(
            "Combine DuckDB catalog, JSON Schema, and sample rows to describe a dataset."
        ),
        http_method="GET",
        http_path="/datasets/{dataset_name}/schema",
        tool_name="dataset_schema",  # if you expose as MCP later
        output_model_name="DatasetSchemaResponse",
        backend_method="dataset_schema",
        required_datasets=[],  # dynamic
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    # Add similar entries for:
    # - architecture.function / architecture.module
    # - profiles.function / profiles.file / profiles.module
    # - subsystems.* operations
    # - ide.hints
    # - health
}
```

Then add simple accessors:

```python
def iter_operation_specs() -> list[OperationSpec]:
    """Return all registered OperationSpec instances."""
    return list(_OPERATION_SPECS.values())


def get_operation_spec(op_id: str) -> OperationSpec | None:
    """Return a single OperationSpec by id, or None if unknown."""
    return _OPERATION_SPECS.get(op_id)
```

> **Follow-on work**: fill in the remaining operations (profiles, architecture, subsystems, ide, health) using the same pattern:
>
> * `http_path` from `serving/http/routes/*.py` router decorators.
> * `tool_name` from MCP tool function names.
> * `backend_method` from the corresponding `QueryService` method.
> * `required_datasets` / `required_graphs` using your analytics + storage knowledge.

---

## 2. Add meta Pydantic models in `serving/mcp/models.py`

We want structured responses for:

* `/meta/datasets` → list of dataset metadata
* `/meta/operations` → list of operation metadata

Add near the bottom of `src/codeintel/serving/mcp/models.py` (close to other response models):

```python
# ... existing imports and models ...

class DatasetMetaResponse(BaseModel):
    """Serving-layer dataset metadata exposed over HTTP and MCP."""

    id: str
    name: str
    table_key: str
    description: str
    schema_version: str | None = None
    family: str | None = None
    is_docs_view: bool = False
    is_read_only: bool = False
    default_limit: int
    max_limit: int


class OperationMetaResponse(BaseModel):
    """Introspectable metadata for a single operation."""

    id: str
    category: str
    summary: str
    description: str | None = None
    http_method: str | None = None
    http_path: str | None = None
    tool_name: str | None = None
    output_model: str  # Pydantic response model class name
    required_datasets: list[str] = []
    required_graphs: list[str] = []
    default_limit: int | None = None
    max_limit: int | None = None
```

And add them to the module’s `__all__` (if you maintain one), so they can be imported from other modules.

---

## 3. HTTP `/meta` endpoints using the registry

### 3.1. New router: `serving/http/routes/meta.py`

**New file**: `src/codeintel/serving/http/routes/meta.py`

```python
"""Meta HTTP routes for dataset and operation introspection."""

from __future__ import annotations

from fastapi import APIRouter

from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.dependencies import ConfigDep, ServiceDep
from codeintel.serving.mcp.models import DatasetMetaResponse, OperationMetaResponse
from codeintel.serving.registry import build_dataset_meta, iter_operation_specs

LOG_ROUTE_PREFIX = "/meta"


def build_meta_router() -> APIRouter:
    """Construct the router exposing meta introspection endpoints."""
    router = APIRouter()

    @router.get(
        f"{LOG_ROUTE_PREFIX}/datasets",
        response_model=list[DatasetMetaResponse],
        summary="List dataset metadata and serving limits.",
    )
    def list_dataset_meta(
        service: ServiceDep,
        cfg: ConfigDep,
    ) -> list[DatasetMetaResponse]:
        limits = BackendLimits.from_config(cfg)
        metas = build_dataset_meta(service, limits)
        return [
            DatasetMetaResponse(
                id=m.id,
                name=m.name,
                table_key=m.table_key,
                description=m.description,
                schema_version=m.schema_version,
                family=m.family,
                is_docs_view=m.is_docs_view,
                is_read_only=m.is_read_only,
                default_limit=m.default_limit,
                max_limit=m.max_limit,
            )
            for m in metas
        ]

    @router.get(
        f"{LOG_ROUTE_PREFIX}/operations",
        response_model=list[OperationMetaResponse],
        summary="List available operations and their characteristics.",
    )
    def list_operation_meta(
        cfg: ConfigDep,
    ) -> list[OperationMetaResponse]:
        limits = BackendLimits.from_config(cfg)
        results: list[OperationMetaResponse] = []
        for spec in iter_operation_specs():
            default_limit = spec.default_limit or limits.default_limit
            max_limit = spec.max_limit or limits.max_rows_per_call
            results.append(
                OperationMetaResponse(
                    id=spec.id,
                    category=spec.category,
                    summary=spec.summary,
                    description=spec.description,
                    http_method=spec.http_method,
                    http_path=spec.http_path,
                    tool_name=spec.tool_name,
                    output_model=spec.output_model_name,
                    required_datasets=list(spec.required_datasets),
                    required_graphs=list(spec.required_graphs),
                    default_limit=default_limit,
                    max_limit=max_limit,
                )
            )
        return results

    return router


__all__ = ["build_meta_router"]
```

### 3.2. Wire the meta router into FastAPI

In `src/codeintel/serving/http/fastapi.py`, you already include routers like:

```python
app.include_router(build_functions_router())
app.include_router(build_profiles_router())
# ...
app.include_router(build_datasets_router())
app.include_router(build_health_router())
```

Add:

```python
from codeintel.serving.http.routes.meta import build_meta_router
```

and include it:

```python
app.include_router(build_meta_router())
```

Now the HTTP surface exposes:

* `GET /meta/datasets`
* `GET /meta/operations`

with typed responses.

---

## 4. MCP meta tools using the registry

### 4.1. New MCP tool module: `serving/mcp/meta_tools.py`

**New file**: `src/codeintel/serving/mcp/meta_tools.py`

```python
"""MCP tools for dataset and operation introspection."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.models import DatasetMetaResponse, OperationMetaResponse, ProblemDetail
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.registry import build_dataset_meta, iter_operation_specs


def register_meta_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """Register meta MCP tools on the given FastMCP instance."""

    @mcp.tool()
    @_wrap
    def list_datasets_meta() -> list[dict[str, object]] | dict[str, ProblemDetail]:
        """
        List dataset metadata and serving limits via MCP.

        Returns
        -------
        list[DatasetMetaResponse]
            One entry per dataset, as dict payloads.
        """
        # Try to get limits from backend if it exposes them; otherwise use defaults
        limits = getattr(backend, "limits", BackendLimits())
        metas = build_dataset_meta(backend.service, limits)
        return [
            DatasetMetaResponse(
                id=m.id,
                name=m.name,
                table_key=m.table_key,
                description=m.description,
                schema_version=m.schema_version,
                family=m.family,
                is_docs_view=m.is_docs_view,
                is_read_only=m.is_read_only,
                default_limit=m.default_limit,
                max_limit=m.max_limit,
            ).model_dump()
            for m in metas
        ]

    @mcp.tool()
    @_wrap
    def list_operations_meta() -> list[dict[str, object]] | dict[str, ProblemDetail]:
        """
        List available operations and their characteristics via MCP.

        Returns
        -------
        list[OperationMetaResponse]
            Operation metadata as dict payloads.
        """
        limits = getattr(backend, "limits", BackendLimits())
        specs = iter_operation_specs()
        payloads: list[OperationMetaResponse] = []
        for spec in specs:
            default_limit = spec.default_limit or limits.default_limit
            max_limit = spec.max_limit or limits.max_rows_per_call
            payloads.append(
                OperationMetaResponse(
                    id=spec.id,
                    category=spec.category,
                    summary=spec.summary,
                    description=spec.description,
                    http_method=spec.http_method,
                    http_path=spec.http_path,
                    tool_name=spec.tool_name,
                    output_model=spec.output_model_name,
                    required_datasets=list(spec.required_datasets),
                    required_graphs=list(spec.required_graphs),
                    default_limit=default_limit,
                    max_limit=max_limit,
                )
            )
        return [p.model_dump() for p in payloads]
```

### 4.2. Hook meta tools into `tools_base.register_tools`

In `src/codeintel/serving/mcp/tools_base.py`, you currently have:

```python
from codeintel.serving.mcp.architecture_tools import register_architecture_tools
from codeintel.serving.mcp.dataset_tools import register_dataset_tools
from codeintel.serving.mcp.function_tools import register_function_tools
from codeintel.serving.mcp.profile_tools import register_profile_tools
from codeintel.serving.mcp.tool_utils import QueryBackendOrService
```

Update imports:

```python
from codeintel.serving.mcp.architecture_tools import register_architecture_tools
from codeintel.serving.mcp.dataset_tools import register_dataset_tools
from codeintel.serving.mcp.function_tools import register_function_tools
from codeintel.serving.mcp.meta_tools import register_meta_tools
from codeintel.serving.mcp.profile_tools import register_profile_tools
from codeintel.serving.mcp.tool_utils import QueryBackendOrService
```

And in `register_tools`:

```python
def register_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """
    Register all MCP tools on the given FastMCP instance.
    """
    register_function_tools(mcp, backend)
    register_profile_tools(mcp, backend)
    register_architecture_tools(mcp, backend)
    register_dataset_tools(mcp, backend)
    register_meta_tools(mcp, backend)  # new
```

No changes needed to `serving/mcp/registry.py` or `server.py` — they already call `tools_base.register_tools`.

---

## 5. (Optional / Future) Using OperationSpec to drive HTTP + MCP

For this epic, the **main value** is meta introspection: agents can discover operations and datasets and understand limits and dependencies.

If you want to go further (either now or as Refactor 3b), `OperationSpec` is intentionally designed so that you can:

* **HTTP**: generate or validate route registrations from specs:

  * In `http/routes/functions.py`, instead of hardcoding `/function/summary`, you could lookup `get_operation_spec("function.summary")` and use its `http_path` and `summary` fields, keeping handler logic in one place.

* **MCP**: generate tools from specs:

  * In `function_tools.py`, instead of hardcoding tool names and backend calls, iterate over a filtered view of `_OPERATION_SPECS` where `category == "functions"` and create tools dynamically with a generic wrapper that calls `getattr(backend, spec.backend_method)(**params)`.

Doing that cleanly probably deserves its own small epic, but the registry you’ve just defined makes it straightforward.

---

## Recap

With this refactor in place you get:

* A **single, structured source of truth** for:

  * What datasets exist (`DatasetMeta`)
  * What operations exist (`OperationSpec`)
  * How they map to HTTP paths and MCP tools
  * What datasets/graphs they touch and what limits apply

* **HTTP** introspection:

  * `GET /meta/datasets` → full dataset view
  * `GET /meta/operations` → full operation catalog

* **MCP** introspection:

  * `list_datasets_meta` and `list_operations_meta` tools that return the same information as MCP payloads.

If you’d like, next I can:

* Enumerate *all* the remaining `OperationSpec` instances (profiles, architecture, subsystems, ide, health) one by one, so you have a complete registry you can just paste in.


# all remaining operationspec instances zoom in #

You’re right to push on this — let’s make the registry actually *complete* so you can just paste it in and tweak details later.

Below is a **drop-in `_OPERATION_SPECS` block** for `serving/registry.py` that includes:

* The **function + dataset** entries we sketched before.
* All the **remaining operations** you asked for:

  * **Profiles** (function/file/module)
  * **Architecture** (function/module)
  * **Subsystems** (list, profiles, coverage, memberships, detail, search)
  * **IDE** (file hints)
  * **Health**
  * **Graph plugin planning**

I’ll show it as a single dict so you can copy the whole thing into `serving/registry.py` (replacing any stub you have there).

You can refine `required_datasets` / `required_graphs` later if you want them to be more precise. For the ones we can infer from your service mixins, I’ve filled them in; for the rest I’ve left them empty (with comments where appropriate).

---

## Full `_OPERATION_SPECS` block with all OperationSpec entries

Assuming you already have in `serving/registry.py`:

```python
from dataclasses import dataclass
from typing import Literal, Sequence

from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.models import DatasetSpecDescriptor
from codeintel.serving.services.query_service import QueryService


@dataclass(frozen=True)
class DatasetMeta:
    # ... as previously defined ...
    ...


@dataclass(frozen=True)
class OperationSpec:
    id: str
    category: str
    summary: str
    description: str | None

    http_method: Literal["GET", "POST"] | None
    http_path: str | None

    tool_name: str | None

    output_model_name: str
    backend_method: str

    required_datasets: Sequence[str]
    required_graphs: Sequence[str]
    default_limit: int | None
    max_limit: int | None
```

You can now define the **complete** `_OPERATION_SPECS` like this:

```python
# Static registry of operations, keyed by operation id.
_OPERATION_SPECS: dict[str, OperationSpec] = {
    # ------------------------------------------------------------------
    # FUNCTION-CENTRIC OPERATIONS
    # ------------------------------------------------------------------

    "function.summary": OperationSpec(
        id="function.summary",
        category="functions",
        summary="Summarize a function by GOID, URN, or source location.",
        description=(
            "Summarize a function using Docs + analytics views, identified by GOID, URN, "
            "qualified name, or file + qualname."
        ),
        http_method="GET",
        http_path="/function/summary",
        tool_name="get_function_summary",
        output_model_name="FunctionSummaryResponse",
        backend_method="get_function_summary",
        required_datasets=[],          # TODO: fill with concrete docs.* view names if desired
        required_graphs=["callgraph"], # depends on call graph analytics
        default_limit=1,
        max_limit=1,
    ),

    "functions.high_risk": OperationSpec(
        id="functions.high_risk",
        category="functions",
        summary="List high-risk functions, optionally restricted to tested ones.",
        description=(
            "Rank functions by risk using analytics.goid_risk_factors and related views, "
            "with filters for minimum risk score and tested-only functions."
        ),
        http_method="GET",
        http_path="/functions/high-risk",
        tool_name="list_high_risk_functions",
        output_model_name="HighRiskFunctionsResponse",
        backend_method="list_high_risk_functions",
        required_datasets=[],          # TODO: e.g. analytics.goid_risk_factors, docs.v_function_summary
        required_graphs=[],
        default_limit=None,            # use BackendLimits
        max_limit=None,
    ),

    "functions.tests": OperationSpec(
        id="functions.tests",
        category="functions",
        summary="List tests that exercise a specific function.",
        description=(
            "Return tests linked to a function via coverage and mapping tables. "
            "Useful for understanding test blast radius."
        ),
        http_method="GET",
        http_path="/function/tests",
        tool_name="get_tests_for_function",
        output_model_name="TestsForFunctionResponse",
        backend_method="get_tests_for_function",
        required_datasets=[],          # TODO: fill with coverage/mapping datasets
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    "graph.call_neighbors": OperationSpec(
        id="graph.call_neighbors",
        category="graph",
        summary="Get call graph neighbors for a function.",
        description=(
            "Return incoming and outgoing neighbors in the call graph for a single function, "
            "with optional direction and limit control."
        ),
        http_method="GET",
        http_path="/function/callgraph",
        tool_name="get_callgraph_neighbors",
        output_model_name="CallGraphNeighborsResponse",
        backend_method="get_callgraph_neighbors",
        required_datasets=["call_graph_nodes"],
        required_graphs=["callgraph"],
        default_limit=None,
        max_limit=None,
    ),

    "graph.call_neighborhood": OperationSpec(
        id="graph.call_neighborhood",
        category="graph",
        summary="Compute a bounded ego neighborhood in the call graph.",
        description=(
            "Return nodes and edges in a radius-bounded ego neighborhood around a function "
            "in the call graph, optionally limited by node count."
        ),
        http_method="GET",
        http_path="/graph/call/neighborhood",
        tool_name="get_callgraph_neighborhood",
        output_model_name="GraphNeighborhoodResponse",
        backend_method="get_callgraph_neighborhood",
        required_datasets=["call_graph_nodes"],
        required_graphs=["callgraph"],
        default_limit=None,
        max_limit=None,
    ),

    "graph.import_boundary": OperationSpec(
        id="graph.import_boundary",
        category="graph",
        summary="List import graph edges crossing a subsystem boundary.",
        description=(
            "Return edges in the import graph that cross the boundary of a given subsystem, "
            "useful for dependency analysis and architecture hygiene."
        ),
        http_method="GET",
        http_path="/graph/import/boundary",
        tool_name="get_import_boundary",
        output_model_name="ImportBoundaryResponse",
        backend_method="get_import_boundary",
        required_datasets=["import_graph_edges"],
        required_graphs=["importgraph"],
        default_limit=None,
        max_limit=None,
    ),

    "file.summary": OperationSpec(
        id="file.summary",
        category="files",
        summary="Get a file summary with function details.",
        description=(
            "Return file-level metrics plus nested function summaries for all functions defined "
            "in a given file."
        ),
        http_method="GET",
        http_path="/file/summary",
        tool_name="get_file_summary",
        output_model_name="FileSummaryResponse",
        backend_method="get_file_summary",
        required_datasets=[],  # TODO: fill if you want to expose underlying docs views
        required_graphs=[],
        default_limit=1,
        max_limit=1,
    ),

    # ------------------------------------------------------------------
    # PROFILE OPERATIONS
    # ------------------------------------------------------------------

    "profiles.function": OperationSpec(
        id="profiles.function",
        category="profiles",
        summary="Get a function profile.",
        description="Return a rich profile for a single function identified by GOID.",
        http_method="GET",
        http_path="/profiles/function",
        tool_name="get_function_profile",
        output_model_name="FunctionProfileResponse",
        backend_method="get_function_profile",
        required_datasets=[],
        required_graphs=["callgraph"],  # may incorporate call graph metrics
        default_limit=1,
        max_limit=1,
    ),

    "profiles.file": OperationSpec(
        id="profiles.file",
        category="profiles",
        summary="Get a file profile.",
        description="Return a profile rollup for a file and its functions.",
        http_method="GET",
        http_path="/profiles/file",
        tool_name="get_file_profile",
        output_model_name="FileProfileResponse",
        backend_method="get_file_profile",
        required_datasets=[],
        required_graphs=[],
        default_limit=1,
        max_limit=1,
    ),

    "profiles.module": OperationSpec(
        id="profiles.module",
        category="profiles",
        summary="Get a module profile.",
        description="Return metrics and rollups for a Python module.",
        http_method="GET",
        http_path="/profiles/module",
        tool_name="get_module_profile",
        output_model_name="ModuleProfileResponse",
        backend_method="get_module_profile",
        required_datasets=[],
        required_graphs=[],
        default_limit=1,
        max_limit=1,
    ),

    # ------------------------------------------------------------------
    # ARCHITECTURE OPERATIONS
    # ------------------------------------------------------------------

    "architecture.function": OperationSpec(
        id="architecture.function",
        category="architecture",
        summary="Get architecture metrics for a function.",
        description=(
            "Return architecture metrics for a function (e.g., fan-in/fan-out, "
            "layer violations, subsystem crossings)."
        ),
        http_method="GET",
        http_path="/architecture/function",
        tool_name="get_function_architecture",
        output_model_name="FunctionArchitectureResponse",
        backend_method="get_function_architecture",
        required_datasets=[],          # TODO: fill with docs/analytics views
        required_graphs=["callgraph", "importgraph"],
        default_limit=1,
        max_limit=1,
    ),

    "architecture.module": OperationSpec(
        id="architecture.module",
        category="architecture",
        summary="Get architecture metrics for a module.",
        description=(
            "Return module-level architecture metrics: dependencies, subsystems, "
            "and cross-layer violations."
        ),
        http_method="GET",
        http_path="/architecture/module",
        tool_name="get_module_architecture",
        output_model_name="ModuleArchitectureResponse",
        backend_method="get_module_architecture",
        required_datasets=[],
        required_graphs=["callgraph", "importgraph"],
        default_limit=1,
        max_limit=1,
    ),

    # ------------------------------------------------------------------
    # SUBSYSTEM OPERATIONS
    # ------------------------------------------------------------------

    "subsystems.list": OperationSpec(
        id="subsystems.list",
        category="subsystems",
        summary="List inferred subsystems.",
        description=(
            "List inferred subsystems with optional filtering by role or search term."
        ),
        http_method="GET",
        http_path="/architecture/subsystems",
        tool_name="list_subsystems",
        output_model_name="SubsystemSummaryResponse",
        backend_method="list_subsystems",
        required_datasets=[],   # TODO: e.g. docs.v_subsystem_summary
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    "subsystems.profiles": OperationSpec(
        id="subsystems.profiles",
        category="subsystems",
        summary="List subsystem profiles.",
        description="List profiles summarizing each subsystem (metrics, size, risk).",
        http_method="GET",
        http_path="/architecture/subsystem-profiles",
        tool_name=None,  # no dedicated MCP tool yet
        output_model_name="SubsystemProfileResponse",
        backend_method="list_subsystem_profiles",
        required_datasets=["docs.v_subsystem_profile"],
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    "subsystems.coverage": OperationSpec(
        id="subsystems.coverage",
        category="subsystems",
        summary="List subsystem coverage rollups.",
        description=(
            "List coverage metrics for subsystems (e.g., number of modules, functions, tests)."
        ),
        http_method="GET",
        http_path="/architecture/subsystem-coverage",
        tool_name=None,  # no dedicated MCP tool yet
        output_model_name="SubsystemCoverageResponse",
        backend_method="list_subsystem_coverage",
        required_datasets=["docs.v_subsystem_coverage"],
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    "subsystems.module_memberships": OperationSpec(
        id="subsystems.module_memberships",
        category="subsystems",
        summary="List subsystem memberships for a module.",
        description="Return which subsystems a given module belongs to.",
        http_method="GET",
        http_path="/architecture/module-subsystems",
        tool_name="get_module_subsystems",
        output_model_name="ModuleSubsystemResponse",
        backend_method="get_module_subsystems",
        required_datasets=[],
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    "subsystems.detail": OperationSpec(
        id="subsystems.detail",
        category="subsystems",
        summary="Get modules and detail for a subsystem.",
        description="Return detailed membership and metrics for a single subsystem.",
        http_method="GET",
        http_path="/architecture/subsystem",
        tool_name="get_subsystem_modules",
        output_model_name="SubsystemModulesResponse",
        backend_method="get_subsystem_modules",
        required_datasets=["docs.v_subsystem_profile"],  # and related tables
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    "subsystems.search": OperationSpec(
        id="subsystems.search",
        category="subsystems",
        summary="Search subsystems by name or role.",
        description="Search subsystems by label or role; MCP-only at present.",
        http_method=None,
        http_path=None,
        tool_name="search_subsystems",
        output_model_name="SubsystemSearchResponse",
        backend_method="search_subsystems",
        required_datasets=[],
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    "subsystems.summarize": OperationSpec(
        id="subsystems.summarize",
        category="subsystems",
        summary="Summarize a subsystem with module details.",
        description=(
            "Summarize a subsystem and return its member modules, limited by an optional "
            "module_limit parameter."
        ),
        http_method=None,   # HTTP uses same /architecture/subsystem endpoint as detail;
                            # you can map this separately if you want 1:1
        http_path=None,
        tool_name="summarize_subsystem",
        output_model_name="SubsystemModulesResponse",
        backend_method="summarize_subsystem",
        required_datasets=["docs.v_subsystem_profile"],
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    # ------------------------------------------------------------------
    # IDE OPERATIONS
    # ------------------------------------------------------------------

    "ide.hints": OperationSpec(
        id="ide.hints",
        category="ide",
        summary="Get IDE hints for a file.",
        description=(
            "Return IDE-friendly hints for a file (e.g., hotspots, missing tests, subsystem "
            "membership hints) suitable for inline surfacing."
        ),
        http_method="GET",
        http_path="/ide/hints",
        tool_name="get_file_hints",
        output_model_name="FileHintsResponse",
        backend_method="get_file_hints",
        required_datasets=[],  # depends on subsystem + risk analytics; keep high-level
        required_graphs=[],
        default_limit=1,
        max_limit=1,
    ),

    # ------------------------------------------------------------------
    # DATASET OPERATIONS
    # ------------------------------------------------------------------

    "datasets.list": OperationSpec(
        id="datasets.list",
        category="datasets",
        summary="List datasets available through the backend.",
        description="List datasets from the dataset registry with basic metadata.",
        http_method="GET",
        http_path="/datasets",
        tool_name="list_datasets",
        output_model_name="DatasetDescriptor",
        backend_method="list_datasets",
        required_datasets=[],
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    "datasets.specs": OperationSpec(
        id="datasets.specs",
        category="datasets",
        summary="Expose dataset contract and registry metadata.",
        description="Return DatasetSpecDescriptor entries describing dataset contracts.",
        http_method="GET",
        http_path="/datasets/specs",
        tool_name=None,  # no dedicated MCP tool yet
        output_model_name="DatasetSpecDescriptor",
        backend_method="dataset_specs",
        required_datasets=[],
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    "datasets.rows": OperationSpec(
        id="datasets.rows",
        category="datasets",
        summary="Read rows from a dataset with limits and messaging.",
        description=(
            "Read rows from a named dataset, applying BackendLimits and returning "
            "clamping/truncation information in ResponseMeta."
        ),
        http_method="GET",
        http_path="/datasets/{dataset_name}",
        tool_name="read_dataset_rows",
        output_model_name="DatasetRowsResponse",
        backend_method="read_dataset_rows",
        required_datasets=[],  # dataset determined dynamically by dataset_name
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    "datasets.schema": OperationSpec(
        id="datasets.schema",
        category="datasets",
        summary="Describe dataset schema and sample rows.",
        description=(
            "Return a composite schema description for a dataset, combining DuckDB "
            "catalog, JSON Schema, and sample rows."
        ),
        http_method="GET",
        http_path="/datasets/{dataset_name}/schema",
        tool_name=None,  # add MCP tool if desired
        output_model_name="DatasetSchemaResponse",
        backend_method="dataset_schema",
        required_datasets=[],  # dataset determined dynamically by dataset_name
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    # ------------------------------------------------------------------
    # GRAPH PLUGIN PLAN
    # ------------------------------------------------------------------

    "graph.plugins.plan": OperationSpec(
        id="graph.plugins.plan",
        category="graph_plugins",
        summary="Compute graph metric plugin execution plan.",
        description=(
            "Compute a graph plugin execution plan (ordering, dependencies, and skips) "
            "using the configured analytics.graphs.plugins registry."
        ),
        http_method=None,
        http_path=None,
        tool_name="graph_plugin_plan",
        output_model_name="GraphPlanResponse",
        backend_method="graph_plugin_plan",  # implemented in architecture_tools via plan_graph_metric_plugins
        required_datasets=[],
        required_graphs=[],  # affects analytics, not raw graph construction
        default_limit=None,
        max_limit=None,
    ),

    # ------------------------------------------------------------------
    # HEALTH / DIAGNOSTICS
    # ------------------------------------------------------------------

    "health.status": OperationSpec(
        id="health.status",
        category="health",
        summary="Health check for CodeIntel API.",
        description=(
            "Return a simple health payload including status, repo, commit, read_only, "
            "and optional limits derived from ServingConfig."
        ),
        http_method="GET",
        http_path="/health",
        tool_name=None,
        output_model_name="HealthPayload",  # non-Pydantic JSON payload
        backend_method="health",           # router-level handler, not QueryService
        required_datasets=[],
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),
}
```

You can then keep the `iter_operation_specs()` / `get_operation_spec()` helpers as:

```python
def iter_operation_specs() -> list[OperationSpec]:
    return list(_OPERATION_SPECS.values())


def get_operation_spec(op_id: str) -> OperationSpec | None:
    return _OPERATION_SPECS.get(op_id)
```

---

If you’d like, next step we can:

* Take one category (e.g. **subsystems**) and wire `OperationSpec` back into the HTTP + MCP code so that the routes and tools use the registry for things like summary text, tags, and even path consistency checking.


# subsystems zoom in #

Gotcha, let’s make subsystems the “proof-of-pattern” category where HTTP + MCP actually *consume* `OperationSpec` instead of each doing their own thing.

I’ll walk through concrete changes in three places:

1. **`serving/registry.py`** – make sure the subsystem `OperationSpec`s are finalized (backend method names, http paths, tool names).
2. **`serving/http/routes/subsystems.py`** – use `OperationSpec` for path, summary, tags, and sanity checks.
3. **`serving/mcp/architecture_tools.py`** – use `OperationSpec` for tool name consistency and (optionally) doc/metadata alignment.

Everything below is designed to be copy-pasta-friendly and mechanically applicable by an agent.

---

## 0. Finalize subsystem `OperationSpec`s in `serving/registry.py`

You already have a `_OPERATION_SPECS` dict; here’s the **subsystems slice** you should drop into that dict (or merge with what you have), making sure `backend_method` and `http_path` match your current code:

```python
# ------------------------------------------------------------------
# SUBSYSTEM OPERATIONS
# ------------------------------------------------------------------

_OPERATION_SPECS: dict[str, OperationSpec] = {
    # ... other entries above ...

    "subsystems.list": OperationSpec(
        id="subsystems.list",
        category="subsystems",
        summary="List inferred subsystems.",
        description=(
            "List inferred subsystems with optional filtering by role or search term. "
            "Results are backed by docs.v_subsystem_summary and related analytics."
        ),
        http_method="GET",
        http_path="/architecture/subsystems",
        tool_name="list_subsystems",                 # MCP tool in architecture_tools
        output_model_name="SubsystemSummaryResponse",
        backend_method="list_subsystems",            # QueryService / delegates
        required_datasets=["docs.v_subsystem_summary"],
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    "subsystems.profiles": OperationSpec(
        id="subsystems.profiles",
        category="subsystems",
        summary="List subsystem profiles.",
        description=(
            "List profile rows that aggregate metrics per subsystem, such as size, "
            "risk, and ownership. HTTP-only at present."
        ),
        http_method="GET",
        http_path="/architecture/subsystem-profiles",
        tool_name=None,                              # HTTP only for now
        output_model_name="SubsystemProfileResponse",
        backend_method="list_subsystem_profiles",    # in _SubsystemQueryDelegates
        required_datasets=["docs.v_subsystem_profile"],
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    "subsystems.coverage": OperationSpec(
        id="subsystems.coverage",
        category="subsystems",
        summary="List subsystem coverage rollups.",
        description=(
            "Summarize coverage-related metrics for each subsystem, such as number of "
            "modules, functions, and test coverage percentages."
        ),
        http_method="GET",
        http_path="/architecture/subsystem-coverage",
        tool_name=None,
        output_model_name="SubsystemCoverageResponse",
        backend_method="list_subsystem_coverage",
        required_datasets=["docs.v_subsystem_coverage"],
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    "subsystems.module_memberships": OperationSpec(
        id="subsystems.module_memberships",
        category="subsystems",
        summary="List subsystem memberships for a module.",
        description="Return which subsystems a given module belongs to.",
        http_method="GET",
        http_path="/architecture/module-subsystems",
        tool_name="get_module_subsystems",           # MCP tool
        output_model_name="ModuleSubsystemResponse",
        backend_method="get_module_subsystems",
        required_datasets=["docs.v_subsystem_memberships"],
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    "subsystems.detail": OperationSpec(
        id="subsystems.detail",
        category="subsystems",
        summary="Get modules and detail for a subsystem.",
        description=(
            "Return detailed membership and metrics for a single subsystem, including "
            "its member modules and rollup metrics."
        ),
        http_method="GET",
        http_path="/architecture/subsystem",
        tool_name="get_subsystem_modules",
        output_model_name="SubsystemModulesResponse",
        backend_method="get_subsystem_modules",
        required_datasets=["docs.v_subsystem_profile"],
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    "subsystems.search": OperationSpec(
        id="subsystems.search",
        category="subsystems",
        summary="Search subsystems by name or role.",
        description="Search subsystems by label or role; currently MCP-only.",
        http_method=None,
        http_path=None,
        tool_name="search_subsystems",
        output_model_name="SubsystemSearchResponse",
        backend_method="search_subsystems",
        required_datasets=["docs.v_subsystem_summary"],
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    "subsystems.summarize": OperationSpec(
        id="subsystems.summarize",
        category="subsystems",
        summary="Summarize a subsystem with module details.",
        description=(
            "Summarize a subsystem and return its member modules, limited by an optional "
            "module_limit parameter. MCP-only convenience variant of subsystem.detail."
        ),
        http_method=None,       # HTTP uses the same /architecture/subsystem endpoint
        http_path=None,
        tool_name="summarize_subsystem",
        output_model_name="SubsystemModulesResponse",
        backend_method="summarize_subsystem",
        required_datasets=["docs.v_subsystem_profile"],
        required_graphs=[],
        default_limit=None,
        max_limit=None,
    ),

    # ... other entries below ...
}
```

Now let’s plug these specs into HTTP + MCP.

---

## 1. Wire subsystems OperationSpec into HTTP routes

### 1.1. Import the registry in `serving/http/routes/subsystems.py`

At the top of `serving/http/routes/subsystems.py`, add:

```python
from codeintel.serving.registry import get_operation_spec
```

Your imports should look roughly like:

```python
from fastapi import APIRouter

from codeintel.serving.http.dependencies import ServiceDep
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import (
    ModuleSubsystemResponse,
    SubsystemCoverageResponse,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemSummaryResponse,
)
from codeintel.serving.registry import get_operation_spec
```

### 1.2. Use specs inside `build_subsystem_router`

Inside `build_subsystem_router`, right at the top of the function, fetch the specs you’ll need:

```python
def build_subsystem_router() -> APIRouter:
    """
    Construct the router for subsystem endpoints.

    Returns
    -------
    APIRouter
        Router exposing subsystem docs views and membership helpers.
    """
    router = APIRouter()

    spec_list = get_operation_spec("subsystems.list")
    spec_profiles = get_operation_spec("subsystems.profiles")
    spec_coverage = get_operation_spec("subsystems.coverage")
    spec_memberships = get_operation_spec("subsystems.module_memberships")
    spec_detail = get_operation_spec("subsystems.detail")

    # Defensive: if a spec is missing or misconfigured, fail fast in dev.
    assert spec_list is not None
    assert spec_profiles is not None
    assert spec_coverage is not None
    assert spec_memberships is not None
    assert spec_detail is not None

    # Optional: runtime invariants to guard against drift
    assert spec_list.http_path == "/architecture/subsystems"
    assert spec_profiles.http_path == "/architecture/subsystem-profiles"
    assert spec_coverage.http_path == "/architecture/subsystem-coverage"
    assert spec_memberships.http_path == "/architecture/module-subsystems"
    assert spec_detail.http_path == "/architecture/subsystem"
```

### 1.3. Replace hard-coded paths/summaries with spec-driven ones

Now swap the decorators for each subsystem route to use `spec.*` instead of hardcoded strings.

#### Before (conceptual)

```python
@router.get(
    "/architecture/subsystems",
    response_model=SubsystemSummaryResponse,
    summary="List inferred subsystems",
)
def list_subsystems(*, service: ServiceDep, limit: int | None = None, role: str | None = None, q: str | None = None) -> SubsystemSummaryResponse:
    return service.list_subsystems(limit=limit, role=role, q=q)
```

#### After (using OperationSpec)

```python
    @router.get(
        spec_list.http_path,
        response_model=SubsystemSummaryResponse,
        summary=spec_list.summary,
        tags=[spec_list.category],
    )
    def list_subsystems(
        *,
        service: ServiceDep,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> SubsystemSummaryResponse:
        return service.list_subsystems(limit=limit, role=role, q=q)
```

Repeat for the other routes.

#### Profiles

```python
    @router.get(
        spec_profiles.http_path,
        response_model=SubsystemProfileResponse,
        summary=spec_profiles.summary,
        tags=[spec_profiles.category],
    )
    def list_subsystem_profiles(
        *,
        service: ServiceDep,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> SubsystemProfileResponse:
        return service.list_subsystem_profiles(limit=limit, role=role, q=q)
```

#### Coverage

```python
    @router.get(
        spec_coverage.http_path,
        response_model=SubsystemCoverageResponse,
        summary=spec_coverage.summary,
        tags=[spec_coverage.category],
    )
    def list_subsystem_coverage(
        *,
        service: ServiceDep,
        limit: int | None = None,
    ) -> SubsystemCoverageResponse:
        return service.list_subsystem_coverage(limit=limit)
```

#### Module memberships

```python
    @router.get(
        spec_memberships.http_path,
        response_model=ModuleSubsystemResponse,
        summary=spec_memberships.summary,
        tags=[spec_memberships.category],
    )
    def module_subsystems(
        *,
        service: ServiceDep,
        module: str,
    ) -> ModuleSubsystemResponse:
        return service.get_module_subsystems(module=module)
```

#### Subsystem detail

```python
    @router.get(
        spec_detail.http_path,
        response_model=SubsystemModulesResponse,
        summary=spec_detail.summary,
        tags=[spec_detail.category],
    )
    def subsystem_modules(
        *,
        service: ServiceDep,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> SubsystemModulesResponse:
        """
        Get modules and detail for a subsystem.

        Raises
        ------
        ProblemError
            If the subsystem cannot be located.
        """
        response = service.get_subsystem_modules(
            subsystem_id=subsystem_id,
            module_limit=module_limit,
        )
        if not response.found or response.subsystem is None:
            message = "Subsystem not found"
            raise errors.not_found(message)
        return response
```

> Note: you can still keep the “not found” behavior exactly as-is; spec is only driving path/summary/tags.

---

## 2. Wire OperationSpec into MCP architecture/subsystem tools

Now we’ll use the same `OperationSpec` entries in the MCP tools, mainly to:

* Check **tool names** stay aligned (`spec.tool_name == function.__name__`).
* Optionally set **docstrings** or metadata from `spec.summary` / `spec.description`.

### 2.1. Import the registry in `serving/mcp/architecture_tools.py`

At the top of `architecture_tools.py`, add:

```python
from codeintel.serving.registry import get_operation_spec
```

Your imports will then look like:

```python
from mcp.server.fastmcp import FastMCP

from codeintel.analytics.graphs.plugins import plan_graph_metric_plugins
from codeintel.serving.mcp.models import (
    FileHintsResponse,
    FunctionArchitectureResponse,
    GraphPlanPluginMetadata,
    GraphPlanResponse,
    GraphPlanSkipped,
    ModuleArchitectureResponse,
    ModuleSubsystemResponse,
    ProblemDetail,
    SubsystemModulesResponse,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
)
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.registry import get_operation_spec
```

### 2.2. Pre-fetch the specs inside `register_architecture_tools`

Inside `register_architecture_tools(mcp: FastMCP, backend: QueryBackendOrService)`, before you start defining tools, fetch the specs you need:

```python
def register_architecture_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """Register architecture and subsystem tools on the given FastMCP instance."""
    spec_list = get_operation_spec("subsystems.list")
    spec_memberships = get_operation_spec("subsystems.module_memberships")
    spec_detail = get_operation_spec("subsystems.detail")
    spec_search = get_operation_spec("subsystems.search")
    spec_summarize = get_operation_spec("subsystems.summarize")

    assert spec_list is not None
    assert spec_memberships is not None
    assert spec_detail is not None
    assert spec_search is not None
    assert spec_summarize is not None
```

You can then reference these specs inside each tool, or just use them for invariants.

### 2.3. Enforce tool name/spec alignment and reuse descriptions

#### Example 1: `list_subsystems`

**Before:**

```python
    @mcp.tool()
    @_wrap
    def list_subsystems(
        limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: SubsystemSummaryResponse = backend.list_subsystems(limit=limit, role=role, q=q)
        return resp.model_dump()
```

**After (using spec):**

```python
    @mcp.tool()
    @_wrap
    def list_subsystems(
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        """
        List inferred subsystems with optional role and query filters.

        This description is kept in sync with OperationSpec["subsystems.list"].
        """
        # Sanity check: keep the registry and tool name in sync.
        spec = get_operation_spec("subsystems.list")
        assert spec is not None and spec.tool_name == "list_subsystems"

        resp: SubsystemSummaryResponse = backend.list_subsystems(
            limit=limit,
            role=role,
            q=q,
        )
        return resp.model_dump()
```

If you’d like, you can even overwrite the docstring from `spec.description` at the end of `register_architecture_tools`:

```python
    # At the bottom of register_architecture_tools:
    if spec_list.description:
        list_subsystems.__doc__ = spec_list.description
```

#### Example 2: `get_module_subsystems` (“module_memberships” op)

```python
    @mcp.tool()
    @_wrap
    def get_module_subsystems(
        module: str,
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        """
        List subsystem memberships for a given module.
        """
        spec = get_operation_spec("subsystems.module_memberships")
        assert spec is not None and spec.tool_name == "get_module_subsystems"

        resp: ModuleSubsystemResponse = backend.get_module_subsystems(module=module)
        return resp.model_dump()
```

#### Example 3: `get_subsystem_modules` (“detail”)

```python
    @mcp.tool()
    @_wrap
    def get_subsystem_modules(
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        """
        Get modules and detail for a subsystem.
        """
        spec = get_operation_spec("subsystems.detail")
        assert spec is not None and spec.tool_name == "get_subsystem_modules"

        resp: SubsystemModulesResponse = backend.get_subsystem_modules(
            subsystem_id=subsystem_id,
            module_limit=module_limit,
        )
        return resp.model_dump()
```

#### Example 4: `search_subsystems` (MCP-only op)

```python
    @mcp.tool()
    @_wrap
    def search_subsystems(
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        """
        Search subsystems by label or role.
        """
        spec = get_operation_spec("subsystems.search")
        assert spec is not None and spec.tool_name == "search_subsystems"

        resp: SubsystemSearchResponse = backend.search_subsystems(
            limit=limit,
            role=role,
            q=q,
        )
        return resp.model_dump()
```

#### Example 5: `summarize_subsystem` (MCP-only op)

```python
    @mcp.tool()
    @_wrap
    def summarize_subsystem(
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        """
        Summarize a subsystem and return its modules.
        """
        spec = get_operation_spec("subsystems.summarize")
        assert spec is not None and spec.tool_name == "summarize_subsystem"

        resp: SubsystemModulesResponse = backend.summarize_subsystem(
            subsystem_id=subsystem_id,
            module_limit=module_limit,
        )
        return resp.model_dump()
```

This keeps **tool name ↔ OperationSpec.tool_name** aligned, and by keeping the docstrings essentially the same as `spec.summary/description`, you reduce drift.

---

## 3. Optional sanity tests

To really lock this in, you can add a couple of cheap tests:

### 3.1. HTTP path ↔ spec consistency

In `tests/serving/test_operation_registry.py` (new file):

```python
from codeintel.serving.registry import get_operation_spec
from codeintel.serving.http.routes.subsystems import build_subsystem_router


def test_subsystems_paths_match_registry() -> None:
    spec_list = get_operation_spec("subsystems.list")
    spec_profiles = get_operation_spec("subsystems.profiles")
    spec_coverage = get_operation_spec("subsystems.coverage")
    spec_memberships = get_operation_spec("subsystems.module_memberships")
    spec_detail = get_operation_spec("subsystems.detail")

    router = build_subsystem_router()
    paths = {route.path for route in router.routes}

    assert spec_list.http_path in paths
    assert spec_profiles.http_path in paths
    assert spec_coverage.http_path in paths
    assert spec_memberships.http_path in paths
    assert spec_detail.http_path in paths
```

### 3.2. MCP tool ↔ spec.tool_name consistency

In `tests/mcp/test_operation_registry_subsystems.py`:

```python
from codeintel.serving.mcp.architecture_tools import register_architecture_tools
from codeintel.serving.mcp.models import ProblemDetail
from codeintel.serving.mcp.tool_utils import QueryBackendOrService
from codeintel.serving.registry import get_operation_spec

from mcp.server.fastmcp import FastMCP


def test_subsystems_tool_names_match_registry(dummy_backend: QueryBackendOrService) -> None:
    mcp = FastMCP("test")
    register_architecture_tools(mcp, dummy_backend)
    tool_names = {tool.name for tool in mcp.tools}

    for op_id in [
        "subsystems.list",
        "subsystems.module_memberships",
        "subsystems.detail",
        "subsystems.search",
        "subsystems.summarize",
    ]:
        spec = get_operation_spec(op_id)
        assert spec is not None
        assert spec.tool_name in tool_names
```

---

### Net result

For **subsystems**, you now have:

* A **single operation description** per operation (`OperationSpec`).
* HTTP routes whose **paths, summaries, and tags** are driven by that spec.
* MCP tools whose **names and semantics** are checked against that spec.
* Meta endpoints (`/meta/operations`) that give agents a complete picture of all subsystem operations (and everything else).

If you’d like to push this pattern even further later, we can do a similar wiring for **functions** or **datasets**, or even auto-generate parts of the FastAPI route registration and MCP tool registration from `OperationSpec`.

# HTTP and MCP pivot to use of OperationSpec #

Totally with you—this is a neat “Refactor 3b” and it’s *very* implementable with what you already have.

I’ll show you **concrete changes** in two places, focusing on the **functions** category as the worked example:

1. **HTTP**: use `OperationSpec` to drive **paths, summaries, tags, and simple invariants** in `serving/http/routes/functions.py`.
2. **MCP**: use `OperationSpec` to **dynamically register tools** in `serving/mcp/function_tools.py`, with a generic wrapper that calls `getattr(backend, spec.backend_method)(**params)`.

I’ll keep everything incremental so it doesn’t blow up what you’ve already implemented.

---

## 1. HTTP: Use OperationSpec to drive the functions router

We’ll wire `OperationSpec` into `serving/http/routes/functions.py` in the same way we already did for subsystems:

* Look up the spec by id.
* Use `spec.http_path`, `spec.summary`, and `spec.category` in the route decorators.
* Assert invariants so spec and code can’t silently drift.

### 1.1. Import the registry at the top of `functions.py`

Edit `src/codeintel/serving/http/routes/functions.py`.

Add:

```python
from codeintel.serving.registry import get_operation_spec
```

So your imports look roughly like:

```python
from fastapi import APIRouter

from codeintel.serving.http.dependencies import ServiceDep
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    FileSummaryResponse,
    FunctionArchitectureResponse,
    FunctionSummaryResponse,
    GraphNeighborhoodResponse,
    GraphScopePayload,
    HighRiskFunctionsResponse,
)
from codeintel.serving.registry import get_operation_spec
```

### 1.2. Fetch specs and assert invariants inside `build_functions_router`

At the top of `build_functions_router`, grab the specs you care about:

```python
def build_functions_router() -> APIRouter:
    """
    Construct the router for function-centric endpoints.
    """
    router = APIRouter()

    spec_summary = get_operation_spec("function.summary")
    spec_high_risk = get_operation_spec("functions.high_risk")
    spec_neighbors = get_operation_spec("graph.call_neighbors")
    spec_neighborhood = get_operation_spec("graph.call_neighborhood")
    spec_import_boundary = get_operation_spec("graph.import_boundary")
    spec_file_summary = get_operation_spec("file.summary")
    spec_func_arch = get_operation_spec("architecture.function")
    spec_mod_arch = get_operation_spec("architecture.module")

    # Fail fast if registry is out of sync in dev/tests
    assert spec_summary is not None
    assert spec_high_risk is not None
    assert spec_neighbors is not None
    assert spec_neighborhood is not None
    assert spec_import_boundary is not None
    assert spec_file_summary is not None
    assert spec_func_arch is not None
    assert spec_mod_arch is not None

    # Optional: invariants to guard against drift
    assert spec_summary.http_path == "/function/summary"
    assert spec_high_risk.http_path == "/functions/high-risk"
    assert spec_neighbors.http_path == "/function/callgraph"
    assert spec_neighborhood.http_path == "/graph/call/neighborhood"
    assert spec_file_summary.http_path == "/file/summary"
    assert spec_func_arch.http_path == "/architecture/function"
    assert spec_mod_arch.http_path == "/architecture/module"
```

> Those asserts are there so that if someone edits a decorator path or spec independently, tests will catch it.

### 1.3. Use OperationSpec in the route decorators

Now swap hard-coded paths/summaries in the decorators for `spec.*`.

#### Function summary

Before (conceptually):

```python
@router.get(
    "/function/summary",
    response_model=FunctionSummaryResponse,
    summary="Summarize a function by GOID, URN, or source location.",
)
def function_summary(..., service: ServiceDep, ...):
    return service.get_function_summary(...)
```

After:

```python
    @router.get(
        spec_summary.http_path,
        response_model=FunctionSummaryResponse,
        summary=spec_summary.summary,
        tags=[spec_summary.category],
    )
    def function_summary(
        *,
        service: ServiceDep,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphScopePayload | None = None,
    ) -> FunctionSummaryResponse:
        return service.get_function_summary(
            urn=urn,
            goid_h128=goid_h128,
            rel_path=rel_path,
            qualname=qualname,
            scope=scope,
        )
```

#### High-risk functions

```python
    @router.get(
        spec_high_risk.http_path,
        response_model=HighRiskFunctionsResponse,
        summary=spec_high_risk.summary,
        tags=[spec_high_risk.category],
    )
    def list_high_risk_functions(
        *,
        service: ServiceDep,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
    ) -> HighRiskFunctionsResponse:
        return service.list_high_risk_functions(
            min_risk=min_risk,
            limit=limit,
            tested_only=tested_only,
        )
```

#### Callgraph neighbors

```python
    @router.get(
        spec_neighbors.http_path,
        response_model=CallGraphNeighborsResponse,
        summary=spec_neighbors.summary,
        tags=[spec_neighbors.category],
    )
    def callgraph_neighbors(
        *,
        service: ServiceDep,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> CallGraphNeighborsResponse:
        return service.get_callgraph_neighbors(
            goid_h128=goid_h128,
            direction=direction,
            limit=limit,
            scope=scope,
        )
```

…and same idea for:

* `graph.call_neighborhood` → `get_callgraph_neighborhood`
* `file.summary` → `get_file_summary`
* `architecture.function` → `get_function_architecture`
* `architecture.module` → `get_module_architecture`

All handler logic stays the same; we’re just **driving path, summary, and tags** from `OperationSpec` and asserting they stay in sync.

---

## 2. MCP: Use OperationSpec to dynamically generate function tools

Now let’s make `serving/mcp/function_tools.py` stop hardcoding tool names and backend calls, and instead **loop over specs** in the `"functions"` category.

We’ll start with a practical v1: each generated tool:

* Has the correct **name** (`spec.tool_name`).
* Exposes the correct **description** (`spec.summary` / `spec.description`).
* Calls the right **backend method** (`getattr(backend, spec.backend_method)`).
* Accepts **keyword arguments** (`**kwargs`) and passes them straight through.

You’ll sacrifice per-parameter JSON schema for now, but it is extremely easy to layer that back in later with dedicated Pydantic arg models.

### 2.1. Import registry + ProblemDetail in `function_tools.py`

Edit `src/codeintel/serving/mcp/function_tools.py`.

Add:

```python
from typing import Any

from codeintel.serving.mcp.models import ProblemDetail
from codeintel.serving.registry import iter_operation_specs
```

Imports will look roughly like:

```python
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.models import (
    FunctionSummaryResponse,
    HighRiskFunctionsResponse,
    CallGraphNeighborsResponse,
    GraphNeighborhoodResponse,
    ProblemDetail,
)
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.registry import iter_operation_specs
```

(You may not actually need the response models anymore after this, but leaving them imported is harmless.)

### 2.2. Replace manual tools with a spec-driven loop

We’ll define a helper inside `register_function_tools` that:

* Captures a single `OperationSpec`.
* Creates a closure that calls `backend.<method>(**kwargs)`.
* Registers it with `mcp.tool(name=..., description=...)`.
* Wraps it in `_wrap` to keep your error handling intact.

**Before (conceptually):**

```python
def register_function_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    @mcp.tool()
    @_wrap
    def get_function_summary(...params...) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: FunctionSummaryResponse = backend.get_function_summary(...params...)
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def list_high_risk_functions(...params...) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: HighRiskFunctionsResponse = backend.list_high_risk_functions(...params...)
        return resp.model_dump()

    # etc...
```

**After: spec-driven tool generator:**

```python
def register_function_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """
    Register function-related MCP tools based on OperationSpec.

    Tools are created dynamically for all operations in the "functions" category
    that define a tool_name and backend_method.
    """

    def _register_tool_for_spec(spec_id: str) -> None:
        # Look up OperationSpec from the registry
        from codeintel.serving.registry import get_operation_spec

        spec = get_operation_spec(spec_id)
        if spec is None:
            raise RuntimeError(f"OperationSpec not found for id={spec_id!r}")
        if spec.tool_name is None:
            # Nothing to register for MCP
            return

        backend_method = getattr(backend, spec.backend_method, None)
        if backend_method is None:
            raise RuntimeError(
                f"Backend {backend!r} does not implement method {spec.backend_method!r} "
                f"for OperationSpec id={spec.id!r}"
            )

        # We generate a tool that forwards **kwargs to the backend method.
        # This keeps things generic; per-parameter schemas can be added later
        # with dedicated Pydantic arg models if desired.
        @_wrap
        def _tool(**kwargs: Any) -> dict[str, object] | dict[str, ProblemDetail]:
            """
            Dynamically generated tool that forwards to backend.<method>(**kwargs).
            """
            response = backend_method(**kwargs)
            # We expect all backend methods to return Pydantic models with model_dump()
            return response.model_dump()

        # Give the tool a nice name and docstring for MCP/OpenAPI introspection.
        _tool.__name__ = spec.tool_name
        _tool.__doc__ = spec.description or spec.summary

        # Register with FastMCP, using the spec name/description
        mcp.tool(
            name=spec.tool_name,
            description=spec.summary,
        )(_tool)

    # Register tools for all "functions" operations that define tool_name
    for spec in iter_operation_specs():
        if spec.category != "functions":
            continue
        if spec.tool_name is None:
            continue
        _register_tool_for_spec(spec.id)
```

That’s it: **no more per-tool boilerplate** for the functions category. The mapping from spec → backend is:

* `OperationSpec.backend_method` → method on `QueryBackendOrService` / `QueryService`.
* `OperationSpec.tool_name` → MCP tool name.

### 2.3. How calls flow now

When a client calls e.g. `list_high_risk_functions` over MCP:

1. FastMCP finds the tool with `name="list_high_risk_functions"`.
2. It calls `_tool(**kwargs)` with the named params from the client.
3. `_tool` calls `backend_method(**kwargs)`, where `backend_method` is `backend.list_high_risk_functions`.
4. The backend method returns a Pydantic model (e.g. `HighRiskFunctionsResponse`).
5. `_tool` returns `response.model_dump()`; `_wrap` handles errors and converts to `ProblemDetail` if needed.

You don’t get auto-generated per-parameter JSON schema (because the wrapper uses `**kwargs`), but you:

* Guarantee that **HTTP + MCP point to the same backend methods** (through `OperationSpec.backend_method`).
* Guarantee that **tool names are aligned with the registry**.
* Have meta introspection (`/meta/operations` and `list_operations_meta`) to see full operation descriptions.

Later, if you want full param schemas, you can:

* Introduce per-operation Pydantic arg models (`FunctionSummaryArgs`, `HighRiskFunctionsArgs`, etc.).
* Keep a mapping `OPERATION_INPUT_MODELS = {"function.summary": FunctionSummaryArgs, ...}`.
* Change `_tool(**kwargs)` to `_tool(args: FunctionSummaryArgs)` and call `backend_method(**args.model_dump())`.

But you don’t *need* that complexity to get the dynamic mapping benefits.

---

## 3. Summary of concrete changes

To implement “OperationSpec-driven HTTP + MCP” for **functions** right now:

1. **HTTP (`serving/http/routes/functions.py`)**

   * Import `get_operation_spec`.

   * At the start of `build_functions_router`, fetch specs for:

     * `"function.summary"`, `"functions.high_risk"`, `"graph.call_neighbors"`,
       `"graph.call_neighborhood"`, `"file.summary"`, `"architecture.function"`, `"architecture.module"`.

   * Assert `spec.http_path` matches the literal path you expect.

   * Update each `@router.get(...)` decorator to use:

     * `path = spec.http_path`
     * `summary = spec.summary`
     * `tags = [spec.category]`

   * Leave handler bodies unchanged.

2. **MCP (`serving/mcp/function_tools.py`)**

   * Import `iter_operation_specs`, `ProblemDetail`, and `Any`.
   * Replace manual `get_function_summary`, `list_high_risk_functions`, etc. with a single `register_function_tools` that:

     * Iterates over `iter_operation_specs()` and filters `spec.category == "functions"` and `spec.tool_name is not None`.
     * For each spec, looks up `backend_method = getattr(backend, spec.backend_method)`.
     * Defines a generic `_tool(**kwargs)` returning `backend_method(**kwargs).model_dump()`.
     * Wraps `_tool` in `_wrap` and decorates it with `mcp.tool(name=spec.tool_name, description=spec.summary)`.

This gives you:

* One **global registry** as the single source of truth.
* HTTP + MCP both wired **through** that registry for paths, names, descriptions, and backend binding.
* Minimal changes to existing handler logic.

If you’d like, we can do the same dynamic MCP registration for **datasets** next (which is an especially nice fit because the tools are very regular: `list_datasets`, `dataset_specs`, `read_dataset_rows`, `dataset_schema`).


# MCP registration for datasets #

Yep, datasets are the perfect place to go fully spec-driven. Let’s wire `OperationSpec` into **MCP dataset tools**, just like we did for functions and subsystems.

I’ll keep this concrete and mechanical:

1. Update your `OperationSpec` definitions for the dataset ops so **all four** have `tool_name` set.
2. Replace the hard-coded tools in `serving/mcp/dataset_tools.py` with a **spec-driven generator** that loops over the “datasets” specs and dynamically registers MCP tools.

---

## 1. Update `OperationSpec` for dataset operations

In `src/codeintel/serving/registry.py`, you already have entries something like this:

```python
"datasets.list": OperationSpec(
    id="datasets.list",
    category="datasets",
    summary="List datasets available through the backend.",
    description="List datasets from the dataset registry with basic metadata.",
    http_method="GET",
    http_path="/datasets",
    tool_name="list_datasets",                # already MCP-exposed
    output_model_name="DatasetDescriptor",
    backend_method="list_datasets",
    required_datasets=[],
    required_graphs=[],
    default_limit=None,
    max_limit=None,
),
"datasets.specs": OperationSpec(
    id="datasets.specs",
    category="datasets",
    summary="Expose dataset contract and registry metadata.",
    description="Return DatasetSpecDescriptor entries describing dataset contracts.",
    http_method="GET",
    http_path="/datasets/specs",
    tool_name=None,                           # <- we’ll change this
    output_model_name="DatasetSpecDescriptor",
    backend_method="dataset_specs",
    required_datasets=[],
    required_graphs=[],
    default_limit=None,
    max_limit=None,
),
"datasets.rows": OperationSpec(
    id="datasets.rows",
    category="datasets",
    summary="Read rows from a dataset with limits and messaging.",
    description=(
        "Read rows from a named dataset, applying BackendLimits and returning "
        "clamping/truncation information in ResponseMeta."
    ),
    http_method="GET",
    http_path="/datasets/{dataset_name}",
    tool_name="read_dataset_rows",            # already MCP-exposed
    output_model_name="DatasetRowsResponse",
    backend_method="read_dataset_rows",
    required_datasets=[],
    required_graphs=[],
    default_limit=None,
    max_limit=None,
),
"datasets.schema": OperationSpec(
    id="datasets.schema",
    category="datasets",
    summary="Describe dataset schema and sample rows.",
    description=(
        "Return a composite schema description for a dataset, combining DuckDB "
        "catalog, JSON Schema, and sample rows."
    ),
    http_method="GET",
    http_path="/datasets/{dataset_name}/schema",
    tool_name=None,                           # <- we’ll change this
    output_model_name="DatasetSchemaResponse",
    backend_method="dataset_schema",
    required_datasets=[],
    required_graphs=[],
    default_limit=None,
    max_limit=None,
),
```

**Change the two `tool_name=None` entries** to add proper MCP tool names:

```python
"datasets.specs": OperationSpec(
    id="datasets.specs",
    category="datasets",
    summary="Expose dataset contract and registry metadata.",
    description="Return DatasetSpecDescriptor entries describing dataset contracts.",
    http_method="GET",
    http_path="/datasets/specs",
    tool_name="dataset_specs",                # <- NEW
    output_model_name="DatasetSpecDescriptor",
    backend_method="dataset_specs",
    required_datasets=[],
    required_graphs=[],
    default_limit=None,
    max_limit=None,
),

"datasets.schema": OperationSpec(
    id="datasets.schema",
    category="datasets",
    summary="Describe dataset schema and sample rows.",
    description=(
        "Return a composite schema description for a dataset, combining DuckDB "
        "catalog, JSON Schema, and sample rows."
    ),
    http_method="GET",
    http_path="/datasets/{dataset_name}/schema",
    tool_name="dataset_schema",               # <- NEW
    output_model_name="DatasetSchemaResponse",
    backend_method="dataset_schema",
    required_datasets=[],
    required_graphs=[],
    default_limit=None,
    max_limit=None,
),
```

So now all dataset operations are:

* `datasets.list`    → `tool_name="list_datasets"`, `backend_method="list_datasets"`
* `datasets.specs`   → `tool_name="dataset_specs"`, `backend_method="dataset_specs"`
* `datasets.rows`    → `tool_name="read_dataset_rows"`, `backend_method="read_dataset_rows"`
* `datasets.schema`  → `tool_name="dataset_schema"`, `backend_method="dataset_schema"`

---

## 2. Rewrite `serving/mcp/dataset_tools.py` to be spec-driven

Right now your `dataset_tools.py` looks like (simplified):

```python
"""Dataset MCP tools."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.models import DatasetRowsResponse, ProblemDetail
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap


def register_dataset_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """Register dataset browsing MCP tools."""

    @mcp.tool()
    @_wrap
    def list_datasets() -> list[dict[str, object]]:
        ...

    @mcp.tool()
    @_wrap
    def read_dataset_rows(...) -> dict[str, object] | dict[str, ProblemDetail]:
        ...
```

We’ll replace this with a **generic generator** that:

* Iterates over `iter_operation_specs()`.
* Filters `category == "datasets"` and `tool_name is not None`.
* For each spec, finds the right backend method (`getattr(backend, spec.backend_method)`).
* Creates a `@mcp.tool(name=spec.tool_name)` that forwards `**kwargs` to that backend method.

### 2.1. Update imports

Edit `src/codeintel/serving/mcp/dataset_tools.py`:

```python
"""Dataset MCP tools."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.models import ProblemDetail
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.registry import iter_operation_specs, get_operation_spec
```

Notes:

* We no longer need to import `DatasetRowsResponse` directly; backend methods return Pydantic models, and we just call `.model_dump()`.
* We pull in `iter_operation_specs` / `get_operation_spec` to drive the registration.

### 2.2. Spec-driven `register_dataset_tools`

Now replace the body of `register_dataset_tools` with:

```python
def register_dataset_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """
    Register dataset-related MCP tools based on OperationSpec.

    For each OperationSpec in the "datasets" category with a non-empty tool_name,
    this will create a tool that forwards directly to the corresponding backend
    method defined by OperationSpec.backend_method.
    """

    def _register_tool_for_spec(spec_id: str) -> None:
        spec = get_operation_spec(spec_id)
        if spec is None:
            raise RuntimeError(f"OperationSpec not found for id={spec_id!r}")
        if spec.tool_name is None:
            # Nothing to expose over MCP for this spec
            return

        backend_method = getattr(backend, spec.backend_method, None)
        if backend_method is None:
            raise RuntimeError(
                f"Backend {backend!r} does not implement method {spec.backend_method!r} "
                f"for OperationSpec id={spec.id!r}"
            )

        @_wrap
        def _tool(**kwargs: Any) -> dict[str, object] | dict[str, ProblemDetail]:
            """
            Dynamically generated dataset tool that forwards to backend.<method>(**kwargs).
            """
            # backend_method is expected to return a Pydantic model with model_dump()
            response = backend_method(**kwargs)
            return response.model_dump()

        # Give the tool the spec-defined name and description for introspection.
        _tool.__name__ = spec.tool_name
        _tool.__doc__ = spec.description or spec.summary

        mcp.tool(
            name=spec.tool_name,
            description=spec.summary,
        )(_tool)

    # Register tools for all dataset operations with a tool_name.
    for spec in iter_operation_specs():
        if spec.category != "datasets":
            continue
        if spec.tool_name is None:
            continue
        _register_tool_for_spec(spec.id)


__all__ = ["register_dataset_tools"]
```

This gives you **four MCP tools** driven purely by `OperationSpec`:

* `list_datasets()`          → `backend.list_datasets()`
* `dataset_specs()`          → `backend.dataset_specs()`
* `read_dataset_rows(**kwargs)` → `backend.read_dataset_rows(dataset_name=..., limit=..., offset=...)`
* `dataset_schema(**kwargs)` → `backend.dataset_schema(dataset_name=..., sample_limit=...)`

### 2.3. How parameters work for `**kwargs`

For datasets, the backend methods are already nicely defined on `QueryBackend`:

```python
class QueryBackend(Protocol):
    def list_datasets(self) -> list[DatasetDescriptor]: ...
    def dataset_specs(self) -> list[DatasetSpecDescriptor]: ...
    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int = 100,
        offset: int = 0,
    ) -> DatasetRowsResponse: ...
    def dataset_schema(
        self,
        *,
        dataset_name: str,
        sample_limit: int = 5,
    ) -> DatasetSchemaResponse: ...
```

So when a client calls MCP tools:

* `list_datasets()` – no args; we call `backend.list_datasets()`.
* `dataset_specs()` – no args; we call `backend.dataset_specs()`.
* `read_dataset_rows(dataset_name="docs.v_function_summary", limit=100, offset=0)`.
* `dataset_schema(dataset_name="docs.v_function_summary", sample_limit=5)`.

Because `_tool(**kwargs)` forwards the kwargs directly to `backend_method(**kwargs)`, the usual MCP parameter names must match the backend’s parameter names (`dataset_name`, `limit`, `offset`, `sample_limit`). That’s already what you use in your HTTP routes and tests, so it aligns naturally.

If you want stricter schemas later, you can introduce Pydantic “args” models per operation (e.g. `ReadDatasetRowsArgs`) and change `_tool` to accept a single typed payload; but you don’t need that to get the spec-driven mapping working.

---

## 3. Optional: tests to enforce registry ↔ tools alignment

To keep this honest, you can add a small test that verifies:

* Every dataset `OperationSpec` with `tool_name` produces an MCP tool with that name.
* `backend` actually has the method referenced by `spec.backend_method`.

Example:

```python
# tests/mcp/test_dataset_tools_registry.py

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.dataset_tools import register_dataset_tools
from codeintel.serving.mcp.tool_utils import QueryBackendOrService
from codeintel.serving.registry import iter_operation_specs


class DummyBackend:
    # Implement the dataset methods with trivial responses for testing
    def list_datasets(self):
        class R:
            def model_dump(self): return {}
        return R()

    def dataset_specs(self):
        class R:
            def model_dump(self): return {}
        return R()

    def read_dataset_rows(self, *, dataset_name: str, limit: int = 100, offset: int = 0):
        class R:
            def model_dump(self): return {}
        return R()

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5):
        class R:
            def model_dump(self): return {}
        return R()


def test_dataset_tools_match_registry() -> None:
    mcp = FastMCP("test")
    backend: QueryBackendOrService = DummyBackend()  # type: ignore[assignment]
    register_dataset_tools(mcp, backend)

    tool_names = {tool.name for tool in mcp.tools}

    for spec in iter_operation_specs():
        if spec.category != "datasets" or spec.tool_name is None:
            continue
        assert spec.tool_name in tool_names
```

---

## 4. Net effect

After these changes:

* **OperationSpec is the single source of truth** for dataset operations:

  * Name (`tool_name`)
  * Backend binding (`backend_method`)
  * HTTP path (`http_path`)
  * Human-readable description (`summary`/`description`)
* **MCP dataset tools become zero-boilerplate**:

  * No duplicated method names.
  * No redundant docstrings.
  * Adding a new dataset operation is literally “add an OperationSpec + backend method,” and the MCP layer just picks it up.
* You’ve now exercised the full spec-driven pattern for:

  * **Subsystems** (HTTP + MCP using OperationSpec for invariants).
  * **Functions** (HTTP + MCP spec-driven).
  * **Datasets** (MCP spec-driven, and HTTP already aligned with the same specs via the `/meta` endpoints and consistent naming).

