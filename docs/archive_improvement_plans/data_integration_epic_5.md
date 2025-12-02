Here’s a concrete, “you can literally follow this” implementation plan for **Epic 3 – Single canonical OperationCatalog**, tailored to your current layout.

I’ll walk through:

1. Target design (what the end state looks like)
2. Phase 0 – Safety net
3. Phase 1 – Introduce canonical `Operation` model & catalog
4. Phase 2 – Wire `serving.registry` to the catalog (keep API stable)
5. Phase 3 – MCP registration and HTTP routes
6. Phase 4 – Make `backend/operations.py` a facade
7. Phase 5 – Strengthen tests (esp. `test_operation_spec_alignment.py`)

---

## 1. Target design (end state)

**Goal:** One canonical source of truth for operations:

```text
codeintel/serving/operations/catalog.py
  - DataSourceType enum
  - Operation dataclass
  - OPERATIONS_BY_ID dict
  - get_operation(id)
  - iter_operations()
```

Everything else becomes a **facade**:

* `serving/registry.py`

  * `OperationSpec` = type alias to `Operation`
  * `iter_operation_specs()` → `iter_operations()`
  * `get_operation_spec()` → `get_operation()`
* `serving/backend/operations.py`

  * `OperationContract` becomes a thin wrapper over `Operation`
  * `OPERATION_CONTRACTS` derived from `OPERATIONS_BY_ID`
* MCP tools (`serving/mcp/*.py`) and HTTP routes already use `OperationSpec`; they now indirectly use the canonical `Operation` under the hood.
* Tests unify around this catalog.

---

## 2. Phase 0 – Safety net & inventory

**Files involved (today):**

* `codeintel/serving/backend/operations.py`

  * `DataSourceType`
  * `OperationContract`
  * `OPERATION_CONTRACTS` / `CONTRACTS`
* `codeintel/serving/registry.py`

  * `DatasetMeta`
  * `OperationSpec`
  * `OPERATION_SPECS` / `_OPERATION_SPECS`
  * `iter_operation_specs`, `get_operation_spec`
* HTTP routes:

  * `serving/http/routes/*.py` (functions, architecture, profiles, datasets, subsystems, ide, health, meta)
* MCP tools & registration:

  * `serving/mcp/function_tools.py`
  * `serving/mcp/profile_tools.py`
  * `serving/mcp/architecture_tools.py`
  * `serving/mcp/dataset_tools.py`
  * `serving/mcp/meta_tools.py`
  * `serving/mcp/tools_base.py`
* Tests:

  * `tests/serving/test_operation_spec_alignment.py`

**Safety net 🎣**

Before changing anything:

* Add an ultra‑simple test that just **asserts current counts**:

```python
# tests/serving/test_operation_catalog_snapshot.py
from codeintel.serving.registry import iter_operation_specs
from codeintel.serving.backend.operations import OPERATION_CONTRACTS

def test_operation_counts_snapshot() -> None:
    specs = iter_operation_specs()
    assert len(specs) >= 1
    assert len(OPERATION_CONTRACTS) >= 1
    # If you want, assert specific numbers once you're comfortable:
    # assert len(specs) == 17
    # assert len(OPERATION_CONTRACTS) == 17
```

This gives you a quick “did we silently drop an operation?” sentinel while you refactor.

---

## 3. Phase 1 – Add canonical `Operation` & `OperationCatalog`

### 3.1. New module: `serving/operations/catalog.py`

Create a new package:

```text
codeintel/serving/operations/__init__.py
codeintel/serving/operations/catalog.py
```

**`codeintel/serving/operations/catalog.py`:**

```python
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class DataSourceType(StrEnum):
    """Classification of data sources for serving operations."""

    VIEW = "view"
    TABLE = "table"
    GRAPH_ENGINE = "graph_engine"
    COMPUTED = "computed"


@dataclass(frozen=True)
class Operation:
    """
    Canonical description of a serving operation across HTTP, MCP, and backend.

    This replaces the separate OperationSpec and OperationContract concepts.
    """

    # Identity & grouping
    id: str                    # e.g. "functions.summary"
    category: str              # "functions", "datasets", "graph", "profiles", ...

    # Human-facing docs
    summary: str
    description: str | None

    # HTTP surface
    http_method: Literal["GET", "POST"] | None
    http_path: str | None

    # MCP surface
    tool_name: str | None
    output_model_name: str

    # Backend method on DuckDBQueryService / QueryService
    backend_method: str

    # Data contract (backend/operations.py + dataset_contract)
    data_source: DataSourceType
    source_name: str | None          # "docs.v_function_summary", "analytics.goid_risk_factors"
    repository_method: str | None    # "FunctionRepository.list_high_risk_functions"

    # Dataset & graph prerequisites
    required_datasets: tuple[str, ...]
    required_graphs: tuple[str, ...]

    # Pagination & limits
    supports_pagination: bool
    default_limit: int | None
    max_limit: int | None


# --- Canonical catalog ---

# NOTE: For the first pass, implement a small subset of operations here (see below),
# then migrate all remaining ones gradually.
OPERATIONS_BY_ID: dict[str, Operation] = {
    # Example: functions.summary
    "functions.summary": Operation(
        id="functions.summary",
        category="functions",
        summary="Summarize a function by GOID, URN, or source location.",
        description=(
            "Summarize a function using docs and analytics views, "
            "identified by GOID, URN, qualified name, or file path."
        ),
        http_method="GET",
        http_path="/functions/summary",
        tool_name="get_function_summary",
        output_model_name="FunctionSummaryResponse",
        backend_method="get_function_summary",
        data_source=DataSourceType.VIEW,
        source_name="docs.v_function_summary",
        repository_method="FunctionRepository.get_function_summary",
        required_datasets=("docs.v_function_summary",),
        required_graphs=("callgraph",),
        supports_pagination=False,
        default_limit=1,
        max_limit=1,
    ),

    # Example: functions.high_risk
    "functions.high_risk": Operation(
        id="functions.high_risk",
        category="functions",
        summary="List high-risk functions, optionally restricted to tested ones.",
        description=(
            "Rank functions by risk using analytics and docs views with optional thresholds "
            "and tested-only filters."
        ),
        http_method="GET",
        http_path="/functions/high-risk",
        tool_name="list_high_risk_functions",
        output_model_name="HighRiskFunctionsResponse",
        backend_method="list_high_risk_functions",
        data_source=DataSourceType.TABLE,
        source_name="analytics.goid_risk_factors",
        repository_method="FunctionRepository.list_high_risk_functions",
        required_datasets=("analytics.goid_risk_factors",),
        required_graphs=(),
        supports_pagination=True,
        default_limit=None,
        max_limit=None,
    ),

    # Example: datasets.list
    "datasets.list": Operation(
        id="datasets.list",
        category="datasets",
        summary="List available datasets in the registry.",
        description="Enumerate datasets with their metadata and serving limits.",
        http_method="GET",
        http_path="/datasets",
        tool_name="list_datasets",
        output_model_name="DatasetListResponse",
        backend_method="list_datasets",
        data_source=DataSourceType.COMPUTED,
        source_name="dataset_registry",
        repository_method=None,
        required_datasets=(),
        required_graphs=(),
        supports_pagination=False,
        default_limit=None,
        max_limit=None,
    ),

    # ...add the rest of your operations here following the same pattern...
}


def get_operation(op_id: str) -> Operation | None:
    """Lookup a single operation by id."""
    return OPERATIONS_BY_ID.get(op_id)


def iter_operations() -> Iterable[Operation]:
    """Iterate over all registered operations."""
    return OPERATIONS_BY_ID.values()
```

**`codeintel/serving/operations/__init__.py`:**

```python
from .catalog import (
    DataSourceType,
    Operation,
    OPERATIONS_BY_ID,
    get_operation,
    iter_operations,
)

__all__ = [
    "DataSourceType",
    "Operation",
    "OPERATIONS_BY_ID",
    "get_operation",
    "iter_operations",
]
```

### 3.2. Populate catalog incrementally

For the first commit, you can:

* Start with a **small subset** of operations (e.g. the three I showed).
* Add a temporary test to ensure the catalog is in sync for those:

```python
# tests/serving/test_operation_catalog_seed.py
from codeintel.serving.operations import get_operation

def test_seed_catalog_has_core_operations() -> None:
    assert get_operation("functions.summary") is not None
    assert get_operation("functions.high_risk") is not None
    assert get_operation("datasets.list") is not None
```

Then, gradually migrate everything from `OPERATION_SPECS` and `OPERATION_CONTRACTS` into `OPERATIONS_BY_ID`.

---

## 4. Phase 2 – Wire `serving.registry` to the catalog

Now make `serving.registry` treat the catalog as canonical, **without** changing external APIs.

### 4.1. Replace `OperationSpec` with a type alias

In `codeintel/serving/registry.py`:

**Before** (simplified):

```python
from dataclasses import dataclass
from typing import Literal, Sequence

from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.models import DatasetSpecDescriptor
from codeintel.serving.services.query_service import QueryService

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

OPERATION_SPECS: dict[str, OperationSpec] = {
    "functions.summary": OperationSpec(...),
    ...
}

def iter_operation_specs() -> list[OperationSpec]:
    return list(_OPERATION_SPECS.values())

def get_operation_spec(op_id: str) -> OperationSpec | None:
    return _OPERATION_SPECS.get(op_id)
```

**After** (canonicalizing on `Operation`):

Keep all the *dataset* stuff intact; we only touch the operation part.

```python
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.models import DatasetSpecDescriptor
from codeintel.serving.operations import Operation, get_operation, iter_operations
from codeintel.serving.services.query_service import QueryService

# --- existing DatasetMeta, dataset registry, etc. stay as-is ---


# OperationSpec is now just an alias for the canonical Operation
OperationSpec = Operation

# Backwards-compatible mapping
_OPERATION_SPECS: dict[str, OperationSpec] = {
    op.id: op for op in iter_operations()
}
OPERATION_SPECS: dict[str, OperationSpec] = _OPERATION_SPECS


def iter_operation_specs() -> list[OperationSpec]:
    """
    Return all registered OperationSpec instances.

    This is now a thin wrapper over the canonical OperationCatalog.
    """
    return list(iter_operations())


def get_operation_spec(op_id: str) -> OperationSpec | None:
    """
    Return a single OperationSpec by id, or None when unknown.
    """
    return get_operation(op_id)
```

**Notes:**

* You keep the **same public symbols**:

  * `OperationSpec`
  * `OPERATION_SPECS`
  * `iter_operation_specs`
  * `get_operation_spec`
* Everything that *used* OperationSpec will now see `Operation` instances with a superset of fields. That’s fine in Python.

### 4.2. Quick regression test

Add a test that ensures the old registry and the catalog are aligned:

```python
# tests/serving/test_operation_catalog_alignment.py
from codeintel.serving.operations import get_operation, iter_operations
from codeintel.serving.registry import OPERATION_SPECS, iter_operation_specs

def test_registry_and_catalog_agree_on_ids() -> None:
    catalog_ids = {op.id for op in iter_operations()}
    registry_ids = set(OPERATION_SPECS.keys())
    assert catalog_ids == registry_ids

def test_registry_uses_catalog_objects() -> None:
    for spec in iter_operation_specs():
        op = get_operation(spec.id)
        assert op is spec  # same object identity
```

---

## 5. Phase 3 – MCP tools & HTTP routes

Because you preserved `OperationSpec`’s **name and shape**, most of this “just works,” but you can tighten alignment.

### 5.1. HTTP routes: rely on canonical spec (no signature changes)

Example: `serving/http/routes/architecture.py` already does:

```python
from codeintel.serving.registry import OperationSpec, get_operation_spec

def _require_spec(op_id: str) -> OperationSpec:
    spec = get_operation_spec(op_id)
    if spec is None:
        raise RuntimeError(f"Missing OperationSpec for {op_id}")
    return spec
```

Keep that, but now you get the **canonical Operation** underneath.

You can add **assertions** (or tests) that:

* `spec.http_path` matches the route path you’re registering.
* `spec.category` matches the tag you’re using.

For example, in `build_architecture_router`:

```python
def build_architecture_router() -> APIRouter:
    router = APIRouter()

    function_spec = _require_spec("architecture.function")
    assert function_spec.http_path == "/architecture/function"

    router.add_api_route(
        function_spec.http_path,
        function_architecture,
        methods=[function_spec.http_method or "GET"],
        summary=function_spec.summary,
        tags=[function_spec.category],
    )
    ...
```

You don't have to wire the router path/method **from** the spec (they’re already in sync); but these checks make the catalog authoritative.

### 5.2. MCP tools: already driven by OperationSpec

Example: `serving/mcp/function_tools.py`:

```python
from codeintel.serving.registry import OperationSpec, iter_operation_specs

FUNCTION_TOOL_CATEGORIES = {"functions", "graph"}

def register_function_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    for spec in iter_operation_specs():
        if spec.category not in FUNCTION_TOOL_CATEGORIES or spec.tool_name is None:
            continue
        tool = _build_function_tool(spec, backend)
        tool.__name__ = spec.tool_name
        tool.__doc__ = spec.description or spec.summary
        mcp.tool(
            name=spec.tool_name,
            description=spec.summary,
        )(tool)
```

No change needed for behavior; it just picks up the canonical `Operation`.

You **can** add simple runtime assertions:

```python
        assert spec.http_method in ("GET", "POST", None)
        assert spec.backend_method.startswith(("get_", "list_"))  # or whatever conventions you enforce
```

And/or additional tests to verify:

* Every `spec.tool_name` is unique.
* Every `spec.tool_name` maps back to exactly one `spec.id`.

---

## 6. Phase 4 – Make `backend/operations.py` a facade on the catalog

Right now `backend/operations.py` defines:

* `DataSourceType` enum
* `OperationContract` dataclass
* A bunch of constants (`FUNCTION_SUMMARY`, `HIGH_RISK_FUNCTIONS`, etc.)
* `OPERATION_CONTRACTS` / `CONTRACTS` dicts

After Phase 1, `DataSourceType` & canonical data live in `serving.operations.catalog`.

### 6.1. Remove local `DataSourceType` & import from operations

In `codeintel/serving/backend/operations.py`:

**Before:**

```python
from dataclasses import dataclass
from enum import StrEnum

class DataSourceType(StrEnum):
    ...
```

**After:**

```python
from __future__ import annotations

from dataclasses import dataclass

from codeintel.serving.operations import DataSourceType, Operation, iter_operations
```

### 6.2. Rework `OperationContract` as a view over `Operation`

Replace the old definition with:

```python
@dataclass(frozen=True)
class OperationContract:
    """
    Backend-centric view of an operation, focused on data sources.

    This is a compatibility wrapper around the canonical Operation.
    """

    name: str
    data_source: DataSourceType
    source_name: str | None
    supports_pagination: bool
    description: str
    repository_method: str | None

    @classmethod
    def from_operation(cls, op: Operation) -> OperationContract:
        return cls(
            name=op.id,
            data_source=op.data_source,
            source_name=op.source_name,
            supports_pagination=op.supports_pagination,
            description=op.description or op.summary,
            repository_method=op.repository_method,
        )
```

Then define the contracts dicts **in terms of the catalog**:

```python
# Canonical mapping from id → OperationContract
OPERATION_CONTRACTS: dict[str, OperationContract] = {
    op.id: OperationContract.from_operation(op) for op in iter_operations()
}

# Backwards-compatible alias kept for now
CONTRACTS: dict[str, OperationContract] = OPERATION_CONTRACTS
```

You can now safely:

* Delete the individual `FUNCTION_SUMMARY`, `HIGH_RISK_FUNCTIONS`, etc. constants.
* Delete the old manual `OPERATION_CONTRACTS = { ... }` block.

Since nothing else imports those constants (only the dict), this won’t break call sites.

### 6.3. Optional: link backend limits directly

If you want, you can later add a helper here to get per-operation limits:

```python
from codeintel.serving.backend.pagination import BackendLimits, LimitClamp

def get_operation_limits(op: Operation, limits: BackendLimits) -> LimitClamp:
    """Derive a LimitClamp suitable for this operation."""
    return LimitClamp(
        default=op.default_limit or limits.default_limit,
        maximum=op.max_limit or limits.max_limit,
    )
```

Then `DuckDBQueryService` can use **both** the canonical Operation and its own BackendLimits to derive clamping behavior.

---

## 7. Phase 5 – Strengthen tests

Now that everything is unified, we can make `tests/serving/test_operation_spec_alignment.py` more powerful.

### 7.1. Ensure every HTTP operation is reachable from a router

Extend `test_operation_spec_alignment.py`:

```python
from fastapi.routing import APIRoute

from codeintel.serving.http.routes.architecture import build_architecture_router
from codeintel.serving.http.routes.datasets import build_datasets_router
from codeintel.serving.http.routes.functions import build_functions_router
from codeintel.serving.http.routes.health import build_health_router
from codeintel.serving.http.routes.ide import build_ide_router
from codeintel.serving.http.routes.profiles import build_profiles_router
from codeintel.serving.http.routes.subsystems import build_subsystems_router
from codeintel.serving.registry import iter_operation_specs

def test_all_http_operations_have_routes() -> None:
    routers = [
        build_architecture_router(),
        build_datasets_router(),
        build_functions_router(),
        build_health_router(),
        build_ide_router(),
        build_profiles_router(),
        build_subsystems_router(),
    ]

    seen: set[tuple[str, str]] = set()
    for router in routers:
        for route in router.routes:
            if not isinstance(route, APIRoute):
                continue
            for method in route.methods:
                seen.add((method.upper(), route.path))

    for spec in iter_operation_specs():
        if spec.http_method is None or spec.http_path is None:
            continue
        pair = (spec.http_method.upper(), spec.http_path)
        assert (
            pair in seen
        ), f"HTTP operation {spec.id} has no matching route: {pair}"
```

### 7.2. Keep & slightly tighten the MCP alignment tests

You already have:

```python
from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.tools_base import QueryBackendOrService, register_tools
from codeintel.serving.registry import iter_operation_specs

def test_mcp_tool_names_match_operation_specs() -> None:
    mcp = FastMCP("test")
    backend = _DummyBackend()
    register_tools(mcp, cast("QueryBackendOrService", backend))
    tools = cast("list[Any]", getattr(mcp, "tools", []))
    tool_names = {cast("str", getattr(tool, "name", "")) for tool in tools}
    tool_names.discard("")

    for spec in iter_operation_specs():
        if spec.tool_name is None:
            continue
        if spec.tool_name not in tool_names:
            pytest.fail(f"MCP tool {spec.tool_name} (spec {spec.id}) not registered")
```

You can add uniqueness checks:

```python
def test_all_operation_tool_names_are_unique() -> None:
    tool_names = [spec.tool_name for spec in iter_operation_specs() if spec.tool_name]
    assert len(tool_names) == len(set(tool_names)), "Duplicate MCP tool names detected"
```

### 7.3. Align `required_datasets` and `source_name` with dataset contracts

Use `config.dataset_contract.DATASET_CONTRACTS_BY_TABLE_KEY` to validate the data contract side:

```python
from codeintel.config.dataset_contract import DATASET_CONTRACTS_BY_TABLE_KEY
from codeintel.serving.operations import iter_operations, DataSourceType

def test_operation_sources_match_dataset_contracts() -> None:
    for op in iter_operations():
        if op.data_source not in {DataSourceType.VIEW, DataSourceType.TABLE}:
            continue
        if op.source_name is None:
            continue

        # Normalize to dataset_contract table_key
        # Your naming already appears like "schema.table"
        table_key = op.source_name
        assert (
            table_key in DATASET_CONTRACTS_BY_TABLE_KEY
        ), f"Operation {op.id} source {table_key} has no DatasetContract"

        # Ensure required_datasets are also valid table_keys
        for dataset in op.required_datasets:
            assert (
                dataset in DATASET_CONTRACTS_BY_TABLE_KEY
            ), f"Operation {op.id} required dataset {dataset} not found in contracts"
```

This ties the **serving** view of datasets to the **storage/config** view, which is exactly the cross-layer robustness you’re aiming for.

---

## Summary of what you get after Epic 3

* A **single, canonical `Operation`** model (`serving/operations/catalog.py`) that:

  * Knows HTTP route/method, MCP tool name, backend method.
  * Carries data contract details (`data_source`, `source_name`, `repository_method`).
  * Encodes which datasets/graphs are required and pagination limits.

* `serving.registry` becomes purely a facade for `OperationCatalog` (plus dataset meta logic):

  * `OperationSpec` = `Operation`
  * `iter_operation_specs` / `get_operation_spec` now just forward to the catalog.

* `backend/operations` becomes a backend-centric **view** over the same operations, no duplicated metadata.

* MCP tool registration and HTTP routes continue to use `OperationSpec`, but they now **implicitly depend on the same canonical source** as the backend.

* Strengthened tests enforcing:

  * HTTP coverage for every HTTP operation.
  * MCP registration for every MCP tool.
  * Consistency between `source_name`/`required_datasets` and `config.dataset_contract`.

