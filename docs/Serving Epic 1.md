Here’s how I’d turn that “extract `serving.backend`” idea into a concrete, implementable refactor, with explicit file-level steps and code you can basically paste in.

I’ll assume the code lives at `src/codeintel/serving/...` as in the zip.

---

## 0. Target shape (what we’re building)

End state for this epic:

```text
codeintel/serving/
  backend/
    __init__.py          # canonical entrypoint
    limits.py            # BackendLimits, ClampResult, clamp_* helpers
    duckdb_service.py    # DuckDBQueryService + helper functions
    datasets.py          # dataset registry helpers (moved from http/datasets.py)

  http/
    datasets.py          # thin compatibility wrapper (or very small shim)
    fastapi.py           # unchanged except for imports

  mcp/
    backend.py           # QueryBackend implementations (unchanged)
    query_service.py     # *shim* that re-exports from serving.backend

  services/
    query_service.py     # now imports from serving.backend
    factory.py           # ditto
    wiring.py            # ditto
```

Data flow:

> storage + analytics → `serving.backend` → `serving.services` → {`serving.http`, `serving.mcp`, future gRPC, CLI, etc.}

---

## 1. Create the `serving.backend` package

**New file**: `src/codeintel/serving/backend/__init__.py`

Purpose: give you a canonical import surface so all other packages can use:

```python
from codeintel.serving.backend import BackendLimits, DuckDBQueryService, clamp_limit_value
```

### Suggested content

```python
"""Transport-agnostic serving backend primitives.

This package hosts the core query backend used by all serving surfaces:

- `DuckDBQueryService` is the DuckDB-backed query runner.
- `BackendLimits` and the clamping helpers centralize safety limits.
- Dataset registry helpers live in `serving.backend.datasets`.
"""

from __future__ import annotations

from .limits import BackendLimits, ClampResult, clamp_limit_value, clamp_offset_value
from .duckdb_service import DuckDBQueryService

__all__ = [
    "BackendLimits",
    "ClampResult",
    "clamp_limit_value",
    "clamp_offset_value",
    "DuckDBQueryService",
]
```

(This is small but important: every other module should import from here unless it needs something very specific.)

---

## 2. Move limits & clamping into `serving.backend.limits`

We’re extracting:

* `BackendLimits`
* `ClampResult`
* `clamp_limit_value`
* `clamp_offset_value`

out of `serving/mcp/query_service.py`.

**New file**: `src/codeintel/serving/backend/limits.py`

### Implementation

Copy the logic from `serving.mcp.query_service` into this file, with only two changes:

1. Different module docstring
2. Imports come from `codeintel.serving.mcp.models` instead of being local

```python
"""Shared safety limits and clamping helpers for serving backends."""

from __future__ import annotations

from dataclasses import dataclass, field

from codeintel.serving.mcp.models import Message


@dataclass(frozen=True)
class BackendLimits:
    """Safety limits applied uniformly across backends."""

    default_limit: int = 50
    max_rows_per_call: int = 500

    @classmethod
    def from_config(cls, cfg: object) -> BackendLimits:
        """
        Build limits from configuration objects exposing default_limit/max_rows_per_call.

        Parameters
        ----------
        cfg :
            Object with optional `default_limit` and `max_rows_per_call` attributes.

        Returns
        -------
        BackendLimits
            Limits derived from the provided configuration.
        """
        default = getattr(cfg, "default_limit", cls.default_limit)
        maximum = getattr(cfg, "max_rows_per_call", cls.max_rows_per_call)
        return cls(default_limit=int(default), max_rows_per_call=int(maximum))


@dataclass(frozen=True)
class ClampResult:
    """Result of clamping limit/offset values with messaging."""

    applied: int
    messages: list[Message] = field(default_factory=list)
    has_error: bool = False


def clamp_limit_value(
    requested: int | None,
    *,
    default: int,
    max_limit: int,
) -> ClampResult:
    """
    Clamp a requested limit to safe bounds, returning warnings instead of raising.

    Parameters
    ----------
    requested:
        Requested limit value; ``None`` means "use default".
    default:
        Default limit to apply when none is requested.
    max_limit:
        Maximum rows allowed for any call.

    Returns
    -------
    ClampResult
        Applied limit plus any informational or error messages.
    """
    messages: list[Message] = []
    limit = default if requested is None else requested

    if limit < 0:
        messages.append(
            Message(
                code="limit_invalid",
                severity="error",
                detail="limit must be non-negative",
                context={"requested": limit},
            )
        )
        return ClampResult(applied=0, messages=messages, has_error=True)

    if limit > max_limit:
        messages.append(
            Message(
                code="limit_clamped",
                severity="warning",
                detail=f"Requested {limit} rows; delivering {max_limit} (max allowed).",
                context={"requested": limit, "applied": max_limit, "max": max_limit},
            )
        )
        limit = max_limit

    return ClampResult(applied=limit, messages=messages, has_error=False)


def clamp_offset_value(offset: int) -> ClampResult:
    """
    Clamp an offset to a non-negative value, returning messaging instead of raising.

    Parameters
    ----------
    offset:
        Requested offset value.

    Returns
    -------
    ClampResult
        Applied offset and any validation messages.
    """
    if offset < 0:
        return ClampResult(
            applied=0,
            messages=[
                Message(
                    code="offset_invalid",
                    severity="error",
                    detail="offset must be non-negative",
                    context={"requested": offset},
                )
            ],
            has_error=True,
        )
    return ClampResult(applied=offset)
```

After adding this file, we’ll remove the duplicate definitions from `mcp/query_service.py` in step 4.

---

## 3. Move dataset registry helpers into `serving.backend.datasets`

Right now, dataset registry / validation lives in `serving/http/datasets.py`. This logic is domain-y, and it’s already shared by HTTP and MCP.

We’ll move it to backend, and then make `http/datasets.py` a thin shim.

**New file**: `src/codeintel/serving/backend/datasets.py`

### Implementation

Take the *body* of `serving/http/datasets.py` and paste into this new file, with two changes:

1. Module docstring + imports switch to backend / limits.
2. The TYPE_CHECKING import for `BackendLimits` points to `serving.backend`.

Concretely:

```python
"""Dataset registry helpers shared by all DuckDB-backed backends."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Literal

from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.storage.gateway import DuckDBConnection, DuckDBError, StorageGateway
from codeintel.storage.views import DOCS_VIEWS as GATEWAY_DOCS_VIEWS

if TYPE_CHECKING:
    # NOTE: import from backend, not MCP
    from codeintel.serving.backend import BackendLimits

DOCS_VIEWS = {view.split(".", maxsplit=1)[1]: view for view in GATEWAY_DOCS_VIEWS}

PREVIEW_COLUMN_COUNT = 5

# ... paste the rest of the helpers:
# - _normalize_dataset_mapping(...)
# - _describe_docs_view(...)
# - describe_dataset(...)
# - _macro_failure_message(...)
# - _collect_dataset_registry_issues(...)
# - build_registry_and_limits(...)
# - validate_dataset_registry(...)
#
# No functional changes needed, just keep them identical.
```

You don’t need to change the function bodies; they are already generic.

---

## 4. Make `serving/http/datasets.py` a thin shim

Now that the real implementations live in `serving.backend.datasets`, we keep HTTP’s module mostly for backwards compatibility and clearer layering.

**Edit**: `src/codeintel/serving/http/datasets.py`

Replace the existing body with:

```python
"""HTTP-facing shim around the shared backend dataset helpers.

Preferred import path for new code:
    `from codeintel.serving.backend.datasets import ...`.
"""

from __future__ import annotations

from codeintel.serving.backend.datasets import (
    DOCS_VIEWS,
    PREVIEW_COLUMN_COUNT,
    build_registry_and_limits,
    describe_dataset,
    validate_dataset_registry,
)

__all__ = [
    "DOCS_VIEWS",
    "PREVIEW_COLUMN_COUNT",
    "build_registry_and_limits",
    "describe_dataset",
    "validate_dataset_registry",
]
```

(If you have any additional helpers in the original file, add them here as well. The goal is: no logic, just re-exports.)

---

## 5. Move `DuckDBQueryService` into `serving.backend.duckdb_service`

This is the main event.

We’re going to:

1. Create `serving/backend/duckdb_service.py` that contains:

   * helper functions: `_fetch_duckdb_schema`, `_load_json_schema`, `_normalize_validation_profile`
   * `@dataclass class DuckDBQueryService` with *all* its methods (function summaries, subsystem APIs, dataset specs/schema, graph plugin listing, etc.)
2. Strip those definitions out of `serving/mcp/query_service.py`.
3. Have `mcp/query_service.py` re-export from backend for backwards compatibility.

### 5.1 New module scaffold

**New file**: `src/codeintel/serving/backend/duckdb_service.py`

Skeleton:

```python
"""DuckDB-backed query service used by all serving surfaces.

This is the transport-agnostic core; HTTP and MCP treat it as a backend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, cast

import networkx as nx

from codeintel.analytics.graphs.plugins import list_graph_metric_plugins
from codeintel.config.steps_graphs import GraphRunScope
from codeintel.graphs.engine import GraphEngine
from codeintel.serving.backend.limits import (
    BackendLimits,
    ClampResult,
    clamp_limit_value,
    clamp_offset_value,
)
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import (
    CallGraphEdgeRow,
    CallGraphNeighborsResponse,
    DatasetRowsResponse,
    DatasetSchemaColumn,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    FileHintsResponse,
    FileProfileResponse,
    FileSummaryResponse,
    FileSummaryRow,
    FunctionArchitectureResponse,
    FunctionProfileResponse,
    FunctionSummaryResponse,
    FunctionSummaryRow,
    GraphNeighborhoodResponse,
    GraphPluginDescriptor,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    Message,
    ModuleArchitectureResponse,
    ModuleArchitectureRow,
    ModuleProfileResponse,
    ModuleSubsystemResponse,
    ModuleWithSubsystemRow,
    ResponseMeta,
    SubsystemCoverageResponse,
    SubsystemCoverageRow,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemProfileRow,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
    SubsystemSummaryRow,
    TestsForFunctionResponse,
    ViewRow,
)
from codeintel.serving.mcp.view_utils import (
    normalize_entrypoints_row,
    normalize_entrypoints_rows,
)
from codeintel.storage.contract_validation import _schema_path
from codeintel.storage.datasets import (
    Dataset,
    dataset_for_name,
    list_dataset_specs,
    load_dataset_registry,
)
from codeintel.storage.gateway import DuckDBConnection, StorageGateway
from codeintel.storage.repositories import (
    DatasetReadRepository,
    FunctionRepository,
    GraphRepository,
    ModuleRepository,
    SubsystemRepository,
    TestRepository,
)

# --- helpers originally at top of serving.mcp.query_service ------------------


def _fetch_duckdb_schema(con: DuckDBConnection, table_key: str) -> list[DatasetSchemaColumn]:
    """
    Return column descriptors for a DuckDB table/view.

    Parameters
    ----------
    con
        DuckDB connection.
    table_key
        Fully qualified table/view name.

    Returns
    -------
    list[DatasetSchemaColumn]
        Column descriptors derived from information_schema.
    """
    rows = con.execute(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = split_part(?, '.', 2)
        ORDER BY ordinal_position
        """,
        [table_key],
    ).fetchall()
    return [
        DatasetSchemaColumn(
            name=row[0],
            duckdb_type=row[1],
            is_nullable=bool(row[2]),
        )
        for row in rows
    ]


def _load_json_schema(ds: Dataset) -> dict[str, object] | None:
    """
    Load a stored JSON Schema for the dataset when available.

    Returns
    -------
    dict[str, object] | None
        Parsed JSON Schema or ``None`` when not present.
    """
    if ds.json_schema_path is None:
        return None

    path = _schema_path(ds.json_schema_path)
    return json.loads(path.read_text(encoding="utf8"))


def _normalize_validation_profile(
    value: object | None,
) -> Literal["strict", "lenient"] | None:
    """
    Restrict validation profile to supported literals.

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


# --- core DuckDB-backed query service ---------------------------------------


@dataclass  # noqa: PLR0904
class DuckDBQueryService:
    """Shared query runner for DuckDB-backed MCP and HTTP surfaces."""

    gateway: StorageGateway
    repo: str
    commit: str
    limits: BackendLimits
    graph_engine: GraphEngine | None = None

    _engine: GraphEngine | None = field(default=None, init=False, repr=False)
    _functions: FunctionRepository | None = field(default=None, init=False, repr=False)
    _modules: ModuleRepository | None = field(default=None, init=False, repr=False)
    _subsystems: SubsystemRepository | None = field(default=None, init=False, repr=False)
    _tests: TestRepository | None = field(default=None, init=False, repr=False)
    _datasets: DatasetReadRepository | None = field(default=None, init=False, repr=False)
    _graphs: GraphRepository | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Paste the *entire* body of DuckDBQueryService from
    # `serving/mcp/query_service.py` here, unchanged, including:
    #
    # - con property
    # - _require_graph_engine / _ensure_* helpers
    # - .functions / .modules / .subsystems / .tests / .datasets / .graphs
    # - all public methods:
    #   * list_high_risk_functions, get_tests_for_function, ...
    #   * get_callgraph_neighborhood, get_import_boundary, ...
    #   * get_file_summary, get_function_profile, get_file_profile, ...
    #   * get_module_profile, get_function_architecture, ...
    #   * list_subsystems, get_module_subsystems, get_file_hints, ...
    #   * get_subsystem_modules, search_subsystems, summarize_subsystem, ...
    #   * dataset_specs, dataset_schema, read_dataset_rows, ...
    #   * list_graph_plugins
    #
    # No logic changes needed – only the imports moved to this module.
    # ------------------------------------------------------------------

    # def con(self) -> DuckDBConnection: ...
    # def _require_graph_engine(self) -> GraphEngine: ...
    # def list_high_risk_functions(...): ...
    # def dataset_specs(...): ...
    # def dataset_schema(...): ...
    # def read_dataset_rows(...): ...
    # def list_graph_plugins(...): ...
```

To actually implement, you’ll:

1. Cut the `@dataclass  # noqa: PLR0904` … `class DuckDBQueryService:` block and *all its methods* from `serving/mcp/query_service.py`.
2. Paste that block under the `DuckDBQueryService` stub above, replacing the commented “Paste…” section.
3. Remove the old helper definitions `_fetch_duckdb_schema`, `_load_json_schema`, `_normalize_validation_profile` from `serving/mcp/query_service.py` and paste them into the new module above the class.

At this point, **the only things left in `serving/mcp/query_service.py` should be imports and the new shim (next step).**

---

## 6. Turn `serving/mcp/query_service.py` into a shim

Now that backend owns the implementation, we want MCP to be a *consumer*, not the owner.

**Edit**: `src/codeintel/serving/mcp/query_service.py`

Delete the body and replace with:

```python
"""Backwards-compat shim for the DuckDB query backend.

The canonical implementation now lives in `codeintel.serving.backend`.
This module re-exports the APIs for compatibility with existing imports.
"""

from __future__ import annotations

from codeintel.serving.backend import (
    BackendLimits,
    ClampResult,
    DuckDBQueryService,
    clamp_limit_value,
    clamp_offset_value,
)

__all__ = [
    "BackendLimits",
    "ClampResult",
    "DuckDBQueryService",
    "clamp_limit_value",
    "clamp_offset_value",
]
```

If you want, you can also keep a short comment at the top of the new backend modules referring back to this decision (helps future-you).

---

## 7. Update internal imports to use `serving.backend`

Now we fix all modules that currently import from `serving.mcp.query_service`.

### 7.1 `serving/services/query_service.py`

**Before** (near the top):

```python
from codeintel.serving.http.datasets import describe_dataset
from codeintel.serving.mcp.models import (
    ...
)
from codeintel.serving.mcp.query_service import (
    BackendLimits,
    DuckDBQueryService,
    clamp_limit_value,
    clamp_offset_value,
)
```

**After**:

```python
from codeintel.serving.backend import (
    BackendLimits,
    DuckDBQueryService,
    clamp_limit_value,
    clamp_offset_value,
)
from codeintel.serving.backend.datasets import describe_dataset
from codeintel.serving.mcp.models import (
    ...
)
```

Key points:

* `BackendLimits` & clamping now come from `serving.backend`.
* `describe_dataset` now comes from `serving.backend.datasets` (or you can keep the http shim if you prefer; I’d favor backend).

No other changes needed; the rest of `HttpQueryService` / `LocalQueryService` will work as-is.

### 7.2 `serving/services/factory.py`

**Before**:

```python
from codeintel.serving.http.datasets import (
    build_registry_and_limits,
    describe_dataset,
    validate_dataset_registry,
)
from codeintel.serving.mcp.query_service import BackendLimits, DuckDBQueryService
```

**After**:

```python
from codeintel.serving.backend import BackendLimits, DuckDBQueryService
from codeintel.serving.backend.datasets import (
    build_registry_and_limits,
    describe_dataset,
    validate_dataset_registry,
)
```

Everything else in this module (building GraphRuntime, `build_local_query_service`, etc.) remains unchanged.

### 7.3 `serving/services/wiring.py`

**Before**:

```python
from codeintel.serving.http.datasets import build_registry_and_limits
from codeintel.serving.mcp.query_service import BackendLimits
from codeintel.serving.services.query_service import (
    LocalQueryService,
    QueryService,
    ServiceObservability,
)
```

**After**:

```python
from codeintel.serving.backend import BackendLimits
from codeintel.serving.backend.datasets import build_registry_and_limits
from codeintel.serving.services.query_service import (
    LocalQueryService,
    QueryService,
    ServiceObservability,
)
```

### 7.4 `serving/mcp/backend.py`

**Before**:

```python
from codeintel.serving.mcp.query_service import BackendLimits, DuckDBQueryService
from codeintel.serving.services.factory import (
    ServiceBuildOptions,
    build_service_from_config,
    get_observability_from_config,
)
```

**After**:

```python
from codeintel.serving.backend import BackendLimits, DuckDBQueryService
from codeintel.serving.services.factory import (
    ServiceBuildOptions,
    build_service_from_config,
    get_observability_from_config,
)
```

Everything else in this file (DuckDBBackend, HttpBackend, DatasetBackendMixin, direction normalization) stays the same; they just point to the new canonical location for the query backend.

### 7.5 `serving/http/datasets.py` (TYPE_CHECKING)

If you kept any TYPE_CHECKING import referencing `codeintel.serving.mcp.query_service.BackendLimits`, switch it to:

```python
if TYPE_CHECKING:
    from codeintel.serving.backend import BackendLimits
```

If you simplified `http/datasets.py` to just a shim (as in step 4), you may not need this at all.

---

## 8. Update tests to the new import path

Now fix tests that import from `codeintel.serving.mcp.query_service`. From the zip, those are:

* `tests/services/test_query_service.py`
* `tests/services/test_backend_limits.py`
* `tests/services/test_typed_docs_responses.py`
* `tests/services/test_subsystem_roundtrip.py`
* `tests/serving/test_dataset_specs.py`
* `tests/mcp/test_registry_limits_parity.py`

### Example updates

#### 8.1 `tests/services/test_backend_limits.py`

**Before**:

```python
from codeintel.serving.mcp.query_service import BackendLimits, clamp_limit_value
from codeintel.serving.services.query_service import HttpQueryService
```

**After**:

```python
from codeintel.serving.backend import BackendLimits, clamp_limit_value
from codeintel.serving.services.query_service import HttpQueryService
```

No other logic changes required.

#### 8.2 `tests/serving/test_dataset_specs.py`

**Before**:

```python
from codeintel.serving.mcp.query_service import BackendLimits, DuckDBQueryService
from codeintel.storage.datasets import DEFAULT_JSONL_FILENAMES
from codeintel.storage.gateway import open_memory_gateway
```

**After**:

```python
from codeintel.serving.backend import BackendLimits, DuckDBQueryService
from codeintel.storage.datasets import DEFAULT_JSONL_FILENAMES
from codeintel.storage.gateway import open_memory_gateway
```

The rest of the test—which calls `DuckDBQueryService.dataset_specs()` etc.—should still pass.

#### 8.3 Same pattern for the others

Everywhere you see:

```python
from codeintel.serving.mcp.query_service import ...
```

change it to:

```python
from codeintel.serving.backend import ...
```

The shim in `mcp/query_service.py` means old imports would technically still work, but switching tests to the new path makes it clear what the canonical direction is.

---

## 9. Sanity checks & sequencing

I’d apply this in roughly this order to keep things smooth:

1. **Add `serving/backend/limits.py`** and **`serving/backend/__init__.py`**.
2. **Move dataset helpers** into `serving/backend/datasets.py` and convert `serving/http/datasets.py` into a shim.
3. **Add `serving/backend/duckdb_service.py`**:

   * copy helpers and `DuckDBQueryService` from `serving/mcp/query_service.py`.
4. **Turn `serving/mcp/query_service.py` into the shim** that re-exports from backend.
5. **Update imports** in:

   * `serving/services/query_service.py`
   * `serving/services/factory.py`
   * `serving/services/wiring.py`
   * `serving/mcp/backend.py`
6. **Fix tests** to import from `codeintel.serving.backend`.
7. Run:

   * `pytest tests/services tests/serving tests/mcp -q`
   * your layering check `tests/test_layering_serving_imports.py`
   * plus any pre-existing CI commands (ruff, pyright, pyrefly) to make sure imports and types are happy.

---

If you’d like, next we can:

* Take one specific file (e.g. `serving/backend/duckdb_service.py`) and I can produce a more concrete “before → after” diff snippet for the top of the class and one or two key methods, so an agent can apply it mechanically across your repo.

# zoom in on specific section to showcase methodology #

Perfect, let’s make this super concrete and mechanical.

Below I’ll:

1. Show what the **code looked like before** in `serving/mcp/query_service.py`.
2. Show what the **code should look like after** in the new `serving/backend/duckdb_service.py`.
3. Use the **top of the file**, the **class header + graph engine helper**, and two key methods (`list_high_risk_functions`, `dataset_specs`/`dataset_schema`) as the pattern.

Key idea: for `DuckDBQueryService` and its methods, you **copy the code verbatim** into `serving/backend/duckdb_service.py`. The only real change is that the **limits types/functions move to `serving.backend.limits` and are imported** instead of defined inline.

---

## 1. Top of module – BEFORE vs AFTER

### BEFORE

`serving/mcp/query_service.py` (simplified to just the relevant parts)

```python
"""
Shared DuckDB query service used by MCP backends and FastAPI surface.

All SQL queries against docs.* and analytics.* views/tables live here.
Other modules must call this service (via LocalQueryService/QueryService)
instead of issuing custom SELECTs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, cast

import networkx as nx

from codeintel.analytics.graphs.plugins import list_graph_metric_plugins
from codeintel.config.steps_graphs import GraphRunScope
from codeintel.graphs.engine import GraphEngine
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import (
    CallGraphEdgeRow,
    CallGraphNeighborsResponse,
    DatasetRowsResponse,
    DatasetSchemaColumn,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    FileHintsResponse,
    FileProfileResponse,
    FileSummaryResponse,
    FileSummaryRow,
    FunctionArchitectureResponse,
    FunctionProfileResponse,
    FunctionSummaryResponse,
    FunctionSummaryRow,
    GraphNeighborhoodResponse,
    GraphPluginDescriptor,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    Message,
    ModuleArchitectureResponse,
    ModuleArchitectureRow,
    ModuleProfileResponse,
    ModuleSubsystemResponse,
    ModuleWithSubsystemRow,
    ResponseMeta,
    SubsystemCoverageResponse,
    SubsystemCoverageRow,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemProfileRow,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
    SubsystemSummaryRow,
    TestsForFunctionResponse,
    ViewRow,
)
from codeintel.serving.mcp.view_utils import (
    normalize_entrypoints_row,
    normalize_entrypoints_rows,
)
from codeintel.storage.contract_validation import _schema_path
from codeintel.storage.datasets import (
    Dataset,
    dataset_for_name,
    list_dataset_specs,
    load_dataset_registry,
)
from codeintel.storage.gateway import DuckDBConnection, StorageGateway
from codeintel.storage.repositories import (
    DatasetReadRepository,
    FunctionRepository,
    GraphRepository,
    ModuleRepository,
    SubsystemRepository,
    TestRepository,
)

# --- limits & clamping lived here before -------------------------------------

@dataclass(frozen=True)
class BackendLimits:
    """Safety limits applied uniformly across backends."""

    default_limit: int = 50
    max_rows_per_call: int = 500

    @classmethod
    def from_config(cls, cfg: object) -> BackendLimits:
        """
        Build limits from configuration objects exposing default_limit/max_rows_per_call.
        """
        default = getattr(cfg, "default_limit", cls.default_limit)
        maximum = getattr(cfg, "max_rows_per_call", cls.max_rows_per_call)
        return cls(default_limit=int(default), max_rows_per_call=int(maximum))


@dataclass(frozen=True)
class ClampResult:
    """Result of clamping limit/offset values with messaging."""

    applied: int
    messages: list[Message] = field(default_factory=list)
    has_error: bool = False


def clamp_limit_value(
    requested: int | None,
    *,
    default: int,
    max_limit: int,
) -> ClampResult:
    """
    Clamp a requested limit to safe bounds, returning warnings instead of errors.
    """
    messages: list[Message] = []
    limit = default if requested is None else requested

    if limit < 0:
        messages.append(
            Message(
                code="limit_invalid",
                severity="error",
                detail="limit must be non-negative",
                context={"requested": limit},
            )
        )
        return ClampResult(applied=0, messages=messages, has_error=True)

    if limit > max_limit:
        messages.append(
            Message(
                code="limit_clamped",
                severity="warning",
                detail=f"Requested {limit} rows; delivering {max_limit} (max allowed).",
                context={"requested": limit, "applied": max_limit, "max": max_limit},
            )
        )
        limit = max_limit

    return ClampResult(applied=limit, messages=messages, has_error=False)


def clamp_offset_value(offset: int) -> ClampResult:
    """
    Clamp an offset to a non-negative value, returning messaging instead of raising.
    """
    if offset < 0:
        return ClampResult(
            applied=0,
            messages=[
                Message(
                    code="offset_invalid",
                    severity="error",
                    detail="offset must be non-negative",
                    context={"requested": offset},
                )
            ],
            has_error=True,
        )
    return ClampResult(applied=offset)

# ... then def _fetch_duckdb_schema, _load_json_schema, _normalize_validation_profile,
#     and finally class DuckDBQueryService ...
```

### AFTER

New file: `serving/backend/duckdb_service.py`

> Pattern:
> **Imports are basically the same**, except:
>
> * We import `BackendLimits`, `ClampResult`, `clamp_limit_value`, `clamp_offset_value` from `serving.backend.limits`.
> * We no longer define those types/functions in this file.

```python
"""
DuckDB-backed query service used by all serving surfaces.

All SQL queries against docs.* and analytics.* views/tables live here.
Other modules must call this service (via LocalQueryService/QueryService)
instead of issuing custom SELECTs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, cast

import networkx as nx

from codeintel.analytics.graphs.plugins import list_graph_metric_plugins
from codeintel.config.steps_graphs import GraphRunScope
from codeintel.graphs.engine import GraphEngine
from codeintel.serving.backend.limits import (
    BackendLimits,
    ClampResult,
    clamp_limit_value,
    clamp_offset_value,
)
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import (
    CallGraphEdgeRow,
    CallGraphNeighborsResponse,
    DatasetRowsResponse,
    DatasetSchemaColumn,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    FileHintsResponse,
    FileProfileResponse,
    FileSummaryResponse,
    FileSummaryRow,
    FunctionArchitectureResponse,
    FunctionProfileResponse,
    FunctionSummaryResponse,
    FunctionSummaryRow,
    GraphNeighborhoodResponse,
    GraphPluginDescriptor,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    Message,
    ModuleArchitectureResponse,
    ModuleArchitectureRow,
    ModuleProfileResponse,
    ModuleSubsystemResponse,
    ModuleWithSubsystemRow,
    ResponseMeta,
    SubsystemCoverageResponse,
    SubsystemCoverageRow,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemProfileRow,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
    SubsystemSummaryRow,
    TestsForFunctionResponse,
    ViewRow,
)
from codeintel.serving.mcp.view_utils import (
    normalize_entrypoints_row,
    normalize_entrypoints_rows,
)
from codeintel.storage.contract_validation import _schema_path
from codeintel.storage.datasets import (
    Dataset,
    dataset_for_name,
    list_dataset_specs,
    load_dataset_registry,
)
from codeintel.storage.gateway import DuckDBConnection, StorageGateway
from codeintel.storage.repositories import (
    DatasetReadRepository,
    FunctionRepository,
    GraphRepository,
    ModuleRepository,
    SubsystemRepository,
    TestRepository,
)

# NOTE: BackendLimits / ClampResult / clamp_* no longer defined here.
# They live in `serving.backend.limits` and are imported above.
```

Then you **copy-paste** these helper functions verbatim from the old file:

```python
def _fetch_duckdb_schema(con: DuckDBConnection, table_key: str) -> list[DatasetSchemaColumn]:
    """
    Return column descriptors for a DuckDB table/view.
    """
    if "." not in table_key:
        return []
    schema_name, table_name = table_key.split(".", maxsplit=1)
    rows = con.execute(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ?
        ORDER BY ordinal_position
        """,
        [schema_name, table_name],
    ).fetchall()
    return [
        DatasetSchemaColumn(
            name=str(col_name),
            type=str(col_type),
            nullable=str(nullable).upper() == "YES",
        )
        for col_name, col_type, nullable in rows
    ]


def _load_json_schema(ds: Dataset) -> dict[str, object] | None:
    """
    Load a JSON Schema document for a dataset if present on disk.
    """
    if ds.json_schema_id is None:
        return None
    schema_path = _schema_path(ds.json_schema_id)
    if not schema_path.exists():
        return None
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _normalize_validation_profile(
    value: str | None,
) -> Literal["strict", "lenient"] | None:
    """
    Restrict validation profile to supported literals.
    """
    if value == "strict":
        return "strict"
    if value == "lenient":
        return "lenient"
    return None
```

These are **byte-for-byte identical** to the originals; the only change is the module they live in.

---

## 2. DuckDBQueryService header + graph engine helper

### BEFORE (in `serving/mcp/query_service.py`)

```python
@dataclass  # noqa: PLR0904
class DuckDBQueryService:
    """Shared query runner for DuckDB-backed MCP and FastAPI surfaces."""

    gateway: StorageGateway
    repo: str
    commit: str
    limits: BackendLimits
    graph_engine: GraphEngine | None = None
    _engine: GraphEngine | None = field(default=None, init=False, repr=False)
    _functions: FunctionRepository | None = field(default=None, init=False, repr=False)
    _modules: ModuleRepository | None = field(default=None, init=False, repr=False)
    _subsystems: SubsystemRepository | None = field(default=None, init=False, repr=False)
    _tests: TestRepository | None = field(default=None, init=False, repr=False)
    _datasets: DatasetReadRepository | None = field(default=None, init=False, repr=False)
    _graphs: GraphRepository | None = field(default=None, init=False, repr=False)

    @property
    def con(self) -> DuckDBConnection:
        """Underlying DuckDB connection."""
        return self.gateway.con

    def _require_graph_engine(self) -> GraphEngine:
        """
        Return the configured graph engine or raise when missing.
        """
        if self._engine is not None:
            return self._engine
        if self.graph_engine is None:
            message = "Graph engine must be provided to DuckDBQueryService."
            raise errors.backend_failure(message)
        self._engine = self.graph_engine
        return self._engine

    @property
    def functions(self) -> FunctionRepository:
        """Lazily construct a function repository."""
        if self._functions is None:
            self._functions = FunctionRepository(self.gateway, self.repo, self.commit)
        return self._functions

    @property
    def modules(self) -> ModuleRepository:
        """Lazily construct a module repository."""
        if self._modules is None:
            self._modules = ModuleRepository(self.gateway, self.repo, self.commit)
        return self._modules

    @property
    def subsystems(self) -> SubsystemRepository:
        """Lazily construct a subsystem repository."""
        if self._subsystems is None:
            self._subsystems = SubsystemRepository(self.gateway, self.repo, self.commit)
        return self._subsystems

    @property
    def tests(self) -> TestRepository:
        """Lazily construct a test repository."""
        if self._tests is None:
            self._tests = TestRepository(self.gateway, self.repo, self.commit)
        return self._tests

    @property
    def datasets(self) -> DatasetReadRepository:
        """Lazily construct a dataset repository."""
        if self._datasets is None:
            self._datasets = DatasetReadRepository(self.gateway, self.repo, self.commit)
        return self._datasets

    @property
    def graphs(self) -> GraphRepository:
        """Lazily construct a graph repository."""
        if self._graphs is None:
            self._graphs = GraphRepository(self.gateway, self.repo, self.commit)
        return self._graphs
```

### AFTER (in `serving/backend/duckdb_service.py`)

> Pattern: **identical code, just a new module**. You can optionally tweak the docstring if you want it to say “serving surfaces”, but that’s cosmetic.

```python
@dataclass  # noqa: PLR0904
class DuckDBQueryService:
    """Shared query runner for DuckDB-backed MCP and FastAPI surfaces."""

    gateway: StorageGateway
    repo: str
    commit: str
    limits: BackendLimits
    graph_engine: GraphEngine | None = None
    _engine: GraphEngine | None = field(default=None, init=False, repr=False)
    _functions: FunctionRepository | None = field(default=None, init=False, repr=False)
    _modules: ModuleRepository | None = field(default=None, init=False, repr=False)
    _subsystems: SubsystemRepository | None = field(default=None, init=False, repr=False)
    _tests: TestRepository | None = field(default=None, init=False, repr=False)
    _datasets: DatasetReadRepository | None = field(default=None, init=False, repr=False)
    _graphs: GraphRepository | None = field(default=None, init=False, repr=False)

    @property
    def con(self) -> DuckDBConnection:
        """Underlying DuckDB connection."""
        return self.gateway.con

    def _require_graph_engine(self) -> GraphEngine:
        """
        Return the configured graph engine or raise when missing.
        """
        if self._engine is not None:
            return self._engine
        if self.graph_engine is None:
            message = "Graph engine must be provided to DuckDBQueryService."
            raise errors.backend_failure(message)
        self._engine = self.graph_engine
        return self._engine

    @property
    def functions(self) -> FunctionRepository:
        """Lazily construct a function repository."""
        if self._functions is None:
            self._functions = FunctionRepository(self.gateway, self.repo, self.commit)
        return self._functions

    @property
    def modules(self) -> ModuleRepository:
        """Lazily construct a module repository."""
        if self._modules is None:
            self._modules = ModuleRepository(self.gateway, self.repo, self.commit)
        return self._modules

    @property
    def subsystems(self) -> SubsystemRepository:
        """Lazily construct a subsystem repository."""
        if self._subsystems is None:
            self._subsystems = SubsystemRepository(self.gateway, self.repo, self.commit)
        return self._subsystems

    @property
    def tests(self) -> TestRepository:
        """Lazily construct a test repository."""
        if self._tests is None:
            self._tests = TestRepository(self.gateway, self.repo, self.commit)
        return self._tests

    @property
    def datasets(self) -> DatasetReadRepository:
        """Lazily construct a dataset repository."""
        if self._datasets is None:
            self._datasets = DatasetReadRepository(self.gateway, self.repo, self.commit)
        return self._datasets

    @property
    def graphs(self) -> GraphRepository:
        """Lazily construct a graph repository."""
        if self._graphs is None:
            self._graphs = GraphRepository(self.gateway, self.repo, self.commit)
        return self._graphs
```

So for the class itself and its helpers, the **mechanical pattern** is:

> *Cut from `serving/mcp/query_service.py`, paste into `serving/backend/duckdb_service.py`, no internal changes.*

---

## 3. Example method 1 – list_high_risk_functions

### BEFORE

```python
def list_high_risk_functions(
    self,
    *,
    min_risk: float = 0.7,
    limit: int | None = None,
    tested_only: bool = False,
    scope: GraphRunScope | None = None,
) -> HighRiskFunctionsResponse:
    """
    List high-risk functions using analytics.goid_risk_factors.

    Parameters
    ----------
    min_risk:
        Minimum risk score threshold.
    limit:
        Maximum number of rows to return.
    tested_only:
        When True, restrict to functions with test coverage.
    scope :
        Optional scope applied to upstream graph execution (unused in lookup).

    Returns
    -------
    HighRiskFunctionsResponse
        Functions, truncation flag, and metadata.
    """
    _ = scope
    applied_limit = self.limits.default_limit if limit is None else limit
    clamp = clamp_limit_value(
        applied_limit,
        default=applied_limit,
        max_limit=self.limits.max_rows_per_call,
    )
    meta = ResponseMeta(
        requested_limit=limit,
        applied_limit=clamp.applied,
        messages=list(clamp.messages),
    )
    if clamp.has_error:
        return HighRiskFunctionsResponse(functions=[], truncated=False, meta=meta)

    _ = scope
    rows = self.functions.list_high_risk_functions(
        min_risk=min_risk,
        limit=clamp.applied,
        tested_only=tested_only,
    )
    models = [ViewRow.model_validate(r) for r in rows]
    truncated = clamp.applied > 0 and len(rows) == clamp.applied
    meta.truncated = truncated
    return HighRiskFunctionsResponse(functions=models, truncated=truncated, meta=meta)
```

### AFTER

> Again: **identical body**, just living in `serving/backend/duckdb_service.py` and using `clamp_limit_value` imported from `serving.backend.limits`.

```python
def list_high_risk_functions(
    self,
    *,
    min_risk: float = 0.7,
    limit: int | None = None,
    tested_only: bool = False,
    scope: GraphRunScope | None = None,
) -> HighRiskFunctionsResponse:
    """
    List high-risk functions using analytics.goid_risk_factors.
    """
    _ = scope
    applied_limit = self.limits.default_limit if limit is None else limit
    clamp = clamp_limit_value(
        applied_limit,
        default=applied_limit,
        max_limit=self.limits.max_rows_per_call,
    )
    meta = ResponseMeta(
        requested_limit=limit,
        applied_limit=clamp.applied,
        messages=list(clamp.messages),
    )
    if clamp.has_error:
        return HighRiskFunctionsResponse(functions=[], truncated=False, meta=meta)

    _ = scope
    rows = self.functions.list_high_risk_functions(
        min_risk=min_risk,
        limit=clamp.applied,
        tested_only=tested_only,
    )
    models = [ViewRow.model_validate(r) for r in rows]
    truncated = clamp.applied > 0 and len(rows) == clamp.applied
    meta.truncated = truncated
    return HighRiskFunctionsResponse(functions=models, truncated=truncated, meta=meta)
```

**Pattern** for other methods (e.g. callgraph neighbors, file summaries, architecture methods):

* Keep the method signature and body **exactly as-is**.
* The method continues to call:

  * `self.functions`, `self.modules`, etc. for repos.
  * `clamp_limit_value` / `clamp_offset_value` imported from backend.limits.
  * Pydantic models from `serving.mcp.models`.

---

## 4. Example method 2 – dataset_specs + dataset_schema

These show the interplay with `_normalize_validation_profile` and `_load_json_schema`, so they’re a good pattern for any DB-ish helpers.

### BEFORE

```python
def dataset_specs(self) -> list[DatasetSpecDescriptor]:
    """
    Return dataset contract entries for the active gateway.
    """
    registry = load_dataset_registry(self.gateway.con)
    specs = list_dataset_specs(registry)
    sorted_specs = sorted(specs, key=lambda spec: cast("str", spec["name"]))
    results: list[DatasetSpecDescriptor] = []
    for spec in sorted_specs:
        normalized: dict[str, object] = dict(spec)
        normalized["schema_columns"] = list(cast("list[str]", spec["schema_columns"]))
        normalized["upstream_dependencies"] = list(
            cast("list[str]", spec.get("upstream_dependencies", []))
        )
        normalized["capabilities"] = dict(cast("dict[str, bool]", spec.get("capabilities", {})))
        normalized["validation_profile"] = _normalize_validation_profile(
            cast("str | None", spec.get("validation_profile"))
        )
        results.append(DatasetSpecDescriptor.model_validate(normalized))
    return results


def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> DatasetSchemaResponse:
    """
    Return a composite schema description for a dataset.
    """
    registry = load_dataset_registry(self.gateway.con)
    try:
        ds = dataset_for_name(registry, dataset_name)
    except KeyError as exc:
        message = f"Unknown dataset: {dataset_name}"
        raise errors.not_found(message) from exc
    duckdb_schema = _fetch_duckdb_schema(self.gateway.con, ds.table_key)
    sample_rows = self.datasets.read_dataset_rows(
        table_key=ds.table_key,
        limit=sample_limit,
        offset=0,
    )
    return DatasetSchemaResponse(
        dataset=dataset_name,
        table_key=ds.table_key,
        duckdb_schema=duckdb_schema,
        json_schema=_load_json_schema(ds),
        sample_rows=[ViewRow.model_validate(row) for row in sample_rows],
        capabilities=ds.capabilities(),
        owner=ds.owner,
        freshness_sla=ds.freshness_sla,
        retention_policy=ds.retention_policy,
        schema_version=ds.schema_version,
        stable_id=ds.stable_id,
        validation_profile=_normalize_validation_profile(ds.validation_profile),
    )
```

### AFTER

Same story: copied into `serving/backend/duckdb_service.py` unchanged:

```python
def dataset_specs(self) -> list[DatasetSpecDescriptor]:
    """
    Return dataset contract entries for the active gateway.
    """
    registry = load_dataset_registry(self.gateway.con)
    specs = list_dataset_specs(registry)
    sorted_specs = sorted(specs, key=lambda spec: cast("str", spec["name"]))
    results: list[DatasetSpecDescriptor] = []
    for spec in sorted_specs:
        normalized: dict[str, object] = dict(spec)
        normalized["schema_columns"] = list(cast("list[str]", spec["schema_columns"]))
        normalized["upstream_dependencies"] = list(
            cast("list[str]", spec.get("upstream_dependencies", []))
        )
        normalized["capabilities"] = dict(cast("dict[str, bool]", spec.get("capabilities", {})))
        normalized["validation_profile"] = _normalize_validation_profile(
            cast("str | None", spec.get("validation_profile"))
        )
        results.append(DatasetSpecDescriptor.model_validate(normalized))
    return results


def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> DatasetSchemaResponse:
    """
    Return a composite schema description for a dataset.
    """
    registry = load_dataset_registry(self.gateway.con)
    try:
        ds = dataset_for_name(registry, dataset_name)
    except KeyError as exc:
        message = f"Unknown dataset: {dataset_name}"
        raise errors.not_found(message) from exc
    duckdb_schema = _fetch_duckdb_schema(self.gateway.con, ds.table_key)
    sample_rows = self.datasets.read_dataset_rows(
        table_key=ds.table_key,
        limit=sample_limit,
        offset=0,
    )
    return DatasetSchemaResponse(
        dataset=dataset_name,
        table_key=ds.table_key,
        duckdb_schema=duckdb_schema,
        json_schema=_load_json_schema(ds),
        sample_rows=[ViewRow.model_validate(row) for row in sample_rows],
        capabilities=ds.capabilities(),
        owner=ds.owner,
        freshness_sla=ds.freshness_sla,
        retention_policy=ds.retention_policy,
        schema_version=ds.schema_version,
        stable_id=ds.stable_id,
        validation_profile=_normalize_validation_profile(ds.validation_profile),
    )
```

---

## 5. How an agent can generalize this pattern

For **`serving/backend/duckdb_service.py`**, the mechanical rules an implementation agent can follow:

1. **Create the file** with:

   * The updated module docstring + imports shown above.
   * Import `BackendLimits`, `ClampResult`, `clamp_limit_value`, `clamp_offset_value` from `serving.backend.limits`.
   * Keep all other imports exactly as in `serving/mcp/query_service.py`.

2. **Copy helpers and class**:

   * Copy `_fetch_duckdb_schema`, `_load_json_schema`, `_normalize_validation_profile` verbatim.
   * Copy the entire `DuckDBQueryService` class (header + all methods) verbatim.

3. **Do not change method bodies**:

   * All usages of `BackendLimits`, `ClampResult`, `clamp_limit_value`, `clamp_offset_value`, repositories, and Pydantic models stay as-is.
   * The only functional change is where those limit types/functions are **defined** (now `serving/backend/limits.py`) and **imported**.

4. In `serving/mcp/query_service.py`, you then replace the original implementation with the tiny shim we discussed earlier:

```python
from __future__ import annotations

from codeintel.serving.backend import (
    BackendLimits,
    ClampResult,
    DuckDBQueryService,
    clamp_limit_value,
    clamp_offset_value,
)

__all__ = [
    "BackendLimits",
    "ClampResult",
    "DuckDBQueryService",
    "clamp_limit_value",
    "clamp_offset_value",
]
```

If you’d like, next step we can do the same “before/after snippet” treatment for `serving/backend/limits.py` + the changes in `services/factory.py` and `services/wiring.py`, so the agent has a fully consistent pattern across all the files touched by this epic.


# zoom in on `serving/backend/limits.py` + the changes in `services/factory.py` and `services/wiring.py` #

Awesome, let’s finish wiring this epic so an implementation agent can follow it mechanically.

I’ll do three things:

1. **`serving/backend/limits.py`** – show how to extract `BackendLimits` + clamping from `mcp/query_service`.
2. **`serving/services/factory.py`** – before/after imports + where they’re used.
3. **`serving/services/wiring.py`** – before/after imports + where they’re used.

I’ll keep the pattern identical to what we did for `duckdb_service.py`: *same code, new home*.

---

## 1. Extract `BackendLimits` into `serving/backend/limits.py`

### 1.1 BEFORE – in `serving/mcp/query_service.py`

These definitions currently live near the top of `serving/mcp/query_service.py`, just after the imports and before `_fetch_duckdb_schema` / `DuckDBQueryService`:

```python
from dataclasses import dataclass, field
from typing import Literal, cast

from codeintel.serving.mcp.models import Message

# ...

@dataclass(frozen=True)
class BackendLimits:
    """Safety limits applied uniformly across backends."""

    default_limit: int = 50
    max_rows_per_call: int = 500

    @classmethod
    def from_config(cls, cfg: object) -> BackendLimits:
        """
        Build limits from configuration objects exposing default_limit/max_rows_per_call.
        """
        default = getattr(cfg, "default_limit", cls.default_limit)
        maximum = getattr(cfg, "max_rows_per_call", cls.max_rows_per_call)
        return cls(default_limit=int(default), max_rows_per_call=int(maximum))


@dataclass(frozen=True)
class ClampResult:
    """Result of clamping limit/offset values with messaging."""

    applied: int
    messages: list[Message] = field(default_factory=list)
    has_error: bool = False


def clamp_limit_value(
    requested: int | None,
    *,
    default: int,
    max_limit: int,
) -> ClampResult:
    """
    Clamp a requested limit to safe bounds, returning warnings instead of raising.
    """
    messages: list[Message] = []
    limit = default if requested is None else requested

    if limit < 0:
        messages.append(
            Message(
                code="limit_invalid",
                severity="error",
                detail="limit must be non-negative",
                context={"requested": limit},
            )
        )
        return ClampResult(applied=0, messages=messages, has_error=True)

    if limit > max_limit:
        messages.append(
            Message(
                code="limit_clamped",
                severity="warning",
                detail=f"Requested {limit} rows; delivering {max_limit} (max allowed).",
                context={"requested": limit, "applied": max_limit, "max": max_limit},
            )
        )
        limit = max_limit

    return ClampResult(applied=limit, messages=messages, has_error=False)


def clamp_offset_value(offset: int) -> ClampResult:
    """
    Clamp an offset to a non-negative value, returning messaging instead of raising.
    """
    if offset < 0:
        return ClampResult(
            applied=0,
            messages=[
                Message(
                    code="offset_invalid",
                    severity="error",
                    detail="offset must be non-negative",
                    context={"requested": offset},
                )
            ],
            has_error=True,
        )
    return ClampResult(applied=offset)
```

### 1.2 AFTER – new file `serving/backend/limits.py`

We move those definitions into a dedicated backend module.

**New file**: `src/codeintel/serving/backend/limits.py`

```python
"""Shared safety limits and clamping helpers for serving backends."""

from __future__ import annotations

from dataclasses import dataclass, field

from codeintel.serving.mcp.models import Message


@dataclass(frozen=True)
class BackendLimits:
    """Safety limits applied uniformly across backends.

    This type is transport-agnostic and can be used by MCP, HTTP, or CLI
    backends to enforce row limits consistently.
    """

    default_limit: int = 50
    max_rows_per_call: int = 500

    @classmethod
    def from_config(cls, cfg: object) -> "BackendLimits":
        """
        Build limits from configuration objects exposing default_limit/max_rows_per_call.

        Parameters
        ----------
        cfg:
            Any object with optional `default_limit` and `max_rows_per_call`
            attributes (e.g. ServingConfig).

        Returns
        -------
        BackendLimits
        """
        default = getattr(cfg, "default_limit", cls.default_limit)
        maximum = getattr(cfg, "max_rows_per_call", cls.max_rows_per_call)
        return cls(default_limit=int(default), max_rows_per_call=int(maximum))


@dataclass(frozen=True)
class ClampResult:
    """Result of clamping limit/offset values with messaging."""

    applied: int
    messages: list[Message] = field(default_factory=list)
    has_error: bool = False


def clamp_limit_value(
    requested: int | None,
    *,
    default: int,
    max_limit: int,
) -> ClampResult:
    """
    Clamp a requested limit to safe bounds, returning warnings instead of raising.

    Parameters
    ----------
    requested:
        Requested limit value; `None` means "use default".
    default:
        Default limit applied when no explicit limit is requested.
    max_limit:
        Maximum rows allowed for any call.

    Returns
    -------
    ClampResult
        Applied limit plus any informational or error messages.
    """
    messages: list[Message] = []
    limit = default if requested is None else requested

    if limit < 0:
        messages.append(
            Message(
                code="limit_invalid",
                severity="error",
                detail="limit must be non-negative",
                context={"requested": limit},
            )
        )
        return ClampResult(applied=0, messages=messages, has_error=True)

    if limit > max_limit:
        messages.append(
            Message(
                code="limit_clamped",
                severity="warning",
                detail=f"Requested {limit} rows; delivering {max_limit} (max allowed).",
                context={"requested": limit, "applied": max_limit, "max": max_limit},
            )
        )
        limit = max_limit

    return ClampResult(applied=limit, messages=messages, has_error=False)


def clamp_offset_value(offset: int) -> ClampResult:
    """
    Clamp an offset to a non-negative value, returning messaging instead of raising.

    Parameters
    ----------
    offset:
        Requested offset value.

    Returns
    -------
    ClampResult
        Applied offset and any validation messages.
    """
    if offset < 0:
        return ClampResult(
            applied=0,
            messages=[
                Message(
                    code="offset_invalid",
                    severity="error",
                    detail="offset must be non-negative",
                    context={"requested": offset},
                )
            ],
            has_error=True,
        )
    return ClampResult(applied=offset)
```

### 1.3 AFTER – wire it through `serving/backend/__init__.py`

We want a nice top-level import:

```python
from codeintel.serving.backend import BackendLimits, clamp_limit_value
```

**File**: `src/codeintel/serving/backend/__init__.py`

```python
from __future__ import annotations

from .limits import BackendLimits, ClampResult, clamp_limit_value, clamp_offset_value
from .duckdb_service import DuckDBQueryService

__all__ = [
    "BackendLimits",
    "ClampResult",
    "clamp_limit_value",
    "clamp_offset_value",
    "DuckDBQueryService",
]
```

### 1.4 AFTER – remove definitions from `mcp/query_service.py`

In `serving/mcp/query_service.py`, delete the old `BackendLimits` / `ClampResult` / `clamp_*` definitions and replace the entire file with the shim (as in the previous message):

```python
"""Backwards-compat shim for the DuckDB query backend.

The canonical implementation now lives in `codeintel.serving.backend`.
This module re-exports the APIs for compatibility with existing imports.
"""

from __future__ import annotations

from codeintel.serving.backend import (
    BackendLimits,
    ClampResult,
    DuckDBQueryService,
    clamp_limit_value,
    clamp_offset_value,
)

__all__ = [
    "BackendLimits",
    "ClampResult",
    "DuckDBQueryService",
    "clamp_limit_value",
    "clamp_offset_value",
]
```

---

## 2. Update `serving/services/factory.py` imports

Now we point the service factory at the new backend package instead of MCP.

### 2.1 BEFORE – top of `serving/services/factory.py`

The imports probably look roughly like this (structure-wise):

```python
from __future__ import annotations

from dataclasses import dataclass

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_runtime import GraphRuntimeOptions, build_graph_runtime
from codeintel.config.serving_config import ServingConfig
from codeintel.serving.http.datasets import (
    build_registry_and_limits,
    describe_dataset,
    validate_dataset_registry,
)
from codeintel.serving.mcp.query_service import BackendLimits, DuckDBQueryService
from codeintel.serving.services.query_service import (
    HttpQueryService,
    LocalQueryService,
    QueryService,
    ServiceObservability,
)
from codeintel.storage.gateway import StorageGateway
```

### 2.2 AFTER – top of `serving/services/factory.py`

We switch:

* `BackendLimits` / `DuckDBQueryService` → `codeintel.serving.backend`.
* Dataset helpers → `codeintel.serving.backend.datasets`.

```python
from __future__ import annotations

from dataclasses import dataclass

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_runtime import GraphRuntimeOptions, build_graph_runtime
from codeintel.config.serving_config import ServingConfig
from codeintel.serving.backend import BackendLimits, DuckDBQueryService
from codeintel.serving.backend.datasets import (
    build_registry_and_limits,
    describe_dataset,
    validate_dataset_registry,
)
from codeintel.serving.services.query_service import (
    HttpQueryService,
    LocalQueryService,
    QueryService,
    ServiceObservability,
)
from codeintel.storage.gateway import StorageGateway
```

### 2.3 Call sites – BEFORE vs AFTER

Most of the body in `factory.py` doesn’t need to change, because it just uses `BackendLimits.from_config` and instantiates `DuckDBQueryService` and `HttpQueryService` / `LocalQueryService`.

Typical pattern (BEFORE):

```python
@dataclass
class ServiceBuildOptions:
    config: ServingConfig
    gateway: StorageGateway
    analytics: AnalyticsContext
    observability: ServiceObservability | None = None


def build_local_query_service(options: ServiceBuildOptions) -> LocalQueryService:
    limits = BackendLimits.from_config(options.config)
    registry, _ = build_registry_and_limits(options.gateway.con, limits)
    validate_dataset_registry(registry)
    backend = DuckDBQueryService(
        gateway=options.gateway,
        repo=options.config.repo,
        commit=options.config.commit,
        limits=limits,
        graph_engine=None,
    )
    return LocalQueryService(
        backend=backend,
        analytics=options.analytics,
        limits=limits,
        observability=options.observability,
        dataset_describer=describe_dataset,
    )
```

AFTER, the body is identical; only the import source changed:

```python
def build_local_query_service(options: ServiceBuildOptions) -> LocalQueryService:
    limits = BackendLimits.from_config(options.config)
    registry, _ = build_registry_and_limits(options.gateway.con, limits)
    validate_dataset_registry(registry)
    backend = DuckDBQueryService(
        gateway=options.gateway,
        repo=options.config.repo,
        commit=options.config.commit,
        limits=limits,
        graph_engine=None,
    )
    return LocalQueryService(
        backend=backend,
        analytics=options.analytics,
        limits=limits,
        observability=options.observability,
        dataset_describer=describe_dataset,
    )
```

Likewise, for `build_http_query_service` or `build_service_from_config`, the new backend lives in `codeintel.serving.backend`, so you don’t touch the function bodies—just the imports.

---

## 3. Update `serving/services/wiring.py` imports

`wiring.py` is the glue between config, the storage gateway, and the actual `QueryService` instance. It also uses `BackendLimits` and the dataset registry.

### 3.1 BEFORE – top of `serving/services/wiring.py`

Something like:

```python
from __future__ import annotations

from dataclasses import dataclass

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_runtime import GraphRuntimeOptions
from codeintel.config.serving_config import ServingConfig
from codeintel.serving.http.datasets import build_registry_and_limits
from codeintel.serving.mcp.query_service import BackendLimits
from codeintel.serving.services.query_service import (
    LocalQueryService,
    QueryService,
    ServiceObservability,
)
from codeintel.storage.gateway import StorageGateway
```

### 3.2 AFTER – top of `serving/services/wiring.py`

We just redirect imports:

```python
from __future__ import annotations

from dataclasses import dataclass

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_runtime import GraphRuntimeOptions
from codeintel.config.serving_config import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.backend.datasets import build_registry_and_limits
from codeintel.serving.services.query_service import (
    LocalQueryService,
    QueryService,
    ServiceObservability,
)
from codeintel.storage.gateway import StorageGateway
```

### 3.3 Call sites – BEFORE vs AFTER

Typical wiring function (BEFORE):

```python
@dataclass
class BackendResource:
    config: ServingConfig
    gateway: StorageGateway
    analytics: AnalyticsContext
    runtime_options: GraphRuntimeOptions
    observability: ServiceObservability | None = None

    def build_local_service(self) -> LocalQueryService:
        limits = BackendLimits.from_config(self.config)
        registry, _ = build_registry_and_limits(self.gateway.con, limits)
        service = LocalQueryService(
            backend=None,  # set later
            analytics=self.analytics,
            limits=limits,
            observability=self.observability,
            dataset_describer=None,
        )
        # ... plus graph runtime wiring, etc.
        return service
```

AFTER, again, the function body doesn’t change; `BackendLimits` and `build_registry_and_limits` just come from `serving.backend` instead of MCP/HTTP.

---

## 4. Summary pattern for the agent

For an implementation agent, the rules across this epic now look like:

1. **Create `serving/backend/limits.py`** and **move** `BackendLimits`, `ClampResult`, `clamp_limit_value`, `clamp_offset_value` there verbatim.

2. **Create/adjust `serving/backend/__init__.py`** to re-export `BackendLimits`, `ClampResult`, `clamp_*`, and `DuckDBQueryService`.

3. **Replace imports**:

   * In `serving/services/factory.py`:

     * `from codeintel.serving.mcp.query_service import BackendLimits, DuckDBQueryService`
       → `from codeintel.serving.backend import BackendLimits, DuckDBQueryService`
     * `from codeintel.serving.http.datasets import ...`
       → `from codeintel.serving.backend.datasets import ...`

   * In `serving/services/wiring.py`:

     * `from codeintel.serving.mcp.query_service import BackendLimits`
       → `from codeintel.serving.backend import BackendLimits`
     * `from codeintel.serving.http.datasets import build_registry_and_limits`
       → `from codeintel.serving.backend.datasets import build_registry_and_limits`

4. **Leave all function bodies unchanged** (they already call `BackendLimits.from_config`, `build_registry_and_limits`, and `DuckDBQueryService` in a transport-agnostic way).

5. **Turn `serving/mcp/query_service.py` into the shim** that re-exports from `serving.backend`.

