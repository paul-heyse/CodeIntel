Here’s a concrete, code‑level migration plan to push serving into a **domain‑first** architecture, specifically targeting `serving/backend` and the query services, with pointers to tests.

I’ll go phase‑by‑phase, and within each phase call out:

* **Files**
* **Exact functions / signatures to change**
* **New tests to add or update**

---

## Phase 0 – Add guardrail tests for domain ↔ MCP conversions

Before touching the backend, lock in your existing domain/transport contracts so refactors can’t subtly break them.

### 0.1 New test module: `tests/serving/test_domain_transport_roundtrip.py`

**File:** `tests/serving/test_domain_transport_roundtrip.py` (new)

**Purpose:**

* For each `dm.*Result` type and its MCP counterpart (e.g. `FunctionSummaryResult` ↔ `FunctionSummaryResponse`), verify:

  * `domain -> MCP -> domain` is stable
  * MCP `.to_domain()` / `.from_domain()` behave as expected

**Concrete tests:**

Use small, synthetic payloads (no DuckDB) and exercise just the conversion code:

* `FunctionSummary`:

  ```python
  from codeintel.serving import domain_models as dm
  from codeintel.serving.mcp.models import FunctionSummaryResponse, ResponseMeta

  def test_function_summary_roundtrip() -> None:
      meta = dm.ResponseMeta(
          requested_limit=10,
          applied_limit=5,
          truncated=True,
          messages=[dm.Message(code="test", severity="info", detail="hi")],
      )
      domain = dm.FunctionSummaryResult(
          found=True,
          summary={"urn": "urn:codeintel:test", "goid_h128": 123},
          meta=meta,
      )

      transport = FunctionSummaryResponse.from_domain(domain)
      back = transport.to_domain()

      assert back.found is True
      assert back.summary == domain.summary
      assert back.meta.truncated is True
      assert back.meta.messages[0].code == "test"
  ```

* Similarly add tests for:

  * `HighRiskFunctionsResult` ↔ `HighRiskFunctionsResponse`
  * `TestsForFunctionResult` ↔ `TestsForFunctionResponse`
  * `CallGraphNeighbors` ↔ `CallGraphNeighborsResponse`
  * `GraphNeighborhood` ↔ `GraphNeighborhoodResponse`
  * `ImportBoundary` ↔ `ImportBoundaryResponse`
  * `FunctionProfileResult` / `ModuleProfileResult` / `FileProfileResult` etc.

You don’t need exhaustive fields here—just enough to confirm basic shapes and metadata survive roundtrip.

---

## Phase 1 – Move backend pagination & metadata to domain types

Right now `serving/backend/pagination.py` is wired to MCP types:

```python
from codeintel.serving.mcp.models import Message, ResponseMeta
```

We want pagination to be **transport‑agnostic**, using `domain_models.Message` / `domain_models.ResponseMeta`.

### 1.1 Switch pagination to domain types

**File:** `serving/backend/pagination.py`

**Changes:**

* Replace import:

  ```python
  # old
  from codeintel.serving.mcp.models import Message, ResponseMeta

  # new
  from codeintel.serving import domain_models as dm
  Message = dm.Message
  ResponseMeta = dm.ResponseMeta
  ```

* Ensure all annotations and default creation use these aliases:

  ```python
  @dataclass
  class BackendLimits:
      default_limit: int = 100
      max_rows_per_call: int = 10_000
      max_offset: int = 1_000_000
  ```

  Any reference to `ResponseMeta()` is now the domain version (which already has `model_dump()`).

The rest of `PaginatedFetch`, `LimitClamp`, `ClampResult`, `clamp_limit_value`, `clamp_offset_value`, `paginate_items` can remain unchanged—they only rely on `.messages` and simple fields that `dm.ResponseMeta` supports.

### 1.2 Adjust import sites to expect domain metadata

**Files:**

* `serving/backend/duckdb_service.py`
* `serving/services/query_service.py` (already aliases `ResponseMeta = dm.ResponseMeta`, so mostly fine)

In `duckdb_service.py`, you *already* import `BackendLimits` from `serving.backend` and `ResponseMeta` / `Message` from MCP:

```python
from codeintel.serving.mcp.models import (
    ...
    Message,
    ...
    ResponseMeta,
    ...
)
```

Change those to domain types:

```python
from codeintel.serving import domain_models as dm

Message = dm.Message
ResponseMeta = dm.ResponseMeta
```

Then:

* All `meta = ResponseMeta()` now create domain metadata.
* All `meta.messages.append(Message(...))` now use domain Message.

**Tests to run / adjust:**

* `tests/services/test_backend_limits.py`
* `tests/services/test_query_service.py`
* `tests/serving/test_serving_runtime_analytics_e2e.py`

They *should* continue to pass, because MCP’s `ResponseMeta.from_domain` / `.to_domain` is already built for this.

---

## Phase 2 – Introduce domain builders for backend results

We want a single place that turns repo rows into **domain** payloads.

### 2.1 New module: `serving/backend/domain_builders.py`

**File:** `serving/backend/domain_builders.py` (new)

**Core structure:**

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from codeintel.serving import domain_models as dm

RowDict = Mapping[str, Any]


def build_function_summary(
    row: RowDict | None,
    meta: dm.ResponseMeta | None = None,
) -> dm.FunctionSummaryResult:
    if meta is None:
        meta = dm.ResponseMeta()
    if row is None:
        return dm.FunctionSummaryResult(found=False, summary=None, meta=meta)
    return dm.FunctionSummaryResult(found=True, summary=dict(row), meta=meta)
```

Repeat for each backend “response” type that `duckdb_service` currently constructs using MCP models:

* **Functions / callgraph:**

  * `build_high_risk_functions(rows: Sequence[RowDict], meta: dm.ResponseMeta) -> dm.HighRiskFunctionsResult`
  * `build_callgraph_neighbors(outgoing: Sequence[RowDict], incoming: Sequence[RowDict], meta: dm.ResponseMeta) -> dm.CallGraphNeighbors`
  * `build_tests_for_function(rows: Sequence[RowDict], meta: dm.ResponseMeta) -> dm.TestsForFunctionResult`
  * `build_graph_neighborhood(nodes: Sequence[RowDict], edges: Sequence[RowDict], meta: dm.ResponseMeta) -> dm.GraphNeighborhood`
  * `build_import_boundary(nodes: Sequence[str], edges: Sequence[RowDict], meta: dm.ResponseMeta) -> dm.ImportBoundary`

* **Summaries & profiles:**

  * `build_file_summary(rows: Sequence[RowDict], rel_path: str, meta: dm.ResponseMeta) -> dm.FileSummaryResult`
  * `build_function_profile(row: RowDict | None, meta) -> dm.FunctionProfileResult`
  * `build_file_profile(row: RowDict | None, meta) -> dm.FileProfileResult`
  * `build_module_profile(row: RowDict | None, meta) -> dm.ModuleProfileResult`

* **Architecture / subsystems:**

  * `build_function_architecture(row: RowDict | None, meta) -> dm.FunctionArchitectureResult`
  * `build_module_architecture(row: RowDict | None, meta) -> dm.ModuleArchitectureResult`
  * `build_subsystem_summary(rows: Sequence[RowDict], meta) -> dm.SubsystemSummaryResult`
  * `build_module_subsystems(rows: Sequence[RowDict], meta) -> dm.ModuleSubsystemResult`
  * `build_file_hints(rows: Sequence[RowDict], rel_path: str, meta) -> dm.FileHintsResult`
  * `build_subsystem_modules(rows: Sequence[RowDict], meta) -> dm.SubsystemModulesResult`
  * `build_subsystem_search(rows: Sequence[RowDict], meta) -> dm.SubsystemSearchResult`
  * `build_subsystem_profile(rows: Sequence[RowDict], meta) -> dm.SubsystemProfileResult`
  * `build_subsystem_coverage(rows: Sequence[RowDict], meta) -> dm.SubsystemCoverageResult`

* (Optional) dataset: `build_dataset_schema` into `dm.DatasetSchema`.

Implementation is intentionally simple: domain models are shape‑compatibile with the `model_dump()` of the Pydantic rows, so each builder just:

* Copies rows into `dict(row)` or `list[dict(row)]`.
* Inserts `dm.ResponseMeta`.

### 2.2 New tests: unit coverage for domain builders

**File:** `tests/serving/test_domain_builders.py` (new)

For each builder, write a tiny test verifying:

* `row=None` → `found=False` / `*Result` with empty lists.
* Non‑empty row(s) → fields appear in domain dicts.
* `meta` is used when provided.

Example:

```python
from codeintel.serving import domain_models as dm
from codeintel.serving.backend.domain_builders import build_high_risk_functions

def test_build_high_risk_functions_marks_truncation() -> None:
    meta = dm.ResponseMeta(truncated=True)
    rows = [{"goid_h128": 1, "qualname": "f", "rel_path": "a.py", "risk_score": 0.9}]
    result = build_high_risk_functions(rows, meta=meta)
    assert result.functions == rows
    assert result.meta.truncated is True
```

---

## Phase 3 – Make `_FunctionQueries`, `_ModuleQueries`, `_SubsystemQueries` return domain results

Now we push domain types down into the DuckDB backend itself.

### 3.1 `_FunctionQueries` → domain return types

**File:** `serving/backend/duckdb_service.py`

**Class:** `class _FunctionQueries:`

For each method, change:

1. **Imports at top of file:**

   * Remove all Pydantic response imports from `codeintel.serving.mcp.models` that are used only for backend responses:

     * `FunctionSummaryResponse`, `HighRiskFunctionsResponse`, `CallGraphNeighborsResponse`, `TestsForFunctionResponse`, `GraphNeighborhoodResponse`, `ImportBoundaryResponse`, etc.

   * Import your new builders:

     ```python
     from codeintel.serving.backend.domain_builders import (
         build_function_summary,
         build_high_risk_functions,
         build_callgraph_neighbors,
         build_tests_for_function,
         build_graph_neighborhood,
         build_import_boundary,
     )
     ```

2. **`get_function_summary`**

   * **Old signature:**

     ```python
     def get_function_summary(... ) -> FunctionSummaryResponse:
     ```

   * **New signature:**

     ```python
     def get_function_summary(... ) -> dm.FunctionSummaryResult:
     ```

   * **Body changes:**

     * Keep identifier resolution and `errors.invalid_argument` logic as‑is.
     * Keep `meta = ResponseMeta()` (now domain) and `Message` usage.
     * Replace the end:

       ```python
       if resolved is None:
           meta.messages.append(...)
           return FunctionSummaryResponse(found=False, summary=None, meta=meta)
       ...
       if row is None:
           meta.messages.append(...)
           return FunctionSummaryResponse(found=False, summary=None, meta=meta)
       return FunctionSummaryResponse(
           found=True, summary=FunctionSummaryRow.model_validate(row), meta=meta
       )
       ```

       with:

       ```python
       if resolved is None or row is None:
           return dm.FunctionSummaryResult(found=False, summary=None, meta=meta)
       return build_function_summary(row, meta=meta)
       ```

3. **`list_high_risk_functions`**

   * **Old signature:**

     ```python
     def list_high_risk_functions(... ) -> HighRiskFunctionsResponse:
     ```

   * **New signature:**

     ```python
     def list_high_risk_functions(... ) -> dm.HighRiskFunctionsResult:
     ```

   * After you compute:

     ```python
     clamp = clamp_limit_value(...)
     rows = self.functions.list_high_risk_functions(...)
     meta = clamp.to_meta() or ResponseMeta(...)  # whatever you currently do
     ```

     Replace Pydantic construction with:

     ```python
     return build_high_risk_functions(rows, meta=meta)
     ```

   * If your current code sets `truncated` separately, ensure the builder takes that into account (pass `meta.truncated` or a separate flag as needed).

4. **`get_callgraph_neighbors`**

   * **Old signature:**

     ```python
     def get_callgraph_neighbors(... ) -> CallGraphNeighborsResponse:
     ```

   * **New signature:**

     ```python
     def get_callgraph_neighbors(... ) -> dm.CallGraphNeighbors:
     ```

   * Build the outgoing / incoming row lists exactly as now, but then:

     ```python
     meta = ResponseMeta(
         requested_limit=limit,
         applied_limit=limit_clamp.applied,
         truncated=limit_clamp.truncated,
     )
     return build_callgraph_neighbors(outgoing, incoming, meta=meta)
     ```

5. **`get_tests_for_function`, `get_callgraph_neighborhood`, `get_import_boundary`**

   * Each of these should:

     * Keep graph/traversal logic exactly as is.
     * Replace `*Response` construction with the appropriate builder call.

### 3.2 `_ModuleQueries`, `_SubsystemQueries` similarly

For:

* `class _ModuleQueries:`
* `class _SubsystemQueries:`

Follow the same pattern:

* Change Pydantic response return types to:

  * `dm.FunctionProfileResult`
  * `dm.FileProfileResult`
  * `dm.ModuleProfileResult`
  * `dm.FunctionArchitectureResult`
  * `dm.ModuleArchitectureResult`
  * `dm.SubsystemSummaryResult`
  * `dm.ModuleSubsystemResult`
  * `dm.FileHintsResult`
  * `dm.SubsystemModulesResult`
  * `dm.SubsystemSearchResult`
  * `dm.SubsystemProfileResult`
  * `dm.SubsystemCoverageResult`

* Replace Pydantic response construction with calls to your domain builders.

### 3.3 `_DatasetQueries` (optional in this pass)

You can leave dataset listing/specs as Pydantic for now; but if you want full domain symmetry:

* Introduce `build_dataset_schema` builder that returns `dm.DatasetSchema`.
* Change `_DatasetQueries.dataset_schema()` to return that.
* Update `DatasetQueriesApi` in the next phase.

### 3.4 Tests

Run and adjust (if needed):

* `tests/serving/test_serving_runtime_analytics_e2e.py`
* `tests/serving/test_dataset_specs.py` (if you touch dataset schema)
* `tests/services/test_query_service.py` (these tests exercise LocalQueryService, which calls these methods under the hood—if you keep QueryService behavior identical, they should remain green).

---

## Phase 4 – Update `DuckDBQueryApi` to use domain types

**File:** `serving/backend/query_api.py`

### 4.1 Replace MCP imports with domain models

At the top:

```python
from codeintel.config.steps_graphs import GraphRunScope
from codeintel.serving.backend.pagination import BackendLimits
from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    FileHintsResponse,
    FileProfileResponse,
    FileSummaryResponse,
    ...
)
```

Change to:

```python
from codeintel.config.steps_graphs import GraphRunScope
from codeintel.serving.backend.pagination import BackendLimits
from codeintel.serving import domain_models as dm
from codeintel.serving.mcp.models import DatasetSpecDescriptor  # if you keep specs as Pydantic
```

### 4.2 Rewrite protocol signatures to domain

In `class FunctionQueriesApi(Protocol):`:

* Change:

  ```python
  def get_function_summary(... ) -> FunctionSummaryResponse: ...
  def list_high_risk_functions(... ) -> HighRiskFunctionsResponse: ...
  def get_callgraph_neighbors(... ) -> CallGraphNeighborsResponse: ...
  def get_tests_for_function(... ) -> TestsForFunctionResponse: ...
  def get_callgraph_neighborhood(... ) -> GraphNeighborhoodResponse: ...
  def get_import_boundary(... ) -> ImportBoundaryResponse: ...
  def get_file_summary(... ) -> FileSummaryResponse: ...
  ```

* To:

  ```python
  def get_function_summary(... ) -> dm.FunctionSummaryResult: ...
  def list_high_risk_functions(... ) -> dm.HighRiskFunctionsResult: ...
  def get_callgraph_neighbors(... ) -> dm.CallGraphNeighbors: ...
  def get_tests_for_function(... ) -> dm.TestsForFunctionResult: ...
  def get_callgraph_neighborhood(... ) -> dm.GraphNeighborhood: ...
  def get_import_boundary(... ) -> dm.ImportBoundary: ...
  def get_file_summary(... ) -> dm.FileSummaryResult: ...
  ```

In `ProfileQueriesApi(Protocol):`:

* `get_function_profile` → `dm.FunctionProfileResult`
* `get_file_profile` → `dm.FileProfileResult`
* `get_module_profile` → `dm.ModuleProfileResult`
* `get_function_architecture` → `dm.FunctionArchitectureResult`
* `get_module_architecture` → `dm.ModuleArchitectureResult`

In `SubsystemQueriesApi(Protocol):`:

* `list_subsystems` → `dm.SubsystemSummaryResult`
* `get_module_subsystems` → `dm.ModuleSubsystemResult`
* `get_file_hints` → `dm.FileHintsResult`
* `get_subsystem_modules` → `dm.SubsystemModulesResult`
* `search_subsystems` → `dm.SubsystemSearchResult`
* `list_subsystem_profiles` → `dm.SubsystemProfileResult`
* `list_subsystem_coverage` → `dm.SubsystemCoverageResult`

In `DatasetQueriesApi(Protocol):`:

* You can leave `list_datasets()` / `dataset_specs()` returning `DatasetDescriptor` / `DatasetSpecDescriptor` for now.
* If you moved `dataset_schema` to domain, change:

  ```python
  def dataset_schema(... ) -> DatasetSchemaResponse: ...
  ```

  to:

  ```python
  def dataset_schema(... ) -> dm.DatasetSchema: ...
  ```

No changes are needed for `DuckDBQueryApi` itself except the type annotations; the concrete `DuckDBQueryService` already implements these names.

---

## Phase 5 – Adapt the query service delegates to be domain‑native

Now that `DuckDBQueryApi` and `_FunctionQueries` / `_ModuleQueries` / `_SubsystemQueries` return domain objects, you can simplify or at least make your delegates domain‑first.

### 5.1 `_FunctionQueryDelegates` – accept domain or MCP, return domain

**File:** `serving/services/functions.py`

**Class:** `_FunctionQueryDelegates`

For each method, update `raw_resp` handling.

#### `get_function_summary`

* **Old:**

  ```python
  def get_function_summary(... ) -> dm.FunctionSummaryResult:
      raw_resp = self._call(
          "get_function_summary",
          lambda: self.query.functions.get_function_summary(...),
      )
      pydantic_resp = (
          raw_resp
          if isinstance(raw_resp, FunctionSummaryResponse)
          else FunctionSummaryResponse.model_validate(raw_resp)
      )
      return pydantic_resp.to_domain()
  ```

* **New (domain‑first):**

  ```python
  def get_function_summary(... ) -> dm.FunctionSummaryResult:
      raw_resp = self._call(
          "get_function_summary",
          lambda: self.query.functions.get_function_summary(
              urn=urn,
              goid_h128=goid_h128,
              rel_path=rel_path,
              qualname=qualname,
              scope=parse_graph_scope(scope),
          ),
      )
      if isinstance(raw_resp, dm.FunctionSummaryResult):
          return raw_resp
      if isinstance(raw_resp, FunctionSummaryResponse):
          return raw_resp.to_domain()
      # Fallback: tolerate dicts etc.
      return FunctionSummaryResponse.model_validate(raw_resp).to_domain()
  ```

Same pattern for:

* `list_high_risk_functions`:

  * Prefer `dm.HighRiskFunctionsResult`
  * Secondary: `HighRiskFunctionsResponse.to_domain()`
  * Fallback: `HighRiskFunctionsResponse.model_validate(...).to_domain()`

* `get_callgraph_neighbors`, `get_tests_for_function`, `get_callgraph_neighborhood`, `get_import_boundary`, `get_file_summary`.

This makes local mode (which now returns domain) the “happy path”, while keeping compatibility for any injection points that still return Pydantic or plain dicts.

### 5.2 `_ProfileQueryDelegates` and `_SubsystemQueryDelegates`

**Files:**

* `serving/services/profiles.py` (for `_ProfileQueryDelegates`)
* `serving/services/subsystems.py` (for `_SubsystemQueryDelegates`)

Apply the same pattern:

* For each method:

  ```python
  raw_resp = self._call(...)
  if isinstance(raw_resp, dm.FunctionProfileResult):
      return raw_resp
  if isinstance(raw_resp, FunctionProfileResponse):
      return raw_resp.to_domain()
  return FunctionProfileResponse.model_validate(raw_resp).to_domain()
  ```

* Do this for all profile/subsystem methods.

### 5.3 `HttpQueryService` and `_Http*Mixin`s

Good news: your `_HttpFunctionQueryMixin`, `_HttpProfileQueryMixin`, and `_HttpSubsystemQueryMixin` are already **domain‑first**:

* They always return `dm.*Result`.
* They accept either domain or Pydantic payload from `request_json`.

No structural changes needed here; just ensure that you still import the Pydantic response types you use inside these mixins.

### 5.4 Tests

Run:

* `tests/services/test_query_service.py`
* `tests/serving/test_query_service_scope.py`

If anything breaks:

* Update type assertions in tests to look at domain dataclasses (e.g., `isinstance(result, dm.FunctionSummaryResult)`) instead of transport models.

---

## Phase 6 – Clean up backend Pydantic usage

Once the above is done:

* `serving/backend/duckdb_service.py` should no longer import any of:

  * `FunctionSummaryResponse`, `FunctionProfileResponse`, `ModuleProfileResponse`, etc.
  * `CallGraphNeighborsResponse`, `GraphNeighborhoodResponse`, `ImportBoundaryResponse`, `TestsForFunctionResponse`…
  * `FileSummaryResponse`, `FileProfileResponse`, `FileHintsResponse`, etc.

* `serving/backend/response_builders.py` will be unused for “real” flows. You have two options:

  1. **Repurpose it** as a thin MCP‑only adapter:

     * Use your domain builders internally.
     * Provide `*_response_from_domain` helpers that are called from FastAPI/MCP routes if you ever want to bypass `QueryService`.

  2. **Remove it** and prune exports from `serving/backend/__init__.py`.

Given you already have `mcp.models.*Response.from_domain`, option (2) is reasonable: you simply don’t need the extra layer.

Don’t forget to update `__all__` in `serving/backend/__init__.py` to drop the removed names.

---

## Phase 7 – (Optional) Make dataset introspection domain‑first

If you want to go all‑in and also make dataset intros domain‑centric:

1. **Extend `domain_models` with a dataset descriptor result:**

   ```python
   @dataclass
   class DatasetDescriptorDomain:
       name: str
       table: str
       description: str
       family: str | None = None
       owner: str | None = None
       schema_version: str | None = None
       stable_id: str | None = None
       is_docs_view: bool = False
       is_read_only: bool = False
   ```

2. **Change `_DatasetQueries` methods:**

   * `list_datasets` → `list[dm.DatasetDescriptorDomain]`
   * `dataset_schema` → `dm.DatasetSchema`

3. **Update `DatasetQueryApi` and `LocalQueryService` / `HttpQueryService`:**

   * `class DatasetQueryApi(Protocol)` → returns domain descriptors.
   * `LocalQueryService.list_datasets()` returns `list[dm.DatasetDescriptorDomain]`.

4. **Fix tests that rely on `.model_dump()`**

   In `tests/services/test_query_service.py`:

   ```python
   def _invoke_list_datasets(
       local: LocalQueryService, _params: dict[str, object]
   ) -> list[dict[str, object]]:
       return [dataclasses.asdict(descriptor) for descriptor in local.list_datasets()]
   ```

   Or simply:

   ```python
   [descriptor.__dict__ for descriptor in local.list_datasets()]
   ```

---

## Quick checklist of functions you’ll touch

For ease of tracking in your editor:

* `serving/backend/pagination.py`

  * **Imports**: `Message`, `ResponseMeta` → domain.

* `serving/backend/domain_builders.py` (new)

  * All `build_*` functions that correspond to the MCP `*Response` types.

* `serving/backend/duckdb_service.py`

  * `_FunctionQueries.get_function_summary`
  * `_FunctionQueries.list_high_risk_functions`
  * `_FunctionQueries.get_callgraph_neighbors`
  * `_FunctionQueries.get_tests_for_function`
  * `_FunctionQueries.get_callgraph_neighborhood`
  * `_FunctionQueries.get_import_boundary`
  * `_FunctionQueries.get_file_summary`
  * `_ModuleQueries.get_function_profile`
  * `_ModuleQueries.get_file_profile`
  * `_ModuleQueries.get_module_profile`
  * `_ModuleQueries.get_function_architecture`
  * `_ModuleQueries.get_module_architecture`
  * `_SubsystemQueries.*` (all public methods)
  * `_DatasetQueries.dataset_schema` (if domain‑izing datasets)

* `serving/backend/query_api.py`

  * All methods in `FunctionQueriesApi`, `ProfileQueriesApi`, `SubsystemQueriesApi`, maybe `DatasetQueriesApi`.

* `serving/services/functions.py`

  * `_FunctionQueryDelegates`: all methods (make domain preferred).
  * `_HttpFunctionQueryMixin`: no semantic change; keep as is.

* `serving/services/profiles.py`

  * `_ProfileQueryDelegates` methods.

* `serving/services/subsystems.py`

  * `_SubsystemQueryDelegates` methods.

* `serving/backend/__init__.py`

  * Remove unused `build_*_response` exports once you decide what to do with `response_builders`.

* Tests:

  * New: `tests/serving/test_domain_transport_roundtrip.py`
  * New: `tests/serving/test_domain_builders.py`
  * Adjust (if needed): `tests/services/test_query_service.py`, `tests/serving/test_query_service_scope.py`, dataset tests if you domain‑ize them.

---


