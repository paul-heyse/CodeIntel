You’ve already got a really nice spine here with `DuckDBQueryService` + `LocalQueryService` + `HttpQueryService`. This refactor is mostly about **locking that in as *the* way to query** and cleaning up the handful of other places where SQL / datasets are defined.

I’ll treat the goal as:

> “All query *semantics* live in `mcp.query_service.DuckDBQueryService` (SQL) and `services.query_service.QueryService` (API). Everything else is either views (`storage`), registry (`server.datasets`), or pure adapters (HTTP, MCP, docs_export).”

I’ll structure this like:

1. Goals & layering
2. Inventory of current query surfaces
3. Solidify the QueryService contract
4. Make DuckDBQueryService the single SQL home
5. Ensure HTTP & MCP use QueryService exclusively
6. Align dataset registry & docs export with QueryService
7. What to do with `server/query_templates.py`
8. Tests & architecture guardrails

---

## 1. Goals & layering

After this refactor:

* **QueryService** (in `services/query_service.py`) is the *only* programmatic query API seen by:

  * FastAPI server (`server/fastapi.py`)
  * MCP server (`mcp/server.py` → `DuckDBBackend`)
  * Any future CLI, agents, adapters.

* **DuckDBQueryService** (in `mcp/query_service.py`) is the *only* place that:

  * Knows which **tables/views** to hit (`docs.v_*`, `analytics.*`, etc.).
  * Encodes SQL for:

    * function summaries
    * file/module profiles
    * architecture metrics
    * call-graph neighbors / neighborhoods
    * dataset browsing

* **Views / tables** are still defined in:

  * `storage/views.py` (`docs.v_*`)
  * `config/schemas/sql_builder.py` & `storage/schemas.py` (ingestion schemas)

* **Dataset registry** (dataset-name → table/view) is defined centrally in:

  * `server/datasets.py` + `StorageGateway.datasets.mapping`, and used by:
  * `DuckDBQueryService` → `LocalQueryService` → QueryService.
  * docs export code shares that registry (instead of ad hoc mappings).

Everything else:

* `server/fastapi.py`, `mcp/backend.py`, `mcp/server.py`, docs_export modules → become thin wrappers that call QueryService or registry, *never* doing their own SQL.

---

## 2. Inventory – what’s doing “query-ish” things now

From the zips:

**Core query layer**

* `mcp/mcp/query_service.py`

  * `DuckDBQueryService` – does almost all the SQL:

    * `get_function_summary`
    * `get_function_profile`
    * `get_file_profile` / `get_module_profile`
    * `get_function_architecture` / `get_module_architecture`
    * `get_callgraph_neighbors`, `get_callgraph_neighborhood`
    * `get_tests_for_function`
    * high-risk functions
    * dataset browsing: `read_dataset_rows` (via arbitrary table)
    * uses `storage.views` / `docs.v_*` as its surfaces.
  * Also uses `graphs.nx_views` for callgraph neighbors (this will later go through `GraphEngine`).

* `services/services/query_service.py`

  * Defines:

    * `FunctionQueryApi`, `ProfileQueryApi`, `SubsystemQueryApi`, `DatasetQueryApi` Protocols.
    * `QueryService` = composite Protocol.
    * `LocalQueryService` – wraps `DuckDBQueryService` with observability and dataset descriptions.
    * `HttpQueryService` – forwards QueryService calls to a remote HTTP server.
  * `LocalQueryService` uses `describe_dataset` and `gateway.datasets.mapping` to implement dataset listing and calling.

* `services/services/factory.py`

  * `build_service_from_config(...)` returns a **QueryService** (local or HTTP), using:

    * `DuckDBQueryService` for local DB.
    * `build_registry_and_limits` / `validate_dataset_registry` from `server.datasets`.

**Adapters**

* `server/server/fastapi.py`

  * Uses dependency injection (`get_service`) to supply a `LocalQueryService | HttpQueryService`.
  * Endpoints call methods like `service.get_function_summary(...)`, `service.list_datasets(...)`, etc.

* `mcp/mcp/backend.py`

  * `DuckDBBackend.service` is a QueryService.
  * MCP methods just forward to service methods.

* `mcp/mcp/server.py`

  * `create_mcp_server()` builds a `BackendResource` via `services.factory.build_backend_resource`, giving it both:

    * `backend: DuckDBBackend`
    * `service: QueryService`

**Dataset registry and views**

* `server/server/datasets.py`

  * `build_registry_and_limits(cfg, include_docs_views=...)` → dataset name → table mapping.
  * `validate_dataset_registry(gateway)` → verifies that registry tables exist and match `TABLE_SCHEMAS`.
  * uses `StorageGateway.datasets.mapping` and `TABLE_SCHEMAS`.
* `storage/storage/views.py` + `storage/docs.v_*`

  * define `docs.v_*` views serving as base surfaces for QueryService.

**Docs export**

* `docs_export/docs_export/export_jsonl.py` & `export_parquet.py`

  * Have their own dictionaries mapping `schema.table` to output filenames.
  * Use `COPY (SELECT * FROM <table>) TO ...` or equivalent to dump entire tables.
  * They do **no custom WHERE/JOIN logic**; just full-table exports.

**Legacy / extra query definitions**

* `server/server/query_templates.py`

  * Large `QUERY_TEMPLATES = {...}` mapping names to SQL queries against `docs.v_*`.
  * Nothing in the zipped packages imports this; historically used for “AI query templates”.

So the main “query language” lives in:

* `DuckDBQueryService` (good)
* `server/query_templates.py` (probably legacy / AI-only)
* dataset registry logic in `server/datasets.py`
* export table mappings in docs_export.

---

## 3. Step 1 – Solidify the QueryService contract

Your `QueryService` Protocol already looks like the right shape. This step is mostly about **documenting and locking it in** as the official API.

In `services/query_service.py` near `class QueryService`, add a docstring that makes this explicit:

```python
class QueryService(
    FunctionQueryApi,
    ProfileQueryApi,
    SubsystemQueryApi,
    DatasetQueryApi,
    Protocol,
):
    """
    Composite query service consumed by HTTP, MCP, and other transports.

    All application surfaces (FastAPI, MCP, CLI) must depend on this interface
    instead of touching DuckDB or raw SQL directly.

    Implementations:
        - LocalQueryService: wraps DuckDBQueryService for local DB access.
        - HttpQueryService: forwards calls to a remote HTTP server.
    """
```

Do a quick pass to ensure QueryService covers everything your frontends need:

* Function-centric:

  * `get_function_summary`, `get_function_profile`, `get_tests_for_function`, neighbors, neighborhoods, high-risk functions.
* Profiles:

  * `get_file_profile`, `get_module_profile`, `get_function_architecture`, `get_module_architecture`.
* Subsystems & hints:

  * `list_subsystems`, `get_module_subsystems`, `get_file_hints`, `get_subsystem_modules`.
* Datasets:

  * `list_datasets`, `read_dataset_rows`.

If anything the frontends need is still being done manually (e.g. a custom SQL query in FastAPI), we’ll add a corresponding method here and implement it in `DuckDBQueryService`.

---

## 4. Step 2 – Make DuckDBQueryService the *only* SQL home

This is mostly true already, but we’ll formalize it:

### 4.1: Audit SQL outside DuckDBQueryService

You’ve already got:

* `storage/views.py` – view definitions (**keep**; they’re schema).
* `server/query_templates.py` – *likely* legacy / AI templates.
* docs_export – `COPY (SELECT * FROM table)` (no custom semantics).

Double-check that:

* FastAPI endpoints **don’t** call `gateway.con.execute(...)` directly (they don’t; they always use `service`).
* MCP server doesn’t either (it doesn’t; it uses `DuckDBBackend` which uses QueryService).
* docs_export doesn’t encode any special filtering/joins; it doesn’t.

So the only place where complex SELECTs against `docs.v_*` live is `DuckDBQueryService` and `server/query_templates.py`.

### 4.2: Keep DuckDBQueryService as the core “query definition” module

Explicitly annotate at the top of `mcp/query_service.py`:

```python
"""Shared DuckDB query service used by MCP backends and FastAPI surface.

All SQL queries against docs.* and analytics.* views/tables live here.
Other modules must call this service (via LocalQueryService / QueryService)
instead of issuing custom SELECTs.
"""
```

If there are any “special” queries defined only in `server/query_templates.py` that you want to support via the API (e.g., impact analysis, “most risky files”), add corresponding methods to `DuckDBQueryService`:

```python
class DuckDBQueryService:
    ...

    def list_impacted_functions(
        self,
        *,
        goid_h128: int,
        radius: int,
        limit: int | None = None,
    ) -> HighRiskFunctionsResponse:
        """
        Example: use templates that rely on docs.v_call_graph_enriched and
        docs.v_function_summary to compute impact.
        """
        # 1) Use clamp_limit_value
        # 2) Build SELECT using docs.v_call_graph_enriched + docs.v_function_summary
        # 3) Return a proper Response model
```

You don’t have to implement all the templates now; just the ones your surfaces actually use.

---

## 5. Step 3 – Ensure HTTP & MCP use QueryService exclusively

You’re already mostly there; we’ll just tighten the story and make it explicit.

### 5.1: FastAPI server

In `server/fastapi.py`:

* You already have:

  ```python
  from codeintel.services.query_service import HttpQueryService, LocalQueryService
  ...
  def get_service(request: Request) -> LocalQueryService | HttpQueryService:
      service = getattr(request.app.state, "service", None)
      ...
  ServiceDep = Annotated[LocalQueryService | HttpQueryService, Depends(get_service)]
  ```

* Endpoints call `service.get_function_summary(...)`, etc. That’s perfect.

To make it more explicit:

* Optionally type alias:

  ```python
  from codeintel.services.query_service import QueryService

  ServiceDep = Annotated[QueryService, Depends(get_service)]
  ```

  (You may need `Protocol` handling tweaks for FastAPI, but at least in type hints this clarifies the intent.)

* Confirm there are **no direct imports** of `DuckDBQueryService` or `StorageGateway` in endpoint handlers. Those should only appear in startup wiring where the backend is constructed.

### 5.2: MCP backend

In `mcp/backend.py`:

* `DuckDBBackend` currently has a `service` attribute that is a QueryService.
* Its methods just call e.g. `self.service.get_function_profile(...)`, `self.service.get_callgraph_neighbors(...)`.

You don’t need to change anything here; just ensure:

* No code in `backend.py` calls `DuckDBQueryService` directly.
* No code in `backend.py` calls `gateway.con.execute(...)` directly.

If you find any direct SQL there, factor it out into a new method on `DuckDBQueryService` and then call it through `LocalQueryService`.

---

## 6. Step 4 – Align dataset registry & docs export with QueryService

Right now you have three “places” that talk about datasets/tables:

1. `StorageGateway.datasets.mapping` (table-level registry).
2. `server/datasets.py` (`build_registry_and_limits`, `DOCS_VIEWS`, `validate_dataset_registry`).
3. `docs_export/export_jsonl.py` & `export_parquet.py` (`TABLE_TO_*_MAP` of table → filename).

And QueryService:

* `LocalQueryService.list_datasets` builds a list of `DatasetDescriptor` using `dataset_tables` (passed from `DatasetRegistryOptions` built off `gateway.datasets.mapping`) and `describe_dataset`.

### 6.1: Make the QueryService dataset registry canonical

Treat the dataset registry used by `LocalQueryService` as **canonical** for named datasets:

* `name` – a logical dataset name (e.g., `"function_profile"`).
* `table` – actual `schema.table` (e.g., `"analytics.function_profile"`).

`server/datasets.build_registry_and_limits()` already constructs that mapping from:

* `TABLE_SCHEMAS` (all tables with JSON schemas).
* `DOCS_VIEWS` (views from `StorageGateway.DOCS_VIEWS`).

`services.factory.build_service_from_config` already uses this:

```python
registry, limits = build_registry_and_limits(cfg)
validate_dataset_registry(gateway)
query = DuckDBQueryService(gateway=gateway, repo=cfg.repo, commit=cfg.commit, limits=limits)
service = LocalQueryService(query=query, dataset_tables=registry, describe_dataset_fn=describe_dataset, ...)
```

So:

* **Dataset names** and **tables** are already centrally defined at `server.datasets` and passed into QueryService.
* QueryService’s `list_datasets` and `read_dataset_rows` use this registry.

### 6.2: Let docs_export share the same registry

Currently:

* `export_jsonl.py` & `export_parquet.py` define their own `TABLE_TO_*_MAP` mapping full table names (like `"analytics.function_profile"`) to filenames.

To centralize:

1. **Keep these as “file naming” maps only**, not as separate registry of what exists.
2. Use the canonical registry for:

   * Validating the requested `dataset` or `table`.
   * Enumerating datasets to export (if you support “export all datasets”).

For example, in `export_jsonl.py`:

* Instead of only accepting a `table_name` parameter, optionally accept a `dataset_name`, and resolve it via registry:

```python
from codeintel.server.datasets import build_registry_and_limits
from codeintel.services.factory import build_service_from_config

def export_dataset_to_jsonl(
    gateway: StorageGateway,
    cfg: ServingConfig,
    *,
    dataset_name: str,
    output_dir: Path,
) -> Path:
    # 1. Build registry via existing helper
    registry, _ = build_registry_and_limits(cfg)
    table = registry.get(dataset_name)
    if table is None:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")

    # 2. Map table -> filename using TABLE_TO_JSONL_MAP
    filename = TABLE_TO_JSONL_MAP.get(table, f"{dataset_name}.jsonl")
    output_path = output_dir / filename

    # 3. Use COPY FROM table to path (existing code)
    _export_table_to_jsonl(gateway.con, table, output_path)
    return output_path
```

For CLI use, you can still accept `schema.table` directly, but validate it against `gateway.datasets.mapping` before using it.

### 6.3: Optional – let export use QueryService instead of raw COPY

For massive tables, `COPY (SELECT * FROM table)` is the right tool. However, you might also want:

* A “small sample” mode where exports go through `QueryService.read_dataset_rows(...)`, so they respect all clamping rules and response models.

You could add:

```python
def export_dataset_via_query_service(
    service: QueryService,
    dataset_name: str,
    output_path: Path,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> None:
    resp = service.read_dataset_rows(dataset_name=dataset_name, limit=limit, offset=offset)
    with output_path.open("w", encoding="utf-8") as f:
        for row in resp.rows:
            f.write(json.dumps(row.model_dump()) + "\n")
```

But that’s optional; the big win is just **sharing the registry**.

---

## 7. Step 5 – Decide what to do with `server/query_templates.py`

`server/query_templates.py` currently contains a big dict of SQL templates:

* e.g. `"function_summary_by_urn"`, `"functions_impacted_by_calls"`, `"files_with_most_high_risk_functions"`, etc., all against `docs.v_*`.

Nothing in your current Python packages imports this file, so it appears to be:

* Either legacy, or
* Used only by some out-of-process AI tooling.

You have three options:

### Option A – Mark as “AI-only, not part of server”

* Leave the file as-is but add a big header comment:

  ```python
  """
  Parameterized SQL templates for AI query tooling.

  NOTE: These templates are NOT used by the HTTP or MCP servers. The canonical
  application query layer is DuckDBQueryService + QueryService. Any server
  additions should first be implemented as methods on DuckDBQueryService and
  exposed via QueryService instead of using these templates directly.
  """
  ```

* Be explicit in your docs that surfaces should **not** use these templates directly; they are for agents / exploration only.

### Option B – Bring select templates into DuckDBQueryService

For the templates that describe behaviors you actually want to support programmatically (e.g., “callgraph neighborhood impact sorted by risk”), implement them as `DuckDBQueryService` methods:

* Any template SQL becomes a regular method:

  ```python
  def list_impacted_functions(...):
      con = self.con
      # use the template SQL with named params or embedded as text
      rows = con.execute("<template-here>", [...]).fetchall()
      return HighRiskFunctionsResponse(...)
  ```

* You can even load the SQL from `query_templates.py` if you want to avoid duplication.

Then add corresponding methods to `FunctionQueryApi` / `QueryService` so HTTP/MCP surfaces can use them in a stable way.

### Option C – Deprecate and delete later

If you’re confident nothing relies on `server/query_templates.py` any more:

* Mark it with a `DeprecationWarning` or a TODO comment.
* Once your AI tooling uses QueryService instead, you can remove the file.

Given we don’t see it imported anywhere in the current tree, Option A is the safest short-term: clearly mark it as AI-only and not part of the server’s canonical query layer.

---

## 8. Step 6 – Add tests & guardrails

Finally, we lock the structure in.

### 8.1: “No direct SQL in adapters” tests

Create a small architecture test to ensure no direct DuckDB queries in server or MCP adapter layers:

```python
# tests/architecture/test_query_layer_boundaries.py

from pathlib import Path

FORBIDDEN = ("duckdb.connect", ".execute(", "FROM docs.v_")

def test_server_does_not_execute_sql_directly() -> None:
    root = Path("src/codeintel/server")
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "datasets.py" in str(path):  # allow schema validation helpers
            continue
        if "fastapi.py" in str(path):
            assert "gateway.con.execute" not in text, f"{path} should use QueryService"
        for snippet in FORBIDDEN:
            assert snippet not in text or "query_service" in text, f"{path} has direct SQL usage: {snippet}"
```

You can tune the heuristics, but the goal is:

* In `server/fastapi.py`, no `gateway.con.execute(...)`.
* In MCP adapter code, no direct `duckdb.connect` or raw `con.execute("SELECT ...")`.

### 8.2: Ensure QueryService covers all routes

Add tests that:

* Spin up a tiny in-memory DB via `StorageGateway`.
* Build a `LocalQueryService` via `build_service_from_config`.
* Call each QueryService method used in FastAPI + MCP and assert it returns the expected response model types (even if the DB is mostly empty).

Example:

```python
def test_local_query_service_function_summary_empty_db(gateway, serving_config) -> None:
    registry, limits = build_registry_and_limits(serving_config)
    validate_dataset_registry(gateway)
    query = DuckDBQueryService(gateway=gateway, repo=serving_config.repo, commit=serving_config.commit, limits=limits)
    service = LocalQueryService(query=query, dataset_tables=registry)

    resp = service.get_function_summary(goid_h128=123)
    assert resp.meta.requested_limit is None  # or whatever semantics you expect
    assert not resp.found
```

### 8.3: Dataset registry / docs_export integration test

Add a test that:

* Creates a temp DB with a couple of tables/views.
* Uses `build_registry_and_limits` to get the registry.
* Uses your docs_export module to export one dataset by **dataset name**, resolving via registry.

This ensures the registry and export code stay aligned.


