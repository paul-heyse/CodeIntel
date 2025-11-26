Absolutely, let’s turn that high-level idea into something an AI programming agent could execute step-by-step.

I’ll structure this as:

1. Goals & invariants
2. Inventory & search tasks
3. API refactor of `StorageGateway` & config
4. Refactor ingestion/analytics entrypoints
5. Refactor Prefect & pipeline wiring
6. Refactor CLI & server surfaces
7. Refactor tests
8. Final cleanup / guardrails

I’ll include snippets that illustrate the *shape* of the change rather than exact final code, so you can hand this directly to an implementation agent.

---

## 1. Goals & invariants for the refactor

**Design goal:** after this refactor:

* **All code outside `storage/` talks to DuckDB only through `StorageGateway`.**

* *No* `duckdb.connect(...)` calls outside `storage.gateway` (and maybe a tiny number of well-documented bootstrap sites like migrations, if any).

* Ingestion, analytics, Prefect, CLI, and the FastAPI server all:

  ```python
  config = StorageConfig(...)
  gateway = open_gateway(config)
  # use gateway everywhere
  ```

* Only `open_gateway` knows about:

  * `read_only`
  * `apply_schema`
  * `ensure_views`
  * `validate_schema`
  * “attach history DB” behaviour

Everything else assumes: **“if you have a `StorageGateway`, the DB is ready to use.”**

---

## 2. Inventory & search tasks (for the agent)

**Task 2.1 – Identify all raw DuckDB usage**

Programmatically (or with search):

* Find all imports of DuckDB types:

  * `from duckdb import DuckDBPyConnection`
  * `import duckdb`

* Find all direct connects:

  * `duckdb.connect(`
  * `DuckDBPyConnection(` if used.

**Deliverable:** a short list of modules & call sites that:

* Accept `DuckDBPyConnection` as a parameter, or
* Call `duckdb.connect` directly.

These are the places we will systematically replace.

**Task 2.2 – Confirm the central gateway module**

In `src/codeintel/storage/gateway.py`, catalog what exists now, e.g.:

* `StorageConfig` dataclass
* `StorageGateway` (class)
* `DatasetRegistry`
* `open_gateway(config: StorageConfig) -> StorageGateway`
* Any helper like `build_snapshot_gateway_resolver`, etc.

Make sure we treat this as **the only true DB boundary.**

---

## 3. Tighten & document the `StorageGateway` API

### 3.1 Clarify `StorageConfig`

**Goal:** `StorageConfig` fully describes *how* to open DBs, with no extra flags leaking into call sites.

Implementation tasks:

1. Ensure `StorageConfig` includes the fields you actually use:

   ```python
   @dataclass(frozen=True)
   class StorageConfig:
       db_path: Path
       read_only: bool = False
       apply_schema: bool = False
       ensure_views: bool = False
       validate_schema: bool = False
       attach_history: bool = False
       history_db_path: Path | None = None
       # add others here if they exist:
       # pragma_settings: dict[str, str] | None = None
   ```

2. Add simple constructors for common modes (classmethods):

   ```python
   @dataclass(frozen=True)
   class StorageConfig:
       ...

       @classmethod
       def for_ingest(cls, db_path: Path, *, history_db_path: Path | None = None) -> "StorageConfig":
           return cls(
               db_path=db_path,
               read_only=False,
               apply_schema=True,
               ensure_views=True,
               validate_schema=True,
               attach_history=history_db_path is not None,
               history_db_path=history_db_path,
           )

       @classmethod
       def for_readonly(cls, db_path: Path) -> "StorageConfig":
           return cls(
               db_path=db_path,
               read_only=True,
               apply_schema=False,
               ensure_views=True,
               validate_schema=True,
           )
   ```

This keeps configuration logic out of Prefect/CLI and makes life easier for agents.

### 3.2 Ensure `StorageGateway` is the only way to get a connection

Inside `storage/gateway.py`:

* Confirm `open_gateway` encapsulates:

  ```python
  def open_gateway(config: StorageConfig) -> StorageGateway:
      con = duckdb.connect(str(config.db_path), read_only=config.read_only)
      # attach history DB if needed
      # apply schema if config.apply_schema
      # create or refresh views if config.ensure_views
      # validate schema if config.validate_schema
      registry = DatasetRegistry(con)
      return StorageGateway(config=config, con=con, registry=registry)
  ```

* Expose a very small surface on `StorageGateway`, e.g.:

  ```python
  @dataclass
  class StorageGateway:
      config: StorageConfig
      con: duckdb.DuckDBPyConnection
      registry: DatasetRegistry

      def execute(self, sql: str, *params: object) -> duckdb.DuckDBPyRelation:
          return self.con.execute(sql, params)

      def table(self, name: str) -> duckdb.DuckDBPyRelation:
          return self.con.table(name)
  ```

**Rule for the rest of the codebase:**

> You may access `gateway.con` in leaf helpers (analytics code, ingestion detail functions), but pipeline/entrypoint functions must accept a `StorageGateway` and never create their own connection.

---

## 4. Refactor ingestion & analytics entrypoints

### 4.1 General pattern for entrypoints

For any ingestion/analytics module that currently has:

```python
def run_something(con: duckdb.DuckDBPyConnection, ctx: SomeContext) -> None:
    ...
```

refactor to:

```python
from codeintel.storage.gateway import StorageGateway

def run_something(gateway: StorageGateway, ctx: SomeContext) -> None:
    con = gateway.con
    ...
```

**Example (illustrative) – function history**

Before:

```python
# analytics/function_history.py
def build_function_history(
    con: duckdb.DuckDBPyConnection,
    *,
    repo: str,
    history_db_path: Path,
) -> None:
    con.execute("ATTACH ...")  # etc
    ...
```

After:

```python
from codeintel.storage.gateway import StorageGateway

def build_function_history(
    gateway: StorageGateway,
    *,
    repo: str,
) -> None:
    con = gateway.con
    # assume history DB is already attached if needed
    con.execute("""
        INSERT INTO analytics.function_history
        SELECT ...
    """)
```

Note: attaching a history DB should be done by `open_gateway` based on `StorageConfig.attach_history`, *not* by this function.

### 4.2 Apply pattern to ingestion entrypoints

Target modules include (names approximate, based on your tree):

* `ingestion/repo_scan.py` – entrypoint that populates `core.modules`, `core.file_state`, etc.
* `ingestion/py_ast_extract.py` – AST ingestion.
* `ingestion/cst_extract.py` – CST ingestion.
* `ingestion/docstrings_ingest.py`
* `ingestion/typing_ingest.py`
* `ingestion/tests_ingest.py`
* `ingestion/coverage_ingest.py`
* `ingestion/scip_ingest.py`
* `ingestion/config_ingest.py`

**Concrete step for each:**

1. Identify the top-level “entry” function (`ingest_*`, `run_*` or similar) that the pipeline calls.
2. Change the signature from `con: DuckDBPyConnection` to `gateway: StorageGateway`.
3. Inside, grab `con = gateway.con` and pass that down to low-level helpers.

Example (simplified) for AST:

```python
# BEFORE
def ingest_python_ast(
    con: duckdb.DuckDBPyConnection,
    modules: Iterable[ModuleRecord],
    ...
) -> None:
    ...

# AFTER
from codeintel.storage.gateway import StorageGateway

def ingest_python_ast(
    gateway: StorageGateway,
    modules: Iterable[ModuleRecord],
    ...
) -> None:
    con = gateway.con
    ...
```

### 4.3 Apply pattern to analytics entrypoints

Similarly, update analytics modules:

* `analytics/function_metrics.py`
* `analytics/function_history.py`
* `analytics/history_timeseries.py`
* `analytics/graph_metrics_*.py`
* `analytics/subsystems.py`, etc.

Anything that `orchestration/steps.py` calls directly should accept a `StorageGateway`.

---

## 5. Refactor Prefect flow & pipeline wiring

### 5.1 Centralize gateway lifecycle in Prefect

In `orchestration/prefect_flow.py` you already have `_get_gateway()` and `_close_gateways()`; the goal is to make these the **only** DB lifecycle functions for the flow.

**Target state:**

```python
_GATEWAY_CACHE: dict[str, StorageGateway] = {}

def _get_gateway(config: StorageConfig) -> StorageGateway:
    key = str(config.db_path) + "|" + str(config.read_only)
    try:
        return _GATEWAY_CACHE[key]
    except KeyError:
        gateway = open_gateway(config)
        _GATEWAY_CACHE[key] = gateway
        return gateway

def _close_gateways() -> None:
    for gateway in _GATEWAY_CACHE.values():
        gateway.con.close()
    _GATEWAY_CACHE.clear()
```

Then in the Prefect flow:

```python
@flow(...)
def export_flow(args: ExportArgs) -> None:
    config = StorageConfig.for_ingest(args.db_path, history_db_path=args.history_db_path)

    try:
        gateway = _get_gateway(config)
        ctx = build_pipeline_context(args, gateway=gateway)
        run_pipeline(ctx)  # uses orchestration.steps
    finally:
        _close_gateways()
```

Key constraints:

* `export_flow` (or equivalent) **never** calls `duckdb.connect` directly.
* All tasks that need DB access receive a `StorageGateway` (or something that contains one, like `PipelineContext`).

### 5.2 Make the steps layer use `StorageGateway`

In `orchestration/steps.py`, check your `PipelineContext`:

* Ensure it contains `gateway: StorageGateway` instead of `con: DuckDBPyConnection`.
* Update each `PipelineStep.run()` to call ingestion/analytics entrypoints with `ctx.gateway`.

Example:

```python
@dataclass
class PipelineContext:
    gateway: StorageGateway
    snapshot: SnapshotConfig
    # other fields...

@dataclass
class AstStep(PipelineStep):
    def run(self, ctx: PipelineContext) -> None:
        modules = fetch_modules(ctx.gateway)  # or fetch via gateway.con
        ingest_python_ast(ctx.gateway, modules=modules, ...)
```

Now the whole data path is:

`Prefect flow -> StorageConfig -> open_gateway -> StorageGateway -> PipelineContext -> Steps -> Ingestion/Analytics`

No bare connections.

---

## 6. Refactor CLI and FastAPI server surfaces

### 6.1 CLI entrypoints

In `src/codeintel/cli/main.py` (Typer app):

* For commands that currently build a DuckDB connection directly, switch to:

  ```python
  @app.command()
  def export(
      db: Path = Option(...),
      read_only: bool = Option(False),
      ...
  ) -> None:
      config = StorageConfig.for_ingest(db)
      gateway = open_gateway(config)
      # either:
      run_export(gateway=gateway, ...)
      # or delegate to Prefect, passing the db path and letting Prefect call open_gateway
  ```

* If the CLI uses Prefect as a subprocess, ensure that you either:

  * Let Prefect manage gateways entirely, or
  * Pass DB path, not a connection.

### 6.2 FastAPI server

In `src/codeintel/server/fastapi.py`:

* In your dependency injection / startup code, make sure you *only* ever construct a `StorageGateway` via `open_gateway`.
* Expose the gateway (or a thin service wrapping it) via FastAPI dependencies:

  ```python
  def get_gateway() -> StorageGateway:
      # could be a cached singleton per process
      return open_gateway(StorageConfig.for_readonly(settings.db_path))

  @router.get("/function/{goid}")
  def get_function(goid: str, gateway: StorageGateway = Depends(get_gateway)):
      row = gateway.con.execute("SELECT ... WHERE goid = ?", [goid]).fetchone()
      ...
  ```

No direct DuckDB connect inside route handlers.

---

## 7. Update tests to use `StorageGateway`

### 7.1 Gateway fixtures

In `tests/_helpers/gateway.py` (or similar):

* Introduce fixtures that build a temporary DB and corresponding `StorageGateway`:

  ```python
  @pytest.fixture
  def gateway(tmp_path: Path) -> Iterator[StorageGateway]:
      db_path = tmp_path / "test.duckdb"
      config = StorageConfig(
          db_path=db_path,
          read_only=False,
          apply_schema=True,
          ensure_views=True,
          validate_schema=True,
      )
      gateway = open_gateway(config)
      try:
          yield gateway
      finally:
          gateway.con.close()
  ```

* Replace tests that use `duckdb.connect` or `DuckDBPyConnection` with these fixtures.

### 7.2 Update analytics/ingestion tests

For tests like:

* `tests/ingestion/test_scip_ingest.py`
* `tests/analytics/test_history_timeseries.py`
* `tests/test_pipeline_smoke.py`

Change:

```python
def test_something(con: DuckDBPyConnection) -> None:
    ingest_python_ast(con, ...)
```

to:

```python
def test_something(gateway: StorageGateway) -> None:
    ingest_python_ast(gateway, ...)
```

If you have helpers in `tests/_helpers/fixtures.py` that currently expose raw connections, change them to expose `StorageGateway` instead.

---

## 8. Final cleanup & guardrails

### 8.1 Delete or wrap remaining `duckdb` imports

Once the above is done:

* Run a repo-wide search for `duckdb.` and `DuckDBPyConnection`.
* For any occurrence outside `storage/gateway.py` (and maybe `storage/schemas.py` if it exists), either:

  * Replace it with usage of `StorageGateway`, or
  * If really necessary (e.g. low-level migration script), add a big comment:

    ```python
    # NOTE: this is one of the very few places allowed to call duckdb.connect directly.
    # All normal runtime code must go through StorageGateway.
    ```

### 8.2 Optional: add a “no-duckdb-imports-outside-storage” test

You can add a simple test or lint rule that asserts the invariant:

* In `tests/architecture/test_duckdb_boundaries.py`:

  ```python
  def test_duckdb_imports_are_localized() -> None:
      sources = _iter_python_files(Path("src/codeintel"))
      for path, text in sources:
          if "storage/gateway.py" in str(path):
              continue
          if "import duckdb" in text or "DuckDBPyConnection" in text:
              pytest.fail(f"duckdb usage not allowed outside storage/gateway.py: {path}")
  ```

This will keep future changes from reintroducing raw connections.

---

If you’d like, next step could be: I can turn this plan into a *concrete patch plan* for one or two specific modules (e.g. “refactor `function_history` + `history_timeseries` to `StorageGateway`”) with more exact signatures and sequences of edits your AI coding agent can apply.
