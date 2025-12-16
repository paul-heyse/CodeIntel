Got it — you want an immediate response. I’ve updated the plan accordingly.


Got it — and yes, given you’re still in design phase, I think the *best* move is a **deliberate serving-layer overhaul** that:

* treats **DuckDB as a read-only “published snapshot” artifact** (never serve from the writer DB),
* makes the serving surface **semantic-view-first** (Hamilton-native, registry-driven),
* removes/avoids the current “backend/service/operation catalog” indirection where it’s now legacy,
* and cleanly supports **FastAPI + FastMCP** in one process with **hot-swap** DB snapshots.

Below is an **exact refactor map** in the mergeable PR style you’ve been using, explicitly incorporating:

* a **read-only DuckDB pool per worker** (multiple connections; no shared single-connection bottleneck),
* a **ServingDBManager** that hot-swaps via an **atomic pointer update**,
* a **semantic view query builder** using DuckDB’s **Relational API** *plus* **parameter binding** for user inputs (no string-format SQL),
* and a **/meta endpoint + MCP tool** exposing semantic layer + schema inventory.

I’m going to propose a **“parallel new stack → cut over → delete legacy”** approach so each PR stays mergeable even if you choose to keep old endpoints temporarily.

---

## Why this overhaul is the right serving shape

### 1) DuckDB concurrency & why you want a pool

DuckDB supports multiple connections (one writer + multiple readers), and within one process you can create additional handles; but **a single connection used across threads serializes queries**, which becomes a bottleneck under HTTP/MCP concurrency. 
So the serving layer should **not** share a single `duckdb.connect()` connection. Instead: **a small pool per Uvicorn worker**.

Also: open serving connections in **read-only mode** to prevent accidental writes. 

### 2) Safety: parameter binding, not interpolated SQL

DuckDB’s Python client supports `?` and `$name` parameter binding; you should avoid Python string formatting for user-provided values. 
That maps perfectly to an MCP tool where the LLM supplies filter values.

### 3) Query construction: Relations (lazy) + controlled execution

DuckDB Relations give you a *structured* query builder and lazy execution; you can chain `.filter()`, `.project()`, `.order()` etc. 
For serving, the best pattern is:

* use Relations for **structural transforms** (select columns, joins if needed, ordering, limiting),
* use parameter binding for **user values** (via `execute(..., params)` on the final SQL or via narrowly controlled filter composition).

### 4) Hamilton-native semantic layer is the “unification point”

Hamilton tags and metadata are intended exactly for this: attach tags (`@tag`) and query/filter nodes by tags; attach lightweight schemas (`@schema`) and validations (`@check_output`). 
And Hamilton’s CLI can compute a stable DAG “version hash” (`hamilton version`) and diff DAGs (`hamilton diff`), which is excellent for your semantic-layer versioning and CI gating. 

### 5) FastAPI + FastMCP: mount the MCP server as an ASGI sub-app

FastMCP supports mounting into FastAPI via `http_app()`, returning an ASGI application that FastAPI can `mount()`. ([FastMCP][1])
FastMCP also supports running over HTTP transport (e.g., `mcp.run(transport="http")`) for remote connectivity. ([DataCamp][2])

---

## Target end-state architecture under `src/codeintel/serving/`

Here’s the shape I’d move you to (the “new core” is **small** and replaces the current layered stack):

```
src/codeintel/serving/
  db/
    pointer.py            # ServingSnapshotPointer (read atomic pointer JSON)
    pool.py               # DuckDBReadPool (read-only, per worker)
    manager.py            # ServingDBManager (hot-swap pools on pointer change)
  semantic/
    registry.py           # SemanticRegistry (loaded from published semantic_registry.json)
    inventory.py          # SchemaInventory (tables/views + columns/types from schema manifest)
    query_builder.py      # Safe query builder: validates, builds SQL + params OR relation chain
    kernel.py             # SemanticQueryKernel (list/describe/query + meta)
    models.py             # Pydantic models for semantic query requests/responses
  http/
    app.py                # create_fastapi_app(): routes + lifespan + state wiring
    routes/
      semantic.py         # /semantic/* endpoints
      meta.py             # /meta endpoint (semantic version + inventory)
      health.py           # /health endpoint (db pointer + ping)
  mcp/
    server.py             # create_mcp_server(kernel) + http_app mount helpers
    tools_semantic.py     # MCP tools: list/describe/query semantic views
    tools_meta.py         # MCP tool: serving_meta (version + inventory)
  legacy/                 # (temporary) old serving stack moved here until deleted
    ...
```

**Key consolidation:** the HTTP and MCP surfaces both talk to the **same** `SemanticQueryKernel` and the **same** `ServingDBManager`.

---

## PR-by-PR refactor board (mergeable)

I’ll number these starting at **PR‑74** (post PR‑73 schema alignment), since that’s where you said you want to land.

---

# PR‑74 — Introduce published serving snapshot pointer + artifact layout

### Goal

Define the **published serving snapshot contract**: a directory containing:

* the read-only DuckDB file,
* semantic registry JSON,
* schema inventory/manifest JSON,
* a single **atomic pointer** (`current.json`) that serving reads.

### Tasks checklist

* [ ] Add `src/codeintel/serving/db/pointer.py`
* [ ] Define `ServingSnapshotPointer`:

  * `db_path`
  * `semantic_registry_path`
  * `schema_manifest_path`
  * `repo`, `commit`, `run_id`
  * `published_at`
  * `semantic_layer_version` (string/hash)
* [ ] Add writer-side publisher (if you don’t already have it under build):

  * Write snapshot DB file to `.../serving_snapshots/<run_id>.duckdb`
  * Write `semantic_registry.json` and `schema_manifest.json`
  * Write `current.json` via **atomic replace**:

    * write `current.json.tmp`
    * `os.replace(tmp, current.json)` (atomic on same filesystem)

### Code sketch

```python
# src/codeintel/serving/db/pointer.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

@dataclass(frozen=True)
class ServingSnapshotPointer:
    db_path: Path
    semantic_registry_path: Path
    schema_manifest_path: Path
    repo: str
    commit: str
    run_id: str
    published_at: datetime
    semantic_layer_version: str

    @classmethod
    def load(cls, path: Path) -> "ServingSnapshotPointer":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            db_path=Path(raw["db_path"]).resolve(),
            semantic_registry_path=Path(raw["semantic_registry_path"]).resolve(),
            schema_manifest_path=Path(raw["schema_manifest_path"]).resolve(),
            repo=raw["repo"],
            commit=raw["commit"],
            run_id=raw["run_id"],
            published_at=datetime.fromisoformat(raw["published_at"]),
            semantic_layer_version=raw["semantic_layer_version"],
        )
```

### Tests checklist

Add under something like `tests/server/` or `tests/serving/`:

* [ ] `test_serving_pointer_loads_and_validates_paths()`

### Snapshot changes

None.

---

# PR‑75 — DuckDBReadPool (read-only) + ServingDBManager hot-swap

### Goal

Implement:

* **DuckDBReadPool**: N read-only connections per worker
* **ServingDBManager**: watches `current.json` and swaps pools safely

### Why

DuckDB can reuse multiple connections; a single connection is thread-safe but serializes concurrent queries, so you want the pool. 
Use `read_only=True` for serving. 

Also configure perf limits at connect-time (`threads`, `memory_limit`) or via `SET`. 

### Tasks checklist

* [ ] Add `src/codeintel/serving/db/pool.py`
* [ ] Add `src/codeintel/serving/db/manager.py`
* [ ] Implement graceful swap:

  * old pool is marked “closing”
  * in-use connections close on release
  * available connections close immediately
  * new pool becomes active

### Code sketch: pool

```python
# src/codeintel/serving/db/pool.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import threading
from queue import LifoQueue, Empty

import duckdb

@dataclass(frozen=True)
class DuckDBPoolConfig:
    size: int = 4
    threads: int | None = None
    memory_limit: str | None = None
    temp_directory: str | None = None

class DuckDBReadPool:
    def __init__(self, db_path: Path, cfg: DuckDBPoolConfig) -> None:
        self._db_path = db_path
        self._cfg = cfg
        self._available: LifoQueue[duckdb.DuckDBPyConnection] = LifoQueue()
        self._lock = threading.Lock()
        self._in_use: set[duckdb.DuckDBPyConnection] = set()
        self._closing = False
        self._init_connections()

    def _open(self) -> duckdb.DuckDBPyConnection:
        config: dict[str, object] = {}
        if self._cfg.threads is not None:
            config["threads"] = self._cfg.threads
        if self._cfg.memory_limit is not None:
            config["memory_limit"] = self._cfg.memory_limit
        con = duckdb.connect(str(self._db_path), read_only=True, config=config)
        if self._cfg.temp_directory is not None:
            con.execute("SET temp_directory = ?", [self._cfg.temp_directory])
        return con

    def _init_connections(self) -> None:
        for _ in range(max(1, self._cfg.size)):
            self._available.put(self._open())

    def acquire(self) -> duckdb.DuckDBPyConnection:
        if self._closing:
            raise RuntimeError("Pool is closing")
        con = self._available.get()
        with self._lock:
            self._in_use.add(con)
        return con

    def release(self, con: duckdb.DuckDBPyConnection) -> None:
        with self._lock:
            self._in_use.discard(con)
            closing = self._closing
        if closing:
            con.close()
            return
        self._available.put(con)

    def close_gracefully(self) -> None:
        with self._lock:
            self._closing = True
        # Close everything that isn't currently borrowed:
        while True:
            try:
                con = self._available.get_nowait()
            except Empty:
                break
            con.close()
```

Notes:

* This uses parameterized `SET temp_directory = ?` and avoids interpolating. 
* Connect-time config for threads/memory is supported. 

### Code sketch: ServingDBManager

```python
# src/codeintel/serving/db/manager.py
from __future__ import annotations
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .pointer import ServingSnapshotPointer
from .pool import DuckDBPoolConfig, DuckDBReadPool

@dataclass
class ServingDBManager:
    pointer_path: Path
    pool_cfg: DuckDBPoolConfig
    poll_interval_s: float = 1.0

    _pointer: ServingSnapshotPointer | None = None
    _pool: DuckDBReadPool | None = None
    _watch_task: asyncio.Task[None] | None = None
    _last_mtime_ns: int | None = None

    async def start(self) -> None:
        await self._reload_if_needed(force=True)
        self._watch_task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        if self._watch_task is not None:
            self._watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch_task
        if self._pool is not None:
            self._pool.close_gracefully()

    def current_pointer(self) -> ServingSnapshotPointer:
        if self._pointer is None:
            raise RuntimeError("ServingDBManager not started")
        return self._pointer

    def acquire(self):
        if self._pool is None:
            raise RuntimeError("ServingDBManager not started")
        return self._pool.acquire()

    def release(self, con) -> None:
        if self._pool is None:
            con.close()
            return
        self._pool.release(con)

    async def _watch_loop(self) -> None:
        while True:
            await self._reload_if_needed(force=False)
            await asyncio.sleep(self.poll_interval_s)

    async def _reload_if_needed(self, *, force: bool) -> None:
        st = self.pointer_path.stat()
        if not force and self._last_mtime_ns == st.st_mtime_ns:
            return
        self._last_mtime_ns = st.st_mtime_ns

        new_ptr = ServingSnapshotPointer.load(self.pointer_path)
        if self._pointer is not None and new_ptr.db_path == self._pointer.db_path:
            self._pointer = new_ptr
            return

        new_pool = DuckDBReadPool(new_ptr.db_path, self.pool_cfg)
        old_pool = self._pool
        self._pool = new_pool
        self._pointer = new_ptr
        if old_pool is not None:
            old_pool.close_gracefully()
```

### Tests checklist

* [ ] `tests/server/test_duckdb_pool_readonly.py`

  * create DB file with a table
  * open pool
  * ensure queries succeed
* [ ] `tests/server/test_serving_db_manager_hot_swap.py`

  * pointer → db1, query returns db1 result
  * atomically replace pointer → db2
  * wait for reload
  * query returns db2 result

### Snapshot changes

None.

---

# PR‑76 — SemanticRegistry + SchemaInventory (loaded from published artifacts)

### Goal

Stop relying on:

* `codeintel.config.datasets.*` (legacy),
* the serving `operations/catalog.py` (legacy),
* ad-hoc dataset registry composition in serving.

Instead, load:

* `semantic_registry.json` (semantic view definitions),
* `schema_manifest.json` (table/view schemas + column types, already aligned via PR‑73).

### Tasks checklist

* [ ] Add `src/codeintel/serving/semantic/registry.py`
* [ ] Add `src/codeintel/serving/semantic/inventory.py`
* [ ] Wire the registry into ServingDBManager pointer (paths come from pointer)
* [ ] Add a small `SemanticLayerMeta` object:

  * version hash
  * counts of views
  * inventory summary

### Hamilton integration note (important)

This is where your Hamilton DAG metadata becomes “serving-native”:

* `@tag(...)` provides semantic grouping/IDs 
* `hamilton version` provides a stable semantic-layer code version 

### Tests checklist

* [ ] `tests/server/test_semantic_registry_load.py`
* [ ] `tests/server/test_schema_inventory_load.py`

### Snapshot changes

None.

---

# PR‑77 — Semantic query builder (Relations + parameter binding) + SemanticQueryKernel

### Goal

Implement the *one* query API both HTTP and MCP use:

* list views
* describe a view (columns, types, grain, docstring)
* query a view with:

  * filters
  * select columns
  * ordering
  * pagination/limits

### Key design constraints

* **No arbitrary SQL input from the LLM**.
* View/table identifiers must come from registry.
* Column names must be validated against schema inventory.
* Values must be passed as **bound params** (no string formatting). 

### Query construction approach

* Use DuckDB’s relational API for query building patterns where it is expressive (`filter`, `project`, `order`, etc.). 
* Use DuckDB parameter binding for user-provided values via `execute(sql, params)`. 

### Code sketch: models

```python
# src/codeintel/serving/semantic/models.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal

Op = Literal["eq","ne","lt","lte","gt","gte","in","contains","startswith","endswith"]

class FilterSpec(BaseModel):
    column: str
    op: Op
    value: object

class SemanticQueryRequest(BaseModel):
    view_id: str
    select: list[str] | None = None
    filters: list[FilterSpec] = Field(default_factory=list)
    order_by: list[str] = Field(default_factory=list)  # e.g. ["-risk_score", "module"]
    limit: int = 50
    offset: int = 0

class SemanticQueryResponse(BaseModel):
    view_id: str
    columns: list[str]
    rows: list[list[object]]
    truncated: bool
```

### Code sketch: safe SQL builder (bound params)

```python
# src/codeintel/serving/semantic/query_builder.py
import re

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def quote_table_key(table_key: str) -> str:
    parts = table_key.split(".")
    if not all(_IDENT.match(p) for p in parts):
        raise ValueError(f"Invalid table key: {table_key}")
    return ".".join(f'"{p}"' for p in parts)

def quote_col(col: str) -> str:
    if not _IDENT.match(col):
        raise ValueError(f"Invalid column: {col}")
    return f'"{col}"'

def build_query_sql(
    *,
    table_key: str,
    select: list[str],
    filters: list[tuple[str, str, object]],
    order_by: list[tuple[str, str]],
    limit: int,
    offset: int,
) -> tuple[str, dict[str, object]]:
    table_sql = quote_table_key(table_key)
    select_sql = ", ".join(quote_col(c) for c in select)

    where_parts: list[str] = []
    params: dict[str, object] = {}
    for i, (col, op, value) in enumerate(filters):
        p = f"p{i}"
        c = quote_col(col)
        if op == "eq":
            where_parts.append(f"{c} = ${p}")
            params[p] = value
        elif op == "in":
            if not isinstance(value, list):
                raise ValueError("IN expects list")
            # build ($p0_0,$p0_1,...) safely
            names = []
            for j, v in enumerate(value):
                pj = f"{p}_{j}"
                names.append(f"${pj}")
                params[pj] = v
            where_parts.append(f"{c} IN ({','.join(names)})")
        else:
            raise ValueError(f"Unsupported op: {op}")

    where_sql = "" if not where_parts else (" WHERE " + " AND ".join(where_parts))

    order_sql = ""
    if order_by:
        bits = []
        for col, direction in order_by:
            bits.append(f"{quote_col(col)} {direction}")
        order_sql = " ORDER BY " + ", ".join(bits)

    sql = f"SELECT {select_sql} FROM {table_sql}{where_sql}{order_sql} LIMIT {limit} OFFSET {offset}"
    return sql, params
```

This is the “hard” part: **validate everything** and bind values (DuckDB supports `$name` + dict). 

### Tests checklist

* [ ] `tests/server/test_semantic_query_builder.py`

  * rejects unknown column
  * binds params correctly
  * `IN` expands safely
* [ ] `tests/server/test_semantic_kernel_query.py`

  * runs against a tiny DuckDB file (single semantic view)

### Snapshot changes

None.

---

# PR‑78 — Refactor FastAPI app to use ServingDBManager + SemanticQueryKernel

### Goal

Stop the FastAPI server from depending on:

* `build_backend_resource`
* the legacy `QueryService` + backend layers
* auto-pipeline

…and instead wire:

* `ServingDBManager` into app state
* `SemanticQueryKernel` into app state
* new `/semantic/*` endpoints
* new `/meta` endpoint (semantic version + inventory)

### Tasks checklist

* [ ] Add `src/codeintel/serving/http/app.py` as the new canonical app factory
* [ ] Update lifespan:

  * start DBManager watcher task
  * load registry+inventory (and reload on swap)
* [ ] Add new router `http/routes/semantic.py`:

  * GET `/semantic/views`
  * GET `/semantic/views/{view_id}`
  * POST `/semantic/query`
* [ ] Update `/meta` and `/health` to use pointer/kernel (not legacy ops catalog)

### Tests checklist

* [ ] `tests/http/test_semantic_endpoints.py` using `TestClient`
* [ ] `tests/http/test_meta_semantic_inventory.py`

### Snapshot changes

None.

---

# PR‑79 — Refactor MCP server: mount into FastAPI + semantic tools + meta tool

### Goal

Replace the old tool surface with:

* `semantic_list_views`
* `semantic_describe_view`
* `semantic_query_view`
* `serving_meta`

And mount MCP into FastAPI for remote serving.

FastMCP mounting into FastAPI is supported by `http_app()`. ([FastMCP][1])
FastMCP can also be run directly in HTTP transport mode (e.g. `mcp.run(transport="http")`) if you want a pure MCP server. ([DataCamp][2])

### Tasks checklist

* [ ] Add `src/codeintel/serving/mcp/tools_semantic.py`
* [ ] Add `src/codeintel/serving/mcp/tools_meta.py`
* [ ] Update `src/codeintel/serving/mcp/server.py`:

  * accept a `SemanticQueryKernel`
  * register semantic tools
* [ ] Add helper to mount MCP app into FastAPI:

  * `mcp_asgi = mcp.http_app(path="/mcp")`
  * `fastapi_app.mount("/mcp", mcp_asgi)`

### Code sketch: FastAPI + MCP mounted in one process

```python
# src/codeintel/serving/http/app.py
from fastapi import FastAPI
from codeintel.serving.mcp.server import create_mcp_server

def create_app(...) -> FastAPI:
    app = FastAPI(...)
    # build db_manager + kernel, attach to app.state, include routers, etc.

    mcp, _close = create_mcp_server(...)

    # FastMCP -> ASGI sub-app mount:
    mcp_app = mcp.http_app(path="/")   # or http_app() depending on your version
    app.mount("/mcp", mcp_app)
    return app
```

### Tests checklist

* [ ] `tests/mcp/test_semantic_tools.py`
* [ ] `tests/mcp/test_meta_tool_semantic_inventory.py`

### Snapshot changes

None.

---

# PR‑80 — Delete/relocate legacy serving layers + remove auto-pipeline from serving

### Goal

Once semantic serving is your only path:

* delete or move the legacy stack to `serving/legacy/` temporarily, then remove it fully:

  * `serving/backend/*` (DuckDBQueryService, repositories usage)
  * `serving/services/*` (QueryService, HttpQueryService, LocalQueryService)
  * `serving/operations/catalog.py`
  * `serving/auto_pipeline.py`
  * dataset registry helpers that depend on `codeintel.config.datasets.*`

### Tasks checklist

* [ ] Remove legacy imports in FastAPI and MCP
* [ ] Replace CLI `op.list/op.call` with `semantic.list/semantic.query` (optional but strongly recommended)
* [ ] Delete tests that only validate old tool surfaces
* [ ] Update docs/OpenAPI

### Tests checklist

* [ ] `tests/architecture/test_layering_serving_imports` updated to reflect new boundaries
* [ ] Ensure `tests/mcp/*` pass after tool surface update

### Snapshot changes

* Potentially update any CLI golden snapshots if you rename commands.

---

## /meta shape: what it should expose (HTTP + MCP)

This is what I’d return from both `/meta` and `serving_meta` tool:

```json
{
  "repo": "my-org/my-repo",
  "commit": "abc123",
  "run_id": "2025-12-15T05:10:22Z__abc123",
  "published_at": "2025-12-15T05:10:25Z",
  "semantic_layer_version": "hamilton:<hash> schema:<hash> registry:<hash>",
  "duckdb": {
    "db_path": "/.../serving_snapshots/<run_id>.duckdb",
    "read_only": true,
    "pool_size": 4
  },
  "semantic_views": [
    {
      "view_id": "functions.summary",
      "table_key": "docs.v_function_summary",
      "grain": "per_function",
      "columns": ["goid_h128", "module", "name", "risk_score", ...]
    }
  ],
  "schema_inventory": {
    "tables": 87,
    "views": 32
  }
}
```

This makes the MCP consumer experience dramatically simpler: the agent can *always* discover “what exists” and “how to query it” without you maintaining a bespoke catalog.

---

## Extra “best-in-class” serving features worth adding (optional but high-value)

These come straight from advanced DuckDB/Hamilton capabilities:

### A) Query explain tool (debug only)

DuckDB supports `EXPLAIN` and `EXPLAIN ANALYZE` for performance debugging. 
Add MCP tool:

* `semantic_explain(view_id, filters, ...) -> plan_text`

Gate it behind `CODEINTEL_SERVING_DEBUG=1`.

### B) Per-connection tuning

Set threads/memory via connect config. 
Expose these via ServingConfig:

* `CODEINTEL_SERVING_DUCKDB_THREADS`
* `CODEINTEL_SERVING_DUCKDB_MEMORY_LIMIT`
* `CODEINTEL_SERVING_DUCKDB_POOL_SIZE`

### C) Hamilton-native view tagging later

When you’re ready for tag taxonomy, the semantic registry compilation becomes “automatic”:

* `@tag(semantic="true", semantic_id="functions.summary", grain="per_function", entity="function")` 
* `hamilton version` baked into semantic layer version string 

---

## Answering your “refactor vs rewrite?” explicitly

Given where you are:

* If you want **mergeability** and minimal disruption → do the PR plan above (parallel new stack, then cut over).
* If you want **fastest path to the cleanest design** → you can *skip straight to PR‑78/79 style*, then delete legacy immediately. You’re not in prod, so that’s viable.

But even in “big bang”, I’d still keep the **ServingDBManager + pointer** approach: it is the clean boundary between build and serve, and it scales.

---

[1]: https://gofastmcp.com/integrations/fastapi?utm_source=chatgpt.com "FastAPI 🤝 FastMCP"
[2]: https://www.datacamp.com/tutorial/building-mcp-server-client-fastmcp?utm_source=chatgpt.com "Building an MCP Server and Client with FastMCP 2.0"
