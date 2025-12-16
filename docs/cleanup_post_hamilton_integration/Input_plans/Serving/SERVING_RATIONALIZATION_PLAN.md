# Serving Module Rationalization Plan

> **Status**: Ready for Implementation  
> **Priority**: Medium  
> **Estimated Effort**: 1-2 days  
> **Dependencies**: None (can proceed independently)  
> **Last Updated**: December 2024

## Executive Summary

The `serving` module has evolved to a clean semantic-first architecture, but contains redundancies with the `storage` layer that should be consolidated. This plan documents opportunities to:

1. Move connection pooling to storage layer (yielding gateways, not raw connections)
2. Eliminate duplicate Ibis connection handling
3. Simplify gateway usage patterns (reuse per-request, not per-call)
4. Use existing storage utilities (`duckdb_schema_exists`)

**Goal**: Reduce serving module complexity by ~15% while improving maintainability through proper layer separation and gateway reuse.

---

## Table of Contents

1. [Current Architecture](#current-architecture)
2. [Storage Module Reference](#storage-module-reference)
3. [Redundancy Analysis](#redundancy-analysis)
4. [Proposed Architecture](#proposed-architecture)
5. [Implementation Phases](#implementation-phases)
6. [File-by-File Changes](#file-by-file-changes)
7. [Migration Guide](#migration-guide)
8. [Testing Strategy](#testing-strategy)
9. [Rollback Plan](#rollback-plan)

---

## Current Architecture

### Serving Module Structure (Post-Cleanup)

```
serving/
├── __init__.py                    # Public API exports
├── settings.py                    # ServingSettings (env config)
├── db/
│   ├── __init__.py
│   ├── manager.py                 # ServingDBManager (hot-swap support)
│   ├── pointer.py                 # ServingSnapshotPointer
│   └── pool.py                    # DuckDBReadPool ← REDUNDANCY
├── semantic/
│   ├── __init__.py
│   ├── kernel.py                  # SemanticQueryKernel ← REDUNDANCY
│   ├── query_builder.py           # Safe Ibis query building
│   ├── registry.py                # SemanticRegistry
│   ├── inventory.py               # SchemaInventory (KEEP AS-IS)
│   └── models.py                  # Pydantic models
├── search/
│   ├── __init__.py
│   └── models.py                  # Search models
├── contracts/
│   └── check_operation_contracts.py
├── http/
│   ├── app.py                     # FastAPI factory
│   └── routes/
│       ├── search.py
│       └── semantic.py
└── mcp/
    ├── app.py                     # FastMCP builder
    └── server.py                  # MCP server factory
```

---

## Storage Module Reference

The storage module has evolved to provide a clean gateway architecture:

```
storage/
├── gateway/
│   ├── protocol.py                # MinimalGateway, StorageGateway protocols
│   ├── minimal.py                 # MinimalStorageGateway (composition root)
│   ├── config.py                  # StorageConfig with for_readonly()
│   ├── connection.py              # connect()
│   ├── factory.py                 # open_gateway(), open_memory_gateway()
│   └── ephemeral.py               # ephemeral_gateway() context manager
├── ibis_adapter.py                # IbisGateway
├── duckdb_policy_backend.py       # DuckDBPolicyBackend + duckdb_schema_exists()
└── serving/
    └── search_index.py            # FTS index building
```

### Key Storage Patterns

1. **MinimalStorageGateway** is the composition root:
   ```python
   class MinimalStorageGateway:
       @property
       def con(self) -> DuckDBPyConnection: ...
       @property
       def ibis(self) -> IbisGateway: ...      # Lazy, cached
       @property
       def policy(self) -> DuckDBPolicyBackend: ...  # Lazy, cached
   ```

2. **IbisGateway.table()** handles qualified names properly:
   ```python
   def table(self, table_name: str) -> it.Table:
       if "." in table_name:
           database, name = table_name.split(".", 1)
           return self.con.table(name, database=database)
       return self.con.table(table_name)
   ```

3. **StorageConfig.for_readonly()** is designed for serving:
   ```python
   @classmethod
   def for_readonly(cls, db_path: Path) -> StorageConfig:
       return cls(db_path=db_path, read_only=True, ...)
   ```

4. **duckdb_schema_exists()** is a standalone utility:
   ```python
   def duckdb_schema_exists(con: DuckDBPyConnection, *, schema: str) -> bool:
       row = con.execute(
           "SELECT 1 FROM information_schema.schemata WHERE schema_name = ? LIMIT 1",
           [schema],
       ).fetchone()
       return row is not None
   ```

---

## Redundancy Analysis

### 1. Connection Pool Yields Raw Connections

**Location**: `serving/db/pool.py`

**Current Implementation**:
```python
class DuckDBReadPool:
    def __init__(self, db_path: Path, cfg: DuckDBPoolConfig) -> None:
        self._available: LifoQueue[DuckDBConnection] = LifoQueue()
        # ...

    def _open(self) -> DuckDBConnection:
        return connect(StorageConfig.for_readonly(self._db_path), duckdb_config=...)

    def acquire(self) -> DuckDBConnection:
        # Returns raw connection
        return self._available.get()
```

**Issue**: Pool correctly uses `StorageConfig.for_readonly()` but yields raw `DuckDBConnection`. Callers must then wrap with `MinimalStorageGateway(con)` on every use, which:
- Creates new `IbisGateway` and `DuckDBPolicyBackend` instances per call
- Violates the composition root pattern
- Wastes CPU/memory on redundant initialization

**Recommendation**: Move pool to storage and yield `MinimalStorageGateway` directly.

---

### 2. Ad-hoc Ibis Connection Creation

**Location**: `serving/semantic/kernel.py:248-257`

**Current Implementation**:
```python
def _execute_semantic_plan(
    self,
    *,
    con: DuckDBConnection,
    plan: SemanticQueryPlan,
) -> list[dict[str, object]]:
    ibis_con = ibis.duckdb.from_connection(con)  # ← Creates new Ibis backend!
    expr = build_query(ibis_con=ibis_con, plan=plan)
    sql = ibis_con.compile(expr)
    return self._execute_sql(con=con, sql=sql)
```

**Issue**: Creates a new Ibis backend connection for every query instead of using `MinimalStorageGateway(con).ibis.con`.

**Impact**:
- Duplicate Ibis backend lifecycle management
- Inconsistent with storage layer patterns
- Potential connection overhead

**Recommendation**: Pass `MinimalStorageGateway` through the call chain, use `gw.ibis.con`.

---

### 3. Gateway Creation Per SQL Call

**Location**: `serving/semantic/kernel.py` at lines 234, 414, 489

**Current Implementation**:
```python
def _execute_sql(
    self,
    *,
    con: DuckDBConnection,
    sql: str,
    params: Sequence[object] | None = None,
) -> list[dict[str, object]]:
    engine = self.settings.result_engine.lower()
    backend = MinimalStorageGateway(con).policy  # ← Creates gateway per call!
    result = backend.execute_sql(sql, params=params)
    # ...
```

Also at line 414 (explain):
```python
raw_rows = MinimalStorageGateway(con).policy.execute_sql(f"EXPLAIN {compiled}").fetchall()
```

And line 489 (search):
```python
backend = MinimalStorageGateway(con).policy
```

**Issue**: Creates a new `MinimalStorageGateway` for every SQL execution, which:
- Reinstantiates `IbisGateway` and `DuckDBPolicyBackend` each time
- Wastes memory and CPU cycles
- Violates the composition root pattern

**Recommendation**: Create gateway once per acquired connection (in pool) and reuse throughout request lifecycle.

---

### 4. Raw SQL for Schema Existence Check

**Location**: `serving/semantic/kernel.py:503-507`

**Current Implementation**:
```python
row = backend.execute_sql(
    "SELECT 1 FROM information_schema.schemata WHERE schema_name = ? LIMIT 1",
    [_SEARCH_FTS_SCHEMA],
).fetchone()
fts_available = row is not None
```

**Issue**: Duplicates logic that already exists in `storage/duckdb_policy_backend.py`:
```python
def duckdb_schema_exists(con: DuckDBPyConnection, *, schema: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM information_schema.schemata WHERE schema_name = ? LIMIT 1",
        [schema],
    ).fetchone()
    return row is not None
```

**Recommendation**: Import and use `duckdb_schema_exists` from storage.

---

### 5. SchemaInventory Parsing — No Change Needed

**Location**: `serving/semantic/inventory.py`

**Assessment**: The `SchemaInventory` class already imports `Column`, `Index`, `TableSchema` from `codeintel.core.schemas.primitives` and constructs them directly. The parsing logic is clean and focused on the serving manifest JSON format.

**Recommendation**: Keep as-is. The ~130 lines are well-organized and serving-specific. Adding a generic `TableSchema.from_dict()` would either be too permissive or duplicate the same validation.

---

### 6. FTS Query SQL Embedded in Kernel — Correctly Placed

**Location**: `serving/semantic/kernel.py:47-105`

**Assessment**: This is **correctly placed** — storage owns index creation (`storage/serving/search_index.py`), serving owns query execution. The SQL is read-only and specific to the serving surface.

**Recommendation**: Keep as-is. This is proper layer separation.

---

## Proposed Architecture

### Target State

```
storage/
├── gateway/
│   ├── protocol.py                # Add: ReadPoolGateway protocol (optional)
│   ├── minimal.py                 # MinimalStorageGateway (unchanged)
│   ├── config.py                  # Add: PoolConfig
│   ├── connection.py              # Unchanged
│   ├── pool.py                    # NEW: ReadPoolGateway implementation
│   ├── ephemeral.py               # Unchanged
│   └── factory.py                 # Unchanged
├── ibis_adapter.py                # Unchanged
├── duckdb_policy_backend.py       # Unchanged (already has duckdb_schema_exists)
└── serving/
    └── search_index.py            # Unchanged

serving/
├── __init__.py                    # Update exports
├── settings.py                    # Unchanged
├── db/
│   ├── __init__.py                # Update exports
│   ├── manager.py                 # REFACTOR: yield MinimalStorageGateway
│   ├── pointer.py                 # Unchanged
│   └── pool.py                    # THIN RE-EXPORT with deprecation
├── semantic/
│   ├── __init__.py                # Unchanged
│   ├── kernel.py                  # REFACTOR: use gateway pattern
│   ├── query_builder.py           # Unchanged
│   ├── registry.py                # Unchanged
│   ├── inventory.py               # Unchanged (correct as-is)
│   └── models.py                  # Unchanged
├── search/                        # Unchanged
├── contracts/                     # Unchanged
├── http/                          # Unchanged
└── mcp/                           # Unchanged
```

### Layer Responsibilities

| Layer | Responsibility | Does NOT Own |
|-------|---------------|--------------|
| `storage.gateway` | Connection lifecycle, pooling, gateway creation | Query semantics |
| `storage.serving` | FTS index building | Query execution |
| `serving.db` | Snapshot pointer, hot-swap coordination | Connection pooling |
| `serving.semantic` | Query building, result extraction | Connection management |
| `serving.http/mcp` | HTTP/MCP surfaces | Business logic |

---

## Implementation Phases

### Phase 1: Create ReadPoolGateway in Storage (2-3 hours)

**Goal**: Establish proper layering by moving connection pooling to storage with gateway yielding.

#### 1.1 Create `storage/gateway/pool.py`

```python
"""Read-only gateway pool for concurrent query execution.

This module provides pooled read-only access for serving scenarios
where multiple concurrent queries need separate connections.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, LifoQueue
from typing import TYPE_CHECKING

from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.gateway.connection import connect
from codeintel.storage.gateway.minimal import MinimalStorageGateway

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["PoolConfig", "ReadPoolGateway"]


@dataclass(frozen=True)
class PoolConfig:
    """Pool configuration parameters.

    Parameters
    ----------
    size
        Number of gateways in the pool.
    threads
        DuckDB threads per connection (None = default).
    memory_limit
        DuckDB memory limit per connection (e.g., "2GB").
    temp_directory
        Temporary directory for spilling.
    """

    size: int = 4
    threads: int | None = None
    memory_limit: str | None = None
    temp_directory: str | None = None


class ReadPoolGateway:
    """Thread-safe pool of read-only MinimalStorageGateway instances.

    Each pooled gateway provides full access to:
    - `gateway.con`: Raw DuckDB connection
    - `gateway.ibis`: IbisGateway for expression queries
    - `gateway.policy`: DuckDBPolicyBackend for SQL execution

    Parameters
    ----------
    db_path
        Path to DuckDB database file.
    cfg
        Pool configuration.

    Examples
    --------
    >>> pool = ReadPoolGateway(Path("catalog.duckdb"), PoolConfig(size=4))
    >>> with pool.acquire() as gw:
    ...     result = gw.policy.execute_sql("SELECT 1").fetchone()
    >>> pool.close_gracefully()
    """

    def __init__(self, db_path: Path, cfg: PoolConfig) -> None:
        self._db_path = db_path
        self._cfg = cfg
        self._available: LifoQueue[MinimalStorageGateway] = LifoQueue()
        self._lock = threading.Lock()
        self._in_use: set[MinimalStorageGateway] = set()
        self._closing = False
        self._init_gateways()

    def _build_duckdb_config(self) -> dict[str, bool | float | int | list[str] | str]:
        """Build DuckDB connection configuration from pool config."""
        duckdb_config: dict[str, bool | float | int | list[str] | str] = {}
        if self._cfg.threads is not None:
            duckdb_config["threads"] = self._cfg.threads
        if self._cfg.memory_limit is not None:
            duckdb_config["memory_limit"] = self._cfg.memory_limit
        if self._cfg.temp_directory is not None:
            duckdb_config["temp_directory"] = self._cfg.temp_directory
        return duckdb_config

    def _open_gateway(self) -> MinimalStorageGateway:
        """Open a new read-only gateway."""
        con = connect(
            StorageConfig.for_readonly(self._db_path),
            duckdb_config=self._build_duckdb_config(),
        )
        return MinimalStorageGateway(con)

    def _init_gateways(self) -> None:
        """Pre-create pool gateways."""
        for _ in range(max(1, self._cfg.size)):
            self._available.put(self._open_gateway())

    @contextmanager
    def acquire(self) -> Iterator[MinimalStorageGateway]:
        """Acquire a gateway from the pool.

        Yields
        ------
        MinimalStorageGateway
            Read-only gateway with ibis and policy access.

        Raises
        ------
        RuntimeError
            If pool is closing.
        """
        with self._lock:
            if self._closing:
                msg = "Pool is closing"
                raise RuntimeError(msg)

        gw = self._available.get()
        with self._lock:
            self._in_use.add(gw)
        try:
            yield gw
        finally:
            self._release(gw)

    def _release(self, gw: MinimalStorageGateway) -> None:
        """Return a gateway to the pool."""
        with self._lock:
            self._in_use.discard(gw)
            closing = self._closing
        if closing:
            gw.close()
            return
        self._available.put(gw)

    def close_gracefully(self) -> None:
        """Mark pool as closing and drain available gateways."""
        with self._lock:
            self._closing = True

        while True:
            try:
                gw = self._available.get_nowait()
            except Empty:
                break
            gw.close()
```

#### 1.2 Update `storage/gateway/__init__.py`

Add exports for new pool module:

```python
# Add to _EXPORTS dict:
"PoolConfig": ("codeintel.storage.gateway.pool", "PoolConfig"),
"ReadPoolGateway": ("codeintel.storage.gateway.pool", "ReadPoolGateway"),

# Add to __all__:
"PoolConfig",
"ReadPoolGateway",
```

---

### Phase 2: Refactor ServingDBManager (1 hour)

**Goal**: Have manager yield `MinimalStorageGateway` instead of raw connections.

#### 2.1 Update `serving/db/manager.py`

```python
"""Serving database manager with hot-swap support.

Watches the pointer file and swaps connection pools when the snapshot changes,
enabling zero-downtime deployments.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.storage.gateway.pool import PoolConfig, ReadPoolGateway

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.storage.gateway.minimal import MinimalStorageGateway


@dataclass
class ServingDBManager:
    """Manage serving database connections with hot-swap support.

    Parameters
    ----------
    pointer_path
        Path to current.json pointer file.
    pool_cfg
        Connection pool configuration.
    poll_interval_s
        Seconds between pointer file checks.
    """

    pointer_path: Path
    pool_cfg: PoolConfig = field(default_factory=PoolConfig)
    poll_interval_s: float = 1.0

    _pointer: ServingSnapshotPointer | None = field(default=None, init=False)
    _pool: ReadPoolGateway | None = field(default=None, init=False)
    _watch_task: asyncio.Task[None] | None = field(default=None, init=False)
    _last_mtime_ns: int | None = field(default=None, init=False)

    async def start(self) -> None:
        """Initialize manager and start watch loop."""
        await self._reload_if_needed(force=True)
        self._watch_task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        """Stop watch loop and close pool."""
        if self._watch_task is not None:
            self._watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch_task
        if self._pool is not None:
            self._pool.close_gracefully()

    def current_pointer(self) -> ServingSnapshotPointer:
        """Return current snapshot pointer.

        Returns
        -------
        ServingSnapshotPointer
            Active snapshot pointer.

        Raises
        ------
        RuntimeError
            If manager not started or pointer not yet available.
        """
        if self._pointer is None:
            msg = "ServingDBManager has no active snapshot pointer"
            raise RuntimeError(msg)
        return self._pointer

    @contextmanager
    def connect(self) -> Iterator[tuple[MinimalStorageGateway, ServingSnapshotPointer]]:
        """Yield a gateway plus the current pointer.

        Yields
        ------
        tuple[MinimalStorageGateway, ServingSnapshotPointer]
            Gateway with ibis/policy access and current pointer.

        Raises
        ------
        RuntimeError
            If manager not started.
        """
        pool = self._pool
        pointer = self._pointer
        if pool is None or pointer is None:
            msg = "ServingDBManager not started"
            raise RuntimeError(msg)

        with pool.acquire() as gw:
            yield gw, pointer

    async def _watch_loop(self) -> None:
        """Background task watching for pointer changes."""
        while True:
            await self._reload_if_needed(force=False)
            await asyncio.sleep(self.poll_interval_s)

    async def _reload_if_needed(self, *, force: bool) -> None:
        """Reload snapshot if pointer file changed."""
        if not self.pointer_path.exists():
            return

        st = self.pointer_path.stat()
        if not force and self._last_mtime_ns == st.st_mtime_ns:
            return
        self._last_mtime_ns = st.st_mtime_ns

        new_ptr = ServingSnapshotPointer.load(self.pointer_path)

        # Skip if same DB path (metadata-only update)
        if self._pointer is not None and new_ptr.db_path == self._pointer.db_path:
            self._pointer = new_ptr
            return

        new_pool = ReadPoolGateway(new_ptr.db_path, self.pool_cfg)
        old_pool = self._pool
        self._pool = new_pool
        self._pointer = new_ptr

        if old_pool is not None:
            old_pool.close_gracefully()


__all__ = ["ServingDBManager"]
```

#### 2.2 Update `serving/db/pool.py` to thin re-export

```python
"""Connection pool re-exports from storage layer.

.. deprecated::
    Import directly from `codeintel.storage.gateway.pool` instead.
"""

from __future__ import annotations

from codeintel.storage.gateway.pool import PoolConfig, ReadPoolGateway

# Backwards compatibility aliases
DuckDBPoolConfig = PoolConfig
DuckDBReadPool = ReadPoolGateway

__all__ = [
    "DuckDBPoolConfig",
    "DuckDBReadPool",
    "PoolConfig",
    "ReadPoolGateway",
]
```

---

### Phase 3: Refactor SemanticQueryKernel (1-2 hours)

**Goal**: Use gateway pattern throughout kernel, eliminating ad-hoc connections.

#### 3.1 Key Changes to `serving/semantic/kernel.py`

**Import changes**:
```python
# Add:
from codeintel.storage.duckdb_policy_backend import duckdb_schema_exists

# Remove need for:
# import ibis  (for ibis.duckdb.from_connection)
```

**Method signature changes**:

```python
# Before:
def _execute_sql(
    self,
    *,
    con: DuckDBConnection,
    sql: str,
    params: Sequence[object] | None = None,
) -> list[dict[str, object]]:
    backend = MinimalStorageGateway(con).policy
    result = backend.execute_sql(sql, params=params)

# After:
def _execute_sql(
    self,
    *,
    gw: MinimalStorageGateway,
    sql: str,
    params: Sequence[object] | None = None,
) -> list[dict[str, object]]:
    result = gw.policy.execute_sql(sql, params=params)
```

```python
# Before:
def _execute_semantic_plan(
    self,
    *,
    con: DuckDBConnection,
    plan: SemanticQueryPlan,
) -> list[dict[str, object]]:
    ibis_con = ibis.duckdb.from_connection(con)
    expr = build_query(ibis_con=ibis_con, plan=plan)
    sql = ibis_con.compile(expr)
    return self._execute_sql(con=con, sql=sql)

# After:
def _execute_semantic_plan(
    self,
    *,
    gw: MinimalStorageGateway,
    plan: SemanticQueryPlan,
) -> list[dict[str, object]]:
    expr = build_query(ibis_con=gw.ibis.con, plan=plan)
    sql = gw.ibis.con.compile(expr)
    return self._execute_sql(gw=gw, sql=sql)
```

**Public method updates**:

```python
# Before:
def query(self, request: SemanticQueryRequest) -> SemanticQueryResponse:
    # ... setup code ...
    with self.db.connect() as (con, pointer):
        rows = self._execute_semantic_plan(con=con, plan=plan)
    # ... response building ...

# After:
def query(self, request: SemanticQueryRequest) -> SemanticQueryResponse:
    # ... setup code ...
    with self.db.connect() as (gw, pointer):
        rows = self._execute_semantic_plan(gw=gw, plan=plan)
    # ... response building ...
```

**Schema existence check**:

```python
# Before:
row = backend.execute_sql(
    "SELECT 1 FROM information_schema.schemata WHERE schema_name = ? LIMIT 1",
    [_SEARCH_FTS_SCHEMA],
).fetchone()
fts_available = row is not None

# After:
fts_available = duckdb_schema_exists(gw.con, schema=_SEARCH_FTS_SCHEMA)
```

---

## File-by-File Changes

### Files to Create

| File | Purpose |
|------|---------|
| `storage/gateway/pool.py` | `ReadPoolGateway` implementation yielding `MinimalStorageGateway` |

### Files to Modify

| File | Changes |
|------|---------|
| `storage/gateway/__init__.py` | Export `PoolConfig`, `ReadPoolGateway` |
| `serving/db/__init__.py` | Update exports for new types |
| `serving/db/pool.py` | Convert to thin re-export with deprecation |
| `serving/db/manager.py` | Use `ReadPoolGateway`, yield `MinimalStorageGateway` |
| `serving/semantic/kernel.py` | Use gateway pattern, import `duckdb_schema_exists` |

### Files Unchanged

| File | Reason |
|------|--------|
| `serving/settings.py` | Environment config is serving-specific |
| `serving/db/pointer.py` | Snapshot pointer is serving-specific |
| `serving/semantic/query_builder.py` | Filter building is serving-specific, correct design |
| `serving/semantic/registry.py` | View registry is serving-specific |
| `serving/semantic/inventory.py` | JSON parsing is serving-specific, correct design |
| `serving/semantic/models.py` | Pydantic models are serving-specific |
| `serving/search/models.py` | Search models are serving-specific |
| `serving/contracts/*` | Contract validation is serving-specific |
| `serving/http/*` | HTTP routes are serving-specific |
| `serving/mcp/*` | MCP tools are serving-specific |

---

## Migration Guide

### For Existing Code Using `DuckDBReadPool`

**Before**:
```python
from codeintel.serving.db.pool import DuckDBPoolConfig, DuckDBReadPool

pool = DuckDBReadPool(db_path, DuckDBPoolConfig(size=4))
con = pool.acquire()
try:
    result = con.execute("SELECT 1").fetchone()
finally:
    pool.release(con)
```

**After**:
```python
from codeintel.storage.gateway.pool import PoolConfig, ReadPoolGateway

pool = ReadPoolGateway(db_path, PoolConfig(size=4))
with pool.acquire() as gw:
    result = gw.policy.execute_sql("SELECT 1").fetchone()
pool.close_gracefully()
```

### For Existing Code Using `db_manager.connect()`

**Before**:
```python
with db_manager.connect() as (con, pointer):
    ibis_con = ibis.duckdb.from_connection(con)
    backend = MinimalStorageGateway(con).policy
    result = backend.execute_sql("SELECT 1")
```

**After**:
```python
with db_manager.connect() as (gw, pointer):
    # gw.ibis.con is the cached Ibis backend
    # gw.policy is the cached DuckDBPolicyBackend
    result = gw.policy.execute_sql("SELECT 1")
```

---

## Testing Strategy

### Unit Tests

1. **Pool Tests** (`tests/storage/gateway/test_pool.py`)
   - Test pool initialization with various sizes
   - Test acquire/release lifecycle via context manager
   - Test graceful shutdown
   - Test concurrent access from multiple threads

2. **Manager Tests** (`tests/serving/db/test_manager.py`)
   - Update to expect `MinimalStorageGateway` from `connect()`
   - Test hot-swap with new pool type
   - Verify gateway reuse within single request

3. **Kernel Tests** (`tests/serving/semantic/test_kernel.py`)
   - Verify queries work with gateway pattern
   - Test that result extraction is unchanged
   - Verify `duckdb_schema_exists` integration

### Integration Tests

1. **HTTP Routes** (`tests/serving/test_semantic_http_routes.py`)
   - Full request/response cycle unchanged
   
2. **MCP Tools** (`tests/serving/test_semantic_mcp_tools.py`)
   - Tool execution unchanged

### Backwards Compatibility

The thin re-export in `serving/db/pool.py` ensures existing imports continue to work:

```python
# Still works (with deprecation warning in future)
from codeintel.serving.db.pool import DuckDBPoolConfig, DuckDBReadPool
```

Note: The API changes from `acquire()/release()` to context manager `acquire()`. Code using the old pattern will need updates.

---

## Rollback Plan

### If Issues Arise

1. **Revert Phase 1**: Delete `storage/gateway/pool.py`, revert `storage/gateway/__init__.py`
2. **Revert Phase 2**: Restore `ServingDBManager` to yield raw connections, restore full `pool.py`
3. **Revert Phase 3**: Restore kernel to create ad-hoc gateways

### Feature Flags (Optional)

Add environment variable for gradual rollout:

```python
import os

USE_GATEWAY_PATTERN = os.environ.get("CODEINTEL_USE_GATEWAY_PATTERN", "1") == "1"
```

---

## Success Criteria

### Code Quality

- [ ] All tests pass
- [ ] No pyright/pyrefly errors
- [ ] No ruff lint issues

### Performance

- [ ] Query latency unchanged (±5%)
- [ ] Memory usage reduced (fewer gateway instantiations)
- [ ] Pool acquisition time < 1ms

### Architecture

- [ ] No `ibis.duckdb.from_connection()` calls in serving kernel
- [ ] No `MinimalStorageGateway()` construction in kernel hot paths
- [ ] All pool management in storage layer
- [ ] Single gateway instance reused throughout request lifecycle

### Documentation

- [ ] Updated docstrings
- [ ] Migration guide complete
- [ ] Deprecation warnings added

---

## Appendix: Metrics

### Lines of Code Impact

| Module | Before | After | Delta |
|--------|--------|-------|-------|
| `serving/db/pool.py` | 139 | 25 | -114 |
| `serving/db/manager.py` | 138 | 115 | -23 |
| `serving/semantic/kernel.py` | 546 | 520 | -26 |
| `storage/gateway/pool.py` | 0 | 100 | +100 |
| **Total** | **823** | **760** | **-63** |

### Dependency Graph Simplification

**Before**:
```
serving.semantic.kernel
  → ibis.duckdb (direct, creates backend per call)
  → storage.gateway.minimal (creates per SQL call)
  → storage.duckdb_policy_backend (creates per SQL call)
```

**After**:
```
serving.semantic.kernel
  → storage.gateway.pool (via manager)
    → storage.gateway.minimal (pooled, one per connection)
      → storage.ibis_adapter (cached on gateway)
      → storage.duckdb_policy_backend (cached on gateway)
```

### Gateway Instantiation Reduction

| Scenario | Before | After |
|----------|--------|-------|
| Single semantic query | 2 gateways | 0 (reuses pooled) |
| Search query | 2 gateways | 0 (reuses pooled) |
| Explain query | 2 gateways | 0 (reuses pooled) |
| Request with 3 queries | 6 gateways | 0 (reuses pooled) |

---

## Related Documents

- [COMBINED_DECOMMISSIONING_PLAN.md](./COMBINED_DECOMMISSIONING_PLAN.md) - Overall decommissioning context
- [BUILD_CONSOLIDATION_AND_ENHANCEMENT_PLAN.md](./BUILD_CONSOLIDATION_AND_ENHANCEMENT_PLAN.md) - Build layer cleanup

