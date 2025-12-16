# Serving Module Rationalization Plan

> **Status**: Ready for Implementation  
> **Priority**: Medium  
> **Estimated Effort**: 2-3 days  
> **Dependencies**: None (can proceed independently)

## Executive Summary

The `serving` module has evolved to a clean semantic-first architecture, but contains redundancies with the `storage` layer that should be consolidated. This plan documents opportunities to:

1. Move connection pooling to storage layer
2. Eliminate duplicate Ibis connection handling
3. Simplify gateway usage patterns
4. Consolidate schema metadata parsing

**Goal**: Reduce serving module complexity by ~30% while improving maintainability through proper layer separation.

---

## Table of Contents

1. [Current Architecture](#current-architecture)
2. [Redundancy Analysis](#redundancy-analysis)
3. [Proposed Architecture](#proposed-architecture)
4. [Implementation Phases](#implementation-phases)
5. [File-by-File Changes](#file-by-file-changes)
6. [Migration Guide](#migration-guide)
7. [Testing Strategy](#testing-strategy)
8. [Rollback Plan](#rollback-plan)

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
│   ├── inventory.py               # SchemaInventory ← REDUNDANCY
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

### Storage Module (Reference)

```
storage/
├── gateway/
│   ├── protocol.py                # DuckDBConnection, StorageGateway, MinimalGateway
│   ├── minimal.py                 # MinimalStorageGateway (composition root)
│   ├── config.py                  # StorageConfig
│   ├── connection.py              # connect()
│   └── factory.py                 # open_gateway(), open_memory_gateway()
├── ibis_adapter.py                # IbisGateway
├── duckdb_policy_backend.py       # DuckDBPolicyBackend
└── serving/
    └── search_index.py            # FTS index building
```

---

## Redundancy Analysis

### 1. Connection Pool Implementation

**Location**: `serving/db/pool.py`

**Current Implementation**:
```python
class DuckDBReadPool:
    def __init__(self, db_path: Path, cfg: DuckDBPoolConfig) -> None:
        self._db_path = db_path
        self._cfg = cfg
        self._available: LifoQueue[DuckDBConnection] = LifoQueue()
        self._in_use: set[DuckDBConnection] = set()
        self._closing = False
        self._init_connections()

    def _open(self) -> DuckDBConnection:
        # Uses storage.gateway.config.StorageConfig.for_readonly()
        # Uses storage.gateway.connection.connect()
        return connect(StorageConfig.for_readonly(self._db_path), duckdb_config=duckdb_config)
```

**Issue**: Pool correctly uses storage primitives but lives in serving layer. This creates a layering inversion where serving owns connection lifecycle.

**Recommendation**: Move `DuckDBReadPool` to `storage/gateway/pool.py` as a reusable `ReadPoolGateway`.

---

### 2. Ad-hoc Ibis Connection Creation

**Location**: `serving/semantic/kernel.py:245-257`

**Current Implementation**:
```python
def _execute_semantic_plan(
    self,
    *,
    con: DuckDBConnection,
    plan: SemanticQueryPlan,
) -> list[dict[str, object]]:
    ibis_con = ibis.duckdb.from_connection(con)  # ← Creates new Ibis connection
    expr = build_query(ibis_con=ibis_con, plan=plan)
    sql = ibis_con.compile(expr)
    return self._execute_sql(con=con, sql=sql)
```

**Issue**: Creates a new Ibis backend connection for every query instead of using `MinimalStorageGateway(con).ibis`.

**Impact**:
- Duplicate connection lifecycle management
- Inconsistent with storage layer patterns
- Potential connection overhead

**Recommendation**: Pass `MinimalStorageGateway` through the call chain instead of raw `DuckDBConnection`.

---

### 3. Gateway Creation Per Call

**Location**: `serving/semantic/kernel.py:217-238`

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

**Issue**: Creates a new `MinimalStorageGateway` for every SQL execution, which:
- Reinstantiates `IbisGateway` and `DuckDBPolicyBackend`
- Wastes memory and CPU cycles
- Violates the composition root pattern

**Recommendation**: Create gateway once per acquired connection and reuse throughout request lifecycle.

---

### 4. Schema Manifest Parsing Duplication

**Location**: `serving/semantic/inventory.py:61-128`

**Current Implementation**:
```python
def _parse_columns(items: object) -> list[Column]:
    cols: list[Column] = []
    for idx, col_obj in enumerate(_expect_list(items, ctx="columns")):
        col = _expect_dict(col_obj, ctx=f"columns[{idx}]")
        col_type = _parse_column_type(col.get("type"), ctx=f"columns[{idx}].type")
        # ... manual parsing
        cols.append(Column(...))
    return cols

def _parse_table(obj: Mapping[str, object]) -> TableSchema:
    # ... manual JSON -> TableSchema conversion
```

**Issue**: Re-implements JSON→`TableSchema` parsing that could use `core.schemas.primitives` directly or be generated from a shared loader.

**Recommendation**: Add `TableSchema.from_dict()` class method to `core.schemas.primitives` and use it in `SchemaInventory`.

---

### 5. FTS Query SQL Embedded in Kernel

**Location**: `serving/semantic/kernel.py:47-105`

**Current State**:
```python
_SQL_SEARCH_FTS = """
SELECT kind, name, module, rel_path, ref_goid_h128, score
FROM (
    SELECT ... fts_docs_search_documents.match_bm25(doc_id, ?) AS score
    FROM docs.search_documents
) ranked
WHERE score IS NOT NULL
ORDER BY score DESC
LIMIT ? OFFSET ?
"""
```

**Assessment**: This is **correctly placed** - storage owns index creation (`storage/serving/search_index.py`), serving owns query execution. The SQL is read-only and specific to the serving surface.

**Recommendation**: Keep as-is. This is proper layer separation.

---

## Proposed Architecture

### Target State

```
storage/
├── gateway/
│   ├── protocol.py                # Add: ReadPoolGateway protocol
│   ├── minimal.py                 # MinimalStorageGateway (unchanged)
│   ├── config.py                  # Add: PoolConfig
│   ├── connection.py              # Unchanged
│   ├── pool.py                    # NEW: ReadPoolGateway implementation
│   └── factory.py                 # Unchanged
├── ibis_adapter.py                # Unchanged
└── serving/
    └── search_index.py            # Unchanged

serving/
├── __init__.py                    # Update exports
├── settings.py                    # Unchanged
├── db/
│   ├── __init__.py                # Update exports
│   ├── manager.py                 # REFACTOR: yield MinimalStorageGateway
│   ├── pointer.py                 # Unchanged
│   └── pool.py                    # DELETE or thin re-export
├── semantic/
│   ├── __init__.py                # Unchanged
│   ├── kernel.py                  # REFACTOR: use gateway pattern
│   ├── query_builder.py           # Unchanged
│   ├── registry.py                # Unchanged
│   ├── inventory.py               # SIMPLIFY: delegate to primitives
│   └── models.py                  # Unchanged
├── search/                        # Unchanged
├── contracts/                     # Unchanged
├── http/                          # Unchanged
└── mcp/                           # Unchanged
```

### Layer Responsibilities

| Layer | Responsibility | Does NOT Own |
|-------|---------------|--------------|
| `storage.gateway` | Connection lifecycle, pooling, Ibis/policy access | Query semantics |
| `storage.serving` | FTS index building | Query execution |
| `serving.db` | Snapshot pointer, hot-swap coordination | Connection pooling |
| `serving.semantic` | Query building, result extraction | Connection management |
| `serving.http/mcp` | HTTP/MCP surfaces | Business logic |

---

## Implementation Phases

### Phase 1: Move Pool to Storage (2-3 hours)

**Goal**: Establish proper layering by moving connection pooling to storage.

#### 1.1 Create `storage/gateway/pool.py`

```python
"""Read-only connection pool with MinimalStorageGateway per connection.

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
        Number of connections in the pool.
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

#### 1.3 Update `serving/db/pool.py`

Convert to thin re-export with deprecation:

```python
"""Connection pool re-exports from storage layer.

.. deprecated::
    Import directly from `codeintel.storage.gateway.pool` instead.
"""

from __future__ import annotations

from codeintel.storage.gateway.pool import PoolConfig, ReadPoolGateway

# Backwards compatibility alias
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

### Phase 2: Refactor ServingDBManager (1-2 hours)

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
        """Return current snapshot pointer."""
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

        new_pool = ReadPoolGateway(new_ptr.db_path, self._pool_cfg)
        old_pool = self._pool
        self._pool = new_pool
        self._pointer = new_ptr

        if old_pool is not None:
            old_pool.close_gracefully()


__all__ = ["ServingDBManager"]
```

---

### Phase 3: Refactor SemanticQueryKernel (2-3 hours)

**Goal**: Use gateway pattern throughout kernel, eliminating ad-hoc connections.

#### 3.1 Key Changes to `serving/semantic/kernel.py`

**Before**:
```python
def _execute_sql(
    self,
    *,
    con: DuckDBConnection,
    sql: str,
    params: Sequence[object] | None = None,
) -> list[dict[str, object]]:
    backend = MinimalStorageGateway(con).policy  # Creates gateway per call
    result = backend.execute_sql(sql, params=params)

def _execute_semantic_plan(
    self,
    *,
    con: DuckDBConnection,
    plan: SemanticQueryPlan,
) -> list[dict[str, object]]:
    ibis_con = ibis.duckdb.from_connection(con)  # Creates Ibis per call
    expr = build_query(ibis_con=ibis_con, plan=plan)
    sql = ibis_con.compile(expr)
    return self._execute_sql(con=con, sql=sql)
```

**After**:
```python
def _execute_sql(
    self,
    *,
    gw: MinimalStorageGateway,
    sql: str,
    params: Sequence[object] | None = None,
) -> list[dict[str, object]]:
    result = gw.policy.execute_sql(sql, params=params)  # Reuse gateway

def _execute_semantic_plan(
    self,
    *,
    gw: MinimalStorageGateway,
    plan: SemanticQueryPlan,
) -> list[dict[str, object]]:
    expr = build_query(ibis_con=gw.ibis.con, plan=plan)  # Use gateway's Ibis
    sql = gw.ibis.con.compile(expr)
    return self._execute_sql(gw=gw, sql=sql)
```

**Public methods update**:
```python
def query(self, request: SemanticQueryRequest) -> SemanticQueryResponse:
    # ... setup code ...
    
    with self.db.connect() as (gw, pointer):  # Now yields gateway
        rows = self._execute_semantic_plan(gw=gw, plan=plan)
    
    # ... response building ...
```

---

### Phase 4: Simplify SchemaInventory (1 hour)

**Goal**: Reduce parsing duplication by using core primitives.

#### 4.1 Add `TableSchema.from_dict()` to `core/schemas/primitives.py`

```python
@classmethod
def from_dict(cls, data: Mapping[str, object]) -> TableSchema:
    """Load TableSchema from dictionary representation.

    Parameters
    ----------
    data
        Dictionary with schema, name, columns, primary_key, indexes, description.

    Returns
    -------
    TableSchema
        Loaded schema instance.
    """
    columns = [
        Column(
            name=str(col.get("name", "")),
            type=str(col.get("type", "VARCHAR")),
            nullable=bool(col.get("nullable", True)),
            description=col.get("description"),
        )
        for col in data.get("columns", [])
    ]
    
    indexes = tuple(
        Index(
            name=str(idx.get("name", "")),
            columns=tuple(str(c) for c in idx.get("columns", [])),
            unique=bool(idx.get("unique", False)),
        )
        for idx in data.get("indexes", [])
    )
    
    pk_raw = data.get("primary_key", [])
    primary_key = tuple(str(k) for k in pk_raw) if isinstance(pk_raw, list) else ()
    
    return cls(
        schema=str(data.get("schema", "")),
        name=str(data.get("name", "")),
        columns=columns,
        primary_key=primary_key,
        indexes=indexes,
        description=data.get("description"),
    )
```

#### 4.2 Simplify `serving/semantic/inventory.py`

```python
"""Schema inventory for serving layer introspection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.schemas.primitives import TableSchema

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["SchemaInventory"]


@dataclass(frozen=True)
class SchemaInventory:
    """Inventory of table and view schemas."""

    schemas: dict[str, TableSchema]

    @classmethod
    def load(cls, path: Path) -> SchemaInventory:
        """Load inventory from schema manifest JSON."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        
        schemas: dict[str, TableSchema] = {}
        for table_data in payload.get("tables", []):
            schema = TableSchema.from_dict(table_data)
            schemas[schema.table_key] = schema
        
        return cls(schemas=schemas)

    def get(self, table_key: str) -> TableSchema | None:
        """Look up schema by table key."""
        return self.schemas.get(table_key)

    def require(self, table_key: str) -> TableSchema:
        """Look up schema by table key, raising if not found."""
        schema = self.get(table_key)
        if schema is None:
            msg = f"Unknown table: {table_key}"
            raise KeyError(msg)
        return schema

    def table_keys(self) -> list[str]:
        """Return all table keys."""
        return list(self.schemas.keys())
```

---

## File-by-File Changes

### Files to Create

| File | Purpose |
|------|---------|
| `storage/gateway/pool.py` | `ReadPoolGateway` implementation |

### Files to Modify

| File | Changes |
|------|---------|
| `storage/gateway/__init__.py` | Export `PoolConfig`, `ReadPoolGateway` |
| `serving/db/__init__.py` | Update exports |
| `serving/db/pool.py` | Convert to re-export with deprecation |
| `serving/db/manager.py` | Yield `MinimalStorageGateway` instead of raw connection |
| `serving/semantic/kernel.py` | Use gateway pattern throughout |
| `serving/semantic/inventory.py` | Simplify using `TableSchema.from_dict()` |
| `core/schemas/primitives.py` | Add `TableSchema.from_dict()` |

### Files Unchanged

| File | Reason |
|------|--------|
| `serving/settings.py` | Environment config is serving-specific |
| `serving/db/pointer.py` | Snapshot pointer is serving-specific |
| `serving/semantic/query_builder.py` | Filter building is serving-specific |
| `serving/semantic/registry.py` | View registry is serving-specific |
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
```

### For Existing Code Using Raw Connections

**Before**:
```python
with db_manager.connect() as (con, pointer):
    ibis_con = ibis.duckdb.from_connection(con)
    backend = MinimalStorageGateway(con).policy
```

**After**:
```python
with db_manager.connect() as (gw, pointer):
    # gw.ibis.con is the Ibis backend
    # gw.policy is the DuckDBPolicyBackend
    result = gw.policy.execute_sql("SELECT 1")
```

---

## Testing Strategy

### Unit Tests

1. **Pool Tests** (`tests/storage/gateway/test_pool.py`)
   - Test pool initialization with various sizes
   - Test acquire/release lifecycle
   - Test graceful shutdown
   - Test concurrent access

2. **Manager Tests** (`tests/serving/db/test_manager.py`)
   - Update to expect `MinimalStorageGateway` from `connect()`
   - Test hot-swap with new pool type

3. **Kernel Tests** (`tests/serving/semantic/test_kernel.py`)
   - Verify queries work with gateway pattern
   - Test result extraction unchanged

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

---

## Rollback Plan

### If Issues Arise

1. **Revert Phase 1**: Restore `serving/db/pool.py` to full implementation
2. **Revert Phase 2**: Restore `ServingDBManager` to yield raw connections
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
- [ ] Memory usage unchanged (±10%)
- [ ] Pool acquisition time < 1ms

### Architecture

- [ ] No `ibis.duckdb.from_connection()` calls in serving (except query_builder)
- [ ] No `MinimalStorageGateway()` construction in kernel hot paths
- [ ] All pool management in storage layer

### Documentation

- [ ] Updated docstrings
- [ ] Migration guide complete
- [ ] Deprecation warnings added

---

## Appendix: Metrics

### Lines of Code Impact

| Module | Before | After | Delta |
|--------|--------|-------|-------|
| `serving/db/pool.py` | 140 | 20 | -120 |
| `serving/db/manager.py` | 139 | 100 | -39 |
| `serving/semantic/kernel.py` | 542 | 500 | -42 |
| `serving/semantic/inventory.py` | 231 | 80 | -151 |
| `storage/gateway/pool.py` | 0 | 140 | +140 |
| `core/schemas/primitives.py` | ~200 | ~240 | +40 |
| **Total** | **1252** | **1080** | **-172** |

### Dependency Graph Simplification

**Before**:
```
serving.semantic.kernel
  → ibis.duckdb (direct)
  → storage.gateway.minimal (per-call)
  → storage.duckdb_policy_backend (per-call)
```

**After**:
```
serving.semantic.kernel
  → storage.gateway.pool (via manager)
    → storage.gateway.minimal (pooled)
      → storage.ibis_adapter
      → storage.duckdb_policy_backend
```

---

## Related Documents

- [COMBINED_DECOMMISSIONING_PLAN.md](./COMBINED_DECOMMISSIONING_PLAN.md) - Overall decommissioning context
- [ANALYTICS_DECOMMISSIONING_PLAN.md](./ANALYTICS_DECOMMISSIONING_PLAN.md) - Analytics layer cleanup
- [GRAPHS_DECOMMISSIONING_PLAN.md](./GRAPHS_DECOMMISSIONING_PLAN.md) - Graph layer cleanup

