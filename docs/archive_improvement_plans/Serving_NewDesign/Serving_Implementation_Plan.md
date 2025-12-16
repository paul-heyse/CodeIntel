# Serving Layer Overhaul: Comprehensive Implementation Plan

> **Status**: Implementation Plan  
> **Created**: 2025-12-15  
> **Target**: Aggressive pivot to semantic-first serving with full legacy deprecation

---

## Executive Summary

This plan details an aggressive pivot from the current serving architecture to a **semantic-first, read-only snapshot** model. The new design:

1. **Decouples build from serve** via immutable published snapshots
2. **Unifies HTTP + MCP** through a single `SemanticQueryKernel`
3. **Eliminates legacy operation catalog** in favor of Hamilton-native semantic tagging
4. **Supports hot-swap** without server restart via atomic pointer updates
5. **Aligns with recent schema work** leveraging `SchemaProvider`, `TableSchema`, and Ibis view registry

### Scope

| Category | Count | Action |
|----------|-------|--------|
| **New modules to create** | ~25 files | Create from scratch |
| **Existing modules to migrate** | ~15 files | Refactor/adapt |
| **Legacy modules to delete** | ~40 files | Remove after cutover |

---

## Part 1: Current State Analysis

### 1.1 Legacy Serving Architecture

The current `src/codeintel/serving/` structure has accumulated technical debt:

```
src/codeintel/serving/
├── backend/                    # LEGACY: To be replaced
│   ├── core.py                 # BackendContext, DuckDBRepositories
│   ├── dataset_backend.py      # Dataset-specific backend
│   ├── datasets.py             # Dataset registry helpers
│   ├── domain_builders.py      # Domain model builders
│   ├── duckdb_service.py       # DuckDBQueryService (monolithic)
│   ├── function_backend.py     # Function-specific queries
│   ├── pagination.py           # BackendLimits (KEEP - reusable)
│   ├── profile_backend.py      # Profile queries
│   ├── query_api.py            # Query API utilities
│   └── subsystem_backend.py    # Subsystem queries
├── services/                   # LEGACY: To be replaced
│   ├── conversion.py           # Model conversion helpers
│   ├── datasets.py             # Dataset service methods
│   ├── errors.py               # Service error types
│   ├── functions.py            # Function service methods
│   ├── http_helpers.py         # HTTP helpers
│   ├── observability.py        # ServiceObservability (KEEP - reusable)
│   ├── profiles.py             # Profile service methods
│   ├── query_service.py        # QueryService protocol (REPLACE)
│   ├── subsystems.py           # Subsystem service methods
│   └── transport.py            # Transport utilities
├── operations/                 # LEGACY: To be deleted
│   └── catalog.py              # Static operation catalog (REMOVE)
├── mcp/                        # REFACTOR: Update for semantic kernel
│   ├── backend.py              # DuckDBBackend, HttpBackend
│   ├── registry.py             # Tool registration
│   ├── server.py               # MCP server creation
│   ├── tools.py                # Individual tool implementations
│   └── ...                     # Various tool utilities
├── http/                       # REFACTOR: New routes for semantic API
│   ├── fastapi.py              # FastAPI app factory
│   └── routes/                 # HTTP route modules
├── bootstrap.py                # REPLACE: New simplified bootstrap
├── auto_pipeline.py            # REMOVE: Legacy auto-pipeline
└── context.py                  # REMOVE: Legacy context
```

### 1.2 Key Problems with Current Design

| Problem | Impact | New Solution |
|---------|--------|--------------|
| **Static operation catalog** | Operations hardcoded in `catalog.py`, not derived from Hamilton | Compile `semantic_registry.json` from Hamilton tags |
| **Tightly coupled to build DB** | Serving uses same connection, can't hot-swap | Read-only snapshot with atomic pointer |
| **Complex 4-layer architecture** | Backend → Service → Query → Repository | Single `SemanticQueryKernel` |
| **Manual SQL in operations** | SQL scattered across repositories | Structured query builder with param binding |
| **No semantic layer** | Views exist but not exposed semantically | Hamilton-tagged semantic views |
| **Legacy config imports** | Imports `codeintel.config.datasets` | Self-contained semantic registry |

### 1.3 Recent Schema Work to Leverage

The codebase has excellent schema infrastructure we'll build on:

| Component | Location | Usage in New Design |
|-----------|----------|---------------------|
| `SchemaProvider` protocol | `core/schemas/provider.py` | Resolve table schemas for validation |
| `TableSchema`, `Column` | `core/schemas/primitives.py` | Schema manifest format |
| `UnifiedSchemaProvider` | `build/schemas/provider_unified.py` | Build-time schema resolution |
| `@register_view` decorator | `storage/views/ibis_registry.py` | View registration (extend for semantic tags) |
| Ibis view builders | `storage/views/ibis_views.py` | ~50+ views already defined |

---

## Part 2: Target Architecture

### 2.1 New Directory Structure

```
src/codeintel/serving/
├── __init__.py                 # Clean public API exports
├── settings.py                 # NEW: Environment-driven serving settings
│
├── db/                         # NEW: Database management layer
│   ├── __init__.py
│   ├── pointer.py              # ServingSnapshotPointer dataclass
│   ├── pool.py                 # DuckDBReadPool (read-only connections)
│   └── manager.py              # ServingDBManager (hot-swap support)
│
├── semantic/                   # NEW: Semantic layer core
│   ├── __init__.py
│   ├── registry.py             # SemanticRegistry (loaded from JSON)
│   ├── inventory.py            # SchemaInventory (table/view catalog)
│   ├── query_builder.py        # Safe SQL builder with param binding
│   ├── kernel.py               # SemanticQueryKernel (unified query API)
│   └── models.py               # Pydantic models for query I/O
│
├── http/                       # REFACTORED: FastAPI with semantic routes
│   ├── __init__.py
│   ├── app.py                  # create_serving_app() factory
│   └── routes/
│       ├── __init__.py
│       ├── semantic.py         # NEW: /semantic/* endpoints
│       ├── meta.py             # UPDATED: /meta with semantic info
│       └── health.py           # UPDATED: /health with snapshot info
│
├── mcp/                        # REFACTORED: Semantic MCP tools
│   ├── __init__.py
│   ├── app.py                  # NEW: build_mcp_app() factory
│   ├── server.py               # UPDATED: stdio/http transport
│   ├── tools_semantic.py       # NEW: semantic_catalog/describe/query
│   └── tools_meta.py           # NEW: serving_meta tool
│
└── _legacy/                    # TEMPORARY: Old code during migration
    └── ...                     # Moved here, then deleted
```

### 2.2 Build-Side Publisher Structure

```
src/codeintel/build/serving/
├── __init__.py
├── publisher.py                # NEW: Snapshot publishing logic
├── manifest.py                 # NEW: ServingSnapshotManifest dataclass
└── semantic_compile.py         # NEW: Compile semantic registry from DAG
```

### 2.3 Core Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          BUILD PHASE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Hamilton DAG          Unified Schema        Ibis Views              │
│  (@semantic_view       Provider              (@register_view)        │
│   decorators)                                                        │
│        │                    │                      │                 │
│        └────────────────────┼──────────────────────┘                 │
│                             ▼                                        │
│                  ┌─────────────────────┐                             │
│                  │   BUILD PIPELINE    │                             │
│                  │   - Run Hamilton    │                             │
│                  │   - Materialize     │                             │
│                  │   - CHECKPOINT DB   │                             │
│                  └──────────┬──────────┘                             │
│                             │                                        │
│                             ▼                                        │
│                  ┌─────────────────────┐                             │
│                  │     PUBLISHER       │                             │
│                  │ publish_serving_    │                             │
│                  │   snapshot()        │                             │
│                  └──────────┬──────────┘                             │
│                             │                                        │
└─────────────────────────────┼────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SERVING SNAPSHOT                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  <serve_dir>/                                                        │
│  ├── current.json              ← Atomic pointer to active snapshot   │
│  └── snapshots/                                                      │
│      └── <run_id>/                                                   │
│          ├── codeintel.duckdb  ← Immutable snapshot DB               │
│          ├── semantic_registry.json  ← Semantic view definitions     │
│          └── schema_manifest.json    ← Table/column schemas          │
│                                                                      │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        SERVE PHASE                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   ServingDBManager                            │   │
│  │  - Watches current.json for changes                          │   │
│  │  - Hot-swaps DuckDBReadPool on pointer update                │   │
│  │  - Manages read-only connection pool                         │   │
│  └──────────────────────┬───────────────────────────────────────┘   │
│                         │                                            │
│                         ▼                                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                 SemanticQueryKernel                           │   │
│  │  - catalog(): List semantic views                            │   │
│  │  - describe(view_id): View schema + metadata                 │   │
│  │  - query(view_id, filters, ...): Execute structured query    │   │
│  └────────────────┬─────────────────────┬───────────────────────┘   │
│                   │                     │                            │
│           ┌───────┴───────┐     ┌───────┴───────┐                   │
│           ▼               ▼     ▼               ▼                   │
│    ┌────────────┐   ┌────────────┐   ┌────────────┐                 │
│    │  FastAPI   │   │  FastMCP   │   │    CLI     │                 │
│    │  /semantic │   │  semantic_ │   │  codeintel │                 │
│    │  /meta     │   │  tools     │   │  serve     │                 │
│    └────────────┘   └────────────┘   └────────────┘                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Implementation Phases

### Phase 0: Preparation (PR-74)

**Goal**: Foundation and infrastructure without breaking existing functionality.

#### 3.0.1 Create Serving Settings Module

**File**: `src/codeintel/serving/settings.py`

```python
"""Environment-driven serving configuration.

Replaces complex ServingConfig with focused serving-only settings.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class ServingSettings:
    """Serving layer configuration loaded from environment variables.

    Parameters
    ----------
    serve_dir
        Root directory for serving snapshots.
    hot_swap
        Enable automatic snapshot hot-swap on pointer change.
    pool_size
        Number of read-only DuckDB connections per worker.
    poll_interval_s
        Seconds between pointer file checks when hot_swap enabled.
    mcp_transport
        MCP transport mode: "stdio" or "http".
    host
        HTTP server bind address.
    port
        HTTP server port.
    auth_token
        Optional bearer token for remote serving.
    """

    serve_dir: Path
    hot_swap: bool = True
    pool_size: int = 4
    poll_interval_s: float = 1.0
    mcp_transport: str = "stdio"
    host: str = "127.0.0.1"
    port: int = 8000
    auth_token: str | None = None

    @classmethod
    def from_env(cls) -> ServingSettings:
        """Load settings from environment variables."""
        return cls(
            serve_dir=Path(os.environ.get(
                "CODEINTEL_SERVE_DIR", ".codeintel/serve"
            )).resolve(),
            hot_swap=os.environ.get("CODEINTEL_SERVE_HOTSWAP", "1") == "1",
            pool_size=int(os.environ.get("CODEINTEL_SERVE_POOL_SIZE", "4")),
            poll_interval_s=float(os.environ.get(
                "CODEINTEL_SERVE_POLL_INTERVAL", "1.0"
            )),
            mcp_transport=os.environ.get("CODEINTEL_MCP_TRANSPORT", "stdio"),
            host=os.environ.get("CODEINTEL_HOST", "127.0.0.1"),
            port=int(os.environ.get("CODEINTEL_PORT", "8000")),
            auth_token=os.environ.get("CODEINTEL_AUTH_TOKEN"),
        )
```

#### 3.0.2 Create Snapshot Pointer Module

**File**: `src/codeintel/serving/db/pointer.py`

```python
"""Serving snapshot pointer for atomic snapshot switching.

The pointer file (current.json) is the single source of truth for
which snapshot is currently active. It is updated atomically via
os.replace() on the same filesystem.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json


@dataclass(frozen=True)
class ServingSnapshotPointer:
    """Pointer to the currently active serving snapshot.

    Parameters
    ----------
    db_path
        Absolute path to the immutable DuckDB snapshot file.
    semantic_registry_path
        Path to semantic_registry.json.
    schema_manifest_path
        Path to schema_manifest.json.
    repo
        Repository identifier.
    commit
        Commit SHA.
    run_id
        Build run identifier.
    published_at
        ISO timestamp when snapshot was published.
    semantic_layer_version
        Version hash of the semantic layer.
    """

    db_path: Path
    semantic_registry_path: Path
    schema_manifest_path: Path
    repo: str
    commit: str
    run_id: str
    published_at: datetime
    semantic_layer_version: str

    @classmethod
    def load(cls, path: Path) -> ServingSnapshotPointer:
        """Load pointer from JSON file.

        Parameters
        ----------
        path
            Path to current.json pointer file.

        Returns
        -------
        ServingSnapshotPointer
            Loaded pointer instance.

        Raises
        ------
        FileNotFoundError
            If pointer file does not exist.
        json.JSONDecodeError
            If pointer file is not valid JSON.
        """
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

    def to_json(self) -> str:
        """Serialize pointer to JSON string.

        Returns
        -------
        str
            JSON representation of this pointer.
        """
        return json.dumps(
            {
                "db_path": str(self.db_path),
                "semantic_registry_path": str(self.semantic_registry_path),
                "schema_manifest_path": str(self.schema_manifest_path),
                "repo": self.repo,
                "commit": self.commit,
                "run_id": self.run_id,
                "published_at": self.published_at.isoformat(),
                "semantic_layer_version": self.semantic_layer_version,
            },
            indent=2,
            sort_keys=True,
        )
```

#### 3.0.3 Tasks Checklist for Phase 0

- [ ] Create `src/codeintel/serving/settings.py`
- [ ] Create `src/codeintel/serving/db/__init__.py`
- [ ] Create `src/codeintel/serving/db/pointer.py`
- [ ] Create `tests/serving/test_settings.py`
- [ ] Create `tests/serving/db/test_pointer.py`
- [ ] Add environment variable documentation

---

### Phase 1: Database Management Layer (PR-75)

**Goal**: Implement read-only connection pool and hot-swap manager.

#### 3.1.1 DuckDB Read Pool

**File**: `src/codeintel/serving/db/pool.py`

Key features:
- Opens N read-only connections per pool
- LIFO queue for connection reuse (warm cache)
- Graceful close drains in-use connections
- Connection-time config for threads/memory

```python
"""Read-only DuckDB connection pool for serving.

DuckDB supports multiple connections; a single connection serializes
queries. This pool provides N read-only handles per worker for
concurrent query execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, LifoQueue
import threading

import duckdb


@dataclass(frozen=True)
class DuckDBPoolConfig:
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


class DuckDBReadPool:
    """Thread-safe pool of read-only DuckDB connections.

    Parameters
    ----------
    db_path
        Path to DuckDB database file.
    cfg
        Pool configuration.
    """

    def __init__(self, db_path: Path, cfg: DuckDBPoolConfig) -> None:
        self._db_path = db_path
        self._cfg = cfg
        self._available: LifoQueue[duckdb.DuckDBPyConnection] = LifoQueue()
        self._lock = threading.Lock()
        self._in_use: set[duckdb.DuckDBPyConnection] = set()
        self._closing = False
        self._init_connections()

    def _open(self) -> duckdb.DuckDBPyConnection:
        """Open a new read-only connection."""
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
        """Pre-create pool connections."""
        for _ in range(max(1, self._cfg.size)):
            self._available.put(self._open())

    def acquire(self) -> duckdb.DuckDBPyConnection:
        """Acquire a connection from the pool.

        Returns
        -------
        duckdb.DuckDBPyConnection
            Read-only database connection.

        Raises
        ------
        RuntimeError
            If pool is closing.
        """
        if self._closing:
            msg = "Pool is closing"
            raise RuntimeError(msg)
        con = self._available.get()
        with self._lock:
            self._in_use.add(con)
        return con

    def release(self, con: duckdb.DuckDBPyConnection) -> None:
        """Return a connection to the pool.

        Parameters
        ----------
        con
            Connection to release.
        """
        with self._lock:
            self._in_use.discard(con)
            closing = self._closing
        if closing:
            con.close()
            return
        self._available.put(con)

    def close_gracefully(self) -> None:
        """Mark pool as closing and drain available connections."""
        with self._lock:
            self._closing = True
        while True:
            try:
                con = self._available.get_nowait()
            except Empty:
                break
            con.close()
```

#### 3.1.2 Serving DB Manager

**File**: `src/codeintel/serving/db/manager.py`

Key features:
- Watches `current.json` for mtime changes
- Swaps pools atomically on pointer update
- Old pool closed gracefully (in-use connections drain)
- Async-compatible watch loop

```python
"""Serving database manager with hot-swap support.

Watches the pointer file and swaps connection pools when the
snapshot changes, enabling zero-downtime deployments.
"""
from __future__ import annotations

import asyncio
import contextlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.serving.db.pool import DuckDBPoolConfig, DuckDBReadPool

if TYPE_CHECKING:
    import duckdb


@dataclass
class ServingDBManager:
    """Manages serving database connections with hot-swap support.

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
    pool_cfg: DuckDBPoolConfig = field(default_factory=DuckDBPoolConfig)
    poll_interval_s: float = 1.0

    _pointer: ServingSnapshotPointer | None = field(default=None, init=False)
    _pool: DuckDBReadPool | None = field(default=None, init=False)
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
            If manager not started.
        """
        if self._pointer is None:
            msg = "ServingDBManager not started"
            raise RuntimeError(msg)
        return self._pointer

    @contextmanager
    def connect(self):
        """Context manager for database connection.

        Yields
        ------
        tuple[duckdb.DuckDBPyConnection, ServingSnapshotPointer]
            Connection and current pointer.
        """
        if self._pool is None:
            msg = "ServingDBManager not started"
            raise RuntimeError(msg)
        con = self._pool.acquire()
        try:
            yield con, self._pointer
        finally:
            self._pool.release(con)

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

        # Skip if same DB path
        if (
            self._pointer is not None
            and new_ptr.db_path == self._pointer.db_path
        ):
            self._pointer = new_ptr
            return

        # Swap pools
        new_pool = DuckDBReadPool(new_ptr.db_path, self.pool_cfg)
        old_pool = self._pool
        self._pool = new_pool
        self._pointer = new_ptr

        if old_pool is not None:
            old_pool.close_gracefully()
```

#### 3.1.3 Tasks Checklist for Phase 1

- [ ] Create `src/codeintel/serving/db/pool.py`
- [ ] Create `src/codeintel/serving/db/manager.py`
- [ ] Create `tests/serving/db/test_pool.py`
  - Test pool creates N connections
  - Test acquire/release cycle
  - Test graceful close behavior
- [ ] Create `tests/serving/db/test_manager.py`
  - Test initial load
  - Test hot-swap on pointer change
  - Test same-path no-swap optimization

---

### Phase 2: Semantic Registry & Inventory (PR-76)

**Goal**: Load semantic layer definitions from published artifacts.

#### 3.2.1 Semantic View Specification

**File**: `src/codeintel/serving/semantic/models.py`

```python
"""Pydantic models for semantic layer queries and responses."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Op = Literal["eq", "ne", "lt", "lte", "gt", "gte", "in", "contains", "startswith"]


class SemanticViewSpec(BaseModel):
    """Specification for a semantic view.

    Parameters
    ----------
    id
        Stable semantic view identifier (e.g., "function.summary").
    kind
        Whether this is a "table" or "view" in DuckDB.
    table_key
        Fully qualified DuckDB object name (e.g., "docs.v_function_summary").
    entity
        Entity type this view represents (e.g., "function", "module").
    grain
        Row granularity (e.g., "per_function", "per_module").
    description
        Human-readable description.
    primary_key
        Column names forming the primary key.
    columns
        Exposed column names.
    joins
        Optional join hints for agents.
    defaults
        Default query parameters (limit, order_by).
    sensitivity
        Data sensitivity level.
    deprecated
        Whether this view is deprecated.
    replaced_by
        Successor view ID if deprecated.
    """

    id: str
    kind: Literal["table", "view"] = "view"
    table_key: str
    entity: str
    grain: str
    description: str | None = None
    primary_key: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    joins: list[dict[str, Any]] = Field(default_factory=list)
    defaults: dict[str, Any] = Field(default_factory=dict)
    sensitivity: str = "internal"
    deprecated: bool = False
    replaced_by: str | None = None


class FilterSpec(BaseModel):
    """Filter specification for semantic queries.

    Parameters
    ----------
    column
        Column name to filter on.
    op
        Filter operation.
    value
        Value to compare against.
    """

    column: str
    op: Op
    value: Any


class SemanticQueryRequest(BaseModel):
    """Request for a semantic view query.

    Parameters
    ----------
    view_id
        Semantic view identifier to query.
    select
        Optional column subset (None = all columns).
    filters
        Filter conditions.
    order_by
        Column ordering (prefix with "-" for DESC).
    limit
        Maximum rows to return.
    offset
        Rows to skip.
    """

    view_id: str
    select: list[str] | None = None
    filters: list[FilterSpec] = Field(default_factory=list)
    order_by: list[str] = Field(default_factory=list)
    limit: int = 200
    offset: int = 0


class SemanticQueryResponse(BaseModel):
    """Response from a semantic view query.

    Parameters
    ----------
    view_id
        Queried view identifier.
    columns
        Column names in result order.
    rows
        Result rows as list of dicts.
    truncated
        Whether results were truncated by limit.
    snapshot
        Snapshot metadata (repo, commit, run_id).
    """

    view_id: str
    columns: list[str]
    rows: list[dict[str, Any]]
    truncated: bool
    snapshot: dict[str, str]
```

#### 3.2.2 Semantic Registry

**File**: `src/codeintel/serving/semantic/registry.py`

```python
"""Semantic view registry loaded from published artifacts.

The registry is the serving-side representation of semantic views,
compiled from Hamilton DAG tags during the build phase.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from codeintel.serving.semantic.models import SemanticViewSpec


@dataclass(frozen=True)
class SemanticRegistry:
    """Registry of semantic views.

    Parameters
    ----------
    version
        Registry schema version.
    views
        Tuple of semantic view specifications.
    """

    version: str
    views: tuple[SemanticViewSpec, ...]

    @classmethod
    def load(cls, path: Path) -> SemanticRegistry:
        """Load registry from JSON file.

        Parameters
        ----------
        path
            Path to semantic_registry.json.

        Returns
        -------
        SemanticRegistry
            Loaded registry instance.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        views = tuple(SemanticViewSpec(**v) for v in payload.get("views", []))
        return cls(version=payload.get("version", "v1"), views=views)

    def by_id(self, view_id: str) -> SemanticViewSpec:
        """Look up view by semantic ID.

        Parameters
        ----------
        view_id
            Semantic view identifier.

        Returns
        -------
        SemanticViewSpec
            Matching view specification.

        Raises
        ------
        KeyError
            If view_id not found.
        """
        for view in self.views:
            if view.id == view_id:
                return view
        msg = f"Unknown semantic view: {view_id}"
        raise KeyError(msg)

    def list_view_ids(self) -> list[str]:
        """Return all registered view IDs."""
        return [v.id for v in self.views]

    def to_json(self) -> str:
        """Serialize registry to JSON string."""
        return json.dumps(
            {
                "version": self.version,
                "views": [v.model_dump() for v in self.views],
            },
            indent=2,
            sort_keys=True,
        )
```

#### 3.2.3 Schema Inventory

**File**: `src/codeintel/serving/semantic/inventory.py`

```python
"""Schema inventory for serving layer introspection.

Provides table/view metadata for agents to understand
available data structures without querying DuckDB catalog.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from codeintel.core.schemas.primitives import TableSchema


@dataclass(frozen=True)
class SchemaInventory:
    """Inventory of table and view schemas.

    Parameters
    ----------
    schemas
        Mapping from table_key to TableSchema.
    """

    schemas: dict[str, TableSchema]

    @classmethod
    def load(cls, path: Path) -> SchemaInventory:
        """Load inventory from schema manifest JSON.

        Parameters
        ----------
        path
            Path to schema_manifest.json.

        Returns
        -------
        SchemaInventory
            Loaded inventory instance.
        """
        from codeintel.core.schemas.primitives import Column

        payload = json.loads(path.read_text(encoding="utf-8"))
        schemas: dict[str, TableSchema] = {}

        for table_data in payload.get("tables", []):
            cols = [
                Column(
                    name=c["name"],
                    type=c["type"],
                    nullable=c.get("nullable", True),
                    description=c.get("description"),
                )
                for c in table_data.get("columns", [])
            ]
            schema = TableSchema(
                schema=table_data["schema"],
                name=table_data["name"],
                columns=cols,
                primary_key=tuple(table_data.get("primary_key", [])),
                description=table_data.get("description"),
            )
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

    def summary(self) -> dict[str, int]:
        """Return summary statistics."""
        tables = sum(1 for k in self.schemas if not k.startswith("docs.v_"))
        views = sum(1 for k in self.schemas if k.startswith("docs.v_"))
        return {"tables": tables, "views": views}
```

#### 3.2.4 Tasks Checklist for Phase 2

- [ ] Create `src/codeintel/serving/semantic/__init__.py`
- [ ] Create `src/codeintel/serving/semantic/models.py`
- [ ] Create `src/codeintel/serving/semantic/registry.py`
- [ ] Create `src/codeintel/serving/semantic/inventory.py`
- [ ] Create `tests/serving/semantic/test_registry.py`
- [ ] Create `tests/serving/semantic/test_inventory.py`
- [ ] Define JSON schema for `semantic_registry.json`
- [ ] Define JSON schema for `schema_manifest.json`

---

### Phase 3: Query Builder & Kernel (PR-77)

**Goal**: Implement safe query construction and the unified kernel API.

#### 3.3.1 Safe Query Builder

**File**: `src/codeintel/serving/semantic/query_builder.py`

Key features:
- Validates all identifiers against allowlist
- Uses parameter binding (no string interpolation)
- Supports standard filter operations
- Returns SQL + params tuple

```python
"""Safe SQL query builder with parameter binding.

All user-provided values are bound as parameters, never interpolated.
Identifiers (table/column names) are validated against the registry
and quoted to prevent injection.
"""
from __future__ import annotations

import re
from typing import Any

from codeintel.serving.semantic.models import FilterSpec

# Valid SQL identifier pattern (letters, numbers, underscores)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Operator to SQL mapping
_OP_SQL = {
    "eq": "=",
    "ne": "!=",
    "lt": "<",
    "lte": "<=",
    "gt": ">",
    "gte": ">=",
}


class QueryBuilderError(Exception):
    """Raised when query construction fails."""


def _quote_ident(name: str) -> str:
    """Quote a single identifier.

    Parameters
    ----------
    name
        Identifier to quote.

    Returns
    -------
    str
        Quoted identifier.

    Raises
    ------
    QueryBuilderError
        If identifier contains invalid characters.
    """
    if not _IDENT.match(name):
        msg = f"Invalid identifier: {name}"
        raise QueryBuilderError(msg)
    return f'"{name}"'


def _quote_table_key(table_key: str) -> str:
    """Quote a fully qualified table key.

    Parameters
    ----------
    table_key
        Table key in "schema.table" format.

    Returns
    -------
    str
        Quoted table reference.
    """
    parts = table_key.split(".")
    return ".".join(_quote_ident(p) for p in parts)


def build_query(
    *,
    table_key: str,
    columns: list[str],
    allowed_columns: set[str],
    filters: list[FilterSpec],
    order_by: list[str],
    limit: int,
    offset: int,
) -> tuple[str, list[Any]]:
    """Build a safe SELECT query with parameter binding.

    Parameters
    ----------
    table_key
        Fully qualified table/view name.
    columns
        Columns to select.
    allowed_columns
        Set of valid column names for validation.
    filters
        Filter specifications.
    order_by
        Order columns (prefix "-" for DESC).
    limit
        Maximum rows.
    offset
        Rows to skip.

    Returns
    -------
    tuple[str, list[Any]]
        SQL query string and parameter values.

    Raises
    ------
    QueryBuilderError
        If any identifier is invalid or column not allowed.
    """
    # Validate columns
    for col in columns:
        if col not in allowed_columns:
            msg = f"Unknown column: {col}"
            raise QueryBuilderError(msg)

    table_sql = _quote_table_key(table_key)
    select_sql = ", ".join(_quote_ident(c) for c in columns)

    # Build WHERE clause
    where_parts: list[str] = []
    params: list[Any] = []

    for f in filters:
        if f.column not in allowed_columns:
            msg = f"Unknown filter column: {f.column}"
            raise QueryBuilderError(msg)

        col_sql = _quote_ident(f.column)

        if f.op in _OP_SQL:
            where_parts.append(f"{col_sql} {_OP_SQL[f.op]} ?")
            params.append(f.value)
        elif f.op == "in":
            if not isinstance(f.value, list):
                msg = "IN operator requires list value"
                raise QueryBuilderError(msg)
            placeholders = ", ".join("?" for _ in f.value)
            where_parts.append(f"{col_sql} IN ({placeholders})")
            params.extend(f.value)
        elif f.op == "contains":
            where_parts.append(f"{col_sql} LIKE ?")
            params.append(f"%{f.value}%")
        elif f.op == "startswith":
            where_parts.append(f"{col_sql} LIKE ?")
            params.append(f"{f.value}%")
        else:
            msg = f"Unsupported operator: {f.op}"
            raise QueryBuilderError(msg)

    where_sql = ""
    if where_parts:
        where_sql = " WHERE " + " AND ".join(where_parts)

    # Build ORDER BY clause
    order_sql = ""
    if order_by:
        order_parts: list[str] = []
        for col in order_by:
            if col.startswith("-"):
                col_name = col[1:]
                direction = "DESC"
            else:
                col_name = col
                direction = "ASC"
            if col_name not in allowed_columns:
                msg = f"Unknown order_by column: {col_name}"
                raise QueryBuilderError(msg)
            order_parts.append(f"{_quote_ident(col_name)} {direction}")
        order_sql = " ORDER BY " + ", ".join(order_parts)

    sql = (
        f"SELECT {select_sql} FROM {table_sql}"
        f"{where_sql}{order_sql} LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])

    return sql, params
```

#### 3.3.2 Semantic Query Kernel

**File**: `src/codeintel/serving/semantic/kernel.py`

```python
"""Semantic query kernel - unified API for HTTP and MCP.

The kernel provides a single entry point for all semantic layer
queries, used by both FastAPI routes and MCP tools.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.semantic.inventory import SchemaInventory
from codeintel.serving.semantic.models import (
    FilterSpec,
    SemanticQueryRequest,
    SemanticQueryResponse,
    SemanticViewSpec,
)
from codeintel.serving.semantic.query_builder import build_query
from codeintel.serving.semantic.registry import SemanticRegistry


@dataclass
class SemanticQueryKernel:
    """Unified query kernel for semantic layer access.

    Parameters
    ----------
    db
        Database manager for connection access.
    """

    db: ServingDBManager

    def _load_registry(self) -> SemanticRegistry:
        """Load semantic registry from current snapshot."""
        pointer = self.db.current_pointer()
        return SemanticRegistry.load(pointer.semantic_registry_path)

    def _load_inventory(self) -> SchemaInventory:
        """Load schema inventory from current snapshot."""
        pointer = self.db.current_pointer()
        return SchemaInventory.load(pointer.schema_manifest_path)

    def catalog(self) -> dict[str, Any]:
        """List all available semantic views.

        Returns
        -------
        dict[str, Any]
            Catalog response with version, snapshot, and views.
        """
        registry = self._load_registry()
        pointer = self.db.current_pointer()

        return {
            "version": registry.version,
            "snapshot": {
                "repo": pointer.repo,
                "commit": pointer.commit,
                "run_id": pointer.run_id,
            },
            "views": [
                {
                    "id": v.id,
                    "table_key": v.table_key,
                    "entity": v.entity,
                    "grain": v.grain,
                    "description": v.description,
                    "column_count": len(v.columns),
                }
                for v in registry.views
                if not v.deprecated
            ],
        }

    def describe(self, view_id: str) -> dict[str, Any]:
        """Describe a single semantic view.

        Parameters
        ----------
        view_id
            Semantic view identifier.

        Returns
        -------
        dict[str, Any]
            View description with schema details.
        """
        registry = self._load_registry()
        inventory = self._load_inventory()
        pointer = self.db.current_pointer()

        view = registry.by_id(view_id)
        table_schema = inventory.get(view.table_key)

        return {
            "id": view.id,
            "table_key": view.table_key,
            "kind": view.kind,
            "entity": view.entity,
            "grain": view.grain,
            "description": view.description,
            "primary_key": view.primary_key,
            "columns": view.columns,
            "column_types": (
                {c.name: c.type for c in table_schema.columns}
                if table_schema
                else {}
            ),
            "joins": view.joins,
            "defaults": view.defaults,
            "deprecated": view.deprecated,
            "replaced_by": view.replaced_by,
            "snapshot": {
                "repo": pointer.repo,
                "commit": pointer.commit,
                "run_id": pointer.run_id,
            },
        }

    def query(self, request: SemanticQueryRequest) -> SemanticQueryResponse:
        """Execute a semantic view query.

        Parameters
        ----------
        request
            Query request with filters, selection, and pagination.

        Returns
        -------
        SemanticQueryResponse
            Query results.
        """
        registry = self._load_registry()
        view = registry.by_id(request.view_id)

        # Determine columns to select
        columns = request.select if request.select else view.columns
        allowed = set(view.columns)

        # Build safe query
        sql, params = build_query(
            table_key=view.table_key,
            columns=columns,
            allowed_columns=allowed,
            filters=request.filters,
            order_by=request.order_by or view.defaults.get("order_by", []),
            limit=request.limit or view.defaults.get("limit", 200),
            offset=request.offset,
        )

        # Execute query
        with self.db.connect() as (con, pointer):
            result = con.execute(sql, params).fetchall()
            rows = [dict(zip(columns, row, strict=True)) for row in result]

        return SemanticQueryResponse(
            view_id=request.view_id,
            columns=columns,
            rows=rows,
            truncated=len(rows) >= request.limit,
            snapshot={
                "repo": pointer.repo,
                "commit": pointer.commit,
                "run_id": pointer.run_id,
            },
        )

    def meta(self) -> dict[str, Any]:
        """Return serving metadata for /meta endpoint and tools.

        Returns
        -------
        dict[str, Any]
            Comprehensive serving metadata.
        """
        registry = self._load_registry()
        inventory = self._load_inventory()
        pointer = self.db.current_pointer()

        return {
            "repo": pointer.repo,
            "commit": pointer.commit,
            "run_id": pointer.run_id,
            "published_at": pointer.published_at.isoformat(),
            "semantic_layer_version": pointer.semantic_layer_version,
            "duckdb": {
                "db_path": str(pointer.db_path),
                "read_only": True,
            },
            "semantic_views": [
                {
                    "id": v.id,
                    "table_key": v.table_key,
                    "entity": v.entity,
                    "grain": v.grain,
                }
                for v in registry.views
                if not v.deprecated
            ],
            "schema_inventory": inventory.summary(),
        }
```

#### 3.3.3 Tasks Checklist for Phase 3

- [ ] Create `src/codeintel/serving/semantic/query_builder.py`
- [ ] Create `src/codeintel/serving/semantic/kernel.py`
- [ ] Create `tests/serving/semantic/test_query_builder.py`
  - Test valid query construction
  - Test parameter binding
  - Test invalid column rejection
  - Test operator validation
- [ ] Create `tests/serving/semantic/test_kernel.py`
  - Integration test with test DuckDB
  - Test catalog/describe/query/meta

---

### Phase 4: Build-Side Publisher (PR-78)

**Goal**: Implement snapshot publishing and semantic registry compilation.

#### 3.4.1 Serving Manifest

**File**: `src/codeintel/build/serving/manifest.py`

```python
"""Serving snapshot manifest dataclass.

Used during build to prepare snapshot metadata before publishing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json


@dataclass(frozen=True)
class ServingSnapshotManifest:
    """Manifest describing a serving snapshot.

    Parameters
    ----------
    run_id
        Unique build run identifier.
    repo
        Repository identifier.
    commit
        Commit SHA.
    created_at
        ISO timestamp when snapshot was created.
    db_path
        Path to DuckDB snapshot file.
    semantic_registry_path
        Path to semantic_registry.json.
    schema_manifest_path
        Path to schema_manifest.json.
    semantic_layer_version
        Version hash of semantic layer.
    """

    run_id: str
    repo: str
    commit: str
    created_at: str
    db_path: str
    semantic_registry_path: str
    schema_manifest_path: str
    semantic_layer_version: str

    def to_json(self) -> str:
        """Serialize manifest to JSON string."""
        return json.dumps(
            {
                "run_id": self.run_id,
                "repo": self.repo,
                "commit": self.commit,
                "created_at": self.created_at,
                "db_path": self.db_path,
                "semantic_registry_path": self.semantic_registry_path,
                "schema_manifest_path": self.schema_manifest_path,
                "semantic_layer_version": self.semantic_layer_version,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_path(cls, path: Path) -> ServingSnapshotManifest:
        """Load manifest from JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)
```

#### 3.4.2 Snapshot Publisher

**File**: `src/codeintel/build/serving/publisher.py`

```python
"""Serving snapshot publisher.

Publishes immutable read-only snapshots from build database
with atomic pointer updates for zero-downtime deployments.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from codeintel.build.serving.manifest import ServingSnapshotManifest
from codeintel.storage.gateway.protocol import StorageGateway


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically using rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", delete=False, encoding="utf-8", dir=str(path.parent)
    ) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _compute_semantic_version(
    registry_path: Path,
    manifest_path: Path,
) -> str:
    """Compute semantic layer version hash."""
    hasher = hashlib.sha256()
    for p in [registry_path, manifest_path]:
        if p.exists():
            hasher.update(p.read_bytes())
    return hasher.hexdigest()[:16]


def publish_serving_snapshot(
    *,
    gateway: StorageGateway,
    run_id: str,
    serve_dir: Path,
    semantic_registry_path: Path,
    schema_manifest_path: Path,
    keep_last: int = 10,
) -> ServingSnapshotManifest:
    """Publish an immutable serving snapshot.

    Parameters
    ----------
    gateway
        Storage gateway with build database.
    run_id
        Unique run identifier.
    serve_dir
        Root serving directory.
    semantic_registry_path
        Path to compiled semantic registry.
    schema_manifest_path
        Path to schema manifest.
    keep_last
        Number of old snapshots to retain.

    Returns
    -------
    ServingSnapshotManifest
        Published snapshot manifest.

    Raises
    ------
    FileNotFoundError
        If build database not found.
    """
    db_path = gateway.config.db_path
    if db_path is None or not db_path.is_file():
        msg = f"Build DB not found: {db_path}"
        raise FileNotFoundError(msg)

    # Checkpoint to flush WAL
    gateway.con.execute("CHECKPOINT")
    gateway.con.commit()

    # Create snapshot directory
    snap_dir = serve_dir / "snapshots" / run_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Copy database
    snap_db = snap_dir / "codeintel.duckdb"
    shutil.copy2(db_path, snap_db)

    # Copy registry and manifest
    snap_registry = snap_dir / "semantic_registry.json"
    shutil.copy2(semantic_registry_path, snap_registry)

    snap_manifest = snap_dir / "schema_manifest.json"
    shutil.copy2(schema_manifest_path, snap_manifest)

    # Compute version hash
    version = _compute_semantic_version(snap_registry, snap_manifest)

    # Build manifest
    manifest = ServingSnapshotManifest(
        run_id=run_id,
        repo=gateway.config.repo or "unknown",
        commit=gateway.config.commit or "unknown",
        created_at=datetime.now(timezone.utc).isoformat(),
        db_path=str(snap_db),
        semantic_registry_path=str(snap_registry),
        schema_manifest_path=str(snap_manifest),
        semantic_layer_version=version,
    )

    # Atomic publish pointer
    current_path = serve_dir / "current.json"
    _atomic_write_text(current_path, manifest.to_json())

    # Retention cleanup
    if keep_last > 0:
        snaps_root = serve_dir / "snapshots"
        if snaps_root.exists():
            dirs = sorted(
                [p for p in snaps_root.iterdir() if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old in dirs[keep_last:]:
                shutil.rmtree(old, ignore_errors=True)

    return manifest
```

#### 3.4.3 Semantic Registry Compiler

**File**: `src/codeintel/build/serving/semantic_compile.py`

```python
"""Compile semantic registry from Hamilton DAG tags.

Scans the Hamilton graph for nodes with semantic layer tags
and produces semantic_registry.json for serving.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codeintel.core.schemas.provider import SchemaProvider


# Tag keys for semantic layer
TAG_OUTPUT_KIND = "output_kind"
TAG_SEMANTIC_ID = "semantic_id"
TAG_SEMANTIC_KIND = "semantic_kind"
TAG_TABLE_KEY = "table_key"
TAG_SEMANTIC_ENTITY = "semantic_entity"
TAG_SEMANTIC_GRAIN = "semantic_grain"
TAG_SEMANTIC_PK = "semantic_primary_key"
TAG_SEMANTIC_COLS = "semantic_columns"
TAG_SEMANTIC_DESC = "semantic_description"
TAG_SEMANTIC_JOINS = "semantic_joins"
TAG_MCP_VISIBLE = "mcp_visible"
TAG_DEFAULT_ORDER = "semantic_default_order_by"
TAG_DEFAULT_LIMIT = "semantic_default_limit"
TAG_SENSITIVITY = "semantic_sensitivity"
TAG_DEPRECATED = "semantic_deprecated"
TAG_REPLACED_BY = "semantic_replaced_by"


def _split_csv(s: str | None) -> list[str]:
    """Split CSV string into list."""
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_json(s: str | None) -> Any:
    """Parse JSON string or return None."""
    if not s:
        return None
    return json.loads(s)


@dataclass(frozen=True)
class CompiledSemanticRegistry:
    """Compiled semantic registry ready for serialization."""

    version: str
    views: list[dict[str, Any]]

    def to_json(self) -> str:
        """Serialize to deterministic JSON."""
        return json.dumps(
            {"version": self.version, "views": self.views},
            indent=2,
            sort_keys=True,
        )


def compile_semantic_registry_from_views(
    *,
    schema_provider: SchemaProvider,
    view_tags: dict[str, dict[str, str]],
    version: str = "v1",
) -> CompiledSemanticRegistry:
    """Compile semantic registry from view tag metadata.

    Parameters
    ----------
    schema_provider
        Provider for resolving table schemas.
    view_tags
        Mapping from view name to tag dict.
    version
        Registry version string.

    Returns
    -------
    CompiledSemanticRegistry
        Compiled registry.
    """
    views: list[dict[str, Any]] = []

    for _view_name, tags in view_tags.items():
        if tags.get(TAG_OUTPUT_KIND) != "semantic":
            continue
        if tags.get(TAG_MCP_VISIBLE, "1") != "1":
            continue

        semantic_id = tags.get(TAG_SEMANTIC_ID)
        table_key = tags.get(TAG_TABLE_KEY)

        if not semantic_id or not table_key:
            continue

        # Resolve columns from schema or explicit tag
        explicit_cols = _split_csv(tags.get(TAG_SEMANTIC_COLS))
        if explicit_cols:
            cols = explicit_cols
        else:
            schema = schema_provider.get_table_schema(table_key)
            cols = schema.column_names() if schema else []

        view_entry = {
            "id": semantic_id,
            "kind": tags.get(TAG_SEMANTIC_KIND, "view"),
            "table_key": table_key,
            "entity": tags.get(TAG_SEMANTIC_ENTITY, "unknown"),
            "grain": tags.get(TAG_SEMANTIC_GRAIN, "unknown"),
            "description": tags.get(TAG_SEMANTIC_DESC),
            "primary_key": _split_csv(tags.get(TAG_SEMANTIC_PK)),
            "columns": cols,
            "joins": _parse_json(tags.get(TAG_SEMANTIC_JOINS)) or [],
            "defaults": {
                "limit": int(tags.get(TAG_DEFAULT_LIMIT, "200")),
                "order_by": _split_csv(tags.get(TAG_DEFAULT_ORDER)),
            },
            "sensitivity": tags.get(TAG_SENSITIVITY, "internal"),
            "deprecated": tags.get(TAG_DEPRECATED, "0") == "1",
            "replaced_by": tags.get(TAG_REPLACED_BY),
        }
        views.append(view_entry)

    # Deterministic ordering
    views.sort(key=lambda v: v["id"])

    return CompiledSemanticRegistry(version=version, views=views)


def write_semantic_registry(
    *,
    registry: CompiledSemanticRegistry,
    out_path: Path,
) -> None:
    """Write semantic registry to file.

    Parameters
    ----------
    registry
        Compiled registry.
    out_path
        Output file path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(registry.to_json(), encoding="utf-8")
```

#### 3.4.4 Tasks Checklist for Phase 4

- [ ] Create `src/codeintel/build/serving/__init__.py`
- [ ] Create `src/codeintel/build/serving/manifest.py`
- [ ] Create `src/codeintel/build/serving/publisher.py`
- [ ] Create `src/codeintel/build/serving/semantic_compile.py`
- [ ] Create `tests/build/serving/test_publisher.py`
- [ ] Create `tests/build/serving/test_semantic_compile.py`
- [ ] Add `@semantic_view` decorator to Hamilton native module
- [ ] Integrate publisher into build pipeline

---

### Phase 5: HTTP & MCP Integration (PR-79)

**Goal**: Wire FastAPI and FastMCP to the semantic kernel.

#### 3.5.1 Semantic HTTP Routes

**File**: `src/codeintel/serving/http/routes/semantic.py`

```python
"""Semantic layer HTTP endpoints.

Provides REST API access to the semantic query kernel.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException

from codeintel.serving.semantic.models import (
    SemanticQueryRequest,
    SemanticQueryResponse,
)

if TYPE_CHECKING:
    from codeintel.serving.semantic.kernel import SemanticQueryKernel

router = APIRouter(prefix="/semantic", tags=["semantic"])


def get_kernel() -> SemanticQueryKernel:
    """Dependency to get kernel from app state."""
    # Will be overridden in app setup
    raise NotImplementedError


@router.get("/views")
async def list_views(
    kernel: SemanticQueryKernel = Depends(get_kernel),
) -> dict[str, Any]:
    """List available semantic views.

    Returns
    -------
    dict[str, Any]
        Catalog of semantic views.
    """
    return kernel.catalog()


@router.get("/views/{view_id}")
async def describe_view(
    view_id: str,
    kernel: SemanticQueryKernel = Depends(get_kernel),
) -> dict[str, Any]:
    """Describe a semantic view.

    Parameters
    ----------
    view_id
        Semantic view identifier.

    Returns
    -------
    dict[str, Any]
        View description.
    """
    try:
        return kernel.describe(view_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/query")
async def query_view(
    request: SemanticQueryRequest,
    kernel: SemanticQueryKernel = Depends(get_kernel),
) -> SemanticQueryResponse:
    """Execute a semantic view query.

    Parameters
    ----------
    request
        Query request.

    Returns
    -------
    SemanticQueryResponse
        Query results.
    """
    try:
        return kernel.query(request)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
```

#### 3.5.2 FastAPI App Factory

**File**: `src/codeintel/serving/http/app.py`

```python
"""FastAPI application factory for semantic serving.

Creates a FastAPI app with semantic routes and optional MCP mount.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.db.pool import DuckDBPoolConfig
from codeintel.serving.http.routes import semantic
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.settings import ServingSettings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


def create_serving_app(
    settings: ServingSettings | None = None,
    *,
    mount_mcp: bool = True,
) -> FastAPI:
    """Create FastAPI serving application.

    Parameters
    ----------
    settings
        Serving settings (defaults to environment).
    mount_mcp
        Whether to mount MCP server at /mcp.

    Returns
    -------
    FastAPI
        Configured application.
    """
    cfg = settings or ServingSettings.from_env()

    # Create manager and kernel
    db_manager = ServingDBManager(
        pointer_path=cfg.serve_dir / "current.json",
        pool_cfg=DuckDBPoolConfig(size=cfg.pool_size),
        poll_interval_s=cfg.poll_interval_s,
    )
    kernel = SemanticQueryKernel(db=db_manager)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Application lifespan handler."""
        await db_manager.start()
        try:
            yield
        finally:
            await db_manager.stop()

    app = FastAPI(
        title="CodeIntel Serving",
        description="Semantic layer API for CodeIntel",
        lifespan=lifespan,
    )

    # Store in state for dependency injection
    app.state.kernel = kernel
    app.state.db_manager = db_manager

    # Override dependency
    def get_kernel() -> SemanticQueryKernel:
        return kernel

    app.dependency_overrides[semantic.get_kernel] = get_kernel

    # Include routers
    app.include_router(semantic.router)

    # Health endpoint
    @app.get("/health")
    async def health() -> dict[str, str]:
        pointer = db_manager.current_pointer()
        return {
            "status": "ok",
            "repo": pointer.repo,
            "commit": pointer.commit,
            "run_id": pointer.run_id,
        }

    # Meta endpoint
    @app.get("/meta")
    async def meta() -> dict:
        return kernel.meta()

    # Mount MCP if requested
    if mount_mcp:
        from codeintel.serving.mcp.app import build_mcp_app

        mcp = build_mcp_app(kernel=kernel)
        mcp_asgi = mcp.http_app(path="/")
        app.mount("/mcp", mcp_asgi)

    return app
```

#### 3.5.3 MCP App Builder

**File**: `src/codeintel/serving/mcp/app.py`

```python
"""FastMCP application builder for semantic tools.

Creates MCP server with semantic layer tools.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from codeintel.serving.semantic.models import SemanticQueryRequest

if TYPE_CHECKING:
    from codeintel.serving.semantic.kernel import SemanticQueryKernel


def build_mcp_app(*, kernel: SemanticQueryKernel) -> FastMCP:
    """Build FastMCP application with semantic tools.

    Parameters
    ----------
    kernel
        Semantic query kernel.

    Returns
    -------
    FastMCP
        Configured MCP server.
    """
    mcp = FastMCP("CodeIntel", json_response=True)

    @mcp.tool
    def semantic_catalog() -> dict[str, Any]:
        """List available semantic views in the CodeIntel database.

        Returns a catalog of all semantic views with their IDs, entities,
        grains, and descriptions. Use this to discover what data is available
        before querying.
        """
        return kernel.catalog()

    @mcp.tool
    def semantic_describe(view_id: str) -> dict[str, Any]:
        """Describe a semantic view's schema and metadata.

        Parameters
        ----------
        view_id
            Semantic view identifier (e.g., "function.summary").

        Returns the view's columns, types, primary key, join hints,
        and default query parameters.
        """
        return kernel.describe(view_id)

    @mcp.tool
    def semantic_query(
        view_id: str,
        filters: list[dict[str, Any]] | None = None,
        select: list[str] | None = None,
        order_by: list[str] | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Query a semantic view with structured filters.

        Parameters
        ----------
        view_id
            Semantic view identifier.
        filters
            List of filter dicts with "column", "op", "value" keys.
            Supported ops: eq, ne, lt, lte, gt, gte, in, contains, startswith.
        select
            Column names to return (None = all columns).
        order_by
            Column names for ordering (prefix "-" for DESC).
        limit
            Maximum rows to return.
        offset
            Rows to skip.

        Returns query results with columns, rows, and snapshot metadata.
        """
        from codeintel.serving.semantic.models import FilterSpec

        request = SemanticQueryRequest(
            view_id=view_id,
            select=select,
            filters=[FilterSpec(**f) for f in (filters or [])],
            order_by=order_by or [],
            limit=limit,
            offset=offset,
        )
        result = kernel.query(request)
        return result.model_dump()

    @mcp.tool
    def serving_meta() -> dict[str, Any]:
        """Get serving layer metadata.

        Returns repository, commit, snapshot info, semantic layer version,
        and inventory summary.
        """
        return kernel.meta()

    return mcp
```

#### 3.5.4 Tasks Checklist for Phase 5

- [ ] Create `src/codeintel/serving/http/routes/semantic.py`
- [ ] Update `src/codeintel/serving/http/app.py`
- [ ] Create `src/codeintel/serving/mcp/app.py`
- [ ] Update `src/codeintel/serving/mcp/server.py`
- [ ] Create `tests/serving/http/test_semantic_routes.py`
- [ ] Create `tests/serving/mcp/test_semantic_tools.py`
- [ ] Integration test: FastAPI + MCP mounted together

---

### Phase 6: Legacy Deletion (PR-80)

**Goal**: Remove deprecated code after semantic serving is stable.

#### 3.6.1 Modules to Delete

| Path | Reason |
|------|--------|
| `serving/backend/core.py` | Replaced by SemanticQueryKernel |
| `serving/backend/dataset_backend.py` | Replaced by semantic query |
| `serving/backend/datasets.py` | Replaced by semantic registry |
| `serving/backend/domain_builders.py` | Not needed for semantic |
| `serving/backend/duckdb_service.py` | Replaced by kernel |
| `serving/backend/function_backend.py` | Replaced by semantic views |
| `serving/backend/profile_backend.py` | Replaced by semantic views |
| `serving/backend/query_api.py` | Replaced by query_builder |
| `serving/backend/subsystem_backend.py` | Replaced by semantic views |
| `serving/services/conversion.py` | Not needed |
| `serving/services/datasets.py` | Replaced by semantic |
| `serving/services/errors.py` | Simplify to kernel errors |
| `serving/services/functions.py` | Replaced by semantic |
| `serving/services/http_helpers.py` | Not needed |
| `serving/services/profiles.py` | Replaced by semantic |
| `serving/services/query_service.py` | Replaced by kernel |
| `serving/services/subsystems.py` | Replaced by semantic |
| `serving/services/transport.py` | Not needed |
| `serving/operations/catalog.py` | **Critical**: Static ops → semantic |
| `serving/auto_pipeline.py` | Not needed for semantic |
| `serving/bootstrap.py` | Replaced by simplified factory |
| `serving/context.py` | Replaced by kernel |
| `serving/contracts/` | Move validation to semantic |
| `serving/mcp/backend.py` | Replaced by kernel |
| `serving/mcp/backend_dispatch.py` | Not needed |
| `serving/mcp/auto_pipeline_wrapper.py` | Delete |
| `serving/mcp/tools.py` | Replaced by tools_semantic |
| `serving/mcp/registry.py` | Simplified |
| `serving/mcp/view_utils.py` | Not needed |
| `serving/http/routes/datasets.py` | Replaced by semantic |
| `serving/http/routes/functions.py` | Replaced by semantic |
| `serving/http/routes/profiles.py` | Replaced by semantic |
| `serving/http/routes/subsystems.py` | Replaced by semantic |
| `serving/http/routes/architecture.py` | Replaced by semantic |
| `serving/http/routes/ide.py` | Evaluate: keep or migrate |

#### 3.6.2 Modules to Keep (Refactored)

| Path | Status |
|------|--------|
| `serving/backend/pagination.py` | **KEEP**: BackendLimits useful |
| `serving/services/observability.py` | **KEEP**: Reuse for kernel |
| `serving/domain_models.py` | **EVALUATE**: May keep for types |
| `serving/types.py` | **KEEP**: Utility types |
| `serving/http/dependencies.py` | **REFACTOR**: For kernel |
| `serving/http/routes/health.py` | **KEEP**: Already simple |
| `serving/http/routes/meta.py` | **REFACTOR**: Use kernel.meta() |
| `serving/mcp/errors.py` | **KEEP**: Error utilities |
| `serving/mcp/models.py` | **REFACTOR**: For semantic |
| `serving/mcp/serialization.py` | **EVALUATE**: May not need |
| `serving/mcp/server.py` | **REFACTOR**: Simplified |

#### 3.6.3 Config Modules to Migrate

| Old Location | New Location | Notes |
|--------------|--------------|-------|
| `config/datasets/` | N/A | Semantic registry replaces |
| `config/datasets/contracts.py` | Build schemas | Already have SchemaProvider |
| `config/datasets/dataflow.py` | N/A | Not needed for serving |
| `config/serving_models.py` | `serving/settings.py` | Simplified |

#### 3.6.4 Deletion Checklist

- [ ] Move legacy modules to `serving/_legacy/` temporarily
- [ ] Update imports in remaining code
- [ ] Run full test suite
- [ ] Remove `serving/_legacy/` directory
- [ ] Remove `config/datasets/` imports from serving
- [ ] Update `serving/__init__.py` exports
- [ ] Update documentation

---

## Part 4: Semantic View Starter Pack

### 4.1 Initial Semantic Views to Tag

These existing Ibis views should be tagged for semantic exposure:

| Semantic ID | Table Key | Entity | Grain |
|-------------|-----------|--------|-------|
| `function.summary` | `docs.v_function_summary` | function | per_function |
| `function.architecture` | `docs.v_function_architecture` | function | per_function |
| `function.hotspots` | `analytics.v_function_hotspots` | function | per_function |
| `module.architecture` | `docs.v_module_architecture` | module | per_module |
| `module.architecture_full` | `docs.v_module_architecture_full` | module | per_module |
| `file.summary` | `docs.v_file_summary` | file | per_file |
| `subsystem.summary` | `docs.v_subsystem_summary` | subsystem | per_subsystem |
| `subsystem.profile` | `docs.v_subsystem_profile` | subsystem | per_subsystem |
| `subsystem.coverage` | `docs.v_subsystem_coverage` | subsystem | per_subsystem |
| `test.architecture` | `docs.v_test_architecture` | test | per_test |
| `test.to_function` | `docs.v_test_to_function` | coverage_edge | per_edge |
| `call_graph.enriched` | `docs.v_call_graph_enriched` | call_edge | per_edge |
| `call_graph.degree` | `graph.v_call_graph_degree` | function | per_function |
| `import_graph.degree` | `graph.v_import_graph_degree` | module | per_module |
| `ide.hints` | `docs.v_ide_hints` | module | per_module |

### 4.2 Tag Decorator Usage

```python
# Example in storage/views/ibis_views.py

from codeintel.build.hamilton.native.semantic_decorators import semantic_view

@register_view("docs.v_function_summary")
@semantic_view(
    semantic_id="function.summary",
    table_key="docs.v_function_summary",
    entity="function",
    grain="per_function",
    primary_key=("function_goid_h128", "repo", "commit"),
    description="Comprehensive function summary with risk, coverage, and typing metrics",
    default_order_by=("-risk_score", "qualname"),
    default_limit=200,
    joins=[
        {"to": "module.architecture", "on": [["module", "module"]]},
        {"to": "call_graph.enriched", "on": [["function_goid_h128", "caller_goid_h128"]]},
    ],
)
def build_docs_function_summary(ibis_gw: IbisViewGateway) -> it.Table:
    # ... existing implementation ...
```

---

## Part 5: Testing Strategy

### 5.1 Unit Tests

| Module | Test File | Coverage |
|--------|-----------|----------|
| `db/pointer.py` | `test_pointer.py` | Load, serialize, validate |
| `db/pool.py` | `test_pool.py` | Acquire, release, close |
| `db/manager.py` | `test_manager.py` | Hot-swap, watch loop |
| `semantic/registry.py` | `test_registry.py` | Load, lookup, serialize |
| `semantic/inventory.py` | `test_inventory.py` | Load, lookup, summary |
| `semantic/query_builder.py` | `test_query_builder.py` | SQL gen, param binding |
| `semantic/kernel.py` | `test_kernel.py` | All kernel methods |

### 5.2 Integration Tests

| Test | Description |
|------|-------------|
| `test_e2e_semantic_serving.py` | Build → Publish → Serve → Query cycle |
| `test_hot_swap.py` | Pointer update triggers pool swap |
| `test_mcp_tools.py` | MCP tool invocation against real DB |
| `test_http_routes.py` | FastAPI TestClient against routes |

### 5.3 Snapshot Tests

| Artifact | Purpose |
|----------|---------|
| `semantic_registry.json` | Detect semantic layer changes |
| `schema_manifest.json` | Detect schema drift |
| `/meta` response | Detect serving metadata changes |

---

## Part 6: Migration Checklist

### 6.1 Pre-Migration

- [ ] Create feature branch `feature/semantic-serving`
- [ ] Document current serving API surface
- [ ] Create backward-compat shims if needed
- [ ] Set up test fixtures with minimal snapshots

### 6.2 During Migration

- [ ] Phase 0: Foundation (PR-74)
- [ ] Phase 1: DB Layer (PR-75)
- [ ] Phase 2: Semantic Registry (PR-76)
- [ ] Phase 3: Query Kernel (PR-77)
- [ ] Phase 4: Build Publisher (PR-78)
- [ ] Phase 5: HTTP/MCP Integration (PR-79)
- [ ] Phase 6: Legacy Deletion (PR-80)

### 6.3 Post-Migration

- [ ] Update CLI commands (`codeintel serve`, `codeintel mcp`)
- [ ] Update OpenAPI spec
- [ ] Update MCP tool documentation
- [ ] Remove deprecated functions
- [ ] Update AGENTS.md with new serving architecture
- [ ] Performance benchmark semantic queries
- [ ] Security review query builder

---

## Part 7: Risk Mitigation

### 7.1 Rollback Strategy

Each phase is designed to be independently mergeable:

1. **Phase 0-3**: No breaking changes to existing serving
2. **Phase 4**: Publisher can coexist with legacy
3. **Phase 5**: New routes alongside old (different paths)
4. **Phase 6**: Only delete after full validation

### 7.2 Feature Flags

```python
# In serving/settings.py
CODEINTEL_SEMANTIC_SERVING = os.environ.get("CODEINTEL_SEMANTIC_SERVING", "0") == "1"
```

Enable new semantic routes only when flag is set during transition.

### 7.3 Monitoring

- [ ] Add metrics for semantic query latency
- [ ] Add metrics for hot-swap events
- [ ] Add metrics for pool utilization
- [ ] Log semantic query patterns for optimization

---

## Part 8: Timeline Estimate

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 0 | 0.5 day | None |
| Phase 1 | 1 day | Phase 0 |
| Phase 2 | 1 day | Phase 1 |
| Phase 3 | 1.5 days | Phase 2 |
| Phase 4 | 1 day | Phase 3 |
| Phase 5 | 1.5 days | Phase 4 |
| Phase 6 | 1 day | Phase 5 + validation |
| **Total** | **~8 days** | Sequential |

---

## Appendix A: Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEINTEL_SERVE_DIR` | `.codeintel/serve` | Serving snapshot directory |
| `CODEINTEL_SERVE_HOTSWAP` | `1` | Enable hot-swap (0/1) |
| `CODEINTEL_SERVE_POOL_SIZE` | `4` | Connections per worker |
| `CODEINTEL_SERVE_POLL_INTERVAL` | `1.0` | Seconds between pointer checks |
| `CODEINTEL_MCP_TRANSPORT` | `stdio` | MCP transport (stdio/http) |
| `CODEINTEL_HOST` | `127.0.0.1` | HTTP bind address |
| `CODEINTEL_PORT` | `8000` | HTTP port |
| `CODEINTEL_AUTH_TOKEN` | None | Optional bearer token |

---

## Appendix B: JSON Schemas

### B.1 semantic_registry.json

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["version", "views"],
  "properties": {
    "version": { "type": "string" },
    "views": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "table_key", "entity", "grain"],
        "properties": {
          "id": { "type": "string" },
          "kind": { "enum": ["table", "view"] },
          "table_key": { "type": "string" },
          "entity": { "type": "string" },
          "grain": { "type": "string" },
          "description": { "type": ["string", "null"] },
          "primary_key": { "type": "array", "items": { "type": "string" } },
          "columns": { "type": "array", "items": { "type": "string" } },
          "joins": { "type": "array" },
          "defaults": { "type": "object" },
          "sensitivity": { "type": "string" },
          "deprecated": { "type": "boolean" },
          "replaced_by": { "type": ["string", "null"] }
        }
      }
    }
  }
}
```

### B.2 current.json (Pointer)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": [
    "db_path", "semantic_registry_path", "schema_manifest_path",
    "repo", "commit", "run_id", "published_at", "semantic_layer_version"
  ],
  "properties": {
    "db_path": { "type": "string" },
    "semantic_registry_path": { "type": "string" },
    "schema_manifest_path": { "type": "string" },
    "repo": { "type": "string" },
    "commit": { "type": "string" },
    "run_id": { "type": "string" },
    "published_at": { "type": "string", "format": "date-time" },
    "semantic_layer_version": { "type": "string" }
  }
}
```

---

## Appendix C: MCP Tool Signatures

### semantic_catalog

```
Name: semantic_catalog
Description: List available semantic views in the CodeIntel database
Parameters: None
Returns: { version, snapshot, views: [{id, table_key, entity, grain, description, column_count}] }
```

### semantic_describe

```
Name: semantic_describe
Description: Describe a semantic view's schema and metadata
Parameters:
  - view_id: string (required)
Returns: { id, table_key, kind, entity, grain, description, primary_key, columns, column_types, joins, defaults, deprecated, replaced_by, snapshot }
```

### semantic_query

```
Name: semantic_query
Description: Query a semantic view with structured filters
Parameters:
  - view_id: string (required)
  - filters: array of {column, op, value} (optional)
  - select: array of column names (optional)
  - order_by: array of column names with optional "-" prefix (optional)
  - limit: integer, default 200 (optional)
  - offset: integer, default 0 (optional)
Returns: { view_id, columns, rows, truncated, snapshot }
```

### serving_meta

```
Name: serving_meta
Description: Get serving layer metadata
Parameters: None
Returns: { repo, commit, run_id, published_at, semantic_layer_version, duckdb, semantic_views, schema_inventory }
```

---

**End of Implementation Plan**






