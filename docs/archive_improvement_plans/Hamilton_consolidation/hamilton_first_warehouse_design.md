# Hamilton-First Warehouse & Semantic Data System (Go-Forward Design)

Status: Draft (proposed target architecture)  
Location: `docs/Hamilton_consolidation/hamilton_first_warehouse_design.md`  
Related docs:
- `docs/Hamilton_consolidation/hamilton_native_implementation_plan.md`
- `docs/Hamilton_consolidation/Remaining_PRs_Semantic_focus.md`
- `docs/python_library_reference/hamilton_advanced.md`
- `docs/python_library_reference/duckdb_advanced.md`

## Executive Summary

This document defines a best-in-class target architecture for CodeIntel’s data system after the
Hamilton-native transition:

- **Hamilton is the single source of truth** for graph structure and node metadata (tags/schema docs).
- **Ibis expressions are the universal data object** used across build, storage, and serving.
- **Storage becomes a “warehouse” API**: a small, typed surface that owns all DuckDB/SQLGlot I/O and
  contract enforcement.
- **Semantic views are Hamilton-discoverable** (tags), compiled into a registry, and safely queried
  through a manifest-driven allowlist.

The goal is to make the system **safe by construction** (schema/allowlist enforcement), **easy to
extend** (new datasets/views automatically appear), and **operationally observable** (telemetry is a
first-class data product).

## Goals

1. **Single set of shared data objects** used by build/storage/serving:
   - identifiers: `TableKey`, `SnapshotRef`
   - inventory: `DatasetSpec`, `SemanticViewSpec`
   - results: `MaterializationResult`, `QueryResult`
   - compute payloads: `ibis.expr.types.Table` (primary), Arrow/Polars/Pandas at boundaries only.
2. **Single I/O boundary**:
   - all writes, schema changes, and view materialization go through a small “warehouse” module.
3. **No bespoke registries or import side effects**:
   - view/semantic discovery uses Hamilton tags and module discovery.
4. **Manifest-driven safety**:
   - serving queries and materializers rely on the schema manifest as canonical truth.
5. **Excellent observability**:
   - Hamilton lifecycle hooks generate telemetry; storage persists it (runs/nodes/assets/profiling).

## Non-Goals (for this design phase)

- Rewriting every existing target at once.
- Adding non-DuckDB backends immediately (but we design for it).
- Introducing a new query language (Ibis + typed request objects remain the interface).

## Principles

1. **Pure compute; explicit side effects**: only materialize nodes perform I/O.
2. **Ibis-first everywhere**: queries are Ibis expressions; SQL strings are an internal backend detail.
3. **Schema is a contract**: the schema manifest/provider is canonical; runtime must not drift.
4. **Discovery > registration**: avoid manual “add to registry” work; derive from graph + tags.
5. **Determinism by default**: ordering, compilation, and materialization should be stable.

---

## Core Shared Types (Object Model)

### Identifiers

- **`TableKey`**: canonical identifier for a table/view, always `"schema.name"` (e.g. `"core.modules"`).
- **`SnapshotRef`**: identifies a repo/commit snapshot (`repo`, `commit`), optionally with `repo_root`.

### Inventory Objects

`DatasetSpec` represents the canonical warehouse inventory row (source of truth is a manifest +
contract provider; the database registry mirrors it for runtime discoverability):

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    table_key: str
    name: str
    kind: str  # "table" | "view"
    family: str | None
    description: str | None

    snapshot_scoped: bool  # has repo+commit columns and is snapshot-addressable
    columns: tuple[str, ...]
    schema_hash: str
    schema_version: str | None

    upstream: tuple[str, ...]  # dataset names or table_keys
    tags: dict[str, object]  # owner/pii/grain/etc (Hamilton tags compatible)
```

`SemanticViewSpec` describes a view exposed to serving/MCP consumers. Source of truth is Hamilton tags;
the JSON registry is a build artifact for serving:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticViewSpec:
    semantic_id: str
    table_key: str  # where the view is materialized (usually docs.* or analytics.*)

    entity: str
    grain: str
    description: str | None

    columns: tuple[str, ...] | None  # None means “all schema manifest columns”
    mcp_visible: bool

    tags: dict[str, object]  # additional structured metadata
```

### Results

All side-effecting operations return structured results:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MaterializationResult:
    table_key: str
    repo: str | None
    commit: str | None

    rows_written: int | None
    started_at: datetime
    completed_at: datetime

    schema_hash: str | None
    schema_version: str | None
    profiling_artifact: str | None  # optional path for profiling output
```

Serving queries also return structured results:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryResult:
    rows: list[dict[str, object]]
    columns: tuple[str, ...]
    execution_ms: float | None
    compiled_sql: str | None  # optional (debug/trace)
```

---

## Layered Architecture

The system is intentionally layered to keep boundaries crisp:

1. **Backend (DuckDB session + primitives)**
2. **Warehouse API (typed core operations)**
3. **Catalog/Contract (inventory + enforcement)**
4. **Hamilton integration (loaders/materializers + discovery)**
5. **Serving (safe semantic querying + result materialization)**

### Target Module Layout (Proposed)

This design is intended to be implemented as a small number of stable entrypoints, while reusing
existing components internally (`IbisGateway`, `DuckDBPolicyBackend`, tracking, etc.).

Target layout (names are illustrative; the boundary is the important part):

- `src/codeintel/storage/warehouse.py` or `src/codeintel/storage/warehouse/`: typed “warehouse” API
  consumed by build + serving (read/exists/count/materialize/view ensure).
- `src/codeintel/storage/backend/duckdb_session.py`: session lifecycle, attach/export/import,
  extensions/secrets, tuning, and concurrency controls (single-writer).
- `src/codeintel/storage/catalog/`: manifest + registry bootstrapping, dataset/view dependency graph.
- `src/codeintel/storage/contracts/`: storage-owned provider interface; build becomes one
  implementation (during transition, keep `src/codeintel/storage/build_bridge.py`).
- `src/codeintel/storage/hamilton_io/`: `@dataloader`/`@datasaver` implementations and/or generated
  `q__...` query nodes from the catalog.
- `src/codeintel/storage/serving/`: semantic query request → Ibis expression building + allowlist
  enforcement + Arrow/Polars extraction.

### 1) Backend: DuckDB Session Management

Backend owns:
- connection open/attach/export/import
- extension management + secrets (httpfs, fts, cloud credentials)
- tuning (threads/memory/temp/profiling), default no-op unless enabled
- concurrency strategy (single-writer, per-thread readers)

Target interface (conceptual):

```python
class DuckDBSession:
    def con(self) -> DuckDBConnection: ...
    def read_con(self) -> DuckDBConnection: ...
    def close(self) -> None: ...
```

Key requirements:
- **No raw SQL outside the backend** except for constrained, audited internal utilities.
- **All non-trivial SQL is generated by SQLGlot** in policy/warehouse layers.

### 2) Warehouse API: Core Storage Operations

Warehouse is the only approved API for:
- reading tables/views as Ibis expressions
- existence/count checks
- writing tables and creating/replacing views
- schema/view/index “ensure” operations
- snapshot deletion operations
- (optional) EXPLAIN/profiling/export helpers

It is implemented on top of the existing components:
- `src/codeintel/storage/ibis_adapter.py` (`IbisGateway`)
- `src/codeintel/storage/duckdb_policy_backend.py` (`DuckDBPolicyBackend`)

#### Explicit API sketch

The goal is a small, typed surface that build/serving depend on, while the internal implementation
can evolve freely:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import ibis.expr.types as ir


@dataclass(frozen=True)
class SnapshotRef:
    repo: str
    commit: str


WriteMode = Literal["append", "replace", "upsert"]


class Warehouse(Protocol):
    def read(self, table_key: str, snapshot: SnapshotRef | None = None) -> ir.Table: ...
    def exists(self, table_key: str, snapshot: SnapshotRef | None = None) -> bool: ...
    def count(self, table_key: str, snapshot: SnapshotRef | None = None) -> int: ...

    def materialize_table(
        self,
        table_key: str,
        expr: ir.Table,
        *,
        snapshot: SnapshotRef | None = None,
        mode: WriteMode = "append",
    ) -> MaterializationResult: ...

    def create_or_replace_view(self, table_key: str, expr: ir.Table) -> None: ...
    def ensure_all_views(self) -> None: ...

    def delete_snapshot(self, snapshot: SnapshotRef) -> None: ...
```

#### Read API

Read is always Ibis-first:

- `read(table_key, snapshot=None) -> ibis.Table`
  - If `snapshot` is provided and dataset is snapshot-scoped, apply `(repo, commit)` filter.
  - If not snapshot-scoped, ignore snapshot safely (or warn in strict modes).

This removes ad-hoc snapshot scoping scattered throughout the codebase.

#### Existence/Count API

`exists()` and `count()` should be implemented once and reused everywhere.
The canonical implementation location is `src/codeintel/storage/queries/safe.py` (expanded to include
snapshot-scoped primitives), with legacy wrappers re-exporting from there.

#### Write API

Writes accept high-level inputs and produce `MaterializationResult`:
- Ibis table expressions (preferred)
- Arrow/Polars/Pandas at boundaries only

Internally:
- INSERT…SELECT paths use Ibis compilation + SQLGlot
- UPSERT/DDL uses SQLGlot (policy backend)
- column order enforced via schema provider

#### View API

Views are created from Ibis expressions with explicit overwrite semantics.
Implementation should mirror the existing Ibis 11+ requirement to pass `database=` for qualified names.

### 3) Catalog/Contract: Inventory + Enforcement

Catalog is the “warehouse contract” layer:
- schema provider + schema manifest generation
- dataset registry table population (`metadata.datasets`) and schema hash registry
- conformance checks (optional sampling)
- dependency graph (datasets + views)

Target properties:
- **Single canonical inventory**: schema manifest + contract provider drive runtime.
- database registry is derived/bootstrapped for introspection and tooling.

Important: storage should own a stable “contract provider interface” even if build remains source of
contracts today. The bridge module is the transition shim:

- `src/codeintel/storage/build_bridge.py` (current) is the temporary adapter.
- target: `codeintel.storage.contracts.Provider` interface so storage can be backend-agnostic.

### 4) Hamilton Integration: Discovery + I/O Boundaries

This layer makes Hamilton-native execution the default:

#### (A) Auto-generated query nodes (`q__...`)

Goal: any dataset in the registry/manifest becomes automatically available as a query node:

- `q__core__modules -> ibis.Table`
- `q__docs__v_function_summary -> ibis.Table`

Implementation options:
1. Generate modules at build time from `DatasetSpec` inventory (preferred for clarity).
2. Dynamically build a Driver module wrapper at runtime (less explicit, but flexible).

Each query node should return the Ibis expression from warehouse:

```python
def q__core__modules(env: BuildEnv) -> ir.Table:
    return env.warehouse.read("core.modules", snapshot=env.snapshot)
```

#### (B) Materializers: a single write boundary

Materialize nodes (tagged `node_type="materialize"`) call warehouse APIs:

- compute nodes return Ibis expressions (or typed dataclasses where needed)
- materialize node writes to DuckDB and returns `MaterializationResult` (and/or manifest records)

This is the natural place to integrate Hamilton’s `@datasaver` (per plan), but the key requirement is:
**all writes go through the same warehouse codepath**.

#### (C) Semantic Views as Hamilton-discoverable nodes

Target end state aligns with PR‑84/85/86 in `docs/Hamilton_consolidation/Remaining_PRs_Semantic_focus.md`:

- semantic metadata is encoded as Hamilton tags
- semantic registry compilation uses Hamilton’s tag discovery
- view materialization uses the same discovery mechanism
- no bespoke registry objects remain

Key design decision for best alignment:
- Prefer defining semantic views as Hamilton nodes with explicit dependencies:

```python
@tag(output_kind="semantic_view", semantic_id="function_summary", table_key="docs.v_function_summary")
def docs__v_function_summary(
    q__analytics__function_metrics: ir.Table,
    q__analytics__function_types: ir.Table,
) -> ir.Table:
    return q__analytics__function_metrics.left_join(q__analytics__function_types, ["function_goid_h128"])
```

This makes dependencies explicit (Hamilton DAG knows lineage) and avoids the “gateway lookup by string”
pattern for view definitions.

### 5) Serving: Safe Semantic Querying

Serving is built on three canonical inputs:
1. `semantic_registry.json` (compiled from Hamilton tags)
2. schema manifest/inventory (canonical columns/types)
3. read-only snapshot gateway/session manager

#### Query request object (typed)

Serving should accept a typed request object and compile it to an Ibis expression, rather than
accepting raw SQL or bespoke string parsing.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ColumnFilter:
    column: str
    op: Literal["eq", "neq", "lt", "lte", "gt", "gte", "in"]
    value: object


@dataclass(frozen=True)
class OrderBy:
    column: str
    direction: Literal["asc", "desc"] = "asc"


@dataclass(frozen=True)
class SemanticQueryRequest:
    semantic_id: str
    snapshot: SnapshotRef | None
    columns: tuple[str, ...] | None
    filters: tuple[ColumnFilter, ...] = ()
    order_by: tuple[OrderBy, ...] = ()
    limit: int | None = None
```

#### Safety model (PR‑87)

The runtime allowlist is derived from the schema manifest:
- if registry provides columns: enforce subset of manifest columns
- if registry provides none: default to manifest columns

Modes:
- strict: error on unknown columns
- warn: intersect and continue with warning
- off: legacy behavior (temporary)

#### Execution model (PR‑88)

Execution stays Ibis-first but avoids pandas for performance:
- compile Ibis expr → SQL
- execute SQL via DuckDB
- fetch as Arrow/Polars
- normalize to JSON-safe rows at the boundary

This becomes the unified “query result extraction” path for serving and agentic consumers.

#### Introspection tools

High-leverage additions:
- `semantic_explain()` returning DuckDB plan text and compiled SQL
- offline CLI utilities to catalog/describe/query semantic views without running the server

---

## Observability & Telemetry

Telemetry is a first-class data product:

- **Node-level telemetry** from Hamilton lifecycle hooks → `build.run_nodes`
  - implemented via `src/codeintel/build/hamilton/hooks/telemetry_hook.py`
- **Pipeline runs/steps** → `metadata.pipeline_runs` / `metadata.pipeline_steps`
  - persisted via storage tracking (`src/codeintel/storage/tracking/run_tracking.py`)
- **Assets/materializations** → `build.assets` and/or manifest-backed tables
  - persisted via storage tracking (`src/codeintel/storage/tracking/asset_tracking.py`)

Target behavior:
- Every materialize node returns `MaterializationResult`.
- The materializer records:
  - the write (rows/schema hash)
  - the asset record (table_key/snapshot)
  - the step/run linkage (for reproducibility)

Optional profiling:
- when enabled, warehouse captures DuckDB profiling output per materialize node and stores a pointer.

---

## Concurrency & Parallel Execution (PR‑96 direction)

Goal: enable safe parallel execution without corrupting DuckDB state:

- compute nodes can run in parallel
- materialize nodes run under a global write lock (single-writer)
- reads use separate connections per thread when needed (snapshot manager / connection pool)
- use early guardrails (cross-thread cache detection) during rollout

This aligns with:
- Hamilton tags: `node_type="materialize"`
- warehouse: explicit “writer” operations vs “reader” operations

---

## Advanced Capabilities to Leverage (Explicitly)

This section makes the “advanced features” commitments concrete, so the implementation can avoid
reinventing primitives already provided by Hamilton + DuckDB.

### Hamilton capabilities to standardize on

- `@tag` and `@schema.output` as the canonical semantic metadata channel (discovery + documentation).
- `@config.when(...)` for backend switching and capability flags (e.g., `backend="duckdb"`).
- `@dataloader`/`@datasaver` (or Builder materializers) as the canonical I/O boundary in DAGs.
- Lifecycle hooks for: run/node telemetry, contract enforcement, and optional profiling artifact
  capture (storage persists; Hamilton emits events).
- Dynamic DAG patterns (`Parallelizable`/`Collect`) for per-snapshot/per-repo fanout work where it
  materially simplifies orchestration (future-facing, but design should not block it).

### DuckDB capabilities to treat as storage primitives

- Multiple connections (per-thread readers) + single-writer discipline for safe parallel execution.
- `ATTACH` for multi-database workflows (history, multi-project, or layered “warehouse + scratch”).
- `EXPORT DATABASE` / `IMPORT DATABASE` for portable snapshot shipping or reproducible artifacts.
- Extension + secret management as part of session lifecycle (httpfs + cloud secrets; fts; parquet).
- Arrow-native result extraction (`fetch_arrow_table`, Arrow/Polars paths) for serving performance.
- Profiling (`PRAGMA enable_profiling`, `profiling_output`) gated by opt-in config for artifacts.

### Ibis + SQLGlot constraints

- Ibis expressions are the only cross-layer query object.
- SQLGlot is the only component allowed to generate non-trivial SQL/DDL outside of Ibis compilation.

---

## Implementation Roadmap (Concrete PR Sequence)

This maps directly to the remaining semantic-focused PRs and the design above.

1. **PR‑84: Semantic metadata as Hamilton tags**
   - tag constants + update semantic decorator to apply `@tag(...)`
2. **PR‑85: Compile semantic registry from Hamilton discovery**
   - Driver module scan + tag filtering + schema manifest column resolution
3. **PR‑86: Remove bespoke view registry**
   - discover view builders via Hamilton (or convert views to Hamilton nodes with explicit deps)
   - delete `register_view` plumbing once migrated
4. **PR‑87: Enforce serving allowed-columns against schema manifest**
   - strict/warn/off modes
5. **PR‑88: Arrow/Polars execution path in serving**
   - keep pandas fallback temporarily behind a flag
6. **PR‑90: FTS search primitive**
   - storage-managed extension + index creation + serving/search API
7. **PR‑96: Safe Hamilton parallel execution**
   - compute parallel + write lock + per-thread readers + deterministic manifests

Cross-cutting (can land anytime, but easiest before PR‑96):
- consolidate all remaining “snapshot exists/count” logic into `storage/queries/safe.py`
- consolidate all remaining build-owned schema/contract imports behind `storage/build_bridge.py`
- remove ad-hoc schema DDL string interpolation; route through policy backend/SQLGlot

---

## Legacy Deletions (Target End State)

When the roadmap completes, the system should not need:
- any plugin subsystem (already removed per context)
- bespoke semantic registries not derived from Hamilton tags
- manual allowlists not derived from the schema manifest
- view registries requiring explicit registration edits
- pandas-first serving execution path

---

## Open Questions / Decisions Needed

1. **Where do semantic view node functions live?**
   - Keep in `src/codeintel/storage/views/...` as Hamilton modules (works well with tag discovery), or
   - move to `src/codeintel/build/hamilton/native/...` to keep “graph content” centralized.

2. **How explicit should view dependencies be?**
   - Best-in-class: view nodes accept `q__...` dependencies directly (explicit lineage).
   - Transitional: keep “gateway.table('schema.table')” lookups but still tag/discover via Hamilton.

3. **Schema provider ownership**
   - Today: build owns schema provider; storage uses `build_bridge`.
   - Target: define a storage-owned provider interface and make build one implementation.

4. **Snapshot scoping behavior**
   - strict: require snapshot for snapshot-scoped datasets in serving/build contexts
   - permissive: treat snapshot as optional and filter only when provided

---

## Appendix: Example End-to-End Flow (Conceptual)

### Build: compute → materialize

1. compute nodes return Ibis expressions
2. materialize node calls warehouse:
   - ensure table exists (schema manifest)
   - write `INSERT…SELECT` or UPSERT
   - record assets/run telemetry
   - return `MaterializationResult`

### Serving: semantic query

1. resolve `SemanticViewSpec` from compiled registry
2. resolve allowed columns from schema manifest (PR‑87)
3. build Ibis expression from request (filters/sorts/limit)
4. compile + execute via DuckDB
5. return Arrow/Polars rows as JSON-safe response (PR‑88)
