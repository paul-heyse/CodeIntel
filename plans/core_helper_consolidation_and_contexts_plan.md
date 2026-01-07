# Core Helper Consolidation + Shared Contexts Plan (Non-Build)

## Goals
- Consolidate duplicated helper logic outside `src/codeintel/build` into canonical core modules.
- Introduce shared context objects to reduce parameter sprawl and enforce consistent behavior.
- Aggressively cut over to the new structure (design phase; breaking changes acceptable).

## Non-Goals
- No changes under `src/codeintel/build`.
- No incremental compatibility shims unless strictly necessary for a staged migration.

## Guiding Principles
1. Core-first: canonical helpers live in `src/codeintel/core`, other packages re-export or wrap.
2. Single source of truth: avoid parallel implementations that drift.
3. Context over parameters: pass a typed context instead of 6-10 loosely related args.
4. Design phase: hard cutovers are fine; remove legacy modules instead of deprecating.

---

## Phase 1: Consolidate duplicate helpers into core
**Status**: completed

### 1.1 Views discovery + inventory
**Problem**: Core and storage duplicate the view discovery inventory stack.

**Canonical module**: `src/codeintel/core/views/*`

**Targets**
- Keep: `src/codeintel/core/views/discovery.py`
- Keep: `src/codeintel/core/views/inventory.py`
- Keep: `src/codeintel/core/views/protocol.py`
- Keep: `src/codeintel/core/views/generated_view_builders.py`
- Replace or delete:
  - `src/codeintel/storage/views/discovery.py`
  - `src/codeintel/storage/views/inventory.py`
  - `src/codeintel/storage/views/protocol.py`
  - `src/codeintel/storage/views/generated_view_builders.py`

**Pattern** (re-export shim, if we keep storage entrypoints):
```python
# src/codeintel/storage/views/inventory.py
from codeintel.core.views.inventory import (
    clear_view_inventory_cache,
    discover_derived_docs_views,
    discover_view_table_keys,
    view_builder_modules,
)

__all__ = [
    "clear_view_inventory_cache",
    "discover_derived_docs_views",
    "discover_view_table_keys",
    "view_builder_modules",
]
```

### 1.2 Dataset paths, manifests, and Parquet metadata
**Problem**: Core and storage duplicate dataset path/manifest helpers.

**Canonical modules**: `src/codeintel/core/datasets/paths.py`, `manifests.py`, `parquet_metadata.py`

**Targets**
- Keep in core:
  - `src/codeintel/core/datasets/paths.py`
  - `src/codeintel/core/datasets/manifests.py`
  - `src/codeintel/core/datasets/parquet_metadata.py`
- Replace or delete:
  - `src/codeintel/storage/datasets/paths.py`
  - `src/codeintel/storage/datasets/manifests.py`
  - `src/codeintel/storage/datasets/parquet_metadata.py`

**Pattern**
```python
# src/codeintel/storage/datasets/paths.py
from codeintel.core.datasets.paths import dataset_snapshot_dir, dataset_table_dir, SnapshotIdError

__all__ = ["dataset_snapshot_dir", "dataset_table_dir", "SnapshotIdError"]
```

### 1.3 Safe query helpers
**Problem**: `core/queries/safe.py` and `storage/queries/safe.py` overlap and diverge.

**Canonical module**: `src/codeintel/core/queries/safe.py`

**Targets**
- Keep: `src/codeintel/core/queries/safe.py`
- Replace or delete: `src/codeintel/storage/queries/safe.py`

**Pattern** (storage wrapper if needed):
```python
# src/codeintel/storage/queries/safe.py
from codeintel.core.queries.safe import *  # re-export canonical API
```

### 1.4 Arrow dataset store
**Problem**: `core/datasets/arrow_store.py` and `storage/datasets/arrow_store.py` diverge.

**Canonical module**: `src/codeintel/core/datasets/arrow_store.py`

**Targets**
- Keep: `src/codeintel/core/datasets/arrow_store.py`
- Replace or delete: `src/codeintel/storage/datasets/arrow_store.py`

**Pattern** (inject storage-specific schema hash or policies):
```python
# In core ArrowStore: accept hash_fn and policy options
@dataclass(frozen=True, slots=True)
class ArrowStoreConfig:
    schema_hash_fn: Callable[[pa.Schema], str] | None = None
    validate_schema: bool = True

# Storage wires schema hash via config
```

### 1.5 Continue re-exports already in place (keep)
- `src/codeintel/storage/sqlglot_tools.py` -> re-export core
- `src/codeintel/storage/duckdb_types.py` -> re-export core
- `src/codeintel/storage/helpers/table_key.py` -> re-export core
- `src/codeintel/storage/query_results.py` -> re-export core
- `src/codeintel/serving/export/formats.py` -> re-export core
- `src/codeintel/storage/validation/columnar.py` -> re-export core validation

---

## Phase 2: Unify runtime bundles
**Status**: completed

**Problem**: Two runtime bundle definitions with overlapping meaning.
- `src/codeintel/runtime/runtime_bundle.py`
- `src/codeintel/core/runtime/bundle.py`

**Plan**
- Promote a single `RuntimeBundle` and `RuntimeSettings` as canonical.
- Rename the Hamilton-heavy runtime bundle to a more explicit name (e.g. `HamiltonRuntimeBundle`).
- Update all call sites to import from the canonical module.

**Targets**
- Canonical: `src/codeintel/core/runtime/bundle.py`
- Move or rename: `src/codeintel/runtime/runtime_bundle.py`
- Update imports in:
  - `src/codeintel/runtime/registry.py`
  - `src/codeintel/runtime/compose.py`
  - `src/codeintel/runtime/registry_service.py`
  - Any serving/CLI/runtime factories that reference the old type

**Pattern**
```python
@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    primitives: RuntimePrimitives
    settings: RuntimeSettings
    # Optional: hamilton-only components in a separate typed bundle
```

---

## Phase 3: Shared context objects
**Status**: pending

### 3.1 RunContext (canonical)
**Problem**: Multiple run identity definitions across CLI and observability.

**Plan**
- Canonicalize on `codeintel.core.execution.context.RunContext`.
- Rename CLI-specific `RunContext` to `CliInvocationContext`.

**Targets**
- Canonical: `src/codeintel/core/execution/context.py`
- Update: `src/codeintel/observability/cli.py`
- Update: `src/codeintel/cli/context.py`

**Pattern**
```python
@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    kind: RunKind
    snapshot: SnapshotRef
    trigger: TriggerKind
```

### 3.2 StorageContext
**Problem**: Storage consumers repeatedly pass duckdb connection, snapshot, dataset root, and schema registry.

**Plan**
- Introduce `StorageContext` under `src/codeintel/core/storage/context.py`.
- Update gateways and repositories to accept a `StorageContext` instead of discrete args.

**Targets**
- New: `src/codeintel/core/storage/context.py`
- Update:
  - `src/codeintel/storage/gateway/*`
  - `src/codeintel/storage/repositories/*`
  - `src/codeintel/storage/warehouse.py`

**Pattern**
```python
@dataclass(frozen=True, slots=True)
class StorageContext:
    snapshot: SnapshotRef
    duckdb: DuckDBContext
    dataset_root: Path
    schema_provider: SchemaProvider
    query_policy: SqlIngressPolicy
```

### 3.3 QueryContext
**Problem**: Query policy and snapshot scoping are spread across helpers.

**Plan**
- Add `QueryContext` in `src/codeintel/core/queries/context.py`.
- Use it in `core/queries/safe.py` and storage repositories.

**Pattern**
```python
@dataclass(frozen=True, slots=True)
class QueryContext:
    snapshot: SnapshotRef
    allowed_tables: frozenset[str] | None
    allowed_schemas: frozenset[str] | None
    allow_cross_db: bool = False
```

### 3.4 ServingContext
**Problem**: Serving runtime/state are split between `serving/runtime.py` and `serving/http/state.py`.

**Plan**
- Merge into `src/codeintel/serving/context.py` and use across HTTP and MCP.

**Targets**
- New: `src/codeintel/serving/context.py`
- Update:
  - `src/codeintel/serving/http/state.py`
  - `src/codeintel/serving/runtime.py`
  - `src/codeintel/serving/mcp/runtime.py`

**Pattern**
```python
@dataclass(frozen=True, slots=True)
class ServingContext:
    settings: ServingSettings
    db_manager: ServingDBManager
    kernel: SemanticQueryKernel
    ops: ServingOperations
```

### 3.5 IngestionContext
**Problem**: Ingestion steps pass a long list of parameters that should be bundled.

**Plan**
- Add `src/codeintel/ingestion/context.py` with a minimal context object.

**Targets**
- New: `src/codeintel/ingestion/context.py`
- Update:
  - `src/codeintel/ingestion/compute/*`
  - `src/codeintel/ingestion/engine/*`
  - `src/codeintel/ingestion/adapters/*`

**Pattern**
```python
@dataclass(frozen=True, slots=True)
class IngestionContext:
    snapshot: SnapshotRef
    repo_root: Path
    scan_profile: ScanProfile
    tools: ToolBinaries
    settings: RuntimeSettings
```

---

## Phase 4: Migration and cleanup (hard cutover)
**Status**: partially completed (storage duplicates removed as part of Phase 1)

**Actions**
- Remove duplicated storage modules (replace with re-exports if needed).
- Update all imports to canonical core modules.
- Delete any now-dead modules to prevent drift.

**Targets**
- Delete or reduce to re-export:
  - `src/codeintel/storage/views/*` (inventory, discovery, protocol, generated)
  - `src/codeintel/storage/datasets/paths.py`
  - `src/codeintel/storage/datasets/manifests.py`
  - `src/codeintel/storage/datasets/parquet_metadata.py`
  - `src/codeintel/storage/queries/safe.py`
  - `src/codeintel/storage/datasets/arrow_store.py`

---

## Phase 5: Validation (design-phase minimal checks)
**Status**: completed

- Import-only smoke check: ensure no broken imports after deletions.
- Run the typing/lint suite after all migrations:
  - `uv run ruff check --fix`
  - `uv run pyright --warnings --pythonversion=3.13`
  - `uv run pyrefly check`

---

## Expected Outcomes
- Single source of truth for dataset paths/manifests/parquet metadata.
- Shared query and view discovery behavior across storage/serving/CLI.
- Fewer context parameters and more consistent, testable entrypoints.
- No redundant helper modules outside `src/codeintel/core`.

---

## Phased Checklist (Exact per-file steps)

### Phase 1 Checklist: Core helper consolidation
- Delete `src/codeintel/storage/views/discovery.py` (imports now point to core).
- Delete `src/codeintel/storage/views/inventory.py` (imports now point to core).
- Delete `src/codeintel/storage/views/protocol.py` (imports now point to core).
- Delete `src/codeintel/storage/views/generated_view_builders.py` (imports now point to core).
- Delete `src/codeintel/storage/datasets/paths.py` (imports now point to core).
- Delete `src/codeintel/storage/datasets/manifests.py` (imports now point to core).
- Delete `src/codeintel/storage/datasets/parquet_metadata.py` (imports now point to core).
- Delete `src/codeintel/storage/queries/safe.py` (imports now point to core).
- Delete `src/codeintel/storage/datasets/arrow_store.py` (imports now point to core).
- Update import sites to use core modules (including tests):
  - `src/codeintel/storage/metadata/*`
  - `src/codeintel/storage/repositories/*`
  - `src/codeintel/storage/views/*`
  - `tests/**`

### Phase 2 Checklist: Runtime bundle unification
- Rename `src/codeintel/runtime/runtime_bundle.py` types to `HamiltonRuntimeBundle`.
- Update `src/codeintel/runtime/registry.py` to use `HamiltonRuntimeBundle`.
- Update `src/codeintel/runtime/compose.py` to return `HamiltonRuntimeBundle`.
- Update `src/codeintel/runtime/registry_service.py` to use `HamiltonRuntimeBundle`.
- Update imports across runtime/build/cli/tests to use `HamiltonRuntimeBundle`.

### Phase 3 Checklist: Shared contexts
- Add `src/codeintel/core/storage/context.py` (StorageContext) and wire into:
  - `src/codeintel/storage/gateway/*`
  - `src/codeintel/storage/warehouse.py`
  - `src/codeintel/storage/repositories/*`
- Add `src/codeintel/core/queries/context.py` (QueryContext) and update:
  - `src/codeintel/core/queries/safe.py`
  - `src/codeintel/storage/queries/safe.py` (re-export or wrapper)
- Add `src/codeintel/serving/context.py` and replace:
  - `src/codeintel/serving/runtime.py`
  - `src/codeintel/serving/http/state.py`
  - `src/codeintel/serving/mcp/runtime.py`
- Add `src/codeintel/ingestion/context.py` and update:
  - `src/codeintel/ingestion/compute/*`
  - `src/codeintel/ingestion/engine/*`
  - `src/codeintel/ingestion/adapters/*`
- Rename CLI-specific `RunContext` in `src/codeintel/observability/cli.py` to `CliInvocationContext` and update references.
- Update `src/codeintel/cli/context.py` to use `core.execution.context.RunContext` for run identity.

### Phase 4 Checklist: Hard cutover cleanup
- Remove legacy storage modules after import updates:
  - `src/codeintel/storage/views/discovery.py`
  - `src/codeintel/storage/views/inventory.py`
  - `src/codeintel/storage/views/protocol.py`
  - `src/codeintel/storage/views/generated_view_builders.py`
  - `src/codeintel/storage/datasets/paths.py`
  - `src/codeintel/storage/datasets/manifests.py`
  - `src/codeintel/storage/datasets/parquet_metadata.py`
  - `src/codeintel/storage/queries/safe.py`
  - `src/codeintel/storage/datasets/arrow_store.py`
- Ensure no remaining imports from removed modules (repo-wide search).

### Phase 5 Checklist: Validation
- Run the repo-level checks:
  - `uv run ruff check --fix`
  - `uv run pyright --warnings --pythonversion=3.13`
  - `uv run pyrefly check`
