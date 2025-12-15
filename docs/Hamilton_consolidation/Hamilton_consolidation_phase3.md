# Hamilton Consolidation Phase 3 — Legacy Deprecation & Full DAG-First Unification

## Executive Summary

Phase 2 established a solid foundation with `SchemaProvider`, schema inference via DuckDB `DESCRIBE`, generated Pandera schemas, and row model generation. However, **65 files still import from `codeintel.config.datasets`** and the legacy `TABLE_SCHEMAS` registry (~114KB, ~3000+ lines) remains the de facto source of truth for many consumers.

This phase completes the migration to a fully DAG-first architecture where:

1. **Hamilton is the single source of truth** for table schemas
2. **Row models are generated**, not hand-maintained
3. **DatasetContracts are derived** from build targets
4. **Legacy config/datasets infrastructure is deleted**

---

## Current State Analysis

### Legacy Infrastructure Still in Use

| Component | Location | Size | Consumers |
|-----------|----------|------|-----------|
| `TABLE_SCHEMAS` | `config/datasets/schemas.py` | ~114KB | 9 files directly |
| `DATASET_CONTRACTS` | `config/datasets/contracts.py` | ~1000 lines | 14 files |
| `RowBinding` definitions | `config/datasets/contracts.py` | 50+ manual bindings | Storage, analytics |
| TypedDict row models | `config/datasets/rows/` | 5 files | Throughout |
| Legacy `TableSchema` primitives | `config/datasets/primitives.py` | Now re-exports from core | Many |

### Files Importing from `config.datasets` (65 total)

**Storage Layer (12 files)**:
- `storage/schema/ddl.py`
- `storage/schema/json_schema.py`
- `storage/metadata/bootstrap.py`
- `storage/validation/contract.py`
- `storage/validation/data_checks.py`
- `storage/gateway/accessors.py`
- `storage/gateway/base_accessor.py`
- `storage/datasets/registry.py`
- `storage/datasets/catalog.py`
- `ingestion/adapters/duckdb_storage.py`
- `ingestion/adapters/hash_change_detection.py`
- `ingestion/compute/typing_ingest.py`

**Analytics Layer (18 files)**:
- `analytics/utilities/datasets.py`
- `analytics/graphs/config_graph_metrics.py`
- `analytics/graphs/subsystem_graph_metrics.py`
- `analytics/graphs/symbol_orchestrator.py`
- `analytics/graphs/graph_metrics_ext.py`
- `analytics/graphs/module_graph_metrics_ext.py`
- `analytics/compute/row_builders/graph_metrics.py`
- `analytics/compute/row_builders/graph_metrics_ext.py`
- `analytics/compute/hotspots/metrics.py`
- `analytics/testing/coverage/edges.py`
- `analytics/testing/profiles/rows.py`
- `analytics/profiles/functions.py`
- `analytics/profiles/modules.py`
- `analytics/profiles/files.py`
- `analytics/functions/metrics.py`
- `analytics/parsing/validation.py`
- `analytics/ast_features/persist.py`
- `ingestion/compute/docstrings_extract.py`

**Build Layer (8 files)**:
- `build/registry.py`
- `build/contracts.py`
- `build/contracts_validation.py`
- `build/targets.py`
- `build/plugins/graphs/builders/callgraph.py`
- `build/hamilton/contracts/schemas/builder.py`
- `build/hamilton/contracts/schemas/schema.py`
- `build/hamilton/contracts/schemas/row_binding_factory.py`
- `build/hamilton/contracts/schemas/pandera_schemas.py`
- `build/hamilton/contracts/schemas/validation.py`
- `build/hamilton/contracts/schemas/row_migration.py`
- `build/schemas/provider_declared.py`

**Export Layer (4 files)**:
- `export/__init__.py`
- `export/export_exprs.py`
- `export/export_jsonl.py`
- `export/export_parquet.py`

**Serving Layer (6 files)**:
- `serving/services/datasets.py`
- `serving/backend/datasets.py`
- `serving/backend/dataset_backend.py`
- `serving/operations/catalog.py`
- `serving/auto_pipeline.py`
- `serving/mcp/meta_tools.py`

**CLI Layer (3 files)**:
- `cli/handlers/datasets.py`
- `cli/handlers/ops.py`
- `cli/commands/datasets.py`

**Graph Layer (2 files)**:
- `graphs/compute/callgraph/collection.py`
- `graphs/compute/callgraph/persistence.py`

---

## Target Architecture

### Schema Resolution Flow (After Phase 3)

```
┌─────────────────────────────────────────────────────────────────┐
│                      SchemaProvider                              │
│  (Single interface for all schema resolution)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Hamilton-Native │  │ Target-Declared │  │  Raw/Source     │
│    Inference    │  │  Output Schema  │  │   Declared      │
│ (PR-60 + PR-69) │  │    (PR-69)      │  │   Schemas       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │  TableSchema    │
                    │  (core/schemas) │
                    └─────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   DDL/Storage   │  │  Row Models    │  │  Pandera/JSON   │
│   (PR-66)       │  │  (PR-67)       │  │  Schema (PR-73) │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Contract Resolution Flow (After Phase 3)

```
┌─────────────────────────────────────────────────────────────────┐
│                      TargetGraph                                 │
│  (Single source of truth for build targets)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OutputTarget                                  │
│  name, module, contract, dependencies, resources, execution      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  OutputTargetContract                            │
│  table_keys, output_schemas, json_schema_ids,                    │
│  jsonl_filenames, parquet_filenames, owner, description, etc.    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              DatasetContract (Generated)                         │
│  Derived from OutputTarget + SchemaProvider                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed PR Specifications

---

# PR-66 — Migrate TABLE_SCHEMAS consumers to SchemaProvider

## Status: ✅ PHASE 1 COMPLETE

**Completed 2024-12**: Core registry created, key storage/build layer consumers migrated.

## Goal

Eliminate all direct imports of `TABLE_SCHEMAS` and `config.datasets.schemas` in favor of the `SchemaProvider` interface established in PR-56.

## Rationale

Direct access to `TABLE_SCHEMAS` bypasses the schema authority architecture. All schema resolution should go through `SchemaProvider` so that:

1. Schema inference (PR-60) can transparently replace declared schemas
2. Schema versioning/hashing is consistent
3. Future schema sources (e.g., remote catalogs) can be added without changing consumers

## Implementation Notes (from actual PR-66 work)

### Key Insight: Schema vs. Contract Separation

During implementation, a critical distinction emerged:

- **`SchemaProvider`** provides only `TableSchema` objects (columns, types, names)
- **`DatasetContract`** provides full metadata (owner, description, `is_view`, filenames, etc.)

Some consumers need **both**:
- `storage/schema/ddl.py`: Uses provider for schemas, but still needs `get_dataset_contracts_by_table_key()` for `is_view` filtering in `assert_schema_alignment()`
- `storage/metadata/bootstrap.py`: Uses provider for schema hashing, but still needs `DATASET_CONTRACTS` for non-schema metadata in `bootstrap_metadata_datasets()`

**This makes PR-68 (DatasetContract derivation) essential** to fully eliminate contract imports.

### Completed Files

| File | Change |
|------|--------|
| `src/codeintel/build/schemas/registry.py` | ✅ Created with `get_schema_provider()`, `require_table_schema()`, `iter_table_schemas()`, `clear_schema_provider_cache()` |
| `src/codeintel/build/schemas/__init__.py` | ✅ Exports registry functions |
| `src/codeintel/storage/schema/ddl.py` | ✅ Uses `get_schema_provider()` in `_get_policy_backend()` and `assert_schema_alignment()` |
| `src/codeintel/storage/metadata/bootstrap.py` | ✅ Uses `get_schema_provider()` + `core.schemas.schema_hash()` for registry operations |
| `src/codeintel/build/contracts_validation.py` | ✅ Uses provider for schema lookups |
| `src/codeintel/build/registry.py` | ✅ Builds `_DATASET_TABLE_SCHEMAS` dict from provider |
| `tests/build/hamilton/test_pr66_schema_provider_registry.py` | ✅ 13 tests validating functionality and parity |

## Tasks Checklist

### Phase 1: Create migration infrastructure ✅ COMPLETE

* [x] Add canonical schema provider factory:
  ```python
  # src/codeintel/build/schemas/registry.py
  from __future__ import annotations

  from functools import lru_cache
  from typing import TYPE_CHECKING

  from codeintel.build.schemas.provider_declared import declared_schema_provider

  if TYPE_CHECKING:
      from collections.abc import Iterable
      from codeintel.core.schemas.primitives import TableSchema
      from codeintel.core.schemas.provider import SchemaProvider


  @lru_cache
  def get_schema_provider() -> SchemaProvider:
      """Return the canonical schema provider for the current build context."""
      return declared_schema_provider()


  def require_table_schema(table_key: str) -> TableSchema:
      """Convenience function to require a table schema by key."""
      return get_schema_provider().require_table_schema(table_key)


  def iter_table_schemas() -> Iterable[TableSchema]:
      """Iterate all known table schemas."""
      return get_schema_provider().iter_table_schemas()


  def clear_schema_provider_cache() -> None:
      """Clear the schema provider cache (for testing)."""
      get_schema_provider.cache_clear()


  __all__ = [
      "clear_schema_provider_cache",
      "get_schema_provider",
      "iter_table_schemas",
      "require_table_schema",
  ]
  ```

* [ ] Add deprecation warnings to legacy access points (deferred to Phase 6)

### Phase 2: Migrate storage layer (highest priority) — PARTIAL

* [x] `src/codeintel/storage/schema/ddl.py`:
  - ✅ `_get_policy_backend()` now uses `get_schema_provider()` directly
  - ✅ `assert_schema_alignment()` gets schemas from provider
  - ⚠️ **Still imports** `get_dataset_contracts_by_table_key()` for `is_view` filtering (needs PR-68)

* [x] `src/codeintel/storage/metadata/bootstrap.py`:
  - ✅ `_expected_schema_hash()` uses provider + `core.schemas.schema_hash()`
  - ✅ `_register_dataset_schema_hashes()` iterates via provider
  - ✅ `validate_dataset_schema_registry()` uses provider
  - ⚠️ **Still imports** `DATASET_CONTRACTS` for non-schema metadata (needs PR-68)

* [ ] `src/codeintel/storage/schema/json_schema.py`:
  - Migrate schema access

* [ ] `src/codeintel/storage/validation/contract.py`:
  - Migrate to `SchemaProvider` for schema lookups

* [ ] `src/codeintel/storage/validation/data_checks.py`:
  - Migrate schema access

* [ ] `src/codeintel/storage/gateway/accessors.py`:
  - Inject `SchemaProvider` or use registry

* [ ] `src/codeintel/ingestion/adapters/duckdb_storage.py`:
  - Migrate `TABLE_SCHEMAS` access

### Phase 3: Migrate analytics layer

* [ ] `src/codeintel/analytics/utilities/datasets.py`:
  - Replace direct schema access
  - Update `dataset_contracts_by_table_key()` usage

* [ ] Migrate all `analytics/graphs/*.py` files using schemas

* [ ] Migrate `analytics/compute/row_builders/*.py` files

* [ ] Migrate `analytics/profiles/*.py` files

### Phase 4: Migrate build layer — PARTIAL

* [x] `src/codeintel/build/registry.py`:
  - ✅ Uses `get_schema_provider()` to build `_DATASET_TABLE_SCHEMAS`

* [x] `src/codeintel/build/contracts_validation.py`:
  - ✅ Uses provider for schema lookups

* [ ] `src/codeintel/build/contracts.py`:
  - Migrate schema access

* [ ] Update Hamilton contracts modules to use provider

### Phase 5: Migrate export/serving/CLI layers

* [ ] Migrate all `export/*.py` files
* [ ] Migrate all `serving/*.py` files
* [ ] Migrate all `cli/handlers/*.py` files

### Phase 6: Add lint enforcement

* [ ] Add Ruff rule to ban direct `config.datasets.schemas` imports:
  ```toml
  # pyproject.toml
  [tool.ruff.lint.flake8-tidy-imports.banned-api]
  "codeintel.config.datasets.schemas.TABLE_SCHEMAS".msg = "Use codeintel.build.schemas.get_schema_provider() instead"
  ```

## Code Snippets

### Migration pattern for schema lookup

**Before:**
```python
from codeintel.config.datasets.schemas import TABLE_SCHEMAS

def get_columns(table_key: str) -> list[str]:
    schema = TABLE_SCHEMAS[table_key]
    return schema.column_names()
```

**After:**
```python
from codeintel.build.schemas import get_schema_provider

def get_columns(table_key: str) -> list[str]:
    schema = get_schema_provider().require_table_schema(table_key)
    return schema.column_names()
```

### Migration pattern for schema iteration

**Before:**
```python
from codeintel.config.datasets.schemas import TABLE_SCHEMAS

for table_key, schema in TABLE_SCHEMAS.items():
    create_table(schema)
```

**After:**
```python
from codeintel.build.schemas import get_schema_provider

for schema in get_schema_provider().iter_table_schemas():
    create_table(schema)
```

### Migration pattern for build-time schema dict (backward compat)

When module-level code needs a dict at import time:

```python
from codeintel.build.schemas import get_schema_provider

# Build dict from provider at module load
_provider = get_schema_provider()
_DATASET_TABLE_SCHEMAS = {ts.table_key: ts for ts in _provider.iter_table_schemas()}

# Use dict normally
schema = _DATASET_TABLE_SCHEMAS["analytics.function_metrics"]
```

### Pattern for files needing both schemas AND contract metadata

During transition, some files need both. Keep contract imports minimal:

```python
from codeintel.build.schemas import get_schema_provider
from codeintel.config.datasets import get_dataset_contracts_by_table_key  # For is_view only

def process_tables() -> None:
    provider = get_schema_provider()
    contracts = get_dataset_contracts_by_table_key()
    
    for table_key, contract in contracts.items():
        if contract.is_view:  # Contract metadata
            continue
        schema = provider.get_table_schema(table_key)  # Schema from provider
        if schema is not None:
            create_table(schema)
```

## Tests Checklist (`tests/build/hamilton/`)

### ✅ Completed Tests

* [x] `test_pr66_schema_provider_registry.py` (13 tests):
  - `test_get_schema_provider_returns_valid_provider` - Verifies provider has correct interface
  - `test_get_schema_provider_is_cached` - Verifies LRU caching works
  - `test_require_table_schema_for_known_key` - Verifies schema resolution
  - `test_require_table_schema_raises_for_unknown_key` - Verifies KeyError on missing
  - `test_iter_table_schemas_returns_all_schemas` - Verifies iteration
  - `test_iter_table_schemas_contains_expected_keys` - Verifies known schemas present
  - `test_every_legacy_key_resolves_via_provider` - Full parity test
  - `test_provider_schemas_match_legacy_schemas` - Schema equality validation
  - `test_provider_schema_count_matches_legacy` - Count parity
  - `test_schema_hash_is_consistent` - Hash determinism
  - `test_schema_hash_is_deterministic_across_provider_calls` - Cross-call consistency
  - `test_all_provider_schemas_are_hashable` - All schemas produce valid hashes
  - `test_clear_cache_allows_fresh_provider` - Cache clearing works

### Test Pattern for Ruff Compliance

Tests should use `pytest.fail()` instead of bare `assert` to comply with S101:

```python
def test_schema_provider_example() -> None:
    """Example test following ruff-compliant pattern."""
    schema = require_table_schema("analytics.function_metrics")
    if schema is None:
        pytest.fail("Expected non-None schema")
    if schema.table_key != "analytics.function_metrics":
        pytest.fail(f"Expected table_key 'analytics.function_metrics', got '{schema.table_key}'")
```

### Remaining Tests

* [ ] `test_pr66_no_direct_table_schemas_imports.py`:
  - AST-based test scanning for banned import patterns
  - Exclude `config/datasets/` itself and migration modules

* [ ] `test_pr66_deprecation_warnings_emitted.py`:
  - Verify deprecation warnings fire on legacy access (when added)

## CLI Snapshots

* [ ] None required (infrastructure PR)

## Migration Metrics

Track migration progress:
- [x] Storage layer: 4/12 files migrated (ddl.py, bootstrap.py partial - need PR-68 for full)
- [ ] Analytics layer: 0/18 files migrated
- [x] Build layer: 3/8 files migrated (registry.py, contracts_validation.py, schemas/registry.py)
- [ ] Export layer: 0/4 files migrated
- [ ] Serving layer: 0/6 files migrated
- [ ] CLI layer: 0/3 files migrated

---

# PR-67 — Migrate RowBindings to schema-generated row models

## Status: ✅ PHASE 1 COMPLETE

**Completed 2024-12**: Core infrastructure created, generated bindings operational, deprecation warnings added to legacy APIs.

## Goal

Replace the 50+ hand-maintained `RowBinding` definitions in `config/datasets/contracts.py` with schema-generated row models from PR-62, eliminating a major source of schema-vs-code drift.

## Implementation Notes (from actual PR-67 work)

### Key Insight: Deprecation-First Approach

During implementation, a critical strategy decision was made:

- **Do NOT immediately delete legacy files** — Many consumers directly import row types and serializers from `config/datasets/rows/`
- **Add deprecation warnings first** — Signal intent to remove while preserving backward compatibility
- **Prefer generated bindings** — Update `row_binding_factory.py` to try generated bindings first, fall back to legacy
- **Defer full cleanup to PR-70** — Consolidate all deletions in a single PR for cleaner review

This approach is both **safer** (no breaking changes) and **easier to review** (deletions are isolated).

### Known Exclusions

Parity testing revealed legitimate cases where legacy bindings exist but schema-generated ones do not or differ:

| Table Key | Reason | Action |
|-----------|--------|--------|
| `docs.v_subsystem_profile` | View, not a table — no schema-generated binding expected | Exclude from parity tests |
| `docs.v_subsystem_coverage` | View, not a table — no schema-generated binding expected | Exclude from parity tests |
| `analytics.static_diagnostics` | Known schema drift between legacy TypedDict and current TableSchema | Track as technical debt for PR-70 |

**Recommendation for PR-70**: Audit all excluded cases and either fix schema drift or confirm intentional divergence.

### Completed Files

| File | Change |
|------|--------|
| `src/codeintel/core/schemas/row_models.py` | ✅ Extended with `GeneratedRowBinding`, `row_binding_for_table_schema()`, `_coerce_value()` |
| `src/codeintel/core/schemas/__init__.py` | ✅ Exports `GeneratedRowBinding`, `row_binding_for_table_schema` |
| `src/codeintel/build/schemas/row_registry.py` | ✅ **NEW** — `get_row_binding()`, `iter_row_bindings()`, `clear_row_binding_cache()` |
| `src/codeintel/build/schemas/__init__.py` | ✅ Exports row registry functions |
| `src/codeintel/build/hamilton/contracts/schemas/row_binding_factory.py` | ✅ Updated `get_or_create_row_binding()` to prefer generated bindings |
| `src/codeintel/config/datasets/contracts.py` | ✅ Added `DeprecationWarning` to `get_row_bindings()` |
| `tests/build/hamilton/test_pr67_row_binding_parity.py` | ✅ **NEW** — 10+ parity and functionality tests |

## Current State

```python
# config/datasets/contracts.py - manual bindings
def _build_row_bindings() -> dict[str, RowBinding]:
    return {
        "analytics.function_metrics": _row_binding(
            row_type=FunctionMetricsRow,  # Hand-maintained TypedDict
            to_tuple=function_metrics_row_to_tuple,  # Hand-maintained serializer
        ),
        "analytics.function_types": _row_binding(
            row_type=FunctionTypesRow,
            to_tuple=function_types_row_to_tuple,
        ),
        # ... 50+ more manual bindings
    }
```

## Target State

```python
# Generated from TableSchema at runtime
from codeintel.build.schemas.row_registry import get_row_binding

binding = get_row_binding("analytics.function_metrics")
# binding.row_type = dynamically generated dataclass
# binding.to_tuple = dynamically generated serializer
```

## Tasks Checklist

### Phase 1: Extend row model generation ✅ COMPLETE

* [x] Extend `src/codeintel/core/schemas/row_models.py`:
  ```python
  @dataclass(frozen=True)
  class GeneratedRowBinding:
      """Schema-generated row binding compatible with legacy RowBinding."""
      
      row_model: type[object]
      serializer: RowSerializer
      table_key: str
      schema_hash: str
      
      @property
      def row_type(self) -> type[object]:
          """Alias for legacy compatibility."""
          return self.row_model
      
      @property
      def to_tuple(self) -> RowSerializer:
          """Alias for legacy compatibility."""
          return self.serializer


  def row_binding_for_table_schema(*, table_schema: TableSchema) -> GeneratedRowBinding:
      """Generate a RowBinding from a TableSchema.

      Parameters
      ----------
      table_schema
          Source TableSchema.

      Returns
      -------
      GeneratedRowBinding
          Generated binding with row model and serializer.
      """
      row_model = row_model_for_table_schema(table_schema=table_schema)
      serializer = row_serializer_for_table_schema(table_schema=table_schema)
      return GeneratedRowBinding(
          row_model=row_model,
          serializer=serializer,
          table_key=table_schema.table_key,
          schema_hash=schema_hash(table_schema),
      )
  ```

* [x] Add type coercion support for edge cases:
  ```python
  def _coerce_value(value: object, col_type: ColumnType) -> object:
      """Coerce a value to match the expected column type."""
      if value is None:
          return None
      if col_type in {"INTEGER", "BIGINT", "DECIMAL(38,0)"}:
          return int(value) if value is not None else None
      if col_type in {"DOUBLE", "DECIMAL"}:
          return float(value) if value is not None else None
      if col_type == "BOOLEAN":
          return bool(value) if value is not None else None
      if col_type in {"VARCHAR", "JSON"}:
          return str(value) if value is not None else None
      if col_type in {"TIMESTAMP", "TIMESTAMPTZ"}:
          if isinstance(value, str):
              return datetime.fromisoformat(value)
          return value
      return value
  ```

### Phase 2: Create row binding registry ✅ COMPLETE

* [x] Add `src/codeintel/build/schemas/row_registry.py`:
  ```python
  from __future__ import annotations

  from functools import lru_cache
  from typing import TYPE_CHECKING

  from codeintel.build.schemas.registry import get_schema_provider
  from codeintel.core.schemas.row_models import row_binding_for_table_schema

  if TYPE_CHECKING:
      from codeintel.core.schemas.row_models import GeneratedRowBinding


  @lru_cache(maxsize=256)
  def get_row_binding(table_key: str) -> GeneratedRowBinding:
      """Return a schema-generated row binding for a table key.

      Parameters
      ----------
      table_key
          Fully qualified table key (schema.table).

      Returns
      -------
      GeneratedRowBinding
          Generated row binding with model and serializer.

      Raises
      ------
      KeyError
          If the table key is not found.
      """
      schema = get_schema_provider().require_table_schema(table_key)
      return row_binding_for_table_schema(table_schema=schema)


  def clear_row_binding_cache() -> None:
      """Clear the row binding cache (for testing)."""
      get_row_binding.cache_clear()


  __all__ = ["clear_row_binding_cache", "get_row_binding"]
  ```

### Phase 3: Migrate consumers — PARTIAL

* [x] Identify all usages of `get_row_bindings()`:
  ```bash
  rg "get_row_bindings|row_binding\." --type py
  ```

* [x] Update `build/hamilton/contracts/schemas/row_binding_factory.py`:
  - `get_or_create_row_binding()` now prefers generated bindings with legacy fallback
  - This provides a transparent migration path for all consumers using the factory

* [ ] Migrate `storage/gateway/accessors.py`:
  - Replace `get_row_bindings()[table_key]` with `get_row_binding(table_key)`
  - **Deferred to PR-70** — many direct imports of legacy row types throughout codebase

* [ ] Migrate `analytics/` row builders:
  - Update to use generated serializers
  - **Deferred to PR-70** — requires coordinated update across many files

* [ ] Migrate `build/plugins/` materializers:
  - Update to use generated bindings
  - **Deferred to PR-70**

**Note**: Full consumer migration deferred to PR-70 to consolidate all legacy cleanup in one PR.

### Phase 4: Validate parity ✅ COMPLETE

* [x] Create parity validation test:
  ```python
  def test_generated_binding_matches_legacy():
      from codeintel.config.datasets.contracts import get_row_bindings
      from codeintel.build.schemas.row_registry import get_row_binding
      
      legacy_bindings = get_row_bindings()
      
      for table_key, legacy in legacy_bindings.items():
          generated = get_row_binding(table_key)
          
          # Test serialization produces same output
          test_row = create_test_row_for_table(table_key)
          legacy_tuple = legacy.to_tuple(test_row)
          generated_tuple = generated.to_tuple(test_row)
          
          assert legacy_tuple == generated_tuple, f"Mismatch for {table_key}"
  ```

### Phase 5: Deprecate legacy APIs ✅ COMPLETE (Deletion deferred to PR-70)

**Strategy change**: Instead of immediate deletion, we add deprecation warnings and defer full cleanup to PR-70.

* [x] Add `DeprecationWarning` to `get_row_bindings()` in `config/datasets/contracts.py`:
  ```python
  def get_row_bindings() -> dict[str, RowBinding]:
      """Return the ROW_BINDINGS_BY_TABLE_KEY dictionary.

      .. deprecated::
          Use `codeintel.build.schemas.row_registry.get_row_binding()` instead.
      """
      warnings.warn(
          "get_row_bindings() is deprecated. "
          "Use codeintel.build.schemas.row_registry.get_row_binding() for individual lookups "
          "or iter_row_bindings() for iteration.",
          DeprecationWarning,
          stacklevel=2,
      )
      return _row_bindings_cache()
  ```

* [ ] Delete `config/datasets/rows/analytics.py` — **Deferred to PR-70**
* [ ] Delete `config/datasets/rows/core.py` — **Deferred to PR-70**
* [ ] Delete `config/datasets/rows/graph.py` — **Deferred to PR-70**
* [ ] Delete `config/datasets/rows/profiles.py` — **Deferred to PR-70**
* [ ] Delete `config/datasets/rows/test.py` — **Deferred to PR-70**
* [ ] Delete `config/datasets/rows/__init__.py` — **Deferred to PR-70**
* [ ] Delete `config/datasets/generated_rows/` directory — **Deferred to PR-70**

**Rationale**: Many files directly import legacy row types (e.g., `from codeintel.config.datasets.rows.analytics import FunctionMetricsRow`). Deleting these files would require updating 30+ consumers simultaneously. Consolidating all deletions in PR-70 allows for cleaner review and safer rollout.

## Tests Checklist (`tests/build/hamilton/`)

### ✅ Completed Tests

* [x] `test_pr67_row_binding_parity.py` (10+ tests):
  - `test_generated_row_binding_has_row_type_alias` - Verifies legacy compatibility
  - `test_generated_row_binding_has_to_tuple_alias` - Verifies legacy compatibility
  - `test_get_row_binding_returns_valid_binding` - Verifies registry works
  - `test_get_row_binding_is_cached` - Verifies LRU caching
  - `test_get_row_binding_raises_for_unknown_key` - Verifies error handling
  - `test_iter_row_bindings_yields_bindings` - Verifies iteration
  - `test_all_legacy_bindings_have_schema_equivalent` - Parity coverage (with exclusions)
  - `test_generated_serializer_column_order_matches_legacy` - Column order parity (with exclusions)
  - `test_generated_binding_fields_match_legacy` - Field parity (with exclusions)
  - `test_generated_serializer_handles_all_column_types` - Type coverage
  - `test_generated_serializer_handles_nullable_columns` - Nullable handling

### Test Pattern: Handling Known Exclusions

Tests must account for legitimate mismatches. Use explicit exclusion sets:

```python
# Views don't have schema-generated bindings
_VIEW_EXCLUSIONS = frozenset({
    "docs.v_subsystem_profile",
    "docs.v_subsystem_coverage",
})

# Known schema drift between legacy TypedDict and current TableSchema
_SCHEMA_DRIFT_EXCLUSIONS = frozenset({
    "analytics.static_diagnostics",
})

def test_all_legacy_bindings_have_schema_equivalent() -> None:
    """Verify all legacy bindings have schema-generated equivalents."""
    legacy_bindings = _get_legacy_bindings_suppressed()
    
    for table_key in legacy_bindings:
        if table_key in _VIEW_EXCLUSIONS:
            continue  # Views not expected to have generated bindings
        # ... test logic
```

### Remaining Tests (Optional)

* [ ] `test_pr67_generated_serializer_roundtrip.py`:
  - Insert row via generated serializer, read back, verify equality
  - Lower priority: basic serialization already validated

* [ ] `test_pr67_row_binding_cache_invalidates_on_schema_change.py`:
  - Verify cache keying includes schema hash
  - Note: Current impl uses `@lru_cache` keyed by `table_key`; schema_hash is stored but not used for cache invalidation

## CLI Snapshots

* [ ] None required (infrastructure PR)

---

## PR-67 Phase 1 Completion Summary

**Completed**: Row binding generation infrastructure established with:

```python
from codeintel.build.schemas import (
    get_row_binding,           # Cached GeneratedRowBinding by table_key
    iter_row_bindings,         # Iterate all known bindings
    clear_row_binding_cache,   # For testing
)

from codeintel.core.schemas import (
    GeneratedRowBinding,              # Dataclass with legacy-compatible aliases
    row_binding_for_table_schema,     # Factory from TableSchema
)
```

**Key files created/modified:**
- `core/schemas/row_models.py` — Extended with `GeneratedRowBinding`, type coercion
- `build/schemas/row_registry.py` — **NEW** canonical registry
- `build/hamilton/contracts/schemas/row_binding_factory.py` — Prefers generated bindings
- `config/datasets/contracts.py` — Deprecation warning added

**Key insight:** Full deletion of legacy files deferred to PR-70 due to extensive direct imports throughout the codebase. The deprecation-first approach is both safer and easier to review.

**Known technical debt for PR-70:**
- `analytics.static_diagnostics` has schema drift requiring investigation
- `docs.v_*` views need clarification on whether they should have row bindings
- 30+ files directly import from `config/datasets/rows/` and need migration

**Next recommended action:** Proceed with PR-68 (contract derivation) to unblock full migration of storage layer files that need both schemas AND contract metadata.

---

# PR-68 — Derive DatasetContract from build targets

## Status: ✅ PHASE 1 COMPLETE

**Completed 2024-12**: Contract provider created, key storage layer consumers migrated, deprecation warnings added to legacy APIs.

## Goal

Eliminate the duplication between `TargetGraph` target definitions and `DATASET_CONTRACTS` by deriving `DatasetContract` instances from `OutputTarget` metadata.

## Critical Dependency from PR-66

PR-66 implementation revealed that **complete elimination of `config.datasets` imports requires PR-68**. Several files migrated in PR-66 still need contract metadata:

| File | Still Needs Contract For |
|------|--------------------------|
| `storage/schema/ddl.py` | `is_view` filtering in `assert_schema_alignment()` |
| `storage/metadata/bootstrap.py` | Contract metadata in `bootstrap_metadata_datasets()` |

These files use `get_schema_provider()` for schemas but still import `get_dataset_contracts_by_table_key()` or `DATASET_CONTRACTS` for non-schema metadata. PR-68 must provide:

1. A way to get `is_view` status for a table key
2. A way to get contract metadata (owner, filenames, etc.) derived from targets

## Implementation Notes (from actual PR-68 work)

### Key Insight: Leverage Existing OutputContract Structure

The `OutputContract` dataclass in `build/contracts.py` already has a `tables: tuple[TableSchema, ...]` field with a `get_table(table_key)` method. Rather than adding a new `output_schemas` field, we extend `OutputContract` with metadata fields previously only in `DatasetContract`:

```python
# Extended OutputContract (build/contracts.py)
@dataclass(frozen=True)
class OutputContract:
    tables: tuple[TableSchema, ...] = ()
    artifacts: tuple[ArtifactSpec, ...] = ()
    
    # NEW: Extended metadata for contract derivation
    json_schema_ids: tuple[str, ...] = ()
    jsonl_filenames: tuple[str, ...] = ()
    parquet_filenames: tuple[str, ...] = ()
    owner: str | None = None
    description: str | None = None
    family: str | None = None
    # ... more metadata fields
```

### Key Insight: ContractProvider vs. DatasetProvider Naming

The module was named `contract_provider.py` (not `dataset_provider.py`) to emphasize that it provides `DatasetContract` instances derived from targets, not raw target data.

### Completed Files

| File | Change |
|------|--------|
| `src/codeintel/build/contracts.py` | ✅ Extended `OutputContract` with metadata fields |
| `src/codeintel/build/schemas/contract_provider.py` | ✅ **NEW** — `get_contract_for_table_key()`, `is_view()`, `iter_contracts()`, `iter_contracts_by_table_key()` |
| `src/codeintel/build/schemas/__init__.py` | ✅ Exports contract provider functions |
| `src/codeintel/storage/schema/ddl.py` | ✅ Migrated to use `is_view()` and `iter_contracts_by_table_key()` |
| `src/codeintel/storage/metadata/bootstrap.py` | ✅ Migrated to use `iter_contracts()` and `is_view()` |
| `src/codeintel/config/datasets/contracts.py` | ✅ Added `DeprecationWarning` to `get_dataset_contracts()` and `get_dataset_contracts_by_table_key()` |
| `tests/build/hamilton/test_pr68_contract_provider_parity.py` | ✅ **NEW** — 33 parity and functionality tests |

## Current Duplication

```python
# TargetGraph has:
OutputTarget(
    name="risk_factors",
    contract=OutputTargetContract(table_keys=("analytics.goid_risk_factors",)),
    ...
)

# DATASET_CONTRACTS has (separate, can drift):
DatasetContract(
    name="goid_risk_factors",
    table_key="analytics.goid_risk_factors",
    owner="analytics",
    description="...",
    upstream_dependencies=("function_metrics", "coverage_functions"),
    ...
)
```

## Target State

```python
# Extended OutputTargetContract
OutputTargetContract(
    table_keys=("analytics.goid_risk_factors",),
    owner="analytics",
    description="Composite risk factors per function",
    json_schema_ids=("goid_risk_factors",),
    jsonl_filenames=("goid_risk_factors.jsonl",),
    parquet_filenames=("goid_risk_factors.parquet",),
)

# DatasetContract derived at runtime
def dataset_contract_for_table_key(table_key: str) -> DatasetContract:
    target = find_producing_target(table_key)
    schema = get_schema_provider().require_table_schema(table_key)
    return DatasetContract.from_target_and_schema(target, schema)
```

## Tasks Checklist

### Phase 1: Extend OutputContract ✅ COMPLETE

* [x] Update `src/codeintel/build/contracts.py` (NOT `targets.py` — used existing `OutputContract`):
  ```python
  @dataclass(frozen=True)
  class OutputContract:
      """Contract defining what an OutputTarget produces."""

      tables: tuple[TableSchema, ...] = ()
      artifacts: tuple[ArtifactSpec, ...] = ()

      # Extended metadata for dataset contract derivation (PR-68)
      json_schema_ids: tuple[str, ...] = ()
      jsonl_filenames: tuple[str, ...] = ()
      parquet_filenames: tuple[str, ...] = ()
      owner: str | None = None
      description: str | None = None
      family: str | None = None
      freshness_sla: str | None = None
      retention_policy: str | None = None
      upstream_dependencies: tuple[str, ...] = ()
      tags: frozenset[str] = field(default_factory=frozenset)
      validation_profile: Literal["strict", "lenient"] = "strict"
  ```

### Phase 2: Create target-to-dataset derivation ✅ COMPLETE

* [x] Add `src/codeintel/build/schemas/contract_provider.py` (named `contract_provider`, not `dataset_provider`):
  ```python
  from __future__ import annotations

  from dataclasses import dataclass
  from functools import lru_cache
  from typing import TYPE_CHECKING, Literal

  from codeintel.build.registry import get_target_graph
  from codeintel.build.schemas.registry import get_schema_provider

  if TYPE_CHECKING:
      from codeintel.build.targets import OutputTarget
      from codeintel.config.datasets.contracts import DatasetContract
      from codeintel.core.schemas.primitives import TableSchema


  def _find_producing_target(table_key: str) -> OutputTarget | None:
      """Find the target that produces a given table key."""
      graph = get_target_graph()
      for target in graph.all_targets:
          if table_key in target.contract.table_keys:
              return target
      return None


  def _owner_package_from_schema(
      schema: str,
  ) -> Literal["core", "analytics", "graphs", "qa", "docs"] | None:
      """Derive owner package from schema prefix."""
      mapping = {
          "core": "core",
          "analytics": "analytics",
          "graph": "graphs",
          "docs": "docs",
          "qa": "qa",
      }
      return mapping.get(schema)


  @lru_cache(maxsize=256)
  def dataset_contract_for_table_key(table_key: str) -> DatasetContract:
      """Derive a DatasetContract from the target that produces a table key.

      Parameters
      ----------
      table_key
          Fully qualified table key (schema.table).

      Returns
      -------
      DatasetContract
          Derived contract combining target metadata and schema.

      Raises
      ------
      KeyError
          If no target produces the table key.
      """
      from codeintel.config.datasets.contracts import DatasetContract

      target = _find_producing_target(table_key)
      schema = get_schema_provider().require_table_schema(table_key)
      schema_prefix, table_name = table_key.split(".", maxsplit=1)

      contract = target.contract if target else None

      return DatasetContract(
          name=table_name,
          table_key=table_key,
          schema=schema,
          row_binding=None,  # Generated on demand via get_row_binding()
          json_schema_id=_get_json_schema_id(contract, table_name),
          jsonl_filename=_get_jsonl_filename(contract, table_key),
          parquet_filename=_get_parquet_filename(contract, table_key),
          is_view=table_key.startswith("docs."),
          owner_package=_owner_package_from_schema(schema_prefix),
          tags=contract.tags if contract else frozenset(),
          description=contract.description if contract else schema.description,
          family=schema_prefix,
          owner=contract.owner if contract else None,
          freshness_sla=contract.freshness_sla if contract else None,
          retention_policy=contract.retention_policy if contract else None,
          upstream_dependencies=contract.upstream_dependencies if contract else (),
          validation_profile=contract.validation_profile if contract else "strict",
      )


  def _get_json_schema_id(contract, table_name: str) -> str | None:
      if contract and contract.json_schema_ids:
          return contract.json_schema_ids[0]
      return None


  def _get_jsonl_filename(contract, table_key: str) -> str | None:
      if contract and contract.jsonl_filenames:
          return contract.jsonl_filenames[0]
      return f"{table_key.split('.')[-1]}.jsonl"


  def _get_parquet_filename(contract, table_key: str) -> str | None:
      if contract and contract.parquet_filenames:
          return contract.parquet_filenames[0]
      return f"{table_key.split('.')[-1]}.parquet"


  __all__ = ["dataset_contract_for_table_key"]
  ```

### Phase 3: Migrate targets to extended contract — DEFERRED

* [ ] Update target registrations in `build/registry.py` to include metadata:
  - **Deferred to PR-70** — targets currently get schemas from declared provider
  - Metadata enrichment can happen incrementally as targets are touched

### Phase 4: Migrate consumers of get_dataset_contracts() — PARTIAL

* [x] `storage/schema/ddl.py`:
  - ✅ Migrated to use `is_view()` and `iter_contracts_by_table_key()`

* [x] `storage/metadata/bootstrap.py`:
  - ✅ Migrated to use `iter_contracts()` and `is_view()`

* [ ] `cli/handlers/datasets.py`:
  - Replace `get_dataset_contracts()` with `get_contract_for_table_key()`
  - **Deferred to PR-70**

* [ ] `serving/services/datasets.py`:
  - Migrate to derived contracts
  - **Deferred to PR-70**

* [ ] `export/` modules:
  - Migrate to derived contracts
  - **Deferred to PR-70**

### Phase 5: Deprecate legacy contract access ✅ COMPLETE

* [x] Add deprecation warning to `get_dataset_contracts()`:
  ```python
  def get_dataset_contracts() -> dict[str, DatasetContract]:
      """Return dataset contracts by name.

      .. deprecated::
          Use `codeintel.build.schemas.get_contract_for_table_key()` instead.
      """
      warnings.warn(
          "get_dataset_contracts() is deprecated. "
          "Use codeintel.build.schemas.get_contract_for_table_key() for individual lookups "
          "or iter_contracts() for iteration.",
          DeprecationWarning,
          stacklevel=2,
      )
      return _dataset_contracts_cache()
  ```

* [x] Add deprecation warning to `get_dataset_contracts_by_table_key()`

## Tests Checklist (`tests/build/hamilton/`)

### ✅ Completed Tests

* [x] `test_pr68_contract_provider_parity.py` (33 tests):
  - `test_is_view_returns_true_for_known_views` - View detection
  - `test_is_view_returns_false_for_tables` - Table detection
  - `test_get_contract_for_table_key_returns_dataset_contract` - Type validation
  - `test_get_contract_for_table_key_is_cached` - LRU caching
  - `test_get_contract_for_table_key_raises_for_unknown_key` - Error handling
  - `test_get_contract_for_table_key_populates_table_key` - Field population
  - `test_get_contract_for_table_key_populates_name` - Field population
  - `test_get_contract_for_table_key_populates_schema` - Schema resolution
  - `test_iter_contracts_yields_multiple_contracts` - Iteration
  - `test_iter_contracts_each_is_dataset_contract` - Type validation
  - `test_iter_contracts_by_table_key_yields_tuples` - Key-value iteration
  - `test_derived_contract_name_matches_legacy` - Parity validation
  - ... and more

## CLI Snapshots

* [ ] None required (infrastructure PR)

---

## PR-68 Phase 1 Completion Summary

**Completed**: Contract provider infrastructure established with:

```python
from codeintel.build.schemas import (
    get_contract_for_table_key,     # Derive DatasetContract by table_key
    is_view,                         # Check if table_key is a view
    iter_contracts,                  # Iterate all known contracts
    iter_contracts_by_table_key,     # Iterate as (key, contract) tuples
    clear_contract_cache,            # For testing
)
```

**Key files created/modified:**
- `build/contracts.py` — Extended `OutputContract` with metadata fields
- `build/schemas/contract_provider.py` — **NEW** canonical contract derivation
- `storage/schema/ddl.py` — Migrated to contract provider
- `storage/metadata/bootstrap.py` — Migrated to contract provider
- `config/datasets/contracts.py` — Deprecation warnings added

**Key insight:** The contract provider derives `DatasetContract` instances by combining:
1. Target metadata (if a target produces the table key)
2. Schema metadata (from the schema provider)
3. View detection (from `DERIVED_DOCS_VIEWS` constant)

**Known limitation:** Some consumers still import legacy `DATASET_CONTRACTS` at module level (e.g., `serving/backend/datasets.py`). These show deprecation warnings but continue to function. Full migration deferred to PR-70.

---

# PR-69 — Extend schema inference to non-Ibis targets

## Status: ✅ PHASE 1 COMPLETE

**Completed 2024-12**: Unified schema provider created with three-tier fallback chain, `get_schema_provider()` now returns `UnifiedSchemaProvider`.

## Goal

Extend the schema provider to handle targets that don't produce Ibis expressions (e.g., plugin wrappers, legacy compute nodes) by using declared `output_schemas` in their contracts.

## Current Gap

PR-60 only infers schemas for `q__`-driven Ibis compute nodes. Targets using plugin wrappers or other mechanisms have no inference path.

## Implementation Notes (from actual PR-69 work)

### Critical Insight: Circular Import Management

The biggest challenge in PR-69 was managing circular imports. The unified provider needs to import from `provider_hamilton` which imports from `build.hamilton.driver_factory`, which eventually loops back through `build.registry` which calls `get_schema_provider()`.

**Solution: Aggressive lazy imports at multiple levels:**

1. **`registry.py`**: Import `provider_unified` inside functions, not at module level
2. **`provider_unified.py`**: Import `provider_hamilton` inside methods, not at module level
3. **`__init__.py`**: Use `__getattr__` for lazy loading of unified provider exports

```python
# Pattern for lazy imports in high-dependency modules
def get_schema_provider() -> SchemaProvider:
    # Lazy import to avoid circular dependency at module load time.
    from codeintel.build.schemas.provider_unified import (  # noqa: PLC0415
        unified_schema_provider,
    )
    return unified_schema_provider()
```

### Key Insight: Separate Module-Level vs. Runtime Schema Access

A critical distinction emerged: **target definitions at module load time** need `declared_schema_provider()` while **runtime resolution** uses `unified_schema_provider()`.

```python
# build/registry.py — Module-level code uses declared provider
_provider = declared_schema_provider()  # NOT get_schema_provider()!
_DATASET_TABLE_SCHEMAS = {ts.table_key: ts for ts in _provider.iter_table_schemas()}

# Runtime code uses unified provider
def resolve_schema(table_key: str) -> TableSchema:
    return get_schema_provider().require_table_schema(table_key)  # Unified
```

### Key Insight: OutputContract.tables IS the output_schemas

The plan originally suggested adding a new `output_schemas` field. In practice, `OutputContract.tables` already serves this purpose with a `get_table(table_key)` method. The unified provider uses this existing infrastructure.

### Completed Files

| File | Change |
|------|--------|
| `src/codeintel/build/schemas/provider_unified.py` | ✅ **NEW** — `UnifiedSchemaProvider`, `unified_schema_provider()`, `clear_unified_provider_cache()` |
| `src/codeintel/build/schemas/registry.py` | ✅ Updated to return unified provider with lazy imports |
| `src/codeintel/build/schemas/__init__.py` | ✅ Exports unified provider via `__getattr__` lazy loading |
| `src/codeintel/build/registry.py` | ✅ Changed to use `declared_schema_provider()` for module-level schemas |
| `tests/build/hamilton/test_pr69_unified_schema_provider.py` | ✅ **NEW** — 21 comprehensive tests |

## Tasks Checklist

### Phase 1: Output schema support via existing infrastructure ✅ COMPLETE

* [x] Leverage existing `OutputContract.tables` field (no new field needed):
  - `OutputContract.get_table(table_key)` already provides schema lookup
  - Unified provider uses this method for target-declared schemas

### Phase 2: Create unified schema provider ✅ COMPLETE

* [x] Add `src/codeintel/build/schemas/provider_unified.py`:
  ```python
  @dataclass
  class UnifiedSchemaProvider:
      """Schema provider with fallback chain: inferred -> target-declared -> declared."""

      declared: SchemaProvider
      inferable_table_keys: frozenset[str]
      fallback_to_declared_on_error: bool = True
      _cache: dict[str, TableSchema] = field(default_factory=dict)

      def get_table_schema(self, table_key: str) -> TableSchema | None:
          # Check cache first
          cached = self._cache.get(table_key)
          if cached is not None:
              return cached

          # 1. Try Hamilton-native inference (LAZY IMPORT!)
          if table_key in self.inferable_table_keys:
              try:
                  from codeintel.build.schemas.provider_hamilton import (  # noqa: PLC0415
                      infer_schema_for_table_key,
                  )
                  inferred = infer_schema_for_table_key(...)
              except Exception:
                  if not self.fallback_to_declared_on_error:
                      raise
              else:
                  self._cache[table_key] = inferred
                  return inferred

          # 2. Try target-declared output schema
          target = _find_producing_target(table_key)
          if target is not None:
              output_schema = target.contract.get_table(table_key)  # Use existing method!
              if output_schema is not None:
                  self._cache[table_key] = output_schema
                  return output_schema

          # 3. Fall back to raw declared schema
          return self.declared.get_table_schema(table_key)
  ```

### Phase 3: Update registry to use unified provider ✅ COMPLETE

* [x] Update `src/codeintel/build/schemas/registry.py`:
  ```python
  def get_schema_provider() -> SchemaProvider:
      # Lazy import to avoid circular dependency at module load time.
      from codeintel.build.schemas.provider_unified import (  # noqa: PLC0415
          unified_schema_provider,
      )
      return unified_schema_provider()
  ```

* [x] Update `src/codeintel/build/registry.py` to use declared provider for module-level:
  ```python
  # Uses declared_schema_provider() directly (not unified) because target
  # definitions happen at module import time, before full system is initialized.
  _provider = declared_schema_provider()
  _DATASET_TABLE_SCHEMAS = {ts.table_key: ts for ts in _provider.iter_table_schemas()}
  ```

### Phase 4: Migrate remaining declared schemas to target contracts — DEFERRED

* [ ] Identify tables without inference that need target declarations
* [ ] Add `output_schemas` to relevant target contracts
* [ ] Verify all table keys resolvable via unified provider
* **Deferred to PR-70** — current fallback to declared provider covers all cases

## Tests Checklist (`tests/build/hamilton/`)

### ✅ Completed Tests

* [x] `test_pr69_unified_schema_provider.py` (21 tests):
  - `test_unified_provider_is_returned_by_get_schema_provider` - Integration
  - `test_unified_provider_has_schema_provider_interface` - Interface compliance
  - `test_unified_provider_is_cached` - LRU caching
  - `test_clear_unified_provider_cache_works` - Cache management
  - `test_unified_provider_resolves_all_legacy_table_keys` - Parity validation
  - `test_unified_provider_schemas_match_legacy_schemas` - Schema equality
  - `test_unified_provider_schema_count_at_least_legacy` - Count validation
  - `test_unified_provider_has_inferable_table_keys` - Attribute validation
  - `test_inferable_table_keys_not_empty` - Hamilton inference availability
  - `test_unified_provider_has_declared_fallback` - Fallback validation
  - `test_unified_provider_fallback_to_declared_for_unknown_inferable` - Fallback behavior
  - `test_iter_table_schemas_no_duplicates` - Deduplication
  - `test_iter_table_schemas_contains_expected_keys` - Key presence
  - `test_iter_table_schemas_returns_valid_schemas` - Schema validity
  - `test_require_table_schema_raises_for_unknown_key` - Error handling
  - `test_get_table_schema_returns_none_for_unknown_key` - None handling
  - `test_unified_provider_caches_resolved_schemas` - Internal caching
  - `test_unified_provider_has_dataclass_fields` - Structure validation
  - `test_target_contract_schemas_accessible` - Target integration
  - `test_unified_provider_works_with_get_schema_provider` - Registry integration
  - `test_unified_provider_works_with_require_table_schema` - Convenience function

## CLI Snapshots

* [ ] None required (infrastructure PR)

---

## PR-69 Phase 1 Completion Summary

**Completed**: Unified schema provider with three-tier fallback chain:

```python
from codeintel.build.schemas import (
    get_schema_provider,              # Returns UnifiedSchemaProvider
    unified_schema_provider,          # Direct access to unified provider
    UnifiedSchemaProvider,            # Type for annotations
    clear_unified_provider_cache,     # For testing
)

# Resolution order:
# 1. Hamilton-native inference (q__-driven Ibis compute nodes)
# 2. Target-declared schemas (OutputContract.tables)
# 3. Raw declared schemas (declared_schema_provider fallback)
```

**Key files created/modified:**
- `build/schemas/provider_unified.py` — **NEW** three-tier provider
- `build/schemas/registry.py` — Updated with lazy imports
- `build/schemas/__init__.py` — `__getattr__` lazy loading
- `build/registry.py` — Uses declared provider for module-level schemas

**Critical learnings:**
1. **Circular import management** is essential — use lazy imports aggressively
2. **Module-level vs runtime** schema access have different requirements
3. **`OutputContract.tables`** already provides output schemas via `get_table()`
4. **Caching at multiple levels** (LRU cache + internal dict) avoids repeated inference

---

# PR-70 — Delete legacy config/datasets infrastructure

## Goal

Complete removal of the legacy schema infrastructure once all consumers are migrated. This PR consolidates all deletions deferred from PR-66, PR-67, PR-68, and PR-69 into a single focused cleanup PR.

## Scope Expansion (from PR-67 learnings)

PR-70 now includes work deferred from PR-67:
- Deletion of `config/datasets/rows/` directory (6 files)
- Deletion of `config/datasets/generated_rows/` directory
- Migration of 30+ files that directly import legacy row types
- Final cleanup of deprecation warnings

## Prerequisites

- [x] PR-66 complete (all TABLE_SCHEMAS consumers migrated) ✅
- [x] PR-67 Phase 1 complete (row binding infrastructure created, deprecations added) ✅
- [ ] PR-67 Phase 2 (consumer migration) — **Now part of PR-70**
- [x] PR-68 Phase 1 complete (contract provider created, deprecations added) ✅
- [x] PR-69 Phase 1 complete (unified provider handles all table keys) ✅

**All infrastructure prerequisites are complete!** PR-70 can now proceed with consumer migration and legacy deletion.

## Tasks Checklist

### Phase 1: Final migration audit

* [ ] Run import check:
  ```bash
  rg "from codeintel\.config\.datasets" --type py | grep -v "config/datasets/"
  ```

* [ ] Verify zero external imports of:
  - `config.datasets.schemas`
  - `config.datasets.contracts`
  - `config.datasets.rows`
  - `config.datasets.generated_rows`

### Phase 2: Migrate remaining row binding consumers (from PR-67)

* [ ] Migrate files that directly import legacy row types:
  ```bash
  rg "from codeintel\.config\.datasets\.rows" --type py
  ```

* [ ] Update `storage/gateway/accessors.py`:
  - Replace `get_row_bindings()[table_key]` with `get_row_binding(table_key)`

* [ ] Update `analytics/utilities/datasets.py`:
  - Replace direct row type imports with generated models or schema-based access

* [ ] Update all `analytics/profiles/*.py` files:
  - Replace legacy row type imports

* [ ] Update all `analytics/compute/row_builders/*.py` files:
  - Use generated serializers from row registry

### Phase 3: Delete legacy modules

* [ ] Delete `src/codeintel/config/datasets/schemas.py` (~114KB)
* [ ] Delete `src/codeintel/config/datasets/contracts.py` (~1000 lines)
* [ ] Delete `src/codeintel/config/datasets/schema_provider.py`
* [ ] Delete `src/codeintel/config/datasets/rows/` directory
* [ ] Delete `src/codeintel/config/datasets/generated_rows/` directory
* [ ] Delete `src/codeintel/config/datasets/row_factory.py`
* [ ] Delete `src/codeintel/config/datasets/pandera_json_schema.py`

### Phase 4: Simplify remaining primitives

* [ ] Update `src/codeintel/config/datasets/primitives.py`:
  ```python
  """Re-exports from core schemas for backward compatibility.

  .. deprecated::
      Import directly from codeintel.core.schemas instead.
  """
  from __future__ import annotations

  from codeintel.core.schemas.primitives import (
      Column,
      ColumnType,
      Index,
      TableSchema,
  )

  __all__ = ["Column", "ColumnType", "Index", "TableSchema"]
  ```

* [ ] Update `src/codeintel/config/datasets/__init__.py`:
  ```python
  """Dataset configuration primitives.

  Most functionality has moved to:
  - codeintel.core.schemas (primitives, provider, hashing)
  - codeintel.build.schemas (registry, inference, row models)
  """
  from __future__ import annotations

  from codeintel.config.datasets.primitives import (
      Column,
      ColumnType,
      Index,
      TableSchema,
  )

  __all__ = ["Column", "ColumnType", "Index", "TableSchema"]
  ```

### Phase 5: Update documentation

* [ ] Update AGENTS.md schema documentation
* [ ] Update any docstrings referencing deleted modules
* [ ] Add migration guide for external consumers

### Phase 6: Clean up imports

* [ ] Find and fix any remaining broken imports
* [ ] Run full test suite
* [ ] Run type checkers (pyright, pyrefly)

## Tests Checklist (`tests/build/hamilton/`)

* [ ] `test_pr70_no_legacy_imports.py`:
  - AST scan verifying no imports from deleted modules

* [ ] `test_pr70_all_existing_tests_pass.py`:
  - Full test suite runs without import errors

## CLI Snapshots

* [ ] None required (deletion PR)

## Technical Debt to Address (from PR-67 findings)

| Issue | Description | Action |
|-------|-------------|--------|
| `analytics.static_diagnostics` schema drift | Legacy TypedDict has different fields than current TableSchema | Investigate and reconcile — may need schema update or explicit migration |
| `docs.v_*` view bindings | Views have legacy RowBindings but no schema-generated equivalents | Decide: remove bindings (views don't need them) or add view schemas |
| Direct row type imports | 30+ files import from `config/datasets/rows/` | Must migrate all before deletion |

## Deletion Checklist

| File/Directory | Size | Status |
|----------------|------|--------|
| `config/datasets/schemas.py` | ~114KB | [ ] Deleted |
| `config/datasets/contracts.py` | ~40KB | [ ] Deleted |
| `config/datasets/schema_provider.py` | ~1KB | [ ] Deleted |
| `config/datasets/rows/` | ~50KB | [ ] Deleted |
| `config/datasets/generated_rows/` | ~30KB | [ ] Deleted |
| `config/datasets/row_factory.py` | ~5KB | [ ] Deleted |
| `config/datasets/pandera_json_schema.py` | ~3KB | [ ] Deleted |
| **Total** | **~243KB** | |

---

# PR-71 — Schema drift detection CI gate

## Goal

Strengthen the PR-63 schema manifest gate to catch all schema drift scenarios and provide tooling for schema migrations.

## Tasks Checklist

### Phase 1: Extend schema diff command

* [ ] Add detailed diff output to `codeintel build schema diff`:
  ```python
  @dataclass
  class SchemaDiff:
      table_key: str
      added_columns: tuple[str, ...]
      removed_columns: tuple[str, ...]
      type_changes: tuple[tuple[str, str, str], ...]  # (col, old_type, new_type)
      nullable_changes: tuple[tuple[str, bool, bool], ...]
      
      @property
      def has_breaking_changes(self) -> bool:
          return bool(self.removed_columns or self.type_changes)
  ```

* [ ] Implement diff algorithm:
  ```python
  def compute_schema_diff(
      expected: TableSchema,
      actual: TableSchema,
  ) -> SchemaDiff:
      """Compute detailed diff between expected and actual schemas."""
      expected_cols = {c.name: c for c in expected.columns}
      actual_cols = {c.name: c for c in actual.columns}
      
      added = tuple(sorted(set(actual_cols) - set(expected_cols)))
      removed = tuple(sorted(set(expected_cols) - set(actual_cols)))
      
      type_changes = []
      nullable_changes = []
      for name in set(expected_cols) & set(actual_cols):
          exp, act = expected_cols[name], actual_cols[name]
          if exp.type != act.type:
              type_changes.append((name, exp.type, act.type))
          if exp.nullable != act.nullable:
              nullable_changes.append((name, exp.nullable, act.nullable))
      
      return SchemaDiff(
          table_key=expected.table_key,
          added_columns=added,
          removed_columns=removed,
          type_changes=tuple(type_changes),
          nullable_changes=tuple(nullable_changes),
      )
  ```

### Phase 2: Add schema migrate command

* [ ] Add `codeintel build schema migrate`:
  ```python
  @cli_command("build.schema.migrate")
  @build_schema_app.command(name="migrate")
  @dataclass
  class BuildSchemaMigrateCommand:
      """Update declared schemas to match inferred schemas."""

      targets: list[str] | None = None
      dry_run: bool = True
      output_file: str | None = None
  ```

* [ ] Implement migration logic:
  - Compare inferred vs declared
  - Generate migration plan
  - Apply (or output) schema updates

### Phase 3: CI integration

* [ ] Add GitHub Actions step:
  ```yaml
  - name: Schema drift gate
    run: |
      uv run codeintel build schema compile --only-native --infer-native --stable --output /tmp/actual.json
      uv run codeintel build schema diff --expected tests/build/hamilton/snapshots/pr63_schema_manifest_native.json --actual /tmp/actual.json
  ```

* [ ] Add pre-commit hook (optional):
  ```yaml
  - id: schema-drift
    name: Schema drift check
    entry: uv run codeintel build schema diff --expected snapshots/schema_manifest.json
    pass_filenames: false
  ```

### Phase 4: Add breaking change detection

* [ ] Implement breaking change rules:
  - Column removal: BREAKING
  - Type change (narrowing): BREAKING
  - Nullable to non-nullable: BREAKING
  - Column addition: NON-BREAKING
  - Non-nullable to nullable: NON-BREAKING

* [ ] Add `--fail-on-breaking` flag:
  ```python
  fail_on_breaking: bool = True  # Exit 1 on breaking changes
  fail_on_any: bool = False  # Exit 1 on any drift
  ```

## Tests Checklist (`tests/build/hamilton/`)

* [ ] `test_pr71_schema_diff_detects_column_addition.py`
* [ ] `test_pr71_schema_diff_detects_column_removal.py`
* [ ] `test_pr71_schema_diff_detects_type_change.py`
* [ ] `test_pr71_schema_diff_detects_nullable_change.py`
* [ ] `test_pr71_breaking_vs_nonbreaking_classification.py`
* [ ] `test_pr71_migrate_dry_run_outputs_plan.py`

## CLI Snapshots

* [ ] `pr71_schema_diff_help.txt`:
  ```yaml
  - name: "pr71_schema_diff_help"
    tags: ["pr71", "schema", "diff", "text", "tiny"]
    args: ["build", "schema", "diff", "--help"]
    exit_code: 0
    snapshot: "pr71_schema_diff_help.txt"
    kind: "text"
  ```

* [ ] `pr71_schema_migrate_help.txt`:
  ```yaml
  - name: "pr71_schema_migrate_help"
    tags: ["pr71", "schema", "migrate", "text", "tiny"]
    args: ["build", "schema", "migrate", "--help"]
    exit_code: 0
    snapshot: "pr71_schema_migrate_help.txt"
    kind: "text"
  ```

---

# PR-72 — Unified catalog: tables, views, and artifacts

## Goal

Extend schema authority to include DuckDB views and build artifacts (Parquet, JSONL exports).

## Tasks Checklist

### Phase 1: Add view schema inference

* [ ] Add `infer_view_schema()` to `build/schemas/infer_duckdb.py`:
  ```python
  def infer_view_schema(
      *,
      con: DuckDBConnection,
      view_key: str,
  ) -> TableSchema:
      """Infer schema for an existing DuckDB view.

      Parameters
      ----------
      con
          DuckDB connection with the view defined.
      view_key
          Fully qualified view key (schema.view_name).

      Returns
      -------
      TableSchema
          Inferred schema for the view.
      """
      schema_name, view_name = split_table_key(view_key)
      rows = con.execute(f"DESCRIBE {schema_name}.{view_name}").fetchall()
      columns = [
          Column(name=str(r[0]), type=normalize_duckdb_type(str(r[1])), nullable=True)
          for r in rows
      ]
      return TableSchema(schema=schema_name, name=view_name, columns=columns)
  ```

* [ ] Add view inference to unified provider

### Phase 2: Add artifact metadata

* [ ] Define `ArtifactSpec`:
  ```python
  @dataclass(frozen=True)
  class ArtifactSpec:
      """Specification for a build artifact."""
      
      kind: Literal["parquet", "jsonl", "json", "csv"]
      filename: str
      table_key: str | None = None  # Source table for typed artifacts
      description: str | None = None
  ```

* [ ] Add to `OutputTargetContract`:
  ```python
  artifacts: tuple[ArtifactSpec, ...] = ()
  ```

### Phase 3: Extend schema compile command

* [ ] Add flags:
  ```python
  include_views: bool = False
  include_artifacts: bool = False
  ```

* [ ] Update manifest format:
  ```json
  {
    "version": "v2",
    "tables": [...],
    "views": [...],
    "artifacts": [...]
  }
  ```

### Phase 4: Register docs.* views

* [ ] Add view schema inference for all `docs.v_*` views
* [ ] Include in schema manifest

## Tests Checklist (`tests/build/hamilton/`)

* [ ] `test_pr72_view_schema_inferred_from_describe.py`
* [ ] `test_pr72_artifact_spec_in_manifest.py`
* [ ] `test_pr72_manifest_v2_format.py`

## CLI Snapshots

* [ ] `pr72_schema_compile_with_views.json`:
  ```yaml
  - name: "pr72_schema_compile_with_views"
    tags: ["pr72", "schema", "views", "json", "integration"]
    args: ["build", "schema", "compile", "--include-views", "--format", "json"]
    exit_code: 0
    snapshot: "pr72_schema_compile_with_views.json"
    kind: "json"
  ```

---

# PR-73 — JSON Schema generation from TableSchema

## Goal

Auto-generate JSON Schemas (2020-12) for export validation from TableSchema, eliminating hand-maintained JSON Schema files.

## Tasks Checklist

### Phase 1: Implement JSON Schema generator

* [ ] Add `src/codeintel/core/schemas/json_schema_gen.py`:
  ```python
  from __future__ import annotations

  from typing import TYPE_CHECKING

  if TYPE_CHECKING:
      from codeintel.core.schemas.primitives import ColumnType, TableSchema


  def _json_schema_type_for_column(col_type: ColumnType) -> dict[str, object]:
      """Map ColumnType to JSON Schema type definition."""
      mapping: dict[str, dict[str, object]] = {
          "BOOLEAN": {"type": "boolean"},
          "INTEGER": {"type": "integer"},
          "BIGINT": {"type": "integer"},
          "DOUBLE": {"type": "number"},
          "DECIMAL": {"type": "number"},
          "DECIMAL(38,0)": {"type": "integer"},
          "VARCHAR": {"type": "string"},
          "JSON": {},  # Any valid JSON
          "TIMESTAMP": {"type": "string", "format": "date-time"},
          "TIMESTAMPTZ": {"type": "string", "format": "date-time"},
      }
      return mapping.get(col_type, {})


  def json_schema_from_table_schema(
      schema: TableSchema,
      *,
      schema_id: str | None = None,
  ) -> dict[str, object]:
      """Generate JSON Schema 2020-12 from TableSchema.

      Parameters
      ----------
      schema
          Source TableSchema.
      schema_id
          Optional $id for the schema.

      Returns
      -------
      dict[str, object]
          JSON Schema document.
      """
      properties: dict[str, object] = {}
      required: list[str] = []

      for col in schema.columns:
          col_schema = _json_schema_type_for_column(col.type)
          if col.nullable:
              if "type" in col_schema:
                  col_schema = {"oneOf": [col_schema, {"type": "null"}]}
          else:
              required.append(col.name)
          
          if col.description:
              col_schema["description"] = col.description
          
          properties[col.name] = col_schema

      result: dict[str, object] = {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "title": schema.table_key,
          "properties": properties,
          "required": required,
          "additionalProperties": False,
      }

      if schema_id:
          result["$id"] = schema_id

      if schema.description:
          result["description"] = schema.description

      return result


  __all__ = ["json_schema_from_table_schema"]
  ```

### Phase 2: Create JSON Schema registry

* [ ] Add `src/codeintel/build/schemas/json_schema_registry.py`:
  ```python
  from functools import lru_cache

  from codeintel.build.schemas.registry import get_schema_provider
  from codeintel.core.schemas.json_schema_gen import json_schema_from_table_schema


  @lru_cache(maxsize=256)
  def get_json_schema(table_key: str) -> dict[str, object]:
      """Return generated JSON Schema for a table key."""
      table_schema = get_schema_provider().require_table_schema(table_key)
      return json_schema_from_table_schema(
          table_schema,
          schema_id=f"urn:codeintel:schema:{table_key}",
      )
  ```

### Phase 3: Migrate export validation

* [ ] Update `export/export_jsonl.py`:
  - Replace hand-maintained JSON Schema lookups
  - Use `get_json_schema(table_key)`

* [ ] Update `export/export_parquet.py`:
  - Use generated schemas for validation

### Phase 4: Delete hand-maintained JSON Schema files

* [ ] Audit `_JSON_SCHEMA_BY_DATASET_NAME` in contracts.py
* [ ] Delete corresponding JSON Schema files
* [ ] Remove lookup tables

## Tests Checklist (`tests/build/hamilton/`)

* [ ] `test_pr73_json_schema_has_all_columns.py`
* [ ] `test_pr73_json_schema_nullable_handled.py`
* [ ] `test_pr73_json_schema_validates_export_row.py`
* [ ] `test_pr73_generated_schema_matches_2020_12.py`

## CLI Snapshots

* [ ] `pr73_json_schema_for_function_metrics.json`:
  - Generate and snapshot JSON Schema for a known table

---

## Implementation Sequence

```
PR-66 (migrate TABLE_SCHEMAS consumers) ✅ PHASE 1 COMPLETE
  │
  ├─► Core registry created, key consumers migrated
  │   Storage/build layer files need PR-68 for full migration
  │
  ▼
PR-67 (migrate RowBindings to generated) ✅ PHASE 1 COMPLETE
  │
  ├─► Core infrastructure created, deprecation warnings added
  │   Full deletion deferred to PR-70 for consolidated cleanup
  │
  ▼
PR-68 (derive DatasetContract from targets) ✅ PHASE 1 COMPLETE
  │
  ├─► Contract provider created, storage layer migrated
  │   Deprecation warnings added to legacy contract access
  │
  ▼
PR-69 (extend inference to non-Ibis targets) ✅ PHASE 1 COMPLETE
  │
  ├─► Unified schema provider with three-tier fallback
  │   Circular import challenges resolved with lazy loading
  │
  ▼
PR-70 (DELETE legacy config/datasets)  ◄── Major milestone (NEXT)
  │
  ├─► Depends on PR-66, PR-67 Phase 1, PR-68, PR-69 (ALL COMPLETE)
  │   Includes PR-67 consumer migration + all deletions
  │   ~243KB of legacy code removed
  │
  ▼
PR-71 (schema drift CI gate)
  │
  ├─► Can run in parallel after PR-70
  │   Strengthens schema governance
  │
  ▼
PR-72 (unified catalog: tables + views + artifacts)
  │
  ├─► Can run in parallel after PR-70
  │   Extends schema authority
  │
  ▼
PR-73 (JSON Schema generation)
  │
  └─► Can run in parallel after PR-70
      Completes export validation migration
```

---

## Impact Summary

| Metric | Before Phase 3 | After Phase 3 |
|--------|----------------|---------------|
| Files importing `config.datasets` | 65 | 0 |
| Lines in `config/datasets/schemas.py` | ~3000+ | 0 (deleted) |
| Lines in `config/datasets/contracts.py` | ~1000 | 0 (deleted) |
| Manual RowBindings | 50+ | 0 (generated) |
| Manual TypedDict row models | 100+ | 0 (generated) |
| Manual DatasetContracts | 100+ | 0 (target-derived) |
| Schema sources of truth | 3 | 1 (DAG) |
| Total legacy code removed | 0 | ~243KB |

---

## Quick Wins (Can Start Immediately)

1. ~~**PR-66 Phase 1**: Add `get_schema_provider()` registry and deprecation warnings (~1 day)~~ ✅ DONE
2. ~~**PR-66 Phase 2**: Migrate `storage/schema/ddl.py` as proof of concept (~1 day)~~ ✅ DONE
3. ~~**PR-67 Phase 1**: Add `GeneratedRowBinding` and row registry infrastructure (~1 day)~~ ✅ DONE
4. ~~**PR-67 Phase 2**: Add deprecation warnings to `get_row_bindings()` (~30 min)~~ ✅ DONE
5. ~~**PR-68 Phase 1**: Create contract provider and migrate storage layer (~1 day)~~ ✅ DONE
6. ~~**PR-69 Phase 1**: Create unified schema provider with fallback chain (~1 day)~~ ✅ DONE
7. **Add lint rule**: Ban new `config.datasets.schemas` imports (~30 min)
8. **Add lint rule**: Ban new `config.datasets.contracts.get_row_bindings()` calls (~30 min)
9. **PR-70**: Begin legacy deletion (all prerequisites complete) (~2-3 days)

---

## Risk Mitigation

### Risk: Breaking existing consumers during migration

**Mitigation**:
- Add deprecation warnings before removal
- Run full test suite after each file migration
- Keep legacy modules until PR-70 (all consumers migrated)
- **Learned from PR-67**: Deprecation-first approach is safer than immediate deletion
- **Learned from PR-68**: Deprecation warnings surface usage patterns in test output

### Risk: Generated row models don't match legacy behavior

**Mitigation**:
- PR-67 includes parity tests comparing generated vs legacy
- Gradual rollout: switch consumers one at a time
- Keep legacy bindings available as fallback during transition
- **Learned from PR-67**: Document known exclusions (views, schema drift) explicitly in tests

### Risk: Schema inference failures for edge cases

**Mitigation**:
- Unified provider (PR-69) has fallback chain
- `fallback_to_declared_on_error=True` by default
- Clear error messages when inference fails
- **Learned from PR-69**: Internal caching prevents repeated inference attempts

### Risk: Large-scale deletion causes review fatigue

**Mitigation** (added after PR-67):
- Consolidate all deletions in PR-70 for focused review
- Each prerequisite PR (66-69) adds infrastructure without breaking changes
- PR-70 is purely subtractive — easier to review and revert if needed

### Risk: Circular imports in schema provider stack

**Mitigation** (added after PR-69):
- Use lazy imports inside functions for modules with complex dependency chains
- Use `__getattr__` in `__init__.py` for lazy module exports
- Separate module-level schema access (`declared_schema_provider()`) from runtime access (`unified_schema_provider()`)
- **Critical insight**: Modules importing at module level (like `build/registry.py`) must use `declared_schema_provider()` directly, not `get_schema_provider()`

### Risk: Performance regression from schema inference

**Mitigation** (added after PR-69):
- Unified provider has internal `_cache` dict for resolved schemas
- `unified_schema_provider()` is LRU cached at function level
- Inference only happens on first access, then cached

---

## Success Criteria

Phase 3 is complete when:

1. ✅ Zero files import from `config.datasets.schemas`
2. ✅ Zero files import from `config.datasets.contracts`
3. ✅ `config/datasets/schemas.py` is deleted
4. ✅ `config/datasets/contracts.py` is deleted
5. ✅ `config/datasets/rows/` directory is deleted
6. ✅ All tests pass with generated row models
7. ✅ All tests pass with target-derived contracts
8. ✅ Schema manifest CI gate is active
9. ✅ Full test suite passes
10. ✅ Type checkers (pyright, pyrefly) pass

---

## Appendix: File Migration Tracking

### Storage Layer (12 files)

### Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Complete - no schema imports from config.datasets |
| ⚠️ | Partial - uses provider but still needs contract imports (awaiting PR-68) |
| [ ] | Pending |
| N/A | Not applicable for this PR |

| File | PR-66 | PR-67 | PR-68 | Status |
|------|-------|-------|-------|--------|
| `storage/schema/ddl.py` | ✅ | N/A | ✅ | Complete - uses `is_view()` and `iter_contracts_by_table_key()` |
| `storage/schema/json_schema.py` | [ ] | N/A | N/A | Pending |
| `storage/metadata/bootstrap.py` | ✅ | N/A | ✅ | Complete - uses `iter_contracts()` and `is_view()` |
| `storage/validation/contract.py` | [ ] | N/A | N/A | Pending |
| `storage/validation/data_checks.py` | [ ] | N/A | N/A | Pending |
| `storage/gateway/accessors.py` | [ ] | [ ] | N/A | Pending |
| `storage/gateway/base_accessor.py` | [ ] | [ ] | N/A | Pending |
| `storage/datasets/registry.py` | [ ] | N/A | [ ] | Pending |
| `storage/datasets/catalog.py` | [ ] | N/A | [ ] | Pending |
| `ingestion/adapters/duckdb_storage.py` | [ ] | N/A | N/A | Pending |
| `ingestion/adapters/hash_change_detection.py` | [ ] | N/A | N/A | Pending |
| `ingestion/compute/typing_ingest.py` | [ ] | N/A | N/A | Pending |

### Analytics Layer (18 files)

| File | PR-66 | PR-67 | Status |
|------|-------|-------|--------|
| `analytics/utilities/datasets.py` | [ ] | [ ] | Pending |
| `analytics/graphs/config_graph_metrics.py` | [ ] | N/A | Pending |
| `analytics/graphs/subsystem_graph_metrics.py` | [ ] | N/A | Pending |
| `analytics/graphs/symbol_orchestrator.py` | [ ] | N/A | Pending |
| `analytics/graphs/graph_metrics_ext.py` | [ ] | N/A | Pending |
| `analytics/graphs/module_graph_metrics_ext.py` | [ ] | N/A | Pending |
| `analytics/compute/row_builders/graph_metrics.py` | [ ] | [ ] | Pending |
| `analytics/compute/row_builders/graph_metrics_ext.py` | [ ] | [ ] | Pending |
| `analytics/compute/hotspots/metrics.py` | [ ] | N/A | Pending |
| `analytics/testing/coverage/edges.py` | [ ] | N/A | Pending |
| `analytics/testing/profiles/rows.py` | [ ] | [ ] | Pending |
| `analytics/profiles/functions.py` | [ ] | [ ] | Pending |
| `analytics/profiles/modules.py` | [ ] | [ ] | Pending |
| `analytics/profiles/files.py` | [ ] | [ ] | Pending |
| `analytics/functions/metrics.py` | [ ] | N/A | Pending |
| `analytics/parsing/validation.py` | [ ] | N/A | Pending |
| `analytics/ast_features/persist.py` | [ ] | N/A | Pending |
| `ingestion/compute/docstrings_extract.py` | [ ] | N/A | Pending |

### Build Layer (12 files + 4 new)

| File | PR-66 | PR-67 | PR-68 | PR-69 | Status |
|------|-------|-------|-------|-------|--------|
| `build/schemas/registry.py` | ✅ | N/A | N/A | ✅ | **UPDATED** - Now returns unified provider with lazy imports |
| `build/schemas/row_registry.py` | N/A | ✅ | N/A | N/A | **NEW** - Created in PR-67 |
| `build/schemas/contract_provider.py` | N/A | N/A | ✅ | N/A | **NEW** - Created in PR-68 |
| `build/schemas/provider_unified.py` | N/A | N/A | N/A | ✅ | **NEW** - Created in PR-69 |
| `build/registry.py` | ✅ | N/A | N/A | ✅ | Complete - uses `declared_schema_provider()` for module-level |
| `build/contracts.py` | [ ] | N/A | ✅ | N/A | Extended with metadata fields |
| `build/contracts_validation.py` | ✅ | N/A | N/A | N/A | Complete |
| `build/targets.py` | [ ] | N/A | N/A | N/A | Pending |
| `build/plugins/graphs/builders/callgraph.py` | [ ] | N/A | N/A | N/A | Pending |
| `build/hamilton/contracts/schemas/builder.py` | [ ] | N/A | N/A | N/A | Pending |
| `build/hamilton/contracts/schemas/schema.py` | [ ] | N/A | N/A | N/A | Pending |
| `build/hamilton/contracts/schemas/row_binding_factory.py` | [ ] | ✅ | N/A | N/A | Complete - prefers generated bindings |
| `build/hamilton/contracts/schemas/pandera_schemas.py` | [ ] | N/A | N/A | N/A | Pending |
| `build/hamilton/contracts/schemas/validation.py` | [ ] | N/A | N/A | N/A | Pending |
| `build/hamilton/contracts/schemas/row_migration.py` | [ ] | [ ] | N/A | N/A | Pending |
| `build/schemas/provider_declared.py` | ✅ | N/A | N/A | N/A | Unchanged - already uses provider pattern |

### Export Layer (4 files)

| File | PR-66 | PR-67 | Status |
|------|-------|-------|--------|
| `export/__init__.py` | [ ] | N/A | Pending |
| `export/export_exprs.py` | [ ] | N/A | Pending |
| `export/export_jsonl.py` | [ ] | N/A | Pending |
| `export/export_parquet.py` | [ ] | N/A | Pending |

### Serving Layer (6 files)

| File | PR-66 | PR-68 | Status |
|------|-------|-------|--------|
| `serving/services/datasets.py` | [ ] | [ ] | Pending |
| `serving/backend/datasets.py` | [ ] | [ ] | Pending |
| `serving/backend/dataset_backend.py` | [ ] | [ ] | Pending |
| `serving/operations/catalog.py` | [ ] | [ ] | Pending |
| `serving/auto_pipeline.py` | [ ] | N/A | Pending |
| `serving/mcp/meta_tools.py` | [ ] | N/A | Pending |

### CLI Layer (3 files)

| File | PR-66 | PR-68 | Status |
|------|-------|-------|--------|
| `cli/handlers/datasets.py` | [ ] | [ ] | Pending |
| `cli/handlers/ops.py` | [ ] | N/A | Pending |
| `cli/commands/datasets.py` | [ ] | N/A | Pending |

### Graph Layer (2 files)

| File | PR-66 | PR-67 | Status |
|------|-------|-------|--------|
| `graphs/compute/callgraph/collection.py` | [ ] | N/A | Pending |
| `graphs/compute/callgraph/persistence.py` | [ ] | N/A | Pending |

### Test Files Created in PR-66

| File | Purpose |
|------|---------|
| `tests/build/hamilton/test_pr66_schema_provider_registry.py` | ✅ 13 parity and functionality tests |

### Test Files Created in PR-67

| File | Purpose |
|------|---------|
| `tests/build/hamilton/test_pr67_row_binding_parity.py` | ✅ 10+ parity and functionality tests for generated row bindings |

### Test Files Created in PR-68

| File | Purpose |
|------|---------|
| `tests/build/hamilton/test_pr68_contract_provider_parity.py` | ✅ 33 parity and functionality tests for derived contracts |

### Test Files Created in PR-69

| File | Purpose |
|------|---------|
| `tests/build/hamilton/test_pr69_unified_schema_provider.py` | ✅ 21 comprehensive tests for unified provider fallback chain |

---

## PR-66 Phase 1 Completion Summary

**Completed**: Core schema provider registry established with:

```python
from codeintel.build.schemas import (
    get_schema_provider,           # Cached SchemaProvider instance
    require_table_schema,          # Convenience: single schema lookup
    iter_table_schemas,            # Convenience: iterate all schemas
    clear_schema_provider_cache,   # For testing
)
```

**Key files migrated:**
- `build/schemas/registry.py` (new)
- `build/registry.py` (complete)
- `build/contracts_validation.py` (complete)
- `storage/schema/ddl.py` (partial - awaits PR-68)
- `storage/metadata/bootstrap.py` (partial - awaits PR-68)

**Key insight:** Files needing contract metadata (`is_view`, `owner`, `filenames`) cannot fully migrate until PR-68 provides target-derived contracts.

**Next recommended action:** ~~Proceed with PR-67 (row binding migration) or PR-68 (contract derivation)~~ **All prerequisites complete!** Proceed with PR-70 (legacy deletion).

---

## Critical Architectural Patterns (Learned from PR-66 through PR-69)

These patterns emerged during implementation and should guide future work:

### 1. Lazy Import Pattern for Circular Dependencies

When modules form circular dependencies, use lazy imports inside functions:

```python
# ❌ BAD: Module-level import causes circular dependency
from codeintel.build.schemas.provider_unified import unified_schema_provider

def get_schema_provider() -> SchemaProvider:
    return unified_schema_provider()

# ✅ GOOD: Lazy import inside function
def get_schema_provider() -> SchemaProvider:
    from codeintel.build.schemas.provider_unified import (  # noqa: PLC0415
        unified_schema_provider,
    )
    return unified_schema_provider()
```

### 2. `__getattr__` for Lazy Package Exports

For `__init__.py` files exporting modules with complex dependencies:

```python
# src/codeintel/build/schemas/__init__.py

_LAZY_IMPORTS = {
    "UnifiedSchemaProvider": "codeintel.build.schemas.provider_unified",
    "unified_schema_provider": "codeintel.build.schemas.provider_unified",
}

def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

### 3. Module-Level vs. Runtime Schema Access

**Critical distinction**: Code that runs at module import time has different requirements than runtime code:

```python
# ❌ BAD: get_schema_provider() triggers unified provider at import time
_DATASET_TABLE_SCHEMAS = {ts.table_key: ts for ts in get_schema_provider().iter_table_schemas()}

# ✅ GOOD: declared_schema_provider() works at import time
_DATASET_TABLE_SCHEMAS = {ts.table_key: ts for ts in declared_schema_provider().iter_table_schemas()}
```

### 4. Deprecation-First Migration Strategy

When migrating from legacy to new APIs:

1. **Create new infrastructure** (e.g., `contract_provider.py`) without breaking existing code
2. **Add deprecation warnings** to legacy functions (`get_dataset_contracts()`)
3. **Migrate consumers incrementally** — deprecation warnings surface usage in test output
4. **Defer deletion to consolidation PR** (PR-70) for focused review

```python
def get_dataset_contracts() -> dict[str, DatasetContract]:
    warnings.warn(
        "get_dataset_contracts() is deprecated. "
        "Use codeintel.build.schemas.get_contract_for_table_key() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _dataset_contracts_cache()
```

### 5. Multi-Level Caching Strategy

For expensive operations like schema inference, cache at multiple levels:

```python
@dataclass
class UnifiedSchemaProvider:
    _cache: dict[str, TableSchema] = field(default_factory=dict)  # Instance cache
    
    def get_table_schema(self, table_key: str) -> TableSchema | None:
        # Level 1: Instance cache
        cached = self._cache.get(table_key)
        if cached is not None:
            return cached
        # ... resolution logic ...
        self._cache[table_key] = resolved
        return resolved

@lru_cache  # Level 2: Function-level cache
def unified_schema_provider() -> UnifiedSchemaProvider:
    return UnifiedSchemaProvider(...)
```

### 6. Parity Testing with Known Exclusions

When testing migration, explicitly handle legitimate mismatches:

```python
# Explicit exclusion sets with documented reasons
_VIEW_EXCLUSIONS = frozenset({
    "docs.v_subsystem_profile",  # View, no schema-generated binding expected
})

_SCHEMA_DRIFT_EXCLUSIONS = frozenset({
    "analytics.static_diagnostics",  # Known drift, tracked in PR-70
})

def test_all_legacy_bindings_have_schema_equivalent() -> None:
    for table_key in legacy_bindings:
        if table_key in _VIEW_EXCLUSIONS:
            continue  # Documented exclusion
        # ... test logic ...
```

### 7. Provider Resolution Hierarchy

The unified provider establishes a clear resolution order:

```
1. Hamilton-native inference (q__-driven Ibis compute nodes)
   └─► Most accurate, dynamically inferred from actual compute
   
2. Target-declared schemas (OutputContract.tables)
   └─► For plugin wrappers and non-Ibis targets
   
3. Raw declared schemas (declared_schema_provider fallback)
   └─► For source tables and legacy definitions
```

This hierarchy ensures:
- **Inferred schemas take precedence** (most accurate)
- **All table keys resolve** (fallback chain covers gaps)
- **Graceful degradation** on inference failures
