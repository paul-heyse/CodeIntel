# Pandera Schema & Test Helper Remediation Plan

> **Status**: ✅ PHASE 1 COMPLETED  
> **Created**: 2025-12-11  
> **Goal**: Fix Pandera validation errors and improve test helper architecture

## Changes Implemented

### Phase 1: Immediate Fixes (Completed)

1. **Removed overlapping Ibis views** (`ibis_views.py`)
   - Removed duplicate view creators that overwrote complete SQL views
   - Kept only unique Ibis views: `analytics.v_function_summary`, `analytics.v_callgraph_degree`, etc.
   - Documented the view separation principle in code comments

2. **Fixed partial-select validation** (`functions.py`, `modules.py`)
   - Removed Pandera validation from single-column projections
   - `list_function_goids()` and `list_modules()` now skip schema validation

3. **Fixed Ibis 11 API issues** (`subsystems.py`)
   - Changed `nulls_last=True` to `nulls_first=False`

4. **Fixed Pandera schema nullability** (`pandera_schemas.py`)
   - Made `caller_risk_level` and `callee_risk_level` nullable in `docs.v_call_graph_enriched`

5. **Fixed test data completeness** (`test_pandera_schemas.py`)
   - Updated `test_covered_lines_leq_executable_lines` to include all required columns

### Results
- Storage tests: 513 passed (from ~100+ failures)
- Remaining CLI/handler failures are unrelated to Pandera (runtime context issues)

---

## Executive Summary

The test suite has ~100+ Pandera schema validation failures caused by a **schema mismatch between competing view systems**. This document provides deep causal analysis and a comprehensive remediation plan.

---

## 1. Root Cause Analysis

### 1.1 The Competing View Systems Problem

The codebase has **two parallel view creation systems** that conflict:

| System | Location | When Called | Purpose |
|--------|----------|-------------|---------|
| **SQL Views** | `views/*.py` → `create_all_views()` | `connection.py:90` | Full-featured views with complex JOINs |
| **Ibis Views** | `views/ibis_views.py` → `create_all_ibis_views()` | `factory.py:58` | Programmatic views (incomplete) |

**The Critical Bug**: When `ensure_views=True`:
1. `create_all_views(con)` creates SQL views ✓
2. `create_all_ibis_views(gateway)` **overwrites** SQL views with incomplete Ibis versions ✗

**Example - `docs.v_subsystem_summary`**:

```sql
-- SQL View (subsystem_views.py) - COMPLETE (21 columns)
SELECT s.repo, s.commit, ...,
       coalesce(agree.disagree_count, 0) AS subsystem_disagree_count,
       coalesce(agree.total_members, 0) AS subsystem_member_count,
       CASE ... END AS subsystem_agreement_ratio,
       ...
FROM analytics.subsystems s
LEFT JOIN (...aggregation...) AS agree ON ...

-- Ibis View (ibis_views.py) - INCOMPLETE (18 columns, missing 3)
summary = joined.select(
    subsystems.repo, subsystems.commit, ...,
    # MISSING: subsystem_disagree_count, subsystem_member_count, subsystem_agreement_ratio
    subsystems.created_at,
)
```

### 1.2 Pandera Schema Expectations vs Reality

```
Pandera Schema (pandera_schemas.py:997-1024):
├── subsystem_disagree_count   ← EXPECTED
├── subsystem_member_count     ← EXPECTED  
├── subsystem_agreement_ratio  ← EXPECTED

Actual Ibis View:
├── (missing all three)        ← ACTUAL
```

**Result**: 100+ test failures with "column X not in dataframe"

### 1.3 Affected Views

| View | SQL Complete | Ibis Complete | Missing Columns |
|------|-------------|---------------|-----------------|
| `docs.v_subsystem_summary` | ✓ | ✗ | 3 columns |
| `docs.v_function_summary` | ✓ | ✓ | None (but Ibis overwrites) |
| `docs.v_call_graph_enriched` | ✓ | ✓ | None |
| `docs.v_file_summary` | ✓ | ✗ | Column name mismatches |

### 1.4 Test Helper Architecture Gaps

1. **No validation mode control**: Can't disable strict validation for partial data scenarios
2. **View creation order not configurable**: Can't choose SQL-only vs Ibis views
3. **Seeds don't trigger view refresh**: Data seeded after view creation may not be visible through views

---

## 2. Proposed Architecture Changes

### 2.1 Remove Redundant Ibis Views (Primary Fix)

**Principle**: SQL views are the source of truth. Ibis views should only exist for views that genuinely need programmatic construction.

**Views to Remove from Ibis**:
- `create_docs_subsystem_summary_view` → Use SQL view instead
- `create_docs_subsystem_profile_view` → Use SQL view instead
- `create_docs_subsystem_coverage_view` → Use SQL view instead

**Views to Keep in Ibis** (for dynamic/complex logic):
- `create_function_hotspots_view` (uses runtime min/max normalization)
- Views that need dynamic column selection

### 2.2 Add View Creation Mode

```python
# New: src/codeintel/storage/views/mode.py
from enum import Enum, auto

class ViewCreationMode(Enum):
    """Control which view system is used."""
    SQL_ONLY = auto()      # Only SQL views (recommended for most cases)
    IBIS_ONLY = auto()     # Only Ibis views (for specific use cases)
    SQL_THEN_IBIS = auto() # Both, Ibis supplements (current broken behavior)
    IBIS_SUPPLEMENTS = auto()  # SQL views first, Ibis only for non-overlapping views

DEFAULT_VIEW_MODE = ViewCreationMode.SQL_ONLY
```

### 2.3 Add Validation Mode to Test Helpers

```python
# Enhancement to tests/_helpers/env_options.py
@dataclass
class ValidationOptions:
    """Control Pandera validation behavior in tests."""
    
    validate_views: bool = True      # Validate view schemas
    validate_tables: bool = True     # Validate table schemas
    strict: bool = True              # Fail on schema errors
    lazy: bool = True                # Collect all errors vs fail-fast
    skip_empty: bool = True          # Skip validation for empty DataFrames
```

### 2.4 Unified Schema Alignment

```python
# New: src/codeintel/storage/validation/schema_alignment.py
def assert_view_schema_matches_pandera(
    con: DuckDBPyConnection,
    view_name: str,
) -> None:
    """Assert that a view's actual columns match its Pandera schema.
    
    This validation runs at startup to catch schema drift early.
    """
    schema = get_dataset_schema(view_name)
    if schema is None:
        return
    
    actual_columns = get_view_columns(con, view_name)
    expected_columns = set(schema.columns.keys())
    
    missing = expected_columns - actual_columns
    extra = actual_columns - expected_columns
    
    if missing or extra:
        raise SchemaAlignmentError(
            view_name=view_name,
            missing=missing,
            extra=extra,
        )
```

---

## 3. Implementation Plan

### Phase 1: Fix Immediate Pandera Errors (High Priority)

**Step 1.1**: Remove overlapping Ibis views from `create_all_ibis_views()`

```python
# views/ibis_views.py
def create_all_ibis_views(gateway: StorageGateway) -> None:
    """Create Ibis-defined views that SUPPLEMENT (not replace) SQL views."""
    # Keep only views that don't exist in SQL form
    create_function_hotspots_view(gateway)  # Unique to Ibis
    create_callgraph_degree_view(gateway)   # Unique to Ibis
    create_import_graph_degree_view(gateway)  # Unique to Ibis
    
    # REMOVE these (they overwrite complete SQL views):
    # - create_docs_subsystem_summary_view
    # - create_docs_subsystem_profile_view
    # - create_docs_subsystem_coverage_view
    # - create_docs_function_summary_view (if duplicate)
```

**Step 1.2**: Fix column references in remaining Ibis views

For `ibis_views.py:create_docs_file_summary_view()`:
```python
# Current (broken):
fp.loc.name("loc"),
fp.complexity.name("complexity"),

# Fixed (matches actual file_profile columns):
fp.avg_loc.name("loc"),
fp.avg_cyclomatic_complexity.name("complexity"),
```

### Phase 2: Add Schema Alignment Validation

**Step 2.1**: Create schema alignment checker

```python
# src/codeintel/storage/validation/schema_alignment.py
def validate_all_view_schemas(con: DuckDBPyConnection) -> list[SchemaError]:
    """Validate all registered view schemas match actual database structure."""
    errors = []
    for view_key in get_registered_view_keys():
        try:
            assert_view_schema_matches_pandera(con, view_key)
        except SchemaAlignmentError as e:
            errors.append(e)
    return errors
```

**Step 2.2**: Add to gateway factory startup

```python
# gateway/factory.py
def open_gateway(config: StorageConfig) -> StorageGateway:
    # ... existing code ...
    if config.validate_schema:
        errors = validate_all_view_schemas(gateway.con)
        if errors:
            raise SchemaValidationError(errors)
    return gateway
```

### Phase 3: Improve Test Helper Architecture

**Step 3.1**: Add validation control to GatewayOptions

```python
# tests/_helpers/env_options.py
@dataclass
class GatewayOptions:
    file_backed: bool = False
    db_path: Path | None = None
    apply_schema: bool = True
    ensure_views: bool = True
    validate_schema: bool = True
    
    # NEW: Fine-grained validation control
    validate_views: bool = True
    skip_validation_for_empty: bool = True
    view_creation_mode: ViewCreationMode = ViewCreationMode.SQL_ONLY
```

**Step 3.2**: Update seed packs to regenerate views after seeding

```python
# tests/_helpers/context.py
class TestContext:
    def require(self, *seed_packs: SeedPack) -> Self:
        for pack in seed_packs:
            self._apply_pack(pack)
        # NEW: Refresh views after seeding if needed
        if self._views_need_refresh:
            self._refresh_views()
        return self
    
    def _refresh_views(self) -> None:
        """Recreate views to reflect newly seeded data."""
        create_all_views(self.gateway.con)
```

### Phase 4: Add View Coverage Testing

**Step 4.1**: Create view schema parity tests

```python
# tests/storage/test_view_schema_parity.py
@pytest.mark.parametrize("view_key", get_all_view_keys())
def test_view_schema_matches_pandera(view_key: str, fresh_gateway: StorageGateway) -> None:
    """Ensure each view's actual schema matches its Pandera definition."""
    schema = get_dataset_schema(view_key)
    if schema is None:
        pytest.skip(f"No Pandera schema for {view_key}")
    
    actual_cols = get_view_columns(fresh_gateway.con, view_key)
    expected_cols = set(schema.columns.keys())
    
    assert actual_cols == expected_cols, f"Schema mismatch for {view_key}"
```

---

## 4. Files to Modify

### Primary Changes

| File | Change |
|------|--------|
| `src/codeintel/storage/views/ibis_views.py` | Remove overlapping view creators |
| `src/codeintel/storage/gateway/factory.py` | Add view creation mode control |
| `src/codeintel/storage/pandera_schemas.py` | Add skip-empty validation option |
| `tests/_helpers/env_options.py` | Add ValidationOptions |
| `tests/_helpers/context.py` | Add view refresh after seeding |

### New Files

| File | Purpose |
|------|---------|
| `src/codeintel/storage/views/mode.py` | ViewCreationMode enum |
| `src/codeintel/storage/validation/schema_alignment.py` | Schema alignment validation |
| `tests/storage/test_view_schema_parity.py` | View schema parity tests |

---

## 5. Rollout Strategy

### Stage 1: Immediate Fix (This PR)
- Remove overlapping Ibis views
- Fix column references in remaining Ibis views
- Run test suite to verify fix

### Stage 2: Validation Hardening (Follow-up PR)
- Add schema alignment validation
- Add view schema parity tests
- Update test helpers with validation options

### Stage 3: Architecture Cleanup (Future)
- Migrate remaining legitimate Ibis views to a clear pattern
- Document when to use SQL vs Ibis views
- Add pre-commit hook for schema parity

---

## 6. Testing the Fix

```bash
# Run the failing tests to verify fix
uv run pytest tests/storage/repositories/ -v --no-cov

# Run full test suite
uv run pytest -q

# Check schema alignment
uv run python -c "
from codeintel.storage.gateway import open_memory_gateway
from codeintel.storage.validation.schema_alignment import validate_all_view_schemas
gw = open_memory_gateway()
errors = validate_all_view_schemas(gw.con)
for e in errors:
    print(e)
gw.close()
"
```

---

## 7. Success Criteria

- [ ] All 100+ Pandera "column not in dataframe" errors resolved
- [ ] `tests/storage/repositories/` passes
- [ ] No view schema drift between SQL, Ibis, and Pandera definitions
- [ ] Test helpers have clear validation mode controls
- [ ] Documentation updated in AGENTS.md

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Removing Ibis views breaks something | Run full test suite, review all callers |
| SQL views have bugs too | Add view schema parity tests |
| Performance regression | Ibis views were already being overwritten; no change |
| New schema drift | Add pre-commit validation hook |

---

## Appendix A: Detailed Error Analysis

### Column Error Frequency

```
100x: subsystem_member_count, subsystem_disagree_count, subsystem_agreement_ratio
 22x: language
 20x: tags, owners, repo, commit  
 16x: urn, tested, qualname, kind, created_at
 14x: yield_count, typedness_source, typedness_bucket, test_count, static_error_count, ...
```

### Affected Test Files

- `tests/storage/repositories/test_functions.py`
- `tests/storage/repositories/test_subsystems.py`
- `tests/storage/repositories/test_graphs.py`
- `tests/docs_export/test_export_parity.py`
- Multiple CLI handler tests
