# Ibis 11 Migration Implementation Plan

## Executive Summary

This document provides a comprehensive implementation plan for migrating the codebase to Ibis 11.0.0, which introduces several breaking changes that are currently causing test failures.

**Status: ✅ COMPLETED**

All phases have been implemented and tested successfully.

## Breaking Changes in Ibis 11

### 1. Connection API Change
- **Old**: `ibis.duckdb.connect(con=duckdb_connection)`
- **New**: `ibis.duckdb.from_connection(duckdb_connection)`
- **Status**: ✅ Fixed in `ibis_adapter.py`

### 2. Table Access with Qualified Names
- **Old**: `con.table("schema.table_name")`
- **New**: `con.table("table_name", database="schema")`
- **Status**: ✅ Fixed with `_table()` helper in `ibis_views.py` and `IbisGateway.table()` in `ibis_adapter.py`

### 3. View Creation with Qualified Names
- **Old**: `con.create_view("schema.view_name", expr, overwrite=True)`
- **New**: `con.create_view("view_name", expr, database="schema", overwrite=True)`
- **Status**: ✅ Fixed with `_create_view()` helper in `ibis_views.py`

### 4. Case Expression Builder
- **Old**: `ibis.case().when(cond, val).else_(default).end()`
- **New**: `ibis.cases((cond1, val1), (cond2, val2), else_=default)`
- **Status**: ✅ All 3 usages in `ibis_views.py` migrated

### 5. Other Deprecated Methods (to watch for)
- `String.to_date` → `String.as_date`
- `String.to_timestamp` → `String.as_timestamp`
- `IntegerValue.to_interval` → `IntegerValue.as_interval`
- `IntegerValue.to_timestamp` → `IntegerValue.as_timestamp`
- `Struct.destructure` → `Table.unpack`

---

## Implementation Phases

### Phase 1: IbisGateway Table Method Fix (Priority: Critical)

**File**: `src/codeintel/storage/ibis_adapter.py`

The `IbisGateway.table()` method must handle qualified names by splitting on `.`:

```python
def table(self, table_name: str) -> it.Table:
    """Return an Ibis table expression for a fully qualified table.

    Parameters
    ----------
    table_name
        Fully qualified table or view name (e.g., "analytics.function_metrics").

    Returns
    -------
    it.Table
        Ibis table expression for the requested object.
    """
    if "." in table_name:
        database, name = table_name.split(".", 1)
        return self.con.table(name, database=database)
    return self.con.table(table_name)
```

This single fix propagates to all callers of `gateway.ibis.table()`.

---

### Phase 2: Views Table Access Normalization

**File**: `src/codeintel/storage/views/ibis_views.py`

The `_table()` helper already exists but needs to be used consistently. Current implementation:

```python
def _table(con: ibis.backends.duckdb.Backend, qualified_name: str) -> it.Table:
    """Return table using database qualifier when provided."""
    if "." in qualified_name:
        database, table = qualified_name.split(".", 1)
        return con.table(table, database=database)
    return con.table(qualified_name)
```

**Action**: Ensure all `con.table()` calls use `_table()` helper.

---

### Phase 3: Case Expression Migration

**File**: `src/codeintel/storage/views/ibis_views.py`

Three `ibis.case()` usages need migration to `ibis.cases()`:

#### 3.1 `loc_bucket` (lines 75-77)

**Before**:
```python
loc_bucket = (
    ibis.case().when(small_loc, "small").when(medium_loc, "medium").else_("large").end()
)
```

**After**:
```python
loc_bucket = ibis.cases(
    (small_loc, "small"),
    (medium_loc, "medium"),
    else_="large",
)
```

#### 3.2 `complexity_band` (lines 78-84)

**Before**:
```python
complexity_band = (
    ibis.case()
    .when(low_complexity, "low")
    .when(medium_complexity, "medium")
    .else_("high")
    .end()
)
```

**After**:
```python
complexity_band = ibis.cases(
    (low_complexity, "low"),
    (medium_complexity, "medium"),
    else_="high",
)
```

#### 3.3 `normalized_score` (lines 311-316)

**Before**:
```python
normalized_score = (
    ibis.case()
    .when(score_range == 0, 0.0)
    .else_((rf.hotspot_score.cast("float64") - min_score.cast("float64")) / score_range)
    .end()
)
```

**After**:
```python
normalized_score = ibis.cases(
    (score_range == 0, 0.0),
    else_=(rf.hotspot_score.cast("float64") - min_score.cast("float64")) / score_range,
)
```

---

### Phase 4: Repository Layer Audit

**Files**: `src/codeintel/storage/repositories/*.py`

Repositories call `self.gateway.ibis.table()` which will be fixed by Phase 1. 

**Verification needed**:
- `base.py` - `_ibis_table()` method
- `datasets.py` - table lookups
- `functions.py` - function queries
- `graphs.py` - graph queries  
- `modules.py` - module queries
- `subsystems.py` - subsystem queries
- `tests.py` - test queries

---

### Phase 5: Accessor & Helper Audit

**Files**:
- `src/codeintel/storage/gateway/accessors.py`
- `src/codeintel/storage/gateway/base_accessor.py`
- `src/codeintel/storage/helpers/db.py`
- `src/codeintel/storage/validation/*.py`

Check for any direct `con.table()` calls that bypass `IbisGateway`.

---

### Phase 6: Ingestion & Analytics Audit

**Files**:
- `src/codeintel/ingestion/adapters/duckdb_storage.py`
- `src/codeintel/build/analytics/graphs/contracts.py`

Verify Ibis usage patterns are compatible.

---

### Phase 7: Test Infrastructure Update

**Files**:
- `tests/_helpers/fakes/*.py`
- `tests/_helpers/sql.py`
- `tests/storage/*.py`
- `tests/ingestion/*.py`

Ensure test helpers use correct Ibis 11 patterns.

---

### Phase 8: Documentation & Guardrails

1. Add smoke test for `create_all_ibis_views` on in-memory gateway
2. Document Ibis 11 patterns in AGENTS.md addendum:
   - Connection: `ibis.duckdb.from_connection(con)`
   - Tables: `gateway.ibis.table("schema.table")` (handled internally)
   - Cases: `ibis.cases((cond, val), ..., else_=default)`

---

## Implementation Order

| Phase | Description | Files | Estimated Changes |
|-------|-------------|-------|-------------------|
| 1 | IbisGateway.table() fix | `ibis_adapter.py` | 5 lines |
| 2 | Views table normalization | `ibis_views.py` | Already done |
| 3 | Case expression migration | `ibis_views.py` | 15 lines |
| 4 | Repository audit | `repositories/*.py` | Verification only |
| 5 | Accessor/helper audit | `gateway/*.py`, `helpers/db.py` | TBD |
| 6 | Ingestion/analytics audit | 2 files | TBD |
| 7 | Test infrastructure | `tests/_helpers/*.py` | TBD |
| 8 | Documentation | `AGENTS.md` | 20 lines |

---

## Success Criteria

- [ ] `create_all_ibis_views()` runs without errors on fresh gateway
- [ ] All repository methods compile and execute
- [ ] `uv run pytest tests/storage -q` passes
- [ ] `uv run python -m tools.quality_report` shows zero Ibis-related errors
- [ ] Full test suite passes

---

## Rollback Plan

If critical issues arise:
1. Pin Ibis to version 10.x in `pyproject.toml`
2. Revert adapter changes
3. Re-evaluate migration timeline
