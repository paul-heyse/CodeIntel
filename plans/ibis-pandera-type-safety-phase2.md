# Ibis + Pandera Type Safety - Phase 2 Implementation Plan

## Executive Summary

This document outlines the fixes for the remaining **18 pyrefly errors** discovered during the Ibis/Pandera migration. These errors fall into 5 distinct categories requiring targeted architectural improvements.

## Root Cause Analysis

### Category 1: TypedDict vs Mapping Mismatch (2 errors)

**Files**: `callgraph_persistence.py`

**Problem**: Functions like `call_graph_edge_to_tuple()` expect typed `CallGraphEdgeRow` (TypedDict) but receive `Mapping[str, object]` after Pandera validation.

**Solution**: Create type-safe row adapter functions that cast dictionaries to TypedDict types.

### Category 2: Redundant Cast Warnings (2 warnings)

**Files**: `ibis_types.py`

**Problem**: In `and_predicates()` and `or_predicates()`, we cast the result of `&` and `|` operations, but the result is already a `BooleanValue`.

**Solution**: Remove the redundant outer cast; only cast the operands.

### Category 3: None Safety in Pandera (1 error)

**Files**: `pandera_schemas.py`

**Problem**: `column.checks` can be `None`, but we iterate over it without a null check.

**Solution**: Add explicit None check before iteration.

### Category 4: Ibis Expression Type Gaps (4 errors)

**Files**: `subsystems.py`, `tests.py`, `ibis_views.py`

**Problem**: 
- `count() > 0` uses comparison operators not recognized by type checker
- `ilike()` method returns `BooleanValue` but type checker doesn't recognize it
- Standard Ibis comparisons (`!=`, `|`) in views aren't typed correctly

**Solution**: Extend `ibis_types.py` with helpers for `count_gt()`, `ilike()`, and ensure all comparisons go through the type-safe wrappers.

### Category 5: Test Helper Type Issues (4 errors)

**Files**: `provisioning.py`, `_wiring.py`, `test_docstrings_plugin.py`, `test_pandera_schemas.py`

**Problem**: Various type mismatches in test infrastructure.

**Solution**: Fix type annotations and add proper protocol implementations.

---

## Implementation

### Phase 2.1: Type-Safe Row Adapters

Add functions to safely convert dictionaries to TypedDict types:

```python
# In codeintel/config/datasets/rows/graph.py

from typing import cast

def dict_to_call_graph_edge(row: Mapping[str, object]) -> CallGraphEdgeRow:
    """Cast a dictionary to CallGraphEdgeRow type."""
    return cast(CallGraphEdgeRow, dict(row))

def dict_to_call_graph_node(row: Mapping[str, object]) -> CallGraphNodeRow:
    """Cast a dictionary to CallGraphNodeRow type."""
    return cast(CallGraphNodeRow, dict(row))
```

### Phase 2.2: Fix ibis_types.py

Remove redundant casts:

```python
def and_predicates(*predicates: object) -> BooleanValue:
    result = ibis_bool(predicates[0])
    for pred in predicates[1:]:
        # Only cast the combined result, not separately
        result = result & ibis_bool(pred)  # & returns BooleanValue already
    return result  # type: ignore[return-value]
```

Add new helpers:

```python
def count_gt(expr: object, value: int) -> BooleanValue:
    """Type-safe count > value comparison."""
    return cast("BooleanValue", expr > value)  # type: ignore[operator]

def ilike(column: object, pattern: str) -> BooleanValue:
    """Type-safe ILIKE pattern match."""
    return cast("BooleanValue", column.ilike(pattern))  # type: ignore[attr-defined]
```

### Phase 2.3: Fix pandera_schemas.py

Add None check:

```python
def _extract_column_constraints(column: Column) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    
    if column.checks is None:
        return constraints
    
    for check in column.checks:
        # ... existing logic
```

### Phase 2.4: Update Ibis Views

Use type-safe helpers in ibis_views.py:

```python
from codeintel.storage.ibis_types import ibis_bool, or_predicates

mismatches = joined.filter(
    or_predicates(
        goids.language != crosswalk.lang,
        goids.rel_path != crosswalk.file_path,
        goids.qualname != crosswalk.ast_qualname,
    )
)
```

---

## Success Criteria

- [ ] Zero pyrefly errors in storage module
- [ ] Zero pyright errors in storage module
- [ ] All tests pass
- [ ] No runtime behavior changes
