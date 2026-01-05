# Ibis + Pandera Type Safety Implementation Plan

## Executive Summary

This document outlines a detailed implementation plan to resolve **317+ type errors** related to Ibis and Pandera in the storage module. The errors fall into two main categories:

1. **Ibis Expression Type Mismatches (38 pyright errors)**: Pyright doesn't understand that Ibis column comparisons (`table.col == value`) return `BooleanValue`, not Python `bool`.

2. **Pandera Column Type Inference (279 pyrefly errors)**: The `_dtype_for_column_type()` function returns `object`, which doesn't satisfy Pandera's `dtype` parameter type signature.

## Root Cause Analysis

### Issue 1: Ibis Expression Types

**Problem**: When writing Ibis filter expressions like:

```python
expr = table.filter(table.repo == self.repo)
```

The static type checkers see `table.repo == self.repo` as returning `bool` (Python's comparison semantics), but Ibis actually returns `BooleanValue` (an Ibis expression type). This causes:

- Pyright: "Argument of type 'bool' cannot be assigned to parameter 'predicates'"
- Pyrefly: "Argument 'bool' is not assignable to parameter '*predicates'"

**Root Cause**: Ibis uses operator overloading (`__eq__`, `__gt__`, etc.) on `Column` objects to return expression objects. Python's type system doesn't understand this without proper type annotations in Ibis's stubs.

### Issue 2: Pandera Column dtype

**Problem**: The function `_dtype_for_column_type()` returns `object`:

```python
def _dtype_for_column_type(col_type: ColumnType) -> object:
    ...
```

But `Column.__init__` expects:

```python
dtype: DataType | ExtensionDtype | dtype[Any] | str | type
```

**Root Cause**: We used `object` as a catch-all return type, but this doesn't satisfy the union type that Pandera expects.

---

## Implementation Plan

### Phase 1: Create Type-Safe Ibis Wrapper Module (Priority: High)

**Goal**: Create a thin wrapper layer that provides proper type hints for Ibis operations.

**Location**: `src/codeintel/storage/ibis_types.py`

**Implementation**:

```python
"""Type-safe Ibis expression helpers.

This module provides wrapper functions that help static type checkers
understand Ibis expression semantics. Ibis uses operator overloading
that returns expression objects (not bools), but Python's type system
doesn't understand this without explicit annotations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import ibis.expr.types as it

if TYPE_CHECKING:
    from ibis.expr.types import BooleanValue, Column, Table

T = TypeVar("T", bound=it.Table)


def eq(column: Column, value: object) -> BooleanValue:
    """Type-safe equality comparison for Ibis columns.

    This wraps column equality to provide correct type hints.
    Ibis's __eq__ returns BooleanValue, not bool.
    """
    return column == value  # type: ignore[return-value]


def ne(column: Column, value: object) -> BooleanValue:
    """Type-safe inequality comparison for Ibis columns."""
    return column != value  # type: ignore[return-value]


def gt(column: Column, value: object) -> BooleanValue:
    """Type-safe greater-than comparison for Ibis columns."""
    return column > value  # type: ignore[return-value]


def ge(column: Column, value: object) -> BooleanValue:
    """Type-safe greater-or-equal comparison for Ibis columns."""
    return column >= value  # type: ignore[return-value]


def lt(column: Column, value: object) -> BooleanValue:
    """Type-safe less-than comparison for Ibis columns."""
    return column < value  # type: ignore[return-value]


def le(column: Column, value: object) -> BooleanValue:
    """Type-safe less-or-equal comparison for Ibis columns."""
    return column <= value  # type: ignore[return-value]


def and_(left: BooleanValue, right: BooleanValue) -> BooleanValue:
    """Type-safe AND for Ibis boolean expressions."""
    return left & right  # type: ignore[return-value]


def or_(left: BooleanValue, right: BooleanValue) -> BooleanValue:
    """Type-safe OR for Ibis boolean expressions."""
    return left | right  # type: ignore[return-value]


def filter_table(table: T, *predicates: BooleanValue) -> T:
    """Type-safe filter that accepts BooleanValue predicates."""
    return table.filter(*predicates)  # type: ignore[return-value]
```

**Migration Pattern**:

Before:
```python
expr = table.filter(
    (table.repo == self.repo) & (table.commit == self.commit)
)
```

After:
```python
from codeintel.storage.ibis_types import eq, and_, filter_table

expr = filter_table(
    table,
    and_(eq(table.repo, self.repo), eq(table.commit, self.commit))
)
```

**Alternative Approach**: Use explicit casts with centralized type comments:

```python
from typing import cast
from ibis.expr.types import BooleanValue

# Cast the comparison result to BooleanValue
predicate = cast(BooleanValue, table.repo == self.repo)
```

### Phase 2: Fix Pandera Column dtype Types (Priority: High)

**Goal**: Make `_dtype_for_column_type()` return a proper type that satisfies Pandera's type requirements.

**Location**: `src/codeintel/storage/pandera_schemas.py`

**Implementation**:

```python
from typing import Union
import numpy as np
from numpy import dtype as NumpyDtype
from pandas.api.types import pandas_dtype

# Define explicit return type
PanderaDtype = Union[type, str, NumpyDtype[np.generic]]

# Type mapping with proper types
_COLUMN_TYPE_TO_DTYPE: dict[str, PanderaDtype] = {
    "VARCHAR": str,
    "TEXT": str,
    "BOOLEAN": bool,
    "INTEGER": np.int64,
    "BIGINT": np.int64,
    "UBIGINT": np.uint64,
    "DOUBLE": np.float64,
    "FLOAT": np.float64,
    "REAL": np.float32,
    "DECIMAL(38,0)": np.int64,
    "TIMESTAMP": "datetime64[ns]",
    "DATE": "datetime64[ns]",
    "JSON": object,  # JSON columns need object type
}

def _dtype_for_column_type(col_type: ColumnType) -> PanderaDtype:
    """Map DuckDB column types to Pandera-compatible dtypes.

    Parameters
    ----------
    col_type
        DuckDB column type string.

    Returns
    -------
    PanderaDtype
        A type that satisfies Pandera's Column dtype parameter.
    """
    normalized = col_type.upper()
    if normalized.startswith("DECIMAL("):
        return _COLUMN_TYPE_TO_DTYPE.get("DECIMAL(38,0)", np.int64)
    return _COLUMN_TYPE_TO_DTYPE.get(normalized, str)
```

### Phase 3: Create Ibis Type Stubs (Priority: Medium)

**Goal**: Create local type stubs that improve Ibis's type hints for our use cases.

**Location**: `stubs/ibis/expr/types/relations.pyi`

**Implementation**:

```python
"""Type stubs for Ibis table expressions."""

from typing import TypeVar, overload
from ibis.expr.types import BooleanValue, Column

T = TypeVar("T", bound="Table")

class Table:
    @overload
    def filter(self, predicate: BooleanValue) -> Table: ...
    @overload
    def filter(self, *predicates: BooleanValue) -> Table: ...
    
    def __getitem__(self, key: str) -> Column: ...
    def __getattr__(self, name: str) -> Column: ...

class Column:
    def __eq__(self, other: object) -> BooleanValue: ...  # type: ignore[override]
    def __ne__(self, other: object) -> BooleanValue: ...  # type: ignore[override]
    def __gt__(self, other: object) -> BooleanValue: ...
    def __ge__(self, other: object) -> BooleanValue: ...
    def __lt__(self, other: object) -> BooleanValue: ...
    def __le__(self, other: object) -> BooleanValue: ...
    def __and__(self, other: BooleanValue) -> BooleanValue: ...
    def __or__(self, other: BooleanValue) -> BooleanValue: ...
    
    def ilike(self, pattern: str) -> BooleanValue: ...
    def is_true(self) -> BooleanValue: ...
```

### Phase 4: Repository Layer Refactoring (Priority: High)

**Goal**: Update all repository methods to use type-safe Ibis patterns.

**Files to Update**:
- `src/codeintel/storage/repositories/functions.py` (10 errors)
- `src/codeintel/storage/repositories/graphs.py` (2 errors)
- `src/codeintel/storage/repositories/modules.py` (6 errors)
- `src/codeintel/storage/repositories/subsystems.py` (14 errors)
- `src/codeintel/storage/repositories/tests.py` (6 errors)

**Pattern to Apply**:

Create a base method in `BaseRepository` that handles the type-safe filter building:

```python
# In base.py

from typing import cast
from ibis.expr.types import BooleanValue

def _ibis_filter(
    self,
    table: it.Table,
    *,
    extra_predicates: list[BooleanValue] | None = None,
) -> it.Table:
    """Build a filtered table with repo/commit predicates.

    This method handles type casting for pyright compatibility.
    """
    # Cast comparisons to BooleanValue for type safety
    repo_match = cast(BooleanValue, table.repo == self.repo)
    commit_match = cast(BooleanValue, table.commit == self.commit)
    
    predicates: list[BooleanValue] = [repo_match, commit_match]
    if extra_predicates:
        predicates.extend(extra_predicates)
    
    return table.filter(predicates)
```

### Phase 5: Views Layer Type Safety (Priority: Medium)

**Goal**: Apply type-safe patterns to `ibis_views.py`.

**Location**: `src/codeintel/storage/views/ibis_views.py`

**Pattern**:

```python
from typing import cast
from ibis.expr.types import BooleanValue

def create_function_summary_view(gateway: StorageGateway) -> None:
    con = gateway.ibis.con
    fm = con.table("analytics.function_metrics")
    ft = con.table("analytics.function_types")

    # Use explicit casts for join predicates
    join_predicates = [
        cast(BooleanValue, fm.repo == ft.repo),
        cast(BooleanValue, fm.commit == ft.commit),
        cast(BooleanValue, fm.function_goid_h128 == ft.function_goid_h128),
    ]
    
    joined = fm.left_join(ft, join_predicates)
    ...
```

---

## Recommended Approach

Given the tradeoffs, I recommend a **pragmatic hybrid approach**:

### Option A: Centralized Type Cast Helper (Recommended)

Create a minimal helper in `base.py` that uses `cast`:

```python
from typing import cast
from ibis.expr.types import BooleanValue, Column

def _bool_expr(expr: object) -> BooleanValue:
    """Cast an Ibis comparison expression to BooleanValue for type safety."""
    return cast(BooleanValue, expr)
```

Then use it consistently:

```python
expr = table.filter(
    _bool_expr(table.repo == self.repo) & _bool_expr(table.commit == self.commit)
)
```

**Pros**:
- Minimal code changes
- Single point of control
- Clear intent

**Cons**:
- Slightly verbose
- Runtime overhead (minimal)

### Option B: Type Comments (Simpler but Less Safe)

Add `# type: ignore[arg-type]` comments where needed:

```python
expr = table.filter(
    (table.repo == self.repo) & (table.commit == self.commit)  # type: ignore[arg-type]
)
```

**Pros**:
- No code changes
- No runtime overhead

**Cons**:
- Suppresses all type checking at that line
- Less maintainable

---

## Implementation Order

1. **Week 1**: 
   - Create `storage/errors.py` (done)
   - Create `storage/ibis_types.py` with helper functions
   - Fix `_dtype_for_column_type()` return type

2. **Week 2**:
   - Update `BaseRepository` with type-safe Ibis helpers
   - Migrate `functions.py` and `graphs.py` repositories

3. **Week 3**:
   - Migrate `modules.py`, `tests.py`, and `subsystems.py`
   - Update `ibis_views.py`

4. **Week 4**:
   - Create Ibis type stubs if needed
   - Final verification with pyright/pyrefly
   - Update documentation

---

## Success Criteria

- [ ] Zero pyright errors in `src/codeintel/storage`
- [ ] Zero pyrefly errors in `src/codeintel/storage`
- [ ] All existing tests pass
- [ ] No runtime behavior changes
- [ ] Code remains readable and maintainable

---

## Appendix: Full Error Summary

### Pyright Errors (38 total)

| File | Error Type | Count |
|------|------------|-------|
| functions.py | bool → BooleanValue | 10 |
| graphs.py | bool → BooleanValue | 2 |
| modules.py | bool → BooleanValue | 6 |
| subsystems.py | bool → BooleanValue, missing attrs | 14 |
| tests.py | bool → BooleanValue | 6 |

### Pyrefly Errors (279 total)

| File | Error Type | Count |
|------|------------|-------|
| pandera_schemas.py | object → dtype | 279 |
| repositories/*.py | bool → BooleanValue | 4 |
