# Storage Module Refactoring Plan

> **Status**: Proposed  
> **Created**: 2025-12-13  
> **Author**: AI Code Review  
> **Scope**: `src/codeintel/storage/`  
> **Estimated Effort**: 20-25 developer hours  
> **Code Reduction Target**: ~20% (~1,300 lines)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Implementation Phases](#3-implementation-phases)
4. [Phase 1: Quick Wins (P0)](#4-phase-1-quick-wins-p0)
5. [Phase 2: Base Class Consolidation (P1)](#5-phase-2-base-class-consolidation-p1)
6. [Phase 3: Generic Infrastructure (P2)](#6-phase-3-generic-infrastructure-p2)
7. [Phase 4: Architectural Improvements (P3)](#7-phase-4-architectural-improvements-p3)
8. [Migration Guide](#8-migration-guide)
9. [Testing Strategy](#9-testing-strategy)
10. [Rollback Plan](#10-rollback-plan)
11. [Success Metrics](#11-success-metrics)
12. [Appendix: Code Examples](#12-appendix-code-examples)

---

## 1. Executive Summary

### Problem Statement

The `src/codeintel/storage/` module has evolved organically and now contains significant functional duplication:

- **Row parsing logic** duplicated across 3 tracking modules (~200+ lines)
- **JSON serialization** reimplemented in multiple files (~100 lines)
- **Timestamp utilities** defined identically 3 times
- **Validated records pattern** copied between repositories
- **Exception handling** split across two files unnecessarily

### Goals

1. **Reduce code duplication** by ~20% through shared abstractions
2. **Improve maintainability** by consolidating related functionality
3. **Enhance type safety** through generic patterns
4. **Simplify testing** via unified base classes
5. **Prevent future drift** by establishing clear patterns

### Non-Goals

- Changing public API contracts (backward-compatible)
- Modifying database schemas
- Altering Ibis view logic
- Restructuring the gateway protocol (deferred to Phase 4)

---

## 2. Current State Analysis

### File Statistics

| Directory | Files | Lines | Notes |
|-----------|-------|-------|-------|
| `gateway/` | 9 | ~1,200 | Core gateway implementation |
| `tracking/` | 4 | ~2,300 | Run, build, asset tracking |
| `repositories/` | 10 | ~800 | Read-only query repositories |
| `views/` | 11 | ~3,200 | Ibis view definitions |
| `validation/` | 4 | ~400 | Schema validation |
| `datasets/` | 4 | ~500 | Dataset registry |
| `schema/` | 3 | ~300 | DDL and JSON Schema |
| `helpers/` | 5 | ~300 | Utilities |
| Root files | 5 | ~2,500 | Core modules |
| **Total** | **55** | **~11,500** | |

### Identified Duplication Patterns

```
Pattern                          | Occurrences | Estimated Lines
---------------------------------|-------------|----------------
Row parsing (tuple → dataclass)  | 6+          | 200+
JSON serialize/deserialize       | 4           | 80
_now() timestamp helper          | 3           | 15
_validated_records()             | 2           | 20
Accessor initialization          | 4           | 40
---------------------------------|-------------|----------------
Total Duplicated                 |             | ~355 lines
```

---

## 3. Implementation Phases

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           IMPLEMENTATION TIMELINE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1 (P0)          Phase 2 (P1)        Phase 3 (P2)    Phase 4 (P3) │
│  Quick Wins            Base Classes        Generic Infra   Architecture │
│  ─────────────         ────────────        ─────────────   ──────────── │
│  [██████]              [████████]          [██████████]    [████████]   │
│  2-3 hours             4-6 hours           6-8 hours       4-6 hours    │
│                                                                          │
│  • Merge exceptions    • tracking/base.py  • RowMapper[T]  • Split views│
│  • Move validated_     • JSON helpers      • TypedColumn   • Slim proto │
│    records             • Time helpers      • Repo factory  • DDL merge  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Dependency Graph

```
Phase 1 ──┬──► Phase 2 ──┬──► Phase 3 ──► Phase 4
          │              │
          │              └──► Phase 4 (parallel possible)
          │
          └──► Phase 3 (some tasks parallel)
```

---

## 4. Phase 1: Quick Wins (P0)

**Estimated Time**: 2-3 hours  
**Risk Level**: Low  
**Dependencies**: None

### Task 1.1: Merge Exception Files

**Files Affected**:
- `storage/errors.py` (DELETE)
- `storage/exceptions.py` (MODIFY)
- Multiple files with imports from `errors.py`

**Current State**:

```python
# errors.py (22 lines)
from duckdb import Error as DuckDBError
DUCKDB_ERRORS: tuple[type[Exception], ...] = (DuckDBError,)

# exceptions.py (37 lines)
class StorageError(Exception): ...
class StorageConnectionError(StorageError): ...
class SchemaError(StorageError): ...
class QueryError(StorageError): ...
```

**Target State**:

```python
# exceptions.py (merged, ~50 lines)
"""Storage layer exceptions and error types.

This module provides exception classes for the storage layer and
re-exports DuckDB error types for consistent error handling.
"""

from __future__ import annotations

from duckdb import Error as DuckDBError

# Re-export DuckDB errors for catch blocks
DUCKDB_ERRORS: tuple[type[Exception], ...] = (DuckDBError,)

__all__ = [
    "DUCKDB_ERRORS",
    "DuckDBError",
    "QueryError",
    "SchemaError",
    "StorageConnectionError",
    "StorageError",
]


class StorageError(Exception):
    """Base exception for storage layer errors.

    This exception wraps database-specific errors (like duckdb.Error)
    to provide a clean abstraction boundary. Code outside the storage
    layer should catch this instead of database-specific exceptions.
    """


class StorageConnectionError(StorageError):
    """Error establishing or maintaining a database connection."""


class SchemaError(StorageError):
    """Error with database schema (tables, views, macros)."""


class QueryError(StorageError):
    """Error executing a database query."""
```

**Migration Steps**:

1. Copy `DUCKDB_ERRORS` and `DuckDBError` re-export to `exceptions.py`
2. Update `__all__` to include all exports
3. Find all imports from `storage/errors.py`:
   ```bash
   grep -r "from codeintel.storage.errors" src/ tests/
   ```
4. Update imports to use `storage.exceptions`
5. Delete `storage/errors.py`
6. Run tests to verify

**Verification**:
```bash
uv run ruff check src/codeintel/storage/
uv run pytest tests/storage/ -q
```

---

### Task 1.2: Move `_validated_records` to BaseRepository

**Files Affected**:
- `repositories/base.py` (MODIFY)
- `repositories/functions.py` (MODIFY)
- `repositories/graphs.py` (MODIFY)

**Current State** (duplicated in functions.py and graphs.py):

```python
@staticmethod
def _validated_records(table_key: str, expr: it.Table) -> list[RowDict]:
    df = pd.DataFrame(expr.execute())
    validated = validate_df(table_key, df)
    return validated.where(pd.notna(validated), None).to_dict(orient="records")
```

**Target State** (in base.py):

```python
# repositories/base.py

class BaseRepository:
    # ... existing methods ...
    
    def _validated_records(
        self,
        table_key: str,
        expr: it.Table,
    ) -> list[RowDict]:
        """Execute expression with Pandera validation and null normalization.

        Parameters
        ----------
        table_key
            Dataset key for Pandera schema lookup.
        expr
            Ibis table expression to execute.

        Returns
        -------
        list[RowDict]
            Validated records with None substituted for missing values.
        """
        df = pd.DataFrame(expr.execute())
        validated = validate_df(table_key, df)
        sanitized = validated.astype("object").where(pd.notna(validated), None)
        return sanitized.to_dict(orient="records")
```

**Migration Steps**:

1. Add `_validated_records` to `BaseRepository` in `repositories/base.py`
2. Remove `@staticmethod` decorator (method needs `self` for consistency)
3. Update `FunctionRepository` to use inherited method
4. Update `GraphRepository` to use inherited method
5. Run type checker and tests

**Verification**:
```bash
uv run pyright src/codeintel/storage/repositories/
uv run pytest tests/storage/repositories/ -q
```

---

## 5. Phase 2: Base Class Consolidation (P1)

**Estimated Time**: 4-6 hours  
**Risk Level**: Low-Medium  
**Dependencies**: Phase 1 complete

### Task 2.1: Create Shared Time Utilities

**Files Affected**:
- `helpers/time.py` (CREATE)
- `helpers/__init__.py` (MODIFY)
- `tracking/run_tracking.py` (MODIFY)
- `tracking/build_tracking.py` (MODIFY)
- `tracking/asset_tracking.py` (MODIFY)

**New File**: `helpers/time.py`

```python
"""Timezone-aware datetime utilities for storage operations.

This module provides shared time utilities used across tracking modules.
All timestamps are UTC-aware following the project's datetime hygiene rules.
"""

from __future__ import annotations

from datetime import UTC, datetime

__all__ = ["utc_now"]


def utc_now() -> datetime:
    """Return current UTC timestamp with timezone info.

    Returns
    -------
    datetime
        Current datetime with UTC timezone attached.

    Examples
    --------
    >>> ts = utc_now()
    >>> ts.tzinfo is not None
    True
    """
    return datetime.now(tz=UTC)
```

**Update `helpers/__init__.py`**:

```python
from codeintel.storage.helpers.time import utc_now

__all__ = [
    # ... existing exports ...
    "utc_now",
]
```

**Migration**: Replace all `_now()` calls with `utc_now()`:

```python
# Before (in each tracking module)
def _now() -> datetime:
    return datetime.now(tz=UTC)

# After
from codeintel.storage.helpers import utc_now
# Use utc_now() directly
```

---

### Task 2.2: Enhance JSON Serialization Helpers

**Files Affected**:
- `helpers/json.py` (MODIFY)
- `tracking/run_tracking.py` (MODIFY)
- `tracking/build_tracking.py` (MODIFY)

**Add to `helpers/json.py`**:

```python
def serialize_str_sequence(items: Sequence[str]) -> str:
    """Serialize a sequence of strings to compact JSON array.

    Parameters
    ----------
    items
        Sequence of strings to serialize.

    Returns
    -------
    str
        JSON-encoded array string.

    Examples
    --------
    >>> serialize_str_sequence(["a", "b", "c"])
    '["a","b","c"]'
    """
    return encode_json_compact(list(items))


def deserialize_str_tuple(raw: str | None) -> tuple[str, ...]:
    """Deserialize JSON array to string tuple.

    Parameters
    ----------
    raw
        JSON-encoded array or None.

    Returns
    -------
    tuple[str, ...]
        Tuple of strings, empty if raw is None or empty.

    Examples
    --------
    >>> deserialize_str_tuple('["a","b"]')
    ('a', 'b')
    >>> deserialize_str_tuple(None)
    ()
    """
    if not raw:
        return ()
    items = decode_json_list(raw)
    return tuple(str(x) for x in items)
```

**Migration**:

```python
# Before (run_tracking.py)
def _serialize_datasets(datasets: Sequence[str]) -> str:
    return encode_json_compact(list(datasets))

def _deserialize_datasets(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    items = decode_json_list(raw)
    return tuple(str(x) for x in items)

# After
from codeintel.storage.helpers.json import (
    serialize_str_sequence,
    deserialize_str_tuple,
)
# Use directly: serialize_str_sequence(datasets)
```

---

### Task 2.3: Create Tracking Base Class

**Files Affected**:
- `tracking/base.py` (CREATE)
- `tracking/__init__.py` (MODIFY)
- `tracking/run_tracking.py` (MODIFY)
- `tracking/build_tracking.py` (MODIFY)
- `tracking/asset_tracking.py` (MODIFY)

**New File**: `tracking/base.py`

```python
"""Base class for tracking accessors.

This module provides shared infrastructure for pipeline run tracking,
build tracking, and asset tracking. All tracking classes inherit from
BaseTracking to ensure consistent patterns for:

- Connection access
- Policy backend access
- Timestamp generation
- JSON serialization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.helpers.json import (
    deserialize_str_tuple,
    encode_json_compact,
    serialize_str_sequence,
)
from codeintel.storage.helpers.time import utc_now

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from duckdb import DuckDBPyConnection

    from codeintel.storage.gateway.protocol import StorageGateway

__all__ = ["BaseTracking"]


@dataclass
class BaseTracking:
    """Base class for tracking accessors.

    Provides shared utilities for all tracking classes:
    - Connection and policy backend access
    - Timestamp generation
    - JSON serialization helpers

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
    """

    gateway: StorageGateway

    @property
    def con(self) -> DuckDBPyConnection:
        """Return the underlying DuckDB connection."""
        return self.gateway.con

    @property
    def backend(self) -> DuckDBPolicyBackend:
        """Return the policy backend for bulk operations."""
        return self.gateway.policy

    @staticmethod
    def now() -> datetime:
        """Return current UTC timestamp."""
        return utc_now()

    @staticmethod
    def serialize_list(items: Sequence[str]) -> str:
        """Serialize string sequence to JSON."""
        return serialize_str_sequence(items)

    @staticmethod
    def deserialize_list(raw: str | None) -> tuple[str, ...]:
        """Deserialize JSON to string tuple."""
        return deserialize_str_tuple(raw)

    @staticmethod
    def serialize_dict(data: Mapping[str, object] | None) -> str:
        """Serialize dict to compact JSON."""
        return encode_json_compact(dict(data) if data else {})
```

**Migrate `BuildTracking`** (example):

```python
# Before
class BuildTracking:
    def __init__(self, gateway: StorageGateway) -> None:
        self._gateway = gateway
        self._con = gateway.con
        self._backend = DuckDBPolicyBackend(gateway)

# After
@dataclass
class BuildTracking(BaseTracking):
    """Accessor for build manifest and run tracking tables."""
    
    # Inherits: gateway, con, backend, now(), serialize_*, deserialize_*
    # Remove: __init__, _gateway, _con, _backend
    
    def save_manifest(self, manifest: OutputManifest) -> None:
        self.backend.upsert(...)  # Use self.backend instead of self._backend
```

---

## 6. Phase 3: Generic Infrastructure (P2)

**Estimated Time**: 6-8 hours  
**Risk Level**: Medium  
**Dependencies**: Phase 2 complete

### Task 3.1: Create Generic Row Mapper

**Files Affected**:
- `helpers/row_parser.py` (CREATE)
- `helpers/__init__.py` (MODIFY)
- `tracking/build_tracking.py` (MODIFY)
- `tracking/asset_tracking.py` (MODIFY)
- `tracking/run_tracking.py` (MODIFY)

**New File**: `helpers/row_parser.py`

```python
"""Generic row-to-dataclass mapping utilities.

This module provides type-safe infrastructure for parsing DuckDB result
tuples into typed dataclasses. It centralizes:

- Column index mapping
- Type coercion
- Null handling
- JSON deserialization

Example
-------
>>> from dataclasses import dataclass
>>> from codeintel.storage.helpers.row_parser import RowMapper, coerce_str
>>>
>>> @dataclass(frozen=True)
... class UserRecord:
...     user_id: str
...     name: str
...     age: int | None
>>>
>>> mapper = RowMapper(
...     UserRecord,
...     columns=["user_id", "name", "age"],
...     coercions={"user_id": coerce_str, "name": coerce_str},
... )
>>> row = ("123", "Alice", 25)
>>> record = mapper.parse(row)
>>> record.name
'Alice'
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar, cast

from codeintel.storage.helpers.json import decode_json_dict, decode_json_list

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "RowMapper",
    "coerce_bool",
    "coerce_datetime",
    "coerce_float",
    "coerce_int",
    "coerce_json_dict",
    "coerce_json_list",
    "coerce_optional_int",
    "coerce_optional_str",
    "coerce_str",
]

T = TypeVar("T")


# ============================================================================
# Coercion Functions
# ============================================================================

def coerce_str(value: object) -> str:
    """Coerce value to string."""
    return str(value)


def coerce_optional_str(value: object) -> str | None:
    """Coerce value to optional string."""
    return str(value) if value is not None else None


def coerce_int(value: object) -> int:
    """Coerce value to integer."""
    return int(value)  # type: ignore[arg-type]


def coerce_optional_int(value: object) -> int | None:
    """Coerce value to optional integer."""
    return int(value) if value is not None else None  # type: ignore[arg-type]


def coerce_float(value: object) -> float:
    """Coerce value to float."""
    return float(value)  # type: ignore[arg-type]


def coerce_bool(value: object) -> bool:
    """Coerce value to boolean."""
    return bool(value)


def coerce_datetime(value: object) -> datetime:
    """Coerce value to datetime (passthrough for DuckDB timestamps)."""
    return cast("datetime", value)


def coerce_json_dict(value: object) -> dict[str, Any] | None:
    """Coerce JSON string to dictionary."""
    if value is None:
        return None
    return decode_json_dict(value) if isinstance(value, str) else None


def coerce_json_list(value: object) -> list[Any] | None:
    """Coerce JSON string to list."""
    if value is None:
        return None
    return decode_json_list(value) if isinstance(value, str) else None


# ============================================================================
# Row Mapper
# ============================================================================

@dataclass(frozen=True)
class RowMapper[T]:
    """Generic mapper from DuckDB row tuples to typed dataclasses.

    Parameters
    ----------
    dataclass_type
        Target dataclass type to instantiate.
    columns
        Column names in the order they appear in the tuple.
    coercions
        Optional mapping of column names to coercion functions.
        Columns not in this mapping are passed through unchanged.

    Examples
    --------
    >>> @dataclass(frozen=True)
    ... class Record:
    ...     id: str
    ...     count: int
    >>>
    >>> mapper = RowMapper(Record, ["id", "count"], {"id": coerce_str})
    >>> mapper.parse(("abc", 42))
    Record(id='abc', count=42)
    """

    dataclass_type: type[T]
    columns: Sequence[str]
    coercions: Mapping[str, Callable[[object], object]] | None = None

    def parse(self, row: tuple[Any, ...]) -> T:
        """Parse a DuckDB row tuple into a typed dataclass instance.

        Parameters
        ----------
        row
            Tuple of values from DuckDB fetchone/fetchall.

        Returns
        -------
        T
            Instance of the target dataclass.

        Raises
        ------
        ValueError
            If row length doesn't match column count.
        """
        if len(row) != len(self.columns):
            message = (
                f"Row has {len(row)} values, expected {len(self.columns)} "
                f"for {self.dataclass_type.__name__}"
            )
            raise ValueError(message)

        kwargs: dict[str, object] = {}
        for idx, col_name in enumerate(self.columns):
            value = row[idx]
            if self.coercions and col_name in self.coercions:
                value = self.coercions[col_name](value)
            kwargs[col_name] = value

        return self.dataclass_type(**kwargs)

    def parse_many(self, rows: Sequence[tuple[Any, ...]]) -> list[T]:
        """Parse multiple rows into dataclass instances.

        Parameters
        ----------
        rows
            Sequence of row tuples.

        Returns
        -------
        list[T]
            List of parsed dataclass instances.
        """
        return [self.parse(row) for row in rows]

    def parse_optional(self, row: tuple[Any, ...] | None) -> T | None:
        """Parse a row that may be None.

        Parameters
        ----------
        row
            Row tuple or None (from fetchone when no results).

        Returns
        -------
        T | None
            Parsed record or None.
        """
        return self.parse(row) if row is not None else None
```

**Usage Example** (migrating `build_tracking.py`):

```python
# Before
def _parse_manifest_row(row: tuple[Any, ...]) -> OutputManifest:
    return OutputManifest(
        target=str(row[0]),
        repo=str(row[1]),
        commit=str(row[2]),
        plugin=str(row[3]),
        computed_at=cast("datetime", row[4]),
        duration_ms=float(row[5]),
        input_hash=str(row[6]),
        output_hash=str(row[7]) if row[7] is not None else None,
        row_count=int(row[8]) if row[8] is not None else None,
        options_hash=str(row[9]) if row[9] is not None else None,
    )

# After
_MANIFEST_MAPPER = RowMapper(
    OutputManifest,
    columns=[
        "target", "repo", "commit", "plugin", "computed_at",
        "duration_ms", "input_hash", "output_hash", "row_count", "options_hash",
    ],
    coercions={
        "target": coerce_str,
        "repo": coerce_str,
        "commit": coerce_str,
        "plugin": coerce_str,
        "computed_at": coerce_datetime,
        "duration_ms": coerce_float,
        "input_hash": coerce_str,
        "output_hash": coerce_optional_str,
        "row_count": coerce_optional_int,
        "options_hash": coerce_optional_str,
    },
)

# Usage
manifest = _MANIFEST_MAPPER.parse(row)
manifests = _MANIFEST_MAPPER.parse_many(rows)
```

---

### Task 3.2: Create Repository Factory

**Files Affected**:
- `repositories/factory.py` (CREATE)
- `repositories/__init__.py` (MODIFY)

**New File**: `repositories/factory.py`

```python
"""Factory for creating snapshot-scoped repositories.

This module provides a unified factory for creating repository instances
bound to a specific repo/commit snapshot. This ensures consistent
initialization and enables lazy repository creation.

Example
-------
>>> from codeintel.storage.repositories import RepositoryFactory
>>>
>>> factory = RepositoryFactory(gateway, repo="org/repo", commit="abc123")
>>> functions = factory.functions
>>> modules = factory.modules
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from codeintel.storage.repositories.dataflow import DataflowRepository
from codeintel.storage.repositories.functions import FunctionRepository
from codeintel.storage.repositories.graphs import GraphRepository
from codeintel.storage.repositories.modules import ModuleRepository
from codeintel.storage.repositories.subsystems import SubsystemRepository
from codeintel.storage.repositories.tests import TestRepository

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import StorageGateway

__all__ = ["RepositoryFactory"]


class RepositoryFactory:
    """Factory for creating snapshot-scoped repositories.

    Creates repository instances lazily and caches them for reuse.
    All repositories share the same gateway/repo/commit binding.

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
    repo
        Repository identifier (e.g., "org/repo").
    commit
        Commit hash for the snapshot.

    Examples
    --------
    >>> factory = RepositoryFactory(gateway, repo="org/repo", commit="abc123")
    >>> summary = factory.functions.get_function_summary_by_goid(goid)
    """

    def __init__(self, gateway: StorageGateway, repo: str, commit: str) -> None:
        """Initialize the repository factory.

        Parameters
        ----------
        gateway
            Storage gateway providing database access.
        repo
            Repository identifier.
        commit
            Commit hash.
        """
        self._gateway = gateway
        self._repo = repo
        self._commit = commit

    @property
    def gateway(self) -> StorageGateway:
        """Return the underlying storage gateway."""
        return self._gateway

    @property
    def repo(self) -> str:
        """Return the repository identifier."""
        return self._repo

    @property
    def commit(self) -> str:
        """Return the commit hash."""
        return self._commit

    @cached_property
    def functions(self) -> FunctionRepository:
        """Return the function repository."""
        return FunctionRepository(self._gateway, self._repo, self._commit)

    @cached_property
    def modules(self) -> ModuleRepository:
        """Return the module repository."""
        return ModuleRepository(self._gateway, self._repo, self._commit)

    @cached_property
    def graphs(self) -> GraphRepository:
        """Return the graph repository."""
        return GraphRepository(self._gateway, self._repo, self._commit)

    @cached_property
    def tests(self) -> TestRepository:
        """Return the test repository."""
        return TestRepository(self._gateway, self._repo, self._commit)

    @cached_property
    def subsystems(self) -> SubsystemRepository:
        """Return the subsystem repository."""
        return SubsystemRepository(self._gateway, self._repo, self._commit)

    @cached_property
    def dataflow(self) -> DataflowRepository:
        """Return the dataflow repository."""
        return DataflowRepository(self._gateway, self._repo, self._commit)
```

---

### Task 3.3: Split ibis_views.py by Domain

**Files Affected**:
- `views/ibis_views.py` (MODIFY - reduce significantly)
- `views/analytics_views.py` (CREATE)
- `views/core_views.py` (CREATE)
- `views/docs_views.py` (CREATE)
- `views/graph_views.py` (CREATE)
- `views/__init__.py` (MODIFY)

**Strategy**:

1. Create domain-specific view files
2. Move `@register_view` functions to appropriate files
3. Keep `ibis_views.py` for shared utilities and legacy `create_*` functions
4. Import all domain files in `__init__.py` to trigger registration

**New File Structure**:

```
views/
├── __init__.py           # Imports all domain modules for registration
├── ibis_registry.py      # VIEW_BUILDERS dict, @register_view decorator
├── analytics_views.py    # analytics.* views (v_function_summary, etc.)
├── core_views.py         # core.* views (if any)
├── docs_views.py         # docs.* views (v_function_summary, v_file_summary, etc.)
├── graph_views.py        # graph.* views (v_call_graph_degree, etc.)
└── ibis_views.py         # Shared utilities, legacy create_* functions
```

**Example**: `views/analytics_views.py`

```python
"""Ibis view builders for analytics schema.

This module contains view definitions for analytics.* views.
Views are registered via the @register_view decorator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import ibis

from codeintel.storage.views.ibis_registry import register_view

if TYPE_CHECKING:
    import ibis.expr.types as it

    from codeintel.storage.views.ibis_registry import IbisViewGateway

__all__ = [
    "build_function_summary",
]

CALLGRAPH_LOC_SMALL = 50
CALLGRAPH_LOC_MEDIUM = 200
COMPLEXITY_LOW_MAX = 5
COMPLEXITY_MEDIUM_MAX = 10


@register_view("analytics.v_function_summary")
def build_function_summary(ibis_gw: IbisViewGateway) -> it.Table:
    """Build the function summary view expression.

    Combines metrics with typedness details and adds lightweight derived
    buckets for complexity and LOC.
    """
    fm: it.Table = ibis_gw.table("analytics.function_metrics")
    ft: it.Table = ibis_gw.table("analytics.function_types").select(
        # ... column selection ...
    )
    # ... rest of view logic ...
```

**Update `views/__init__.py`**:

```python
"""Ibis view definitions and registry.

This package contains all Ibis-defined views organized by schema:

- analytics_views: analytics.* views
- core_views: core.* views
- docs_views: docs.* views
- graph_views: graph.* views

Views are auto-registered when this package is imported.
"""

from __future__ import annotations

# Import all domain modules to trigger @register_view registration
from codeintel.storage.views import (
    analytics_views,
    core_views,
    docs_views,
    graph_views,
)
from codeintel.storage.views.ibis_registry import (
    VIEW_BUILDERS,
    IbisViewGateway,
    ViewBuilder,
    get_registered_views,
    register_view,
)

# Silence unused import warnings
_ = (analytics_views, core_views, docs_views, graph_views)

__all__ = [
    "VIEW_BUILDERS",
    "IbisViewGateway",
    "ViewBuilder",
    "get_registered_views",
    "register_view",
]
```

---

## 7. Phase 4: Architectural Improvements (P3)

**Estimated Time**: 4-6 hours  
**Risk Level**: Medium-High  
**Dependencies**: Phase 2 complete (Phase 3 optional)

### Task 4.1: Merge schema/ddl.py into DuckDBPolicyBackend

**Files Affected**:
- `schema/ddl.py` (DELETE after migration)
- `schema/__init__.py` (MODIFY)
- `duckdb_policy_backend.py` (MODIFY)

**Current State**: `schema/ddl.py` is a thin wrapper:

```python
def _get_policy_backend(con: DuckDBPyConnection) -> DuckDBPolicyBackend:
    return DuckDBPolicyBackend(gateway=MinimalStorageGateway(con))

def apply_all_schemas(con: DuckDBPyConnection, extra_ddl: Iterable[str] | None = None) -> None:
    backend = _get_policy_backend(con)
    backend.ensure_all_schemas(drop_existing=True, extra_ddl=extra_ddl)
```

**Target State**: Add class methods to `DuckDBPolicyBackend`:

```python
# duckdb_policy_backend.py

class DuckDBPolicyBackend:
    # ... existing methods ...
    
    @classmethod
    def from_connection(cls, con: DuckDBPyConnection) -> DuckDBPolicyBackend:
        """Create a policy backend from a raw DuckDB connection.

        This factory method wraps the connection in a MinimalStorageGateway
        to satisfy the gateway protocol. Use this when you have a raw
        connection and need DDL operations.

        Parameters
        ----------
        con
            Raw DuckDB connection.

        Returns
        -------
        DuckDBPolicyBackend
            Policy backend instance.
        """
        from codeintel.storage.gateway.minimal import MinimalStorageGateway
        return cls(gateway=MinimalStorageGateway(con))
```

**Update `schema/__init__.py`**:

```python
# Thin wrappers for backward compatibility
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

def apply_all_schemas(con, extra_ddl=None):
    DuckDBPolicyBackend.from_connection(con).ensure_all_schemas(
        drop_existing=True, extra_ddl=extra_ddl
    )

def ensure_schemas_preserve(con, extra_ddl=None):
    DuckDBPolicyBackend.from_connection(con).ensure_schemas_preserve(
        extra_ddl=extra_ddl
    )
```

---

### Task 4.2: Refactor MinimalStorageGateway

**Files Affected**:
- `gateway/minimal.py` (MODIFY)

**Current Issues**:
- Uses `Any` typed stubs that lose type safety
- 10+ attributes set to `None`

**Target State**:

```python
"""Minimal gateway for schema-only operations.

This module provides a lightweight gateway implementation that satisfies
the minimum requirements for DuckDBPolicyBackend and view creation,
without the overhead of full accessor initialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from duckdb import DuckDBPyConnection, DuckDBPyRelation

    from codeintel.storage.ibis_adapter import IbisGateway

__all__ = ["MinimalStorageGateway"]


class MinimalStorageGateway:
    """Lightweight gateway for DDL and view operations.

    This gateway provides only the essentials:
    - con: DuckDB connection
    - ibis: IbisGateway (lazy)
    - execute/table methods

    Use DuckDBGateway for full functionality including accessors.

    Parameters
    ----------
    connection
        Raw DuckDB connection to wrap.
    """

    __slots__ = ("_con", "_ibis")

    def __init__(self, connection: DuckDBPyConnection) -> None:
        """Initialize minimal gateway."""
        self._con = connection
        self._ibis: IbisGateway | None = None

    @property
    def con(self) -> DuckDBPyConnection:
        """Return the underlying DuckDB connection."""
        return self._con

    @property
    def ibis(self) -> IbisGateway:
        """Return an Ibis gateway (lazy initialization)."""
        if self._ibis is None:
            from codeintel.storage.ibis_adapter import IbisGateway
            self._ibis = IbisGateway(self)  # type: ignore[arg-type]
        return self._ibis

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._con.close()

    def execute(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> DuckDBPyConnection:
        """Execute SQL against the underlying connection."""
        return self._con.execute(sql, params)

    def table(self, name: str) -> DuckDBPyRelation:
        """Return a relation for the specified table or view."""
        return self._con.table(name)

    # Note: Other attributes (analytics, core, etc.) are not provided.
    # Code requiring those should use DuckDBGateway instead.
```

---

## 8. Migration Guide

### For Existing Code Using `_now()`

```python
# Before
from datetime import UTC, datetime

def _now() -> datetime:
    return datetime.now(tz=UTC)

# After
from codeintel.storage.helpers import utc_now
# Use utc_now() directly
```

### For Existing Code Using `_serialize_*` / `_deserialize_*`

```python
# Before
def _serialize_datasets(datasets: Sequence[str]) -> str:
    return encode_json_compact(list(datasets))

# After
from codeintel.storage.helpers.json import serialize_str_sequence
# Use serialize_str_sequence(datasets)
```

### For Existing Code Using Row Parsing

```python
# Before
def _parse_row(row: tuple[Any, ...]) -> MyRecord:
    return MyRecord(
        field1=str(row[0]),
        field2=int(row[1]) if row[1] else None,
    )

# After
from codeintel.storage.helpers.row_parser import RowMapper, coerce_str, coerce_optional_int

_MAPPER = RowMapper(
    MyRecord,
    columns=["field1", "field2"],
    coercions={"field1": coerce_str, "field2": coerce_optional_int},
)
# Use _MAPPER.parse(row)
```

### For Existing Code Importing from `storage.errors`

```python
# Before
from codeintel.storage.errors import DUCKDB_ERRORS

# After
from codeintel.storage.exceptions import DUCKDB_ERRORS
```

---

## 9. Testing Strategy

### Unit Tests for New Helpers

```python
# tests/storage/helpers/test_time.py
from codeintel.storage.helpers.time import utc_now

def test_utc_now_has_timezone() -> None:
    ts = utc_now()
    assert ts.tzinfo is not None

def test_utc_now_is_utc() -> None:
    from datetime import UTC
    ts = utc_now()
    assert ts.tzinfo == UTC
```

```python
# tests/storage/helpers/test_row_parser.py
from dataclasses import dataclass
from codeintel.storage.helpers.row_parser import RowMapper, coerce_str, coerce_optional_int

@dataclass(frozen=True)
class SampleRecord:
    name: str
    count: int | None

def test_row_mapper_basic() -> None:
    mapper = RowMapper(
        SampleRecord,
        columns=["name", "count"],
        coercions={"name": coerce_str, "count": coerce_optional_int},
    )
    record = mapper.parse(("test", 42))
    assert record.name == "test"
    assert record.count == 42

def test_row_mapper_null_handling() -> None:
    mapper = RowMapper(
        SampleRecord,
        columns=["name", "count"],
        coercions={"name": coerce_str, "count": coerce_optional_int},
    )
    record = mapper.parse(("test", None))
    assert record.count is None
```

### Integration Tests

Ensure existing functionality is preserved:

```bash
# Run all storage tests after each phase
uv run pytest tests/storage/ -v

# Run type checking
uv run pyright src/codeintel/storage/

# Run linting
uv run ruff check src/codeintel/storage/
```

### Regression Tests

For each migrated module, ensure existing tests pass:

```bash
uv run pytest tests/storage/tracking/ -v
uv run pytest tests/storage/repositories/ -v
uv run pytest tests/storage/gateway/ -v
```

---

## 10. Rollback Plan

### Phase 1 Rollback

If issues arise after merging exceptions:

1. Restore `errors.py` from git
2. Revert import changes
3. Keep both files temporarily

### Phase 2-4 Rollback

New files are additive; rollback by:

1. Reverting changes to existing files
2. Removing new files (`tracking/base.py`, `helpers/time.py`, etc.)
3. Running tests to verify

### Git Strategy

```bash
# Create feature branch for each phase
git checkout -b refactor/storage-phase-1
git checkout -b refactor/storage-phase-2
# etc.

# Merge phases individually after validation
git checkout main
git merge refactor/storage-phase-1
# Validate
git merge refactor/storage-phase-2
# etc.
```

---

## 11. Success Metrics

### Quantitative

| Metric | Before | Target | How to Measure |
|--------|--------|--------|----------------|
| Total Lines | ~11,500 | ~9,500 | `find src/codeintel/storage -name "*.py" \| xargs wc -l` |
| Duplicate Functions | 15+ | <5 | Manual audit |
| Files | 55 | ~50 | `find src/codeintel/storage -name "*.py" \| wc -l` |
| Type Errors | 0 | 0 | `uv run pyright src/codeintel/storage/` |
| Lint Errors | 0 | 0 | `uv run ruff check src/codeintel/storage/` |

### Qualitative

- [ ] All tracking modules inherit from `BaseTracking`
- [ ] No duplicate `_now()` functions
- [ ] No duplicate `_serialize_*` / `_deserialize_*` functions
- [ ] No duplicate `_validated_records` methods
- [ ] Single exception file (`exceptions.py`)
- [ ] View files organized by domain
- [ ] `RowMapper` used for all row parsing

---

## 12. Appendix: Code Examples

### A. Complete `helpers/time.py`

```python
"""Timezone-aware datetime utilities for storage operations.

This module provides shared time utilities used across tracking modules.
All timestamps are UTC-aware following the project's datetime hygiene rules.

Example
-------
>>> from codeintel.storage.helpers.time import utc_now
>>> ts = utc_now()
>>> ts.tzinfo is not None
True
"""

from __future__ import annotations

from datetime import UTC, datetime

__all__ = ["utc_now"]


def utc_now() -> datetime:
    """Return current UTC timestamp with timezone info.

    Returns
    -------
    datetime
        Current datetime with UTC timezone attached.

    Examples
    --------
    >>> ts = utc_now()
    >>> ts.tzinfo is not None
    True
    >>> from datetime import UTC
    >>> ts.tzinfo == UTC
    True
    """
    return datetime.now(tz=UTC)
```

### B. Complete `tracking/base.py`

```python
"""Base class for tracking accessors.

This module provides shared infrastructure for pipeline run tracking,
build tracking, and asset tracking. All tracking classes inherit from
BaseTracking to ensure consistent patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.helpers.json import (
    decode_json_dict,
    decode_json_list,
    deserialize_str_tuple,
    encode_json_compact,
    serialize_str_sequence,
)
from codeintel.storage.helpers.time import utc_now

if TYPE_CHECKING:
    from collections.abc import Any, Mapping, Sequence
    from datetime import datetime

    from duckdb import DuckDBPyConnection

    from codeintel.storage.gateway.protocol import StorageGateway

__all__ = ["BaseTracking"]


@dataclass
class BaseTracking:
    """Base class for tracking accessors.

    Provides shared utilities for all tracking classes:
    - Connection and policy backend access
    - Timestamp generation
    - JSON serialization helpers

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
    """

    gateway: StorageGateway

    @property
    def con(self) -> DuckDBPyConnection:
        """Return the underlying DuckDB connection."""
        return self.gateway.con

    @property
    def backend(self) -> DuckDBPolicyBackend:
        """Return the policy backend for bulk operations."""
        return self.gateway.policy

    @staticmethod
    def now() -> datetime:
        """Return current UTC timestamp."""
        return utc_now()

    @staticmethod
    def serialize_list(items: Sequence[str]) -> str:
        """Serialize string sequence to compact JSON array."""
        return serialize_str_sequence(items)

    @staticmethod
    def deserialize_list(raw: str | None) -> tuple[str, ...]:
        """Deserialize JSON array to string tuple."""
        return deserialize_str_tuple(raw)

    @staticmethod
    def serialize_dict(data: Mapping[str, Any] | None) -> str:
        """Serialize dict to compact JSON string."""
        return encode_json_compact(dict(data) if data else {})

    @staticmethod
    def deserialize_dict(raw: str | None) -> dict[str, Any] | None:
        """Deserialize JSON string to dict."""
        return decode_json_dict(raw) if raw else None

    @staticmethod
    def deserialize_json_list(raw: str | None) -> list[Any] | None:
        """Deserialize JSON string to list."""
        return decode_json_list(raw) if raw else None
```

### C. Updated `helpers/__init__.py`

```python
"""Storage helper utilities.

This package provides various helper functions for DuckDB operations.

Submodules
----------
helpers.json
    JSON encode/decode helpers for DuckDB column values.

helpers.time
    Timezone-aware timestamp utilities.

helpers.row_parser
    Generic row-to-dataclass mapping utilities.

helpers.table_key
    Table key parsing utilities.

helpers.profiling
    Docs view profiling utilities (import directly).

helpers.module_index
    Module metadata helpers (import directly).
"""

from __future__ import annotations

from codeintel.storage.errors import DUCKDB_ERRORS
from codeintel.storage.helpers.json import (
    decode_json,
    decode_json_dict,
    decode_json_list,
    deserialize_str_tuple,
    encode_json_compact,
    serialize_str_sequence,
)
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.helpers.time import utc_now

__all__ = [
    "DUCKDB_ERRORS",
    "decode_json",
    "decode_json_dict",
    "decode_json_list",
    "deserialize_str_tuple",
    "encode_json_compact",
    "serialize_str_sequence",
    "split_table_key",
    "utc_now",
]
```

---

## Document History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2025-12-13 | 1.0 | AI Code Review | Initial plan created |

---

## References

- [AGENTS.md](../../AGENTS.md) - Project coding standards
- [Ibis 11 Patterns](../ibis_pandera_implementation.md) - Ibis usage patterns
- [Dataset Contracts](../dataset_contract_migration_summary.md) - Dataset contract system

