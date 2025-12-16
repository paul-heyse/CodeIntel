# Serving Module Rationalization and Enhancement Plan

> **Status**: Implementation Complete (E7 & E13)  
> **Priority**: High  
> **Estimated Effort**: 3-5 days  
> **Dependencies**: None (can proceed independently)  
> **Last Updated**: December 2025
>
> ### Implementation Progress
>
> | Enhancement | Status | Notes |
> |------------|--------|-------|
> | E1-E6 | ✅ Complete | Core FastAPI patterns implemented |
> | E7 | ✅ Complete | Background tasks for metrics/logging |
> | E8-E12 | ✅ Complete | API versioning, middleware, correlation IDs |
> | E13 | ✅ Complete | Export endpoints (JSON/NDJSON/Parquet/Arrow) |
> | E14 | ✅ Complete | API key security dependency |

## Executive Summary

This plan consolidates two complementary improvement tracks for the `serving` module:

1. **Rationalization** (from storage layer alignment): Make `Warehouse` the serving I/O boundary, move pooling to storage, and eliminate per-call gateway creation.
2. **Enhancement** (from FastAPI advanced features): Implement typed dependencies, RFC 9457 errors, typed lifespan state, correlation ID middleware, and safe concurrency guardrails.

**Combined Goals**:
- Reduce serving module complexity by ~20%
- Achieve best-in-class FastAPI patterns
- Improve type safety and IDE support
- Standardize error handling across HTTP/MCP surfaces
- Enable future API versioning
- Preserve strict DuckDB boundary discipline (no `duckdb` imports outside `codeintel.storage`)

---

## Table of Contents

### Part I: Storage Layer Rationalization
1. [Current Architecture](#current-architecture)
2. [Storage Module Reference](#storage-module-reference)
3. [Redundancy Analysis](#redundancy-analysis)
4. [Proposed Architecture](#proposed-architecture)

### Part II: FastAPI Advanced Enhancements
5. [Enhancement Overview](#enhancement-overview)
6. [High Impact Enhancements](#high-impact-enhancements)
   - [E1: Typed Dependency Injection with Annotated](#e1-typed-dependency-injection-with-annotated)
   - [E2: RFC 9457 Problem Details for Errors](#e2-rfc-9457-problem-details-for-errors)
   - [E3: Response Model Configuration](#e3-response-model-configuration)
   - [E4: Pydantic v2 Validation Features](#e4-pydantic-v2-validation-features)
   - [E5: Typed Lifespan State](#e5-typed-lifespan-state)
   - [E10: Correlation ID + Logging Context Middleware](#e10-correlation-id--logging-context-middleware)
   - [E12: Concurrency Guardrails + Timeouts](#e12-concurrency-guardrails--timeouts)
7. [Medium Impact Enhancements](#medium-impact-enhancements)
   - [E6: Custom Exception Handlers](#e6-custom-exception-handlers)
   - [E7: Background Tasks for Metrics/Logging](#e7-background-tasks-for-metricslogging)
   - [E8: API Versioning Strategy](#e8-api-versioning-strategy)
   - [E9: OpenAPI Schema Customization](#e9-openapi-schema-customization)
   - [E11: Middleware Hardening (GZip/CORS/TrustedHost/Timing)](#e11-middleware-hardening-gzipcorstrustedhosttiming)
   - [E13: Optional Streaming/Export Endpoints (NDJSON/Arrow/Parquet)](#e13-optional-streamingexport-endpoints-ndjsonarrowparquet)
   - [E14: Optional API-Key Security Dependency](#e14-optional-api-key-security-dependency)

### Part III: Implementation
8. [Implementation Phases](#implementation-phases)
9. [File-by-File Changes](#file-by-file-changes)
10. [Migration Guide](#migration-guide)
11. [Testing Strategy](#testing-strategy)
12. [Rollback Plan](#rollback-plan)
13. [Success Criteria](#success-criteria)

---

# Part I: Storage Layer Rationalization

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
│   ├── inventory.py               # SchemaInventory (KEEP AS-IS)
│   └── models.py                  # Pydantic models
├── search/
│   ├── __init__.py
│   └── models.py                  # Search models
├── contracts/
│   └── check_operation_contracts.py
├── http/
│   ├── app.py                     # FastAPI factory ← ENHANCEMENT TARGET
│   └── routes/
│       ├── search.py              # ← ENHANCEMENT TARGET
│       └── semantic.py            # ← ENHANCEMENT TARGET
└── mcp/
    ├── app.py                     # FastMCP builder
    └── server.py                  # MCP server factory
```

---

## Storage Module Reference

The storage module provides a clean gateway architecture:

```
storage/
├── gateway/
│   ├── protocol.py                # MinimalGateway, StorageGateway protocols
│   ├── minimal.py                 # MinimalStorageGateway (composition root)
│   ├── config.py                  # StorageConfig with for_readonly()
│   ├── connection.py              # connect()
│   ├── factory.py                 # open_gateway(), open_memory_gateway()
│   └── ephemeral.py               # ephemeral_gateway() context manager
├── warehouse.py                   # Warehouse I/O boundary (read/write/explain)
├── ibis_adapter.py                # IbisGateway
├── duckdb_policy_backend.py       # DuckDBPolicyBackend (execute_sql escape hatch)
└── serving/
    └── search_index.py            # FTS index building
```

### Key Storage Patterns

1. **Warehouse** is the serving-facing I/O boundary:
   ```python
   from codeintel.storage.warehouse import Warehouse

   warehouse = Warehouse(gateway)
   expr = warehouse.read("docs.v_function_summary", snapshot=snapshot)
   plan = warehouse.explain_table("docs.v_function_summary", limit=50)
   ```

2. **MinimalStorageGateway** is the composition root (for raw connections):
   ```python
   class MinimalStorageGateway:
       @property
       def con(self) -> DuckDBPyConnection: ...
       @property
       def ibis(self) -> IbisGateway: ...      # Lazy, cached
       @property
       def policy(self) -> DuckDBPolicyBackend: ...  # Lazy, cached
   ```

3. **IbisGateway.table()** handles qualified names properly:
   ```python
   def table(self, table_name: str) -> it.Table:
       if "." in table_name:
           database, name = table_name.split(".", 1)
           return self.con.table(name, database=database)
       return self.con.table(table_name)
   ```

4. **StorageConfig.for_readonly()** is designed for serving:
   ```python
   @classmethod
   def for_readonly(cls, db_path: Path) -> StorageConfig:
       return cls(db_path=db_path, read_only=True, ...)
   ```

5. **Catalog/introspection checks should be parameterized via the policy backend**:
   ```python
   row = warehouse.gateway.policy.execute_sql(
       "SELECT 1 FROM information_schema.schemata WHERE schema_name = ? LIMIT 1",
       [schema],
   ).fetchone()
   exists = row is not None
   ```
   Note: repository DuckDB-boundary tests are substring-based; avoid importing symbols whose
   import lines contain `"import duckdb"` outside `codeintel.storage`.

---

## Redundancy Analysis

### R1. Connection Pool Yields Raw Connections

**Location**: `serving/db/pool.py`

**Issue**: Pool yields raw `DuckDBConnection`. Callers must then wrap it repeatedly, recreating gateway/ibis/policy objects and blurring the “Warehouse is the I/O boundary” principle.

**Recommendation**: Move pool to storage and yield a request-scoped `Warehouse` (or a context containing `Warehouse` + snapshot metadata) directly.

### R2. Ad-hoc Ibis Connection Creation

**Location**: `serving/semantic/kernel.py:248-257`

**Issue**: Creates a new Ibis backend connection for every query instead of using a request-scoped cached `IbisGateway` from the warehouse/gateway.

**Recommendation**: Pass `Warehouse` (or `StorageGateway`) through the call chain and use `warehouse.gateway.ibis.con`.

### R3. Gateway Creation Per SQL Call

**Location**: `serving/semantic/kernel.py` at lines 234, 414, 489

**Issue**: Creates a new gateway wrapper for every SQL execution.

**Recommendation**: Create `Warehouse` once per acquired pool handle and reuse throughout the request lifecycle.

### R4. Raw SQL for Schema Existence Check

**Location**: `serving/semantic/kernel.py:503-507`

**Issue**: Duplicates schema existence logic.

**Recommendation**: Keep the check parameterized via the policy backend (`warehouse.gateway.policy.execute_sql(...)`) and/or add a storage-owned helper with a non-`duckdb_*` name (e.g., `codeintel.storage.introspection.schema_exists`) to avoid tripping the repository’s DuckDB boundary tests.

### R5. Repeated JSON Artifact Loads Per Request

**Location**: `serving/semantic/kernel.py` (`_load_registry`, `_load_inventory`, `_load_buildspec`)

**Issue**: Registry/inventory/buildspec are read from disk on every request even though they are snapshot-scoped and already hot-swapped by `ServingDBManager`.

**Recommendation**: Introduce a `ServingSnapshotContext` cached in memory and refreshed when the pointer changes:
- `pointer: ServingSnapshotPointer`
- `registry: SemanticRegistry`
- `inventory: SchemaInventory`
- `buildspec: BuildSpec`

`SemanticQueryKernel` should read from context instead of rereading JSON.

### R6. FTS Read Logic Split Across Serving/Storage

**Location**: `serving/semantic/kernel.py` and `storage/serving/search_index.py`

**Issue**: Storage owns FTS “build/ensure”, but serving owns “detect/dispatch”, so DuckDB/FTS details leak into serving.

**Recommendation**: Move FTS schema naming + availability helpers into `codeintel.storage.serving.search_index` and have serving call that storage-owned API.

---

## Proposed Architecture

### Target State (Post-Rationalization + Enhancement)

```
storage/
├── gateway/
│   ├── protocol.py                # Add: ReadPoolWarehouse protocol (optional)
│   ├── minimal.py                 # MinimalStorageGateway (unchanged)
│   ├── config.py                  # Add: PoolConfig
│   ├── connection.py              # Unchanged
│   ├── pool.py                    # NEW: ReadPoolWarehouse implementation (yields Warehouse)
│   ├── ephemeral.py               # Unchanged
│   └── factory.py                 # Unchanged
├── warehouse.py                   # Warehouse I/O boundary (read/exists/count/materialize/explain)
└── serving/
    └── search_index.py            # FTS index helpers

serving/
├── __init__.py                    # Update exports
├── settings.py                    # Unchanged
├── db/
│   ├── __init__.py                # Update exports
│   ├── manager.py                 # REFACTOR: yield Warehouse (+ snapshot context)
│   ├── pointer.py                 # Unchanged
│   └── pool.py                    # THIN RE-EXPORT with deprecation
├── semantic/
│   ├── __init__.py                # Unchanged
│   ├── kernel.py                  # REFACTOR: use Warehouse/context pattern
│   ├── query_builder.py           # Unchanged
│   ├── registry.py                # Unchanged
│   ├── inventory.py               # Unchanged
│   └── models.py                  # ENHANCE: Pydantic v2 features
├── search/
│   ├── models.py                  # ENHANCE: Pydantic v2 features
├── contracts/                     # Unchanged
├── http/
│   ├── __init__.py                # NEW: Shared types and dependencies
│   ├── app.py                     # ENHANCE: Typed lifespan, exception handlers
│   ├── errors.py                  # NEW: RFC 9457 Problem Details
│   ├── dependencies.py            # NEW: Annotated dependency definitions
│   ├── middleware.py              # NEW: Correlation ID + timing middleware
│   └── routes/
│       ├── v1/                    # NEW: Versioned routes
│       │   ├── __init__.py
│       │   ├── search.py          # ENHANCE: Typed deps, response models
│       │   └── semantic.py        # ENHANCE: Typed deps, response models
│       └── __init__.py            # Router aggregation
└── mcp/                           # Unchanged (uses kernel directly)
```

### Layer Responsibilities

| Layer | Responsibility | Does NOT Own |
|-------|---------------|--------------|
| `storage.gateway` | Connection lifecycle, pooling, gateway creation | Query semantics |
| `storage.warehouse` | Single I/O boundary for storage ops (read/exists/count/materialize/explain) | HTTP concerns |
| `storage.serving` | FTS index building | Query execution |
| `serving.db` | Snapshot pointer, hot-swap coordination, snapshot context cache | Connection pooling primitives |
| `serving.semantic` | Query building, result extraction | Connection management |
| `serving.http` | HTTP surfaces, error handling, API versioning | Business logic |
| `serving.mcp` | MCP tool surfaces | HTTP concerns |

---

# Part II: FastAPI Advanced Enhancements

## Enhancement Overview

The following enhancements integrate advanced FastAPI features to achieve best-in-class API design:

| ID | Enhancement | Impact | Effort | Dependencies |
|----|-------------|--------|--------|--------------|
| E1 | Typed Dependency Injection with `Annotated` | High | Low | None |
| E2 | RFC 9457 Problem Details for Errors | High | Medium | E6 |
| E3 | Response Model Configuration | High | Low | E4 |
| E4 | Pydantic v2 Validation Features | High | Low | None |
| E5 | Typed Lifespan State | High | Medium | E1 |
| E6 | Custom Exception Handlers | Medium | Low | None |
| E7 | Background Tasks for Metrics/Logging | Medium | Low | None |
| E8 | API Versioning Strategy | Medium | Medium | E1 |
| E9 | OpenAPI Schema Customization | Medium | Low | E4 |
| E10 | Correlation ID + Logging Context Middleware | High | Low | E2 |
| E11 | Middleware Hardening (GZip/CORS/TrustedHost/Timing) | Medium | Low | None |
| E12 | Concurrency Guardrails + Timeouts | High | Medium | E1 |
| E13 | Optional Streaming/Export Endpoints (NDJSON/Arrow/Parquet) | Medium | Medium | E3 |
| E14 | Optional API-Key Security Dependency | Medium | Low | E1 |

---

## High Impact Enhancements

### E1: Typed Dependency Injection with Annotated

**Current State** (`serving/http/routes/semantic.py`):
```python
from fastapi import Depends

def get_kernel() -> SemanticQueryKernel:
    raise NotImplementedError("Override in app factory")

_KERNEL_DEPENDENCY = Depends(get_kernel)

@router.get("/views")
async def list_views(kernel: SemanticQueryKernel = _KERNEL_DEPENDENCY) -> dict[str, object]:
    return kernel.catalog()
```

**Issues**:
- Default value pattern is non-idiomatic
- Type checker doesn't understand runtime override
- Dependency definition scattered across modules

**Enhanced State** (`serving/http/dependencies.py`):
```python
"""Shared typed dependencies for HTTP routes.

All dependencies are defined as Annotated types for reuse across routes.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from codeintel.serving.semantic.kernel import SemanticQueryKernel


def _get_kernel(request: Request) -> SemanticQueryKernel:
    """Extract kernel from application state.
    
    Returns
    -------
    SemanticQueryKernel
        The semantic query kernel from app state.
    
    Raises
    ------
    RuntimeError
        If kernel not configured in app state.
    """
    kernel = getattr(request.app.state, "kernel", None)
    if kernel is None:
        msg = "SemanticQueryKernel not configured"
        raise RuntimeError(msg)
    return kernel


# Reusable typed dependency
Kernel = Annotated[SemanticQueryKernel, Depends(_get_kernel)]
```

**Enhanced Route** (`serving/http/routes/v1/semantic.py`):
```python
from codeintel.serving.http.dependencies import Kernel

@router.get("/views")
async def list_views(kernel: Kernel) -> dict[str, object]:
    """List available semantic views."""
    return kernel.catalog()
```

**Benefits**:
- Clean, reusable type alias
- Better IDE autocomplete
- Centralized dependency definitions
- No more `= Depends()` in signatures

---

### E2: RFC 9457 Problem Details for Errors

**Current State**:
```python
@router.get("/views/{view_id}")
async def describe_view(view_id: str, kernel: Kernel) -> dict[str, object]:
    try:
        return kernel.describe(view_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
```

**Issues**:
- Non-standard error format
- No machine-readable error types
- Inconsistent across endpoints
- No stable correlation IDs (per-request)

**Enhanced State** (`serving/http/errors.py`):
```python
"""RFC 9457 Problem Details implementation for serving errors.

This module provides structured error responses conforming to RFC 9457,
enabling machine-readable error handling for API clients.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class ProblemType(StrEnum):
    """Standard problem types for the serving API."""
    
    VIEW_NOT_FOUND = "/problems/view-not-found"
    INVALID_QUERY = "/problems/invalid-query"
    VALIDATION_ERROR = "/problems/validation-error"
    INTERNAL_ERROR = "/problems/internal-error"
    SEARCH_UNAVAILABLE = "/problems/search-unavailable"


class ProblemDetail(BaseModel):
    """RFC 9457 Problem Details response model.
    
    Attributes
    ----------
    type
        URI reference identifying the problem type.
    title
        Short human-readable summary.
    status
        HTTP status code.
    detail
        Human-readable explanation specific to this occurrence.
    instance
        URI reference identifying this specific occurrence.
    correlation_id
        Request correlation ID for tracing.
    """
    
    type: str = Field(
        default=ProblemType.INTERNAL_ERROR,
        description="URI reference identifying the problem type",
    )
    title: str = Field(description="Short human-readable summary")
    status: int = Field(description="HTTP status code")
    detail: str | None = Field(
        default=None,
        description="Human-readable explanation specific to this occurrence",
    )
    instance: str | None = Field(
        default=None,
        description="URI reference identifying this specific occurrence",
    )
    correlation_id: str = Field(description="Request correlation ID for tracing")
    
    # Extension fields
    errors: list[dict[str, Any]] | None = Field(
        default=None,
        description="Detailed validation errors",
    )

    model_config = {"extra": "allow"}


class ServingError(Exception):
    """Base exception for serving errors with Problem Details support.
    
    Parameters
    ----------
    problem_type
        The problem type URI.
    title
        Short summary.
    status
        HTTP status code.
    detail
        Detailed explanation.
    errors
        Optional list of sub-errors (for validation).
    """
    
    def __init__(
        self,
        problem_type: ProblemType,
        title: str,
        status: int,
        detail: str | None = None,
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(detail or title)
        self.problem_type = problem_type
        self.title = title
        self.status = status
        self.detail = detail
        self.errors = errors


class ViewNotFoundError(ServingError):
    """Raised when a requested view does not exist."""
    
    def __init__(self, view_id: str) -> None:
        super().__init__(
            problem_type=ProblemType.VIEW_NOT_FOUND,
            title="View Not Found",
            status=404,
            detail=f"Semantic view '{view_id}' does not exist",
        )
        self.view_id = view_id


class InvalidQueryError(ServingError):
    """Raised when a query is malformed or invalid."""
    
    def __init__(self, detail: str) -> None:
        super().__init__(
            problem_type=ProblemType.INVALID_QUERY,
            title="Invalid Query",
            status=400,
            detail=detail,
        )


class SearchUnavailableError(ServingError):
    """Raised when FTS search is not available."""
    
    def __init__(self) -> None:
        super().__init__(
            problem_type=ProblemType.SEARCH_UNAVAILABLE,
            title="Search Unavailable",
            status=503,
            detail="Full-text search index is not available for this snapshot",
        )


def problem_response(
    request: Request,
    error: ServingError,
    *,
    correlation_id: str,
) -> JSONResponse:
    """Create a Problem Details JSON response.
    
    Parameters
    ----------
    request
        The incoming request (for instance URI).
    error
        The serving error.
    
    Returns
    -------
    JSONResponse
        RFC 9457 compliant error response.
    """
    problem = ProblemDetail(
        type=error.problem_type,
        title=error.title,
        status=error.status,
        detail=error.detail,
        instance=str(request.url),
        correlation_id=correlation_id,
        errors=error.errors,
    )
    return JSONResponse(
        status_code=error.status,
        content=problem.model_dump(mode="json", exclude_none=True),
        media_type="application/problem+json",
    )
```

**Enhanced Route**:
```python
from codeintel.serving.http.errors import ViewNotFoundError

@router.get("/views/{view_id}")
async def describe_view(view_id: str, kernel: Kernel) -> dict[str, object]:
    """Describe a semantic view."""
    try:
        return kernel.describe(view_id)
    except KeyError as exc:
        raise ViewNotFoundError(view_id) from exc
```

**Benefits**:
- Machine-readable error types
- Correlation IDs for debugging (provided by middleware; stable per request)
- Standard `application/problem+json` content type
- Extensible error model

---

### E3: Response Model Configuration

**Current State**:
```python
@router.post("/query")
async def query_view(
    payload: dict[str, object] = _QUERY_BODY,
    kernel: Kernel = _KERNEL_DEPENDENCY,
) -> dict[str, object]:
    """Execute a semantic view query."""
    # ...
```

**Issues**:
- Return type `dict[str, object]` provides no schema
- No serialization configuration
- Client doesn't know response structure

**Enhanced State** (`serving/semantic/models.py`):
```python
from pydantic import BaseModel, ConfigDict, Field


class SemanticQueryResponse(BaseModel):
    """Response for semantic layer queries.
    
    Attributes
    ----------
    view_id
        The queried view identifier.
    columns
        List of column names in result.
    rows
        Query result rows.
    total_count
        Total matching rows (before limit).
    truncated
        Whether results were truncated.
    """
    
    model_config = ConfigDict(
        # Exclude fields that weren't explicitly set
        exclude_unset=True,
        # Use enum values, not names
        use_enum_values=True,
        # Validate on assignment for safety
        validate_assignment=True,
    )
    
    view_id: str
    columns: list[str]
    rows: list[dict[str, object]]
    total_count: int | None = None
    truncated: bool = False
    
    # Computed at serialization time
    row_count: int = Field(default=0, exclude=True)
    
    def model_post_init(self, __context: object) -> None:
        """Set computed fields after initialization."""
        object.__setattr__(self, "row_count", len(self.rows))


class SemanticCatalogResponse(BaseModel):
    """Response for view catalog listing."""
    
    model_config = ConfigDict(extra="forbid")
    
    views: list[SemanticViewSummary]
    total_count: int


class SemanticViewSummary(BaseModel):
    """Summary of a semantic view for catalog listing."""
    
    model_config = ConfigDict(extra="forbid")
    
    view_id: str
    description: str | None = None
    columns: list[str]
    source_table: str | None = None
```

**Enhanced Route**:
```python
@router.get(
    "/views",
    response_model=SemanticCatalogResponse,
    response_model_exclude_unset=True,
)
async def list_views(kernel: Kernel) -> SemanticCatalogResponse:
    """List available semantic views."""
    catalog = kernel.catalog()
    return SemanticCatalogResponse.model_validate(catalog)
```

**Benefits**:
- Auto-generated OpenAPI schemas
- Client code generation support
- Consistent serialization
- Validation on output

---

### E4: Pydantic v2 Validation Features

**Current State** (`serving/semantic/models.py`):
```python
class SemanticQueryRequest(BaseModel):
    view_id: str
    filters: list[dict[str, object]] | None = None
    columns: list[str] | None = None
    limit: int | None = None
    offset: int | None = None
```

**Issues**:
- No field validation
- No custom validators
- No computed fields
- Loose filter typing

**Enhanced State**:
```python
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class FilterOperator(StrEnum):
    """Supported filter operators."""
    
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GE = "ge"
    LT = "lt"
    LE = "le"
    IN = "in"
    NOT_IN = "not_in"
    LIKE = "like"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class SemanticFilter(BaseModel):
    """A single filter condition for semantic queries.
    
    Attributes
    ----------
    column
        Column name to filter on.
    operator
        Comparison operator.
    value
        Value to compare against (not required for null checks).
    """
    
    model_config = ConfigDict(extra="forbid")
    
    column: str = Field(min_length=1, max_length=128)
    operator: FilterOperator
    value: str | int | float | bool | list[str | int | float] | None = None
    
    @model_validator(mode="after")
    def validate_value_for_operator(self) -> SemanticFilter:
        """Ensure value is appropriate for operator."""
        null_ops = {FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL}
        if self.operator in null_ops and self.value is not None:
            msg = f"Operator {self.operator} does not accept a value"
            raise ValueError(msg)
        if self.operator not in null_ops and self.value is None:
            msg = f"Operator {self.operator} requires a value"
            raise ValueError(msg)
        return self


# Constrained types for query parameters
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
QueryLimit = Annotated[int, Field(gt=0, le=10000)]
ColumnName = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")]


class SemanticQueryRequest(BaseModel):
    """Request model for semantic layer queries.
    
    Attributes
    ----------
    view_id
        Target semantic view identifier.
    filters
        Optional list of filter conditions.
    columns
        Optional list of columns to return (default: all).
    limit
        Maximum rows to return (default: 1000, max: 10000).
    offset
        Number of rows to skip (for pagination).
    """
    
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )
    
    view_id: str = Field(min_length=1, max_length=256)
    filters: list[SemanticFilter] | None = None
    columns: list[ColumnName] | None = None
    limit: QueryLimit = 1000
    offset: NonNegativeInt = 0
    
    @field_validator("view_id")
    @classmethod
    def validate_view_id(cls, v: str) -> str:
        """Validate view_id format."""
        if not v.replace("_", "").replace(".", "").isalnum():
            msg = "view_id must be alphanumeric with underscores and dots"
            raise ValueError(msg)
        return v
    
    @model_validator(mode="after")
    def validate_pagination(self) -> SemanticQueryRequest:
        """Ensure pagination parameters are sensible."""
        if self.offset > 0 and self.limit > 5000:
            msg = "When using offset, limit must be <= 5000"
            raise ValueError(msg)
        return self
```

**Benefits**:
- Strong input validation
- Self-documenting constraints
- Type-safe filter operators
- Cross-field validation

---

### E5: Typed Lifespan State

**Current State** (`serving/http/app.py`):
```python
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    await db_manager.start()
    try:
        yield
    finally:
        await db_manager.stop()

app = FastAPI(lifespan=lifespan)
app.state.kernel = kernel
app.state.db_manager = db_manager
```

**Issues**:
- `app.state` is untyped `State` object
- No IDE autocomplete for state attributes
- Runtime errors if attribute missing
- State management scattered

**Enhanced State** (`serving/http/state.py`):
```python
"""Typed application state for the serving HTTP layer.

This module provides a typed state container that replaces FastAPI's
untyped `app.state` with compile-time type safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.serving.db.manager import ServingDBManager
    from codeintel.serving.semantic.kernel import SemanticQueryKernel


@dataclass
class ServingState:
    """Typed container for serving application state.
    
    Attributes
    ----------
    kernel
        The semantic query kernel.
    db_manager
        The database manager with hot-swap support.
    """
    
    kernel: SemanticQueryKernel
    db_manager: ServingDBManager


# Key used to store state in app.state
SERVING_STATE_KEY = "_serving_state"
```

**Enhanced App Factory** (`serving/http/app.py`):
```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request

from codeintel.serving.http.state import SERVING_STATE_KEY, ServingState


def get_serving_state(request: Request) -> ServingState:
    """Extract typed state from request.
    
    Parameters
    ----------
    request
        The incoming HTTP request.
    
    Returns
    -------
    ServingState
        The typed application state.
    
    Raises
    ------
    RuntimeError
        If state not configured.
    """
    state = getattr(request.app.state, SERVING_STATE_KEY, None)
    if state is None:
        msg = "ServingState not configured"
        raise RuntimeError(msg)
    return state


def create_serving_app(
    settings: ServingSettings | None = None,
    *,
    mount_mcp: bool = True,
) -> FastAPI:
    """Create the serving FastAPI application.
    
    Parameters
    ----------
    settings
        Serving configuration (default: from environment).
    mount_mcp
        Whether to mount MCP server at /mcp.
    
    Returns
    -------
    FastAPI
        Configured application instance.
    """
    cfg = settings or ServingSettings.from_env()

    db_manager = ServingDBManager(
        pointer_path=cfg.serve_dir / "current.json",
        pool_cfg=PoolConfig(size=cfg.pool_size),
        poll_interval_s=cfg.poll_interval_s,
    )
    kernel = SemanticQueryKernel(db=db_manager, settings=cfg)
    
    # Create typed state container
    serving_state = ServingState(kernel=kernel, db_manager=db_manager)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        # Store typed state
        setattr(app.state, SERVING_STATE_KEY, serving_state)
        
        await db_manager.start()
        try:
            yield
        finally:
            await db_manager.stop()

    app = FastAPI(
        title="CodeIntel Serving",
        description="Semantic layer API for CodeIntel",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Configure exception handlers (E6)
    configure_exception_handlers(app)
    
    # Include versioned routers (E8)
    app.include_router(v1_router, prefix="/v1")

    # ... rest of setup
    return app
```

**Enhanced Dependencies** (`serving/http/dependencies.py`):
```python
from typing import Annotated

from fastapi import Depends, Request

from codeintel.serving.http.state import ServingState, get_serving_state
from codeintel.serving.semantic.kernel import SemanticQueryKernel


def _get_state(request: Request) -> ServingState:
    """Extract typed state from request."""
    return get_serving_state(request)


def _get_kernel(state: Annotated[ServingState, Depends(_get_state)]) -> SemanticQueryKernel:
    """Extract kernel from typed state."""
    return state.kernel


# Reusable typed dependencies
State = Annotated[ServingState, Depends(_get_state)]
Kernel = Annotated[SemanticQueryKernel, Depends(_get_kernel)]
```

**Benefits**:
- Compile-time type checking for state
- IDE autocomplete for state attributes
- Single source of truth for state structure
- Easier testing via state injection

---

### E10: Correlation ID + Logging Context Middleware

Add a small middleware that:
- Accepts an inbound `X-Correlation-Id` header (or generates one).
- Stores it on request state (and optionally a `contextvars.ContextVar`) so handlers, exception
  handlers, and background tasks can reuse the same ID.
- Echoes the correlation ID back in the response headers, and optionally sets a process-time header.

This enables:
- Stable correlation IDs in RFC 9457 error responses (E2).
- Structured logging with per-request context (use `extra={"correlation_id": ...}` at minimum).

**Sketch** (`serving/http/middleware.py`):
```python
from __future__ import annotations

import contextvars
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Final

from fastapi import FastAPI, Request, Response

CORRELATION_ID_HEADER: Final[str] = "X-Correlation-Id"
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "codeintel_serving_correlation_id",
    default="",
)


def get_correlation_id(request: Request) -> str:
    value = getattr(request.state, "correlation_id", "")
    if isinstance(value, str) and value:
        return value
    return correlation_id_var.get() or "unknown"


def install_serving_middlewares(app: FastAPI) -> None:
    @app.middleware("http")
    async def correlation_id_and_timing(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        correlation_id = request.headers.get(CORRELATION_ID_HEADER) or str(uuid.uuid4())
        token = correlation_id_var.set(correlation_id)
        request.state.correlation_id = correlation_id
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
        finally:
            correlation_id_var.reset(token)
        response.headers.setdefault(CORRELATION_ID_HEADER, correlation_id)
        response.headers["X-Process-Time"] = f"{(time.perf_counter() - start):0.4f}"
        return response
```

---

### E12: Concurrency Guardrails + Timeouts

Serving uses a read-only DuckDB pool and executes blocking I/O (DuckDB + Ibis). To keep FastAPI
responsive under concurrency:

- Ensure blocking kernel calls run off the event loop:
  - Prefer `def` endpoints (FastAPI runs them in a threadpool), or
  - In `async def` endpoints, wrap heavy calls with `starlette.concurrency.run_in_threadpool`.
- Add a per-process limiter (semaphore) sized to the pool to avoid overcommitting threads/handles.
- Optional: add a soft timeout; on timeout, close (not release) the underlying DB handle so it
  doesn’t return to the pool in an unknown state.

This can be implemented as a dependency providing a shared semaphore and a helper like
`run_bounded(...)` that acquires the semaphore and executes the blocking callable in the threadpool.

---
## Medium Impact Enhancements

### E6: Custom Exception Handlers

**Implementation** (`serving/http/app.py`):
```python
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from codeintel.serving.http.errors import (
    ProblemDetail,
    ProblemType,
    ServingError,
    problem_response,
)
from codeintel.serving.http.middleware import get_correlation_id


def configure_exception_handlers(app: FastAPI) -> None:
    """Configure centralized exception handlers.
    
    Parameters
    ----------
    app
        FastAPI application to configure.
    """

    logger = logging.getLogger(__name__)
    
    @app.exception_handler(ServingError)
    async def serving_error_handler(
        request: Request,
        exc: ServingError,
    ) -> JSONResponse:
        """Handle serving-specific errors with Problem Details."""
        return problem_response(request, exc, correlation_id=get_correlation_id(request))
    
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Handle request validation errors with Problem Details."""
        errors = [
            {
                "loc": list(err["loc"]),
                "msg": err["msg"],
                "type": err["type"],
            }
            for err in exc.errors()
        ]
        problem = ProblemDetail(
            type=ProblemType.VALIDATION_ERROR,
            title="Validation Error",
            status=422,
            detail="Request validation failed",
            instance=str(request.url),
            correlation_id=get_correlation_id(request),
            errors=errors,
        )
        return JSONResponse(
            status_code=422,
            content=problem.model_dump(mode="json", exclude_none=True),
            media_type="application/problem+json",
        )
    
    @app.exception_handler(Exception)
    async def unhandled_error_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle unexpected errors with Problem Details."""
        logger.exception("Unhandled error", extra={"path": str(request.url)})
        
        problem = ProblemDetail(
            type=ProblemType.INTERNAL_ERROR,
            title="Internal Server Error",
            status=500,
            detail="An unexpected error occurred",
            instance=str(request.url),
            correlation_id=get_correlation_id(request),
        )
        return JSONResponse(
            status_code=500,
            content=problem.model_dump(mode="json", exclude_none=True),
            media_type="application/problem+json",
        )
```

**Benefits**:
- Consistent error responses across all endpoints
- Automatic Problem Details for validation errors
- Safe handling of unhandled exceptions
- Centralized logging for errors

---

### E7: Background Tasks for Metrics/Logging

**Implementation** (`serving/http/routes/v1/semantic.py`):
```python
import logging
import time

from fastapi import BackgroundTasks, Request
from starlette.concurrency import run_in_threadpool

from codeintel.serving.http.dependencies import Kernel
from codeintel.serving.http.middleware import get_correlation_id

logger = logging.getLogger("codeintel.serving.metrics")

async def _log_query_metrics(
    view_id: str,
    row_count: int,
    duration_ms: float,
    correlation_id: str,
) -> None:
    """Log query metrics in background.
    
    Parameters
    ----------
    view_id
        The queried view.
    row_count
        Number of rows returned.
    duration_ms
        Query duration in milliseconds.
    correlation_id
        Request correlation ID.
    """
    logger.info(
        "query_executed",
        extra={
            "view_id": view_id,
            "row_count": row_count,
            "duration_ms": duration_ms,
            "correlation_id": correlation_id,
        },
    )


@router.post(
    "/query",
    response_model=SemanticQueryResponse,
)
async def query_view(
    query: SemanticQueryRequest,
    kernel: Kernel,
    background: BackgroundTasks,
    request: Request,
) -> SemanticQueryResponse:
    """Execute a semantic view query."""
    correlation_id = get_correlation_id(request)
    start = time.perf_counter()

    response: SemanticQueryResponse | None = None
    try:
        response = await run_in_threadpool(kernel.query, query)
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        background.add_task(
            _log_query_metrics,
            view_id=query.view_id,
            row_count=len(response.rows) if response else 0,
            duration_ms=duration_ms,
            correlation_id=correlation_id,
        )
```

**Benefits**:
- Non-blocking metrics collection
- Reduced response latency
- Clean separation of concerns
- Easy to extend for additional background work

---

### E8: API Versioning Strategy

**Implementation** (`serving/http/routes/__init__.py`):
```python
"""Route aggregation with API versioning.

This module provides versioned routers for the serving API.
The default version (v1) is also mounted at the root for backwards compatibility.
"""

from __future__ import annotations

from fastapi import APIRouter

from codeintel.serving.http.routes.v1 import router as v1_router

# Versioned router
router = APIRouter()
router.include_router(v1_router, prefix="/v1", tags=["v1"])

# Root alias for backwards compatibility (points to v1)
router.include_router(v1_router, tags=["default"])

__all__ = ["router", "v1_router"]
```

**V1 Routes** (`serving/http/routes/v1/__init__.py`):
```python
"""V1 API routes for semantic serving."""

from __future__ import annotations

from fastapi import APIRouter

from codeintel.serving.http.routes.v1.search import router as search_router
from codeintel.serving.http.routes.v1.semantic import router as semantic_router

router = APIRouter()
router.include_router(semantic_router, prefix="/semantic", tags=["semantic"])
router.include_router(search_router, prefix="/search", tags=["search"])

__all__ = ["router"]
```

**App Integration** (`serving/http/app.py`):
```python
from codeintel.serving.http.routes import router as api_router

app.include_router(api_router)
# Routes available at:
# - /v1/semantic/views (versioned)
# - /semantic/views (root alias, same as v1)
```

**Benefits**:
- Future-proof API evolution
- Non-breaking changes for existing clients
- Clear versioning in OpenAPI docs
- Easy to add v2 when needed

---

### E9: OpenAPI Schema Customization

**Implementation** (`serving/http/app.py`):
```python
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def custom_openapi(app: FastAPI) -> dict[str, object]:
    """Generate customized OpenAPI schema.
    
    Parameters
    ----------
    app
        FastAPI application.
    
    Returns
    -------
    dict
        OpenAPI schema with customizations.
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="CodeIntel Serving API",
        version="1.0.0",
        summary="Semantic layer and search API for CodeIntel analytics",
        description="""
## Overview

The CodeIntel Serving API provides semantic access to code intelligence data.

### Key Features

- **Semantic Views**: Query pre-defined analytical views with filtering and pagination
- **Full-Text Search**: Search code metadata with relevance ranking
- **Hot-Swap**: Zero-downtime updates when new snapshots are published

### Error Handling

All errors are returned as RFC 9457 Problem Details with content type `application/problem+json`.

### Rate Limits

- Default: 100 requests/minute
- Burst: 20 requests/second
        """,
        routes=app.routes,
        tags=[
            {
                "name": "semantic",
                "description": "Semantic view queries and metadata",
            },
            {
                "name": "search",
                "description": "Full-text search over code metadata",
            },
            {
                "name": "health",
                "description": "Health and status endpoints",
            },
        ],
    )
    
    # Add Problem Details schema to components
    openapi_schema["components"]["schemas"]["ProblemDetail"] = {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "format": "uri",
                "description": "URI reference identifying the problem type",
            },
            "title": {
                "type": "string",
                "description": "Short human-readable summary",
            },
            "status": {
                "type": "integer",
                "description": "HTTP status code",
            },
            "detail": {
                "type": "string",
                "description": "Human-readable explanation",
            },
            "instance": {
                "type": "string",
                "format": "uri",
                "description": "URI reference for this occurrence",
            },
            "correlation_id": {
                "type": "string",
                "description": "Request correlation ID for tracing",
            },
        },
        "required": ["type", "title", "status"],
    }
    
    # Add example responses
    openapi_schema["components"]["responses"] = {
        "NotFound": {
            "description": "Resource not found",
            "content": {
                "application/problem+json": {
                    "schema": {"$ref": "#/components/schemas/ProblemDetail"},
                    "example": {
                        "type": "/problems/view-not-found",
                        "title": "View Not Found",
                        "status": 404,
                        "detail": "Semantic view 'unknown_view' does not exist",
                        "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
                    },
                },
            },
        },
        "ValidationError": {
            "description": "Request validation failed",
            "content": {
                "application/problem+json": {
                    "schema": {"$ref": "#/components/schemas/ProblemDetail"},
                    "example": {
                        "type": "/problems/validation-error",
                        "title": "Validation Error",
                        "status": 422,
                        "detail": "Request validation failed",
                        "errors": [
                            {
                                "loc": ["body", "limit"],
                                "msg": "Input should be less than or equal to 10000",
                                "type": "less_than_equal",
                            },
                        ],
                    },
                },
            },
        },
    }
    
    app.openapi_schema = openapi_schema
    return openapi_schema


def create_serving_app(...) -> FastAPI:
    # ... existing setup ...
    
    # Override OpenAPI schema generator
    app.openapi = lambda: custom_openapi(app)
    
    return app
```

**Benefits**:
- Rich API documentation
- Problem Details in schema
- Example responses for each error type
- Better client code generation

---

### E11: Middleware Hardening (GZip/CORS/TrustedHost/Timing)

Add a small, explicit middleware stack for “default safe” serving behavior:

- `GZipMiddleware` for large JSON responses (tunable minimum size).
- Optional `CORSMiddleware` (only if this API is called from browsers).
- `TrustedHostMiddleware` to reject invalid `Host` headers in production deployments.
- Request timing headers (already covered in E10; keep those in one place).

Wire this in the app factory so behavior is consistent for HTTP and does not leak into the
kernel/business logic.

---

### E13: Optional Streaming/Export Endpoints (NDJSON/Arrow/Parquet)

If clients need large resultsets (beyond “interactive JSON”), add opt-in endpoints that stream:

- NDJSON (`application/x-ndjson`) for row streams.
- Arrow IPC / Parquet downloads for analytical clients.

This aligns well with DuckDB/Arrow strengths and avoids buffering huge payloads in memory. When
streaming, be mindful of dependency teardown semantics (see `FastAPI_advanced.md`): don’t stream
while holding resources that will be cleaned up immediately after the response is returned.

---

### E14: Optional API-Key Security Dependency

Even for “internal-only” deployments, add an optional API-key dependency so operators can lock down
the HTTP surface without rewriting routes later. Apply it as a router-level dependency and integrate
it with RFC 9457 errors (E2) for consistent responses.

# Part III: Implementation

## Implementation Phases

### Phase 1: Foundation (Day 1)

**Goal**: Establish core infrastructure for both rationalization and enhancement.

#### 1.1 Create ReadPoolWarehouse in Storage (2-3 hours)

Create `storage/gateway/pool.py` with `PoolConfig` and `ReadPoolWarehouse` yielding a request-scoped `Warehouse`.

```python
# See full implementation in Rationalization section above
```

#### 1.2 Create HTTP Infrastructure (1-2 hours)

Create new files:
- `serving/http/state.py` - Typed state container
- `serving/http/errors.py` - RFC 9457 Problem Details
- `serving/http/dependencies.py` - Typed dependencies
- `serving/http/middleware.py` - Correlation ID + timing middleware (E10)

### Phase 2: Refactor Core Components (Day 1-2)

#### 2.1 Refactor ServingDBManager (1 hour)

Update `serving/db/manager.py` to yield `Warehouse` (and optionally a cached `ServingSnapshotContext`) instead of raw connections.

#### 2.2 Refactor SemanticQueryKernel (1-2 hours)

Update `serving/semantic/kernel.py`:
- Use Warehouse/context pattern throughout
- Avoid importing `duckdb_schema_exists` into serving (DuckDB boundary tests are substring-based)
- Update method signatures to accept `warehouse: Warehouse` (or context containing it)
- Use `warehouse.gateway.policy.execute_sql(...)` for parameterized execution
- Move snapshot-scoped JSON artifact loads behind `ServingSnapshotContext` caching (R5)
- Move FTS schema naming/availability helpers behind `codeintel.storage.serving.search_index` (R6)

#### 2.3 Enhance Pydantic Models (1 hour)

Update `serving/semantic/models.py` and `serving/search/models.py`:
- Add field validators
- Add constrained types
- Add model configuration

### Phase 3: Enhance HTTP Layer (Day 2-3)

#### 3.1 Update App Factory (2 hours)

Update `serving/http/app.py`:
- Implement typed lifespan state (E5)
- Configure exception handlers (E6)
- Add OpenAPI customization (E9)
- Install correlation ID + timing middleware (E10)
- Add middleware hardening (GZip/CORS/TrustedHost) when configured (E11)
- Add concurrency guardrails + threadpool offload for blocking calls (E12)

#### 3.2 Implement API Versioning (1 hour)

Create versioned route structure:
- `serving/http/routes/v1/__init__.py`
- `serving/http/routes/v1/semantic.py`
- `serving/http/routes/v1/search.py`
- Update `serving/http/routes/__init__.py`

#### 3.3 Enhance Routes (2-3 hours)

Update all routes with:
- Typed dependencies (E1)
- Response models (E3)
- Background tasks (E7)
- Serving error types (E2)
- Optional API-key security dependency (E14)
- Optional streaming/export endpoints if needed (E13)

### Phase 4: Deprecation and Cleanup (Day 3-4)

#### 4.1 Create Deprecation Shims

Update `serving/db/pool.py` to thin re-export with deprecation warnings.

#### 4.2 Update Exports

Update `__init__.py` files to export new types.

#### 4.3 Update Tests

Update all serving tests to use new patterns.

### Phase 5: Documentation and Validation (Day 4-5)

#### 5.1 Run Quality Gates

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

#### 5.2 Update Documentation

- Update docstrings
- Create migration examples

---

## File-by-File Changes

### Files to Create

| File | Purpose |
|------|---------|
| `storage/gateway/pool.py` | `ReadPoolWarehouse` implementation (yields Warehouse) |
| `serving/http/state.py` | Typed state container |
| `serving/http/errors.py` | RFC 9457 Problem Details |
| `serving/http/dependencies.py` | Typed dependency definitions |
| `serving/http/middleware.py` | Correlation ID + timing middleware |
| `serving/http/routes/v1/__init__.py` | V1 router aggregation |
| `serving/http/routes/v1/semantic.py` | V1 semantic routes |
| `serving/http/routes/v1/search.py` | V1 search routes |

### Files to Modify

| File | Changes |
|------|---------|
| `storage/gateway/__init__.py` | Export `PoolConfig`, `ReadPoolWarehouse` |
| `serving/db/__init__.py` | Update exports |
| `serving/db/pool.py` | Convert to thin re-export |
| `serving/db/manager.py` | Use `ReadPoolWarehouse`, yield `Warehouse` (+ snapshot context cache) |
| `serving/semantic/kernel.py` | Use Warehouse/context pattern; avoid `duckdb_schema_exists` imports; push FTS helpers into storage |
| `serving/semantic/models.py` | Pydantic v2 features, validators |
| `serving/search/models.py` | Pydantic v2 features |
| `serving/http/app.py` | Typed lifespan, exception handlers, OpenAPI, middleware, concurrency guardrails |
| `serving/http/routes/__init__.py` | Router aggregation with versioning |

### Files Unchanged

| File | Reason |
|------|--------|
| `serving/settings.py` | Environment config is serving-specific |
| `serving/db/pointer.py` | Snapshot pointer is serving-specific |
| `serving/semantic/query_builder.py` | Filter building is correct design |
| `serving/semantic/registry.py` | View registry is serving-specific |
| `serving/semantic/inventory.py` | JSON parsing is serving-specific |
| `serving/contracts/*` | Contract validation is serving-specific |
| `serving/mcp/*` | MCP uses kernel directly |

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
from codeintel.storage.gateway.pool import PoolConfig, ReadPoolWarehouse

pool = ReadPoolWarehouse(db_path, PoolConfig(size=4))
with pool.acquire() as warehouse:
    result = warehouse.gateway.policy.execute_sql("SELECT 1").fetchone()
pool.close_gracefully()
```

### For Existing Code Using `db_manager.connect()`

**Before**:
```python
with db_manager.connect() as (con, pointer):
    ibis_con = ibis.duckdb.from_connection(con)
    backend = MinimalStorageGateway(con).policy
    result = backend.execute_sql("SELECT 1")
```

**After**:
```python
with db_manager.connect() as (warehouse, pointer):
    result = warehouse.gateway.policy.execute_sql("SELECT 1")
```

### For Existing HTTP Route Handlers

**Before**:
```python
@router.get("/views/{view_id}")
async def describe_view(
    view_id: str,
    kernel: SemanticQueryKernel = Depends(get_kernel),
) -> dict[str, object]:
    try:
        return kernel.describe(view_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
```

**After**:
```python
from codeintel.serving.http.dependencies import Kernel
from codeintel.serving.http.errors import ViewNotFoundError

@router.get(
    "/views/{view_id}",
    response_model=SemanticViewDetail,
    responses={404: {"model": ProblemDetail}},
)
async def describe_view(view_id: str, kernel: Kernel) -> SemanticViewDetail:
    """Describe a semantic view."""
    try:
        result = kernel.describe(view_id)
        return SemanticViewDetail.model_validate(result)
    except KeyError as exc:
        raise ViewNotFoundError(view_id) from exc
```

---

## Testing Strategy

### Unit Tests

1. **Pool Tests** (`tests/storage/gateway/test_pool.py`)
   - Test pool initialization
   - Test acquire/release lifecycle
   - Test graceful shutdown
   - Test concurrent access

2. **Error Tests** (`tests/serving/http/test_errors.py`)
   - Test Problem Details serialization
   - Test each error type
   - Test correlation ID generation

3. **Dependency Tests** (`tests/serving/http/test_dependencies.py`)
   - Test state extraction
   - Test kernel extraction
   - Test error cases

4. **Model Tests** (`tests/serving/semantic/test_models.py`)
   - Test validation
   - Test constrained types
   - Test cross-field validation

### Integration Tests

1. **HTTP Routes** (`tests/serving/test_semantic_http_routes.py`)
   - Full request/response cycle
   - Error response format
   - Background task execution

2. **API Versioning** (`tests/serving/http/test_versioning.py`)
   - V1 routes accessible
   - Root routes accessible
   - Same behavior

### Backwards Compatibility

- Old pool imports still work (with deprecation)
- Root routes mirror v1
- HTTPException still handled (converted to Problem Details)

---

## Rollback Plan

### If Rationalization Issues Arise

1. **Revert Phase 1**: Delete `storage/gateway/pool.py`, revert exports
2. **Revert Phase 2**: Restore `ServingDBManager` and kernel

### If Enhancement Issues Arise

1. **Revert HTTP changes**: Restore original routes
2. **Keep rationalization**: Warehouse/pool refactor is independent
3. **Remove versioning**: Flatten routes back to root

### Feature Flags

```python
import os

USE_WAREHOUSE_PATTERN = os.environ.get("CODEINTEL_USE_WAREHOUSE_PATTERN", "1") == "1"
USE_PROBLEM_DETAILS = os.environ.get("CODEINTEL_USE_PROBLEM_DETAILS", "1") == "1"
```

---

## Success Criteria

### Code Quality

- [ ] All tests pass
- [ ] No pyright/pyrefly errors
- [ ] No ruff lint issues
- [ ] Docstrings complete

### Performance

- [ ] Query latency unchanged (±5%)
- [ ] Memory usage reduced
- [ ] Pool acquisition < 1ms
- [ ] Background tasks don't block

### Architecture

- [ ] No direct `ibis.duckdb.from_connection()` calls in serving hot paths (use gateway/warehouse caching)
- [ ] No per-call `MinimalStorageGateway()` in hot paths (request-scoped Warehouse only)
- [ ] All pool management in storage
- [ ] Serving does not introduce `duckdb` imports outside `codeintel.storage`
- [ ] Blocking kernel calls do not run on the event loop (sync endpoints or threadpool offload)
- [ ] Typed dependencies throughout
- [ ] RFC 9457 errors on all paths
- [ ] Correlation ID available in error payloads, logs, and response headers

### Documentation

- [ ] OpenAPI schema complete
- [ ] Problem Details in schema
- [ ] Migration guide complete
- [ ] All routes documented

---

## Appendix: Metrics

### Lines of Code Impact

| Module | Before | After | Delta |
|--------|--------|-------|-------|
| `serving/db/pool.py` | 139 | 25 | -114 |
| `serving/db/manager.py` | 138 | 115 | -23 |
| `serving/semantic/kernel.py` | 546 | 520 | -26 |
| `serving/semantic/models.py` | 85 | 150 | +65 |
| `storage/gateway/pool.py` | 0 | 100 | +100 |
| `serving/http/state.py` | 0 | 35 | +35 |
| `serving/http/errors.py` | 0 | 130 | +130 |
| `serving/http/dependencies.py` | 0 | 45 | +45 |
| `serving/http/app.py` | 75 | 140 | +65 |
| `serving/http/routes/v1/*` | 0 | 200 | +200 |
| **Total** | **983** | **1460** | **+477** |

Note: LOC increase is due to:
- Explicit type definitions (+200)
- RFC 9457 implementation (+130)
- Comprehensive validation (+65)
- OpenAPI customization (+65)

This is expected and represents investment in type safety, error handling, and documentation.

### Gateway Instantiation Reduction

| Scenario | Before | After |
|----------|--------|-------|
| Single semantic query | 2 gateways | 0 (reuses pooled) |
| Search query | 2 gateways | 0 (reuses pooled) |
| Request with 3 queries | 6 gateways | 0 (reuses pooled) |

### Type Safety Improvement

| Metric | Before | After |
|--------|--------|-------|
| Typed dependencies | 0% | 100% |
| Response models | 0% | 100% |
| Error responses | Unstructured | RFC 9457 |
| State access | Untyped | Fully typed |

---

## Related Documents

- [SERVING_RATIONALIZATION_PLAN.md](./SERVING_RATIONALIZATION_PLAN.md) - Original rationalization plan
- [COMBINED_DECOMMISSIONING_PLAN.md](./COMBINED_DECOMMISSIONING_PLAN.md) - Overall decommissioning context
- [BUILD_CONSOLIDATION_AND_ENHANCEMENT_PLAN.md](./BUILD_CONSOLIDATION_AND_ENHANCEMENT_PLAN.md) - Build layer cleanup
- [FastAPI_advanced.md](../python_library_reference/FastAPI_advanced.md) - FastAPI advanced features reference
