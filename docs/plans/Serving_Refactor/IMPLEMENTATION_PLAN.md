# Serving Module Refactoring Implementation Plan

> **Document Version:** 1.0  
> **Created:** 2025-12-13  
> **Status:** Proposed  
> **Scope:** `src/codeintel/serving/`

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Phase 1: Quick Wins](#phase-1-quick-wins-low-risk-high-value)
4. [Phase 2: Transport Consolidation](#phase-2-transport-consolidation)
5. [Phase 3: Backend Consolidation](#phase-3-backend-consolidation)
6. [Phase 4: Layer Simplification](#phase-4-layer-simplification-long-term)
7. [Testing Strategy](#testing-strategy)
8. [Migration Checklist](#migration-checklist)
9. [Appendix: Code Patterns](#appendix-code-patterns)

---

## Executive Summary

### Goals

1. **Reduce code duplication** by ~2,000 lines (~30% of serving module)
2. **Improve maintainability** through common abstractions
3. **Harden the codebase** with consistent patterns
4. **Increase extensibility** by completing half-finished refactors

### Impact Summary

| Phase | Lines Reduced | Risk Level | Effort | Priority | Status |
|-------|---------------|------------|--------|----------|--------|
| Phase 1 | ~630 lines | Low | 2-3 days | High | ✅ Complete |
| Phase 2 | ~150 lines | Medium | 1-2 days | Medium | ✅ Complete |
| Phase 3 | ~960 lines | Medium-High | 4-5 days | Medium | Pending |
| Phase 4 | ~500 lines | High | 5+ days | Low | Pending |
| **Total** | **~2,240 lines** | - | **12-16 days** | - |

> **Phase 3/4 Updated:** Added new optimization items discovered during Phase 1/2 implementation.

---

## Current State Analysis

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Transport Layer                                                 │
│  - HTTP Routes (http/routes/*.py)                               │
│  - MCP Tools (mcp/tools.py)                                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  MCP/HTTP Backend Layer                                          │
│  - DuckDBBackend (mcp/backend.py)                               │
│  - HttpBackend (mcp/backend.py)                                 │
│  - ~1,200 lines, 20+ methods each                               │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  Service Layer                                                   │
│  - LocalQueryService (services/query_service.py)                │
│  - HttpQueryService (services/query_service.py)                 │
│  - Mixins: functions.py, profiles.py, subsystems.py, datasets.py│
│  - ~1,600 lines total                                           │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  Query Layer                                                     │
│  - DuckDBQueryService (backend/duckdb_service.py)               │
│  - FunctionQueryLayer, ProfileQueryLayer, etc.                  │
│  - ~2,200 lines total                                           │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  Repository Layer                                                │
│  - FunctionRepository, ModuleRepository, etc.                   │
│  - (in storage/repositories/)                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Files Affected

| File | Lines | Primary Issues |
|------|-------|----------------|
| `mcp/backend.py` | 1,188 | Duplicate method patterns |
| `services/functions.py` | 438 | Repeated conversion pattern |
| `services/profiles.py` | 160 | Repeated conversion pattern |
| `services/subsystems.py` | 327 | Repeated conversion pattern |
| `services/datasets.py` | 399 | Repeated conversion + duplicate function |
| `services/base.py` | 393 | Unused abstract classes |
| `services/transport.py` | 239 | Unused transport adapters |
| `services/http_transport.py` | 86 | Duplicate of transport.py |
| `types.py` | 683 | Protocol explosion |
| `bootstrap.py` | 863 | Overlapping entry points |
| `auto_pipeline.py` | 852 | Duplicate prereq functions |
| `backend/dataset_backend.py` | 347 | Source of duplicate function |

---

## Phase 1: Quick Wins (Low Risk, High Value)

### 1.1 Create Response Conversion Helper

**Priority:** P0 (Highest)  
**Estimated Effort:** 4 hours  
**Lines Saved:** ~200

#### Problem

Every service method repeats this pattern:

```python
raw_resp = self._call("method_name", lambda: self.query.functions.some_method(...))
if isinstance(raw_resp, dm.SomeResult):
    return raw_resp
if isinstance(raw_resp, SomeResponse):
    return raw_resp.to_domain()
return SomeResponse.model_validate(raw_resp).to_domain()
```

#### Solution

Create `services/conversion.py`:

```python
"""Response conversion utilities for domain/transport model interop."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codeintel.serving import domain_models as dm

D = TypeVar("D")  # Domain model type


@runtime_checkable
class HasToDomain(Protocol[D]):
    """Protocol for response models with to_domain() method."""
    
    def to_domain(self) -> D:
        """Convert response model to domain model."""
        ...

    @classmethod
    def model_validate(cls, obj: object) -> "HasToDomain[D]":
        """Validate and construct from arbitrary object."""
        ...


def to_domain_result(
    raw: object,
    domain_type: type[D],
    response_type: type[HasToDomain[D]],
) -> D:
    """
    Convert raw response to domain model with type coercion.
    
    This function handles three cases:
    1. raw is already the domain type → return as-is
    2. raw is the response type → call to_domain()
    3. raw is dict/other → validate as response, then to_domain()
    
    Parameters
    ----------
    raw
        Raw response from query layer or HTTP.
    domain_type
        Expected domain model type (e.g., dm.FunctionSummaryResult).
    response_type
        Pydantic response model type (e.g., FunctionSummaryResponse).
    
    Returns
    -------
    D
        Domain model instance.
    
    Examples
    --------
    >>> result = to_domain_result(raw, dm.FunctionSummaryResult, FunctionSummaryResponse)
    """
    if isinstance(raw, domain_type):
        return raw
    if isinstance(raw, response_type):
        return raw.to_domain()
    return response_type.model_validate(raw).to_domain()


__all__ = ["HasToDomain", "to_domain_result"]
```

#### Migration Steps

1. Create `services/conversion.py` with the helper
2. Update `services/functions.py` methods:

**Before:**
```python
def get_function_summary(self, ...) -> dm.FunctionSummaryResult:
    raw_resp = self._call(
        "get_function_summary",
        lambda: self.query.functions.get_function_summary(...),
    )
    if isinstance(raw_resp, dm.FunctionSummaryResult):
        return raw_resp
    if isinstance(raw_resp, FunctionSummaryResponse):
        return raw_resp.to_domain()
    return FunctionSummaryResponse.model_validate(raw_resp).to_domain()
```

**After:**
```python
def get_function_summary(self, ...) -> dm.FunctionSummaryResult:
    raw = self._call(
        "get_function_summary",
        lambda: self.query.functions.get_function_summary(...),
    )
    return to_domain_result(raw, dm.FunctionSummaryResult, FunctionSummaryResponse)
```

3. Repeat for `profiles.py`, `subsystems.py`, `datasets.py`
4. Run tests: `uv run pytest tests/serving/ -v`

---

### 1.2 Delete Duplicate `_normalize_validation_profile`

**Priority:** P0  
**Estimated Effort:** 15 minutes  
**Lines Saved:** ~15

#### Problem

Identical function defined in two files:
- `services/datasets.py:29-44`
- `backend/dataset_backend.py:125-141`

#### Solution

1. Keep the function in `backend/dataset_backend.py`
2. Import in `services/datasets.py`:

```python
from codeintel.serving.backend.dataset_backend import _normalize_validation_profile
```

3. Delete the duplicate definition from `services/datasets.py`

---

### 1.3 Merge `ensure_prereqs_for_http` and `ensure_prereqs_for_mcp`

**Priority:** P0  
**Estimated Effort:** 30 minutes  
**Lines Saved:** ~25

#### Problem

Both functions in `auto_pipeline.py` have identical implementations.

#### Solution

Replace both with a single function:

```python
def ensure_prereqs(
    *,
    op_id: str,
    config: ServingConfig,
    backend: QueryBackend,
    transport: Literal["http", "mcp"] = "http",
) -> HamiltonBuildResult | None:
    """
    Ensure prerequisites are run for an operation if needed.
    
    This function is called before serving HTTP requests or executing MCP tools.
    If auto-pipeline is enabled and no previous successful run exists, it will
    execute the necessary pipeline stages.
    
    Parameters
    ----------
    op_id
        Operation identifier.
    config
        Serving configuration.
    backend
        Query backend (must be DuckDBBackend for local_db mode).
    transport
        Transport type for logging ("http" or "mcp").
    
    Returns
    -------
    HamiltonBuildResult | None
        The build result if a run was executed, None if skipped.
    """
    should_run, gateway, skip_reason = should_run_auto_pipeline(config, backend)
    if not should_run or gateway is None:
        LOG.debug("auto_pipeline skipped (%s): %s", transport, skip_reason)
        return None
    
    if has_successful_prereq_run(gateway.runs, repo=config.repo, commit=config.commit, op_id=op_id):
        LOG.debug("auto_pipeline skipped (%s): prereqs already satisfied for %s", transport, op_id)
        return None
    
    return _run_prereqs_build(op_id=op_id, config=config, gateway=gateway)


# Backward-compatible aliases (deprecated)
def ensure_prereqs_for_http(
    *,
    op_id: str,
    config: ServingConfig,
    backend: QueryBackend,
) -> HamiltonBuildResult | None:
    """Ensure prerequisites for HTTP. Deprecated: use ensure_prereqs()."""
    return ensure_prereqs(op_id=op_id, config=config, backend=backend, transport="http")


def ensure_prereqs_for_mcp(
    *,
    op_id: str,
    config: ServingConfig,
    backend: QueryBackend,
) -> HamiltonBuildResult | None:
    """Ensure prerequisites for MCP. Deprecated: use ensure_prereqs()."""
    return ensure_prereqs(op_id=op_id, config=config, backend=backend, transport="mcp")
```

---

### 1.4 Complete or Remove `services/base.py`

**Priority:** P1  
**Estimated Effort:** 2-4 hours  
**Lines Saved/Used:** 393 lines

#### Problem

The file defines abstract base classes that are never inherited:

```python
class BaseFunctionQueries(ABC):
    """Abstract base class for function query operations."""
    # ... 80 lines of abstract methods
```

The docstring explicitly states these are not yet used.

#### Options

**Option A: Remove Until Needed (Recommended for Phase 1)**
- Delete `services/base.py` entirely
- Simpler codebase now
- Can be re-added when the full refactor happens

**Option B: Complete the Migration**
- Have `_FunctionQueryDelegates` inherit from `BaseFunctionQueries`
- Have `_HttpFunctionQueryMixin` inherit from `BaseFunctionQueries`
- Repeat for other domains

#### Recommendation

For Phase 1, choose **Option A** (remove). The abstract classes can be reintroduced in Phase 2/3 when the transport consolidation happens.

---

## Phase 2: Transport Consolidation

### 2.1 Merge Transport Handling

**Priority:** P1  
**Estimated Effort:** 1-2 days  
**Lines Saved:** ~150

#### Problem

Two modules do similar observability wrapping:
- `services/transport.py` - defines `LocalTransport`, `HttpTransport` (currently unused!)
- `services/http_transport.py` - defines `_HttpTransportMixin` (actually used)

#### Solution

1. **Keep `services/transport.py`** as the canonical location
2. **Migrate `_HttpTransportMixin` logic** into `HttpTransport`
3. **Delete `services/http_transport.py`**
4. **Update service mixins** to use `TransportAdapter`

#### New Transport Architecture

```python
# services/transport.py (consolidated)

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

from codeintel.serving.backend import BackendLimits
from codeintel.serving.services.observability import ServiceCallContext, _observe_call

if TYPE_CHECKING:
    from collections.abc import Callable
    from codeintel.serving.backend.query_api import DuckDBQueryApi
    from codeintel.serving.services.observability import ServiceObservability

T = TypeVar("T")


class TransportAdapter(ABC):
    """Abstract transport adapter for query execution."""
    
    @property
    @abstractmethod
    def limits(self) -> BackendLimits:
        """Return backend limits configuration."""
        ...
    
    @abstractmethod
    def call(
        self,
        operation: str,
        executor: Callable[[], T],
        *,
        dataset: str | None = None,
        schema_version: str | None = None,
        retries: int | None = None,
    ) -> T:
        """Execute a query operation through the transport."""
        ...


@dataclass
class LocalTransport(TransportAdapter):
    """Transport adapter for local DuckDB queries."""
    
    query: DuckDBQueryApi
    observability: ServiceObservability | None = None
    _limits: BackendLimits = field(default_factory=BackendLimits)
    
    @property
    def limits(self) -> BackendLimits:
        return getattr(self.query, "limits", self._limits)
    
    def call(
        self,
        operation: str,
        executor: Callable[[], T],
        *,
        dataset: str | None = None,
        schema_version: str | None = None,
        retries: int | None = None,
    ) -> T:
        return _observe_call(
            self.observability,
            transport="local",
            name=operation,
            context=ServiceCallContext(
                dataset=dataset,
                schema_version=schema_version,
                retries=retries,
            ),
            func=executor,
        )


@dataclass
class HttpTransport(TransportAdapter):
    """Transport adapter for HTTP API queries."""
    
    request_json: Callable[[str, dict[str, object]], object]
    _limits: BackendLimits = field(default_factory=BackendLimits)
    observability: ServiceObservability | None = None
    
    @property
    def limits(self) -> BackendLimits:
        return self._limits
    
    def call(
        self,
        operation: str,
        executor: Callable[[], T],
        *,
        dataset: str | None = None,
        schema_version: str | None = None,
        retries: int | None = None,
    ) -> T:
        # Get retry info from backend if available
        backend = getattr(self.request_json, "__self__", None)
        actual_retries = retries or getattr(backend, "last_retry_attempts", None)
        
        result = _observe_call(
            self.observability,
            transport="http",
            name=operation,
            context=ServiceCallContext(
                dataset=dataset,
                schema_version=schema_version,
                retries=actual_retries if isinstance(actual_retries, int) else None,
            ),
            func=executor,
        )
        
        # Record retry metrics if applicable
        if actual_retries and self.observability is not None:
            from codeintel.serving.services.observability import ServiceCallMetrics
            self.observability.record(
                ServiceCallMetrics(
                    name=f"{operation}_retries",
                    transport="http",
                    duration_ms=0.0,
                    dataset=dataset,
                    retries=actual_retries,
                    schema_version=schema_version,
                )
            )
        
        return result
```

---

## Phase 3: Backend Consolidation

> **Note:** Updated based on learnings from Phase 1/2 implementation.

### 3.0 Add `to_response_result()` Helper (NEW)

**Priority:** P1 (enables 3.1)  
**Estimated Effort:** 1-2 hours  
**Lines Saved:** ~50 (from backend methods)

#### Problem

The MCP backend methods have the inverse conversion pattern of service methods:
- Service methods: raw → domain (handled by `to_domain_result()`)
- Backend methods: domain → response (needs `to_response_result()`)

#### Solution

Add to `services/conversion.py`:

```python
@runtime_checkable
class HasFromDomain(Protocol[D_co]):
    """Protocol for response models with from_domain class method."""
    
    @classmethod
    def from_domain(cls, domain: D_co) -> HasFromDomain[D_co]:
        """Create response model from domain model."""
        ...


def to_response_result(
    raw: object,
    response_type: type[HasFromDomain[D]],
) -> HasFromDomain[D]:
    """
    Convert domain model to response model with type coercion.
    
    This function handles two cases:
    1. raw is already the response type → return as-is
    2. raw is a domain model → call from_domain()
    
    Used by MCP backends to ensure consistent response serialization.
    """
    if isinstance(raw, response_type):
        return raw
    return response_type.from_domain(raw)
```

This creates symmetry with `to_domain_result()` and enables cleaner backend code.

---

### 3.1 Create `BackendDispatchMixin`

**Priority:** P2  
**Estimated Effort:** 3-4 days  
**Lines Saved:** ~600

#### Problem

`DuckDBBackend` and `HttpBackend` in `mcp/backend.py` have 20+ methods each following identical patterns:

```python
# DuckDBBackend pattern
def get_function_summary(self, ...) -> FunctionSummaryResponse:
    scope_payload = scope if isinstance(scope, GraphScopePayload) else None
    try:
        domain_result = self.service.get_function_summary(
            urn=urn, goid_h128=goid_h128, rel_path=rel_path,
            qualname=qualname, scope=scope_payload,
        )
    except ProblemError as exc:
        raise errors.McpError(exc.detail) from exc
    return FunctionSummaryResponse.from_domain(domain_result)

# HttpBackend pattern
def get_function_summary(self, ...) -> FunctionSummaryResponse:
    result = self.service.get_function_summary(
        urn=urn, goid_h128=goid_h128, rel_path=rel_path,
        qualname=qualname, scope=scope if isinstance(scope, GraphScopePayload) else None,
    )
    if isinstance(result, FunctionSummaryResponse):
        return result
    return FunctionSummaryResponse.from_domain(result)
```

#### Solution

Create a common dispatch mechanism:

```python
# mcp/backend_base.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar, Generic

from codeintel.serving.mcp import errors
from codeintel.serving.services.errors import ProblemError

if TYPE_CHECKING:
    from collections.abc import Callable
    from codeintel.serving.services.query_service import QueryService

D = TypeVar("D")  # Domain model
R = TypeVar("R")  # Response model


class HasFromDomain(Generic[D]):
    """Protocol for response models with from_domain class method."""
    
    @classmethod
    def from_domain(cls, domain: D) -> "HasFromDomain[D]":
        ...


class BackendDispatchMixin(ABC):
    """Mixin providing common dispatch pattern for backend methods."""
    
    @property
    @abstractmethod
    def service(self) -> QueryService:
        """Return the underlying query service."""
        ...
    
    @property
    @abstractmethod
    def is_local(self) -> bool:
        """Return True if this is a local (DuckDB) backend."""
        ...
    
    def _dispatch(
        self,
        method_name: str,
        response_type: type[R],
        *args,
        **kwargs,
    ) -> R:
        """
        Dispatch a method call to the service with error handling and conversion.
        
        Parameters
        ----------
        method_name
            Name of the method on self.service to call.
        response_type
            Pydantic response model type for conversion.
        *args, **kwargs
            Arguments to pass to the service method.
        
        Returns
        -------
        R
            Response model instance.
        """
        method = getattr(self.service, method_name)
        
        if self.is_local:
            try:
                domain_result = method(*args, **kwargs)
            except ProblemError as exc:
                raise errors.McpError(exc.detail) from exc
            return response_type.from_domain(domain_result)
        else:
            result = method(*args, **kwargs)
            if isinstance(result, response_type):
                return result
            return response_type.from_domain(result)
    
    def _normalize_scope(self, scope: object | None) -> GraphScopePayload | None:
        """Convert scope to GraphScopePayload if applicable."""
        from codeintel.serving.mcp.models import GraphScopePayload
        return scope if isinstance(scope, GraphScopePayload) else None
```

#### Refactored Backend Methods

```python
# mcp/backend.py (refactored)

@dataclass
class DuckDBBackend(BackendDispatchMixin, DatasetBackendMixin):
    """DuckDB-backed implementation of QueryBackend."""
    
    service: QueryService
    gateway: StorageGateway
    # ... other fields ...
    
    @property
    def is_local(self) -> bool:
        return True
    
    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: object | None = None,
    ) -> FunctionSummaryResponse:
        _require_identifier(urn=urn, goid_h128=goid_h128, rel_path=rel_path)
        return self._dispatch(
            "get_function_summary",
            FunctionSummaryResponse,
            urn=urn,
            goid_h128=goid_h128,
            rel_path=rel_path,
            qualname=qualname,
            scope=self._normalize_scope(scope),
        )
    
    # ... other methods use _dispatch similarly ...


@dataclass
class HttpBackend(BackendDispatchMixin, DatasetBackendMixin):
    """HTTP-backed QueryBackend."""
    
    # ... fields ...
    
    @property
    def is_local(self) -> bool:
        return False
    
    # Methods now use _dispatch, identical to DuckDBBackend!
```

---

### 3.2 Consolidate Protocols in `types.py`

**Priority:** P2  
**Estimated Effort:** 1 day  
**Lines Saved:** ~250

#### Problem

15+ protocols with significant overlap:

| Current | Methods |
|---------|---------|
| `QueryBackendProtocol` | repo, commit |
| `QueryServiceProtocol` | repo, commit |
| `RepositoryProtocol` | repo, commit |
| `FunctionBackendProtocol` | 7 methods |
| `FunctionQueryProtocol` | 7 methods (same!) |
| ... | ... |

#### Solution: Domain-Based Protocols

```python
# types.py (refactored)

from __future__ import annotations
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.serving import domain_models as dm
    from codeintel.serving.mcp.models import GraphScopePayload


class RepoCommitProtocol(Protocol):
    """Base protocol for repo/commit identification."""
    
    @property
    def repo(self) -> str:
        """Repository identifier."""
        ...
    
    @property
    def commit(self) -> str:
        """Commit hash."""
        ...


class FunctionQueryable(Protocol):
    """Unified protocol for function query operations."""
    
    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.FunctionSummaryResult:
        ...
    
    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphScopePayload | None = None,
    ) -> dm.HighRiskFunctionsResult:
        ...
    
    # ... other function methods ...


class ProfileQueryable(Protocol):
    """Unified protocol for profile query operations."""
    
    def get_function_profile(self, *, goid_h128: int) -> dm.FunctionProfileResult:
        ...
    
    def get_file_profile(self, *, rel_path: str) -> dm.FileProfileResult:
        ...
    
    def get_module_profile(self, *, module: str) -> dm.ModuleProfileResult:
        ...
    
    # ... other profile methods ...


class SubsystemQueryable(Protocol):
    """Unified protocol for subsystem query operations."""
    # ... subsystem methods ...


class DatasetQueryable(Protocol):
    """Unified protocol for dataset query operations."""
    # ... dataset methods ...


class QueryService(
    FunctionQueryable,
    ProfileQueryable,
    SubsystemQueryable,
    DatasetQueryable,
    Protocol,
):
    """Composite query service interface."""


class QueryBackend(
    FunctionQueryable,
    ProfileQueryable,
    SubsystemQueryable,
    DatasetQueryable,
    RepoCommitProtocol,
    Protocol,
):
    """Composite backend interface for MCP tools."""
    
    service: QueryService
```

This reduces from 15+ protocols to 6 core protocols.

---

### 3.3 Centralize Scope Normalization (NEW)

**Priority:** P2  
**Estimated Effort:** 2-3 hours  
**Lines Saved:** ~30

#### Problem

Scope normalization appears in multiple places:
- `parse_graph_scope()` in `mcp/models.py`
- Inline `isinstance(scope, GraphScopePayload)` checks in backend methods
- `_normalize_scope()` proposed for `BackendDispatchMixin`

#### Solution

Consolidate into a single utility in `mcp/models.py`:

```python
def normalize_scope(scope: object | None) -> GraphScopePayload | None:
    """
    Normalize scope parameter to GraphScopePayload or None.
    
    Parameters
    ----------
    scope
        Raw scope value from API calls.
    
    Returns
    -------
    GraphScopePayload | None
        Normalized scope for backend operations.
    """
    if scope is None:
        return None
    if isinstance(scope, GraphScopePayload):
        return scope
    if isinstance(scope, dict):
        return GraphScopePayload.model_validate(scope)
    return None
```

Update all callers to use this single function.

---

### 3.4 HTTP Mixin Limit/Error Pattern Consolidation (NEW)

**Priority:** P2  
**Estimated Effort:** 3-4 hours  
**Lines Saved:** ~80

#### Problem

HTTP mixins (`_HttpFunctionQueryMixin`, `_HttpSubsystemQueryMixin`, etc.) repeat this pattern:

```python
def some_method(self, *, limit: int | None = None, ...) -> dm.SomeResult:
    def _run() -> SomeResponse:
        applied_limit = self.limits.default_limit if limit is None else limit
        clamp = clamp_limit(
            applied_limit,
            default=applied_limit,
            max_limit=self.limits.max_rows_per_call,
        )
        if clamp.has_error:
            return SomeResponse(items=[], meta=ResponseMeta())  # Empty response
        # ... actual HTTP call ...
```

#### Solution

Create a helper that encapsulates this pattern:

```python
# services/http_helpers.py

@dataclass
class ClampedCall(Generic[T]):
    """Result of a clamped HTTP call."""
    
    result: T
    clamped_limit: ClampResult
    clamped_offset: ClampResult | None = None


def with_clamped_limits(
    limits: BackendLimits,
    limit: int | None,
    offset: int | None = None,
    *,
    empty_response_factory: Callable[[], T],
) -> tuple[int, int, list[dm.Message]] | T:
    """
    Apply limit/offset clamping and return early if errors.
    
    Returns tuple of (applied_limit, applied_offset, messages) if valid,
    or the empty response if clamping failed.
    """
    applied_limit = limits.default_limit if limit is None else limit
    clamp = clamp_limit(applied_limit, default=applied_limit, max_limit=limits.max_rows_per_call)
    
    messages = list(clamp.messages)
    applied_offset = 0
    
    if offset is not None:
        offset_clamp = clamp_offset(offset)
        messages.extend(offset_clamp.messages)
        applied_offset = offset_clamp.applied
        if offset_clamp.has_error:
            return empty_response_factory()
    
    if clamp.has_error:
        return empty_response_factory()
    
    return (clamp.applied, applied_offset, messages)
```

---

## Phase 4: Layer Simplification (Long-term)

> **Note:** Updated based on learnings from Phase 1/2 implementation.

### 4.1 Simplify Bootstrap Entry Points

**Priority:** P3  
**Estimated Effort:** 2 days  
**Lines Saved:** ~200

#### Problem

7+ overlapping entry points in `bootstrap.py`:
- `build_service_stack()`
- `build_backend_resource()`
- `build_service_from_config()`
- `build_local_query_service()`
- `build_http_query_service()`
- `_build_local_resource()`
- `_build_remote_resource()`

#### Solution

Consolidate to 2 main entry points:

```python
# bootstrap.py (simplified)

@dataclass
class ServingStack:
    """Complete serving stack with lifecycle management."""
    
    service: QueryService
    backend: QueryBackend | None
    context: BackendContext | None
    close: Callable[[], None]


def build_serving_stack(
    config: ServingConfig,
    *,
    gateway: StorageGateway | None = None,
    http_client: httpx.Client | None = None,
    options: BootstrapOptions | None = None,
    include_backend: bool = False,
) -> ServingStack:
    """
    Build a complete serving stack from configuration.
    
    This is the primary entry point for constructing the serving layer.
    
    Parameters
    ----------
    config
        Serving configuration.
    gateway
        StorageGateway for local_db mode.
    http_client
        HTTP client for remote_api mode.
    options
        Optional bootstrap configuration.
    include_backend
        If True, also construct MCP backend.
    
    Returns
    -------
    ServingStack
        Complete serving stack ready for use.
    """
    if config.mode == "local_db":
        return _build_local_stack(config, gateway=gateway, options=options, include_backend=include_backend)
    elif config.mode == "remote_api":
        return _build_remote_stack(config, http_client=http_client, options=options, include_backend=include_backend)
    else:
        raise ValueError(f"Unsupported mode: {config.mode}")
```

### 4.2 Canonical Import Documentation (NEW)

**Priority:** P2  
**Estimated Effort:** 2-3 hours  
**Lines Saved:** N/A (prevents import errors)

#### Problem

During Phase 1/2 implementation, discovered that test files were importing `BackendResource` from `codeintel.serving.http.fastapi` instead of the canonical location `codeintel.serving.bootstrap`. This is a documentation/discovery issue.

#### Solution

1. **Update module docstrings** to specify canonical import locations
2. **Add re-exports** to convenient locations with deprecation warnings:

```python
# http/fastapi.py - add explicit note
"""
FastAPI server exposing MCP-aligned queries over DuckDB.

Note: Import ``BackendResource`` from ``codeintel.serving.bootstrap``,
not from this module.
"""
```

3. **Add to `services/__init__.py`** (already done in Phase 1):
   - `BackendResource` is now exported from services for convenience

4. **Update test fixtures** to use canonical imports (completed in Phase 1/2)

---

### 4.3 Transport Adapter Migration (NEW)

**Priority:** P3  
**Estimated Effort:** 1-2 days  
**Lines Saved:** ~100

#### Problem

After Phase 2 consolidation, we have both:
- `LocalTransport` / `HttpTransport` classes in `transport.py`
- `_HttpTransportMixin` class also in `transport.py`

The mixin is currently used by HTTP service classes, while `HttpTransport` is unused.

#### Options

**Option A: Migrate to Transport Adapters**
- Update HTTP mixins to use `HttpTransport.call()` instead of `_http_call()`
- Deprecate `_HttpTransportMixin`
- Pro: Cleaner abstraction
- Con: Larger change

**Option B: Keep Both, Document Purpose**
- `_HttpTransportMixin`: For service mixins that inherit from it
- `HttpTransport`: For composition-based usage
- Pro: No changes needed
- Con: Conceptual overlap

#### Recommendation

Choose Option B for now. The mixins work well for the current architecture. Consider Option A when doing a larger service layer refactor.

---

### 4.4 Consider Layer Collapse

**Priority:** P4 (Future)  
**Estimated Effort:** 5+ days  
**Risk:** High

#### Analysis

Currently, MCP Backend → Service → Query Layer all do thin wrapping.

**Potential Collapse:**
- MCP Backend could directly use Query Layer
- Service layer could be absorbed

**Recommendation:** Defer this to a future major version. The current three-layer design provides good separation of concerns, even if verbose.

---

### 4.5 Dead Code Audit (NEW)

**Priority:** P3  
**Estimated Effort:** 2-3 hours  
**Lines Saved:** Unknown

#### Problem

During Phase 1 implementation, discovered that `_normalize_validation_profile` in `services/datasets.py` was completely unused. There may be other dead code in the serving module.

#### Solution

Run a dead code audit:

```bash
uv run vulture src/codeintel/serving/ --min-confidence 90
```

Review results and remove confirmed dead code. Candidates to check:
- Unused imports (already handled by ruff)
- Functions defined but never called
- Classes defined but never instantiated
- Protocol methods never implemented

---

## Testing Strategy

### Unit Test Updates

For each phase, update corresponding tests:

| Phase | Test Files |
|-------|------------|
| 1.1 | `tests/serving/services/test_conversion.py` (new) |
| 1.2 | No test changes needed |
| 1.3 | `tests/serving/test_auto_pipeline.py` |
| 1.4 | Delete `tests/serving/services/test_base.py` if exists |
| 2 | `tests/serving/services/test_transport.py` |
| 3 | `tests/serving/mcp/test_backend.py` |

### Integration Tests

Run full integration suite after each phase:

```bash
uv run pytest tests/serving/ -v --tb=short
uv run pytest tests/mcp/ -v --tb=short
```

### Type Checking

After each change:

```bash
uv run pyright src/codeintel/serving/
uv run pyrefly check
uv run ruff check src/codeintel/serving/ --fix
```

---

## Migration Checklist

### Phase 1 Checklist

- [x] Create `services/conversion.py` with `to_domain_result()`
- [x] Update `services/functions.py` to use `to_domain_result()`
- [x] Update `services/profiles.py` to use `to_domain_result()`
- [x] Update `services/subsystems.py` to use `to_domain_result()`
- [x] Update `services/datasets.py` to use `to_domain_result()`
- [x] Delete duplicate `_normalize_validation_profile` from `services/datasets.py`
- [x] Create unified `ensure_prereqs()` in `auto_pipeline.py`
- [x] Deprecate `ensure_prereqs_for_http()` and `ensure_prereqs_for_mcp()`
- [x] Delete or document `services/base.py`
- [x] Run tests, type checking, and linting
- [x] Update `__all__` exports

### Phase 2 Checklist

- [x] Consolidate `transport.py` and `http_transport.py`
- [x] Update service mixins to use `_HttpTransportMixin` from `transport.py`
- [x] Delete `services/http_transport.py`
- [x] Run tests, type checking, and linting

### Phase 3 Checklist

- [ ] Add `HasFromDomain` protocol and `to_response_result()` to `services/conversion.py`
- [ ] Create `mcp/backend_base.py` with `BackendDispatchMixin`
- [ ] Refactor `DuckDBBackend` to use `_dispatch()`
- [ ] Refactor `HttpBackend` to use `_dispatch()`
- [ ] Consolidate protocols in `types.py`
- [ ] Create `normalize_scope()` utility and update callers
- [ ] Create HTTP limit/error helpers (optional, can defer)
- [ ] Run tests, type checking, and linting

### Phase 4 Checklist

- [ ] Document canonical import locations in module docstrings
- [ ] Run dead code audit with vulture
- [ ] Remove confirmed dead code
- [ ] Consolidate bootstrap entry points
- [ ] Update all callers to new entry points
- [ ] Deprecate old entry points
- [ ] Run full test suite

---

## Appendix: Code Patterns

### Pattern A: Response Conversion (Current)

```python
# ~40 occurrences of this pattern
raw_resp = self._call("method_name", lambda: self.query.functions.method(...))
if isinstance(raw_resp, dm.DomainResult):
    return raw_resp
if isinstance(raw_resp, ResponseModel):
    return raw_resp.to_domain()
return ResponseModel.model_validate(raw_resp).to_domain()
```

### Pattern B: Response Conversion (After Phase 1)

```python
# Single line with helper
raw = self._call("method_name", lambda: self.query.functions.method(...))
return to_domain_result(raw, dm.DomainResult, ResponseModel)
```

### Pattern C: Backend Method (Current)

```python
# DuckDBBackend (~20 methods)
def get_something(self, ...) -> SomeResponse:
    try:
        domain = self.service.get_something(...)
    except ProblemError as exc:
        raise errors.McpError(exc.detail) from exc
    return SomeResponse.from_domain(domain)
```

### Pattern D: Backend Method (After Phase 3)

```python
# Both backends (~2 lines per method)
def get_something(self, ...) -> SomeResponse:
    return self._dispatch("get_something", SomeResponse, ...)
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-13 | AI Assistant | Initial comprehensive plan |
| 1.1 | 2025-12-13 | AI Assistant | Completed Phase 1 and Phase 2 implementation |
| 1.2 | 2025-12-13 | AI Assistant | Updated Phase 3/4 with learnings from implementation: added `to_response_result()`, scope normalization, HTTP helpers, canonical imports, transport migration options, dead code audit |

